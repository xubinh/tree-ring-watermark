import copy
import json
import random
from typing import Any, Mapping, Optional

import numpy as np
import scipy
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageFilter
from torchvision import transforms

from inverse_stable_diffusion import InversableStableDiffusionPipeline


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed: int):
    r"""手动设置大部分常用部件的 seed"""

    random.seed(seed + 0)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)


def transform_image(image: torch.FloatTensor, target_size=512) -> torch.FloatTensor:
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )

    image = transform(image)

    image = 2.0 * image - 1.0  # type: ignore

    return image


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def distort_image(
    img1, img2, seed, args
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """根据命令行参数中指定的攻击方法对图像进行处理. 如果没有指定任何攻击方法则不进行任何处理"""

    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(
            img1.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(
            img2.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img2)

    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2  # type: ignore


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    """利用 CLIP 计算所有图像关于给定提示词的相似度"""

    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        # 形状: (num_images, embedding_dim)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    r"""获取训练集 `dataset`, 以及提示词的 Python 字典的键 `prompt_key`"""

    if "laion" in args.dataset:
        dataset_dict = load_dataset(args.dataset)

        assert isinstance(dataset_dict, DatasetDict)

        dataset = dataset_dict["train"]
        prompt_key = "TEXT"

    elif "coco" in args.dataset:
        # 这里需要手动下载, 见 README.md
        with open("fid_outputs/coco/meta_data.json") as f:
            dataset = json.load(f)
            dataset = dataset["annotations"]
            prompt_key = "caption"

    else:
        dataset_dict = load_dataset(args.dataset)

        assert isinstance(dataset_dict, DatasetDict)

        dataset = dataset_dict["test"]
        prompt_key = "Prompt"

    return dataset, prompt_key


def get_circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    """返回一个边长为 `size` 的布尔方阵, 其中距离方阵中心点小于等于 r 的点置为 True, 作为水印掩码

    见 <https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3>

    """

    # 方阵中心点
    x0 = y0 = size // 2

    # 可选偏移
    x0 += x_offset
    y0 += y_offset

    # 产生开放网格，适合于创建稀疏的坐标网格
    y, x = np.ogrid[:size, :size]

    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2


def get_watermarking_masks(shape_of_latents: torch.Size, args, device):
    """生成一个水印掩码 Tensor 方阵, 其中水印区域置 1, 其余置 0"""

    watermarking_masks = torch.zeros(shape_of_latents, dtype=torch.bool).to(device)

    # 默认
    if args.w_mask_shape == "circle":
        circle_mask = get_circle_mask(shape_of_latents[-1], r=args.w_radius)
        circle_mask = torch.tensor(circle_mask).to(device)

        if args.w_channel >= 0:
            watermarking_masks[:, args.w_channel] = circle_mask

        else:
            watermarking_masks[:, :] = circle_mask

    elif args.w_mask_shape == "square":
        anchor_p = shape_of_latents[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_masks[
                :,
                :,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
        else:
            watermarking_masks[
                :,
                args.w_channel,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True

    elif args.w_mask_shape == "no":
        pass

    else:
        raise NotImplementedError(f"w_mask_shape: {args.w_mask_shape}")

    return watermarking_masks


def get_watermarked_fourier_latents(
    pipe: InversableStableDiffusionPipeline, batch_size, args, device
):
    """生成一个随机的 (有可能是傅里叶空间中的) 树环水印 latent"""

    set_random_seed(args.w_seed)

    initial_latents = pipe.get_random_latents(batch_size=batch_size)

    # 默认
    if args.w_pattern == "ring":
        watermarked_fourier_latents: torch.Tensor = torch.fft.fftshift(
            torch.fft.fft2(initial_latents), dim=(-1, -2)
        )

        copy_of_watermarked_fourier_latents = copy.deepcopy(watermarked_fourier_latents)

        for current_radius in range(args.w_radius, 0, -1):
            circle_mask = get_circle_mask(initial_latents.shape[-1], r=current_radius)
            circle_mask = torch.tensor(circle_mask).to(device)

            for current_channel_index in range(watermarked_fourier_latents.shape[1]):
                chosen_value = copy_of_watermarked_fourier_latents[
                    0, current_channel_index, 0, current_radius
                ].item()

                watermarked_fourier_latents[:, current_channel_index, circle_mask] = (
                    chosen_value
                )

    elif "seed_ring" in args.w_pattern:
        watermarked_fourier_latents = initial_latents

        copy_of_watermarked_fourier_latents = copy.deepcopy(watermarked_fourier_latents)

        for current_radius in range(args.w_radius, 0, -1):
            circle_mask = get_circle_mask(initial_latents.shape[-1], r=current_radius)
            circle_mask = torch.tensor(circle_mask).to(device)

            for current_channel_index in range(watermarked_fourier_latents.shape[1]):
                chosen_value = copy_of_watermarked_fourier_latents[
                    0, current_channel_index, 0, current_radius
                ].item()

                watermarked_fourier_latents[:, current_channel_index, circle_mask] = (
                    chosen_value
                )

    elif "zeros" in args.w_pattern:
        watermarked_fourier_latents = (
            torch.fft.fftshift(torch.fft.fft2(initial_latents), dim=(-1, -2)) * 0
        )

    elif "seed_zeros" in args.w_pattern:
        watermarked_fourier_latents = initial_latents * 0

    elif "rand" in args.w_pattern:
        watermarked_fourier_latents = torch.fft.fftshift(
            torch.fft.fft2(initial_latents), dim=(-1, -2)
        )
        watermarked_fourier_latents[:] = watermarked_fourier_latents[0]

    elif "seed_rand" in args.w_pattern:
        watermarked_fourier_latents = initial_latents

    elif "const" in args.w_pattern:
        watermarked_fourier_latents = (
            torch.fft.fftshift(torch.fft.fft2(initial_latents), dim=(-1, -2)) * 0
        )
        watermarked_fourier_latents += args.w_pattern_const

    return watermarked_fourier_latents


def inject_watermark(
    initial_latents_to_be_watermarked,
    watermarking_masks,
    watermarked_fourier_latents,
    args,
):
    r"""将 `watermarked_fourier_latents` 中对应于 `watermarking_mask` 中置 1 的部分复制到 `initial_latents_to_be_watermarked` 中"""

    fourier_initial_latents_to_be_watermarked = torch.fft.fftshift(
        torch.fft.fft2(initial_latents_to_be_watermarked), dim=(-1, -2)
    )

    # 默认
    if args.w_injection == "complex":
        fourier_initial_latents_to_be_watermarked[watermarking_masks] = (
            watermarked_fourier_latents[watermarking_masks].clone()
        )

    elif args.w_injection == "seed":
        initial_latents_to_be_watermarked[watermarking_masks] = (
            watermarked_fourier_latents[watermarking_masks].clone()
        )
        return initial_latents_to_be_watermarked

    else:
        NotImplementedError(f"w_injection: {args.w_injection}")

    # 再转回时域
    watermarked_initial_latents = torch.fft.ifft2(
        torch.fft.ifftshift(fourier_initial_latents_to_be_watermarked, dim=(-1, -2))
    ).real  # 注意到这里直接截断为了实部, 虚部被丢掉了

    return watermarked_initial_latents


def get_metrics(
    reversed_attacked_clean_initial_latents,
    reversed_attacked_watermarked_initial_latents,
    watermarking_masks,
    watermarked_fourier_latents,
    args,
):
    """计算经过攻击与 DDIM inversion 之后的初始噪声中的水印部分与原始水印之间的距离 (默认使用 MAE, 即平均绝对误差 (Mean absolute error))"""

    # `w_measurement` 默认为 "l1_complex"
    if "complex" in args.w_measurement:
        reversed_attacked_clean_fourier_initial_latents = torch.fft.fftshift(
            torch.fft.fft2(reversed_attacked_clean_initial_latents), dim=(-1, -2)
        )
        reversed_attacked_watermarked_fourier_initial_latents = torch.fft.fftshift(
            torch.fft.fft2(reversed_attacked_watermarked_initial_latents), dim=(-1, -2)
        )

    elif "seed" in args.w_measurement:
        reversed_attacked_clean_fourier_initial_latents = (
            reversed_attacked_clean_initial_latents
        )
        reversed_attacked_watermarked_fourier_initial_latents = (
            reversed_attacked_watermarked_initial_latents
        )

    else:
        NotImplementedError(f"w_measurement: {args.w_measurement}")

    if "l1" in args.w_measurement:
        l1_norm_of_clean_images = (
            torch.abs(
                reversed_attacked_clean_fourier_initial_latents[watermarking_masks]
                - watermarked_fourier_latents[watermarking_masks]
            )
            .mean()
            .item()
        )

        l1_norm_of_watermarked_images = (
            torch.abs(
                reversed_attacked_watermarked_fourier_initial_latents[
                    watermarking_masks
                ]
                - watermarked_fourier_latents[watermarking_masks]
            )
            .mean()
            .item()
        )

        metric_of_clean_images = l1_norm_of_clean_images
        metric_of_watermarked_images = l1_norm_of_watermarked_images

    else:
        NotImplementedError(f"w_measurement: {args.w_measurement}")

    return metric_of_clean_images, metric_of_watermarked_images


def get_p_value(
    reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
):
    # assume it's Fourier space wm
    reversed_attacked_clean_fourier_initial_latents = torch.fft.fftshift(
        torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2)
    )[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(
        torch.fft.fft2(reversed_latents_w), dim=(-1, -2)
    )[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])

    # no_w
    reversed_attacked_clean_fourier_initial_latents = torch.concatenate(
        [
            reversed_attacked_clean_fourier_initial_latents.real,
            reversed_attacked_clean_fourier_initial_latents.imag,
        ]
    )
    sigma_no_w = reversed_attacked_clean_fourier_initial_latents.std()
    lambda_no_w = (target_patch**2 / sigma_no_w**2).sum().item()
    x_no_w = (
        (
            (
                (reversed_attacked_clean_fourier_initial_latents - target_patch)
                / sigma_no_w
            )
            ** 2
        )
        .sum()
        .item()
    )
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate(
        [reversed_latents_w_fft.real, reversed_latents_w_fft.imag]
    )
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch**2 / sigma_w**2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
