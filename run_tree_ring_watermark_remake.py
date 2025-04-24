import argparse
import copy
import time
from datetime import datetime

import numpy as np
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from sklearn import metrics
from tqdm import tqdm

import optim_utils
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from modified_stable_diffusion import ModifiedStableDiffusionPipelineOutput

BATCH_SIZE: int = 1


def get_current_timestamp() -> str:
    r"""返回当前系统时间的字符串表示，格式为 'YYYY-MM-DD HH:MM:SS.mmmmmm'"""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def log_and_print(
    msg: str,
    log_file_path: str = "log.txt",
) -> None:
    r"""先输出日志, 再打印到终端"""

    msg = f"[{get_current_timestamp()}] {msg}"

    with open(log_file_path, "a", encoding="utf8") as file:
        file.write(msg + "\n")

    print(f"\033[37m{msg}\033[0m", flush=True)


def main(args):
    log_and_print(f"💬 Mission {args.run_name} started...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_and_print("💬 Loading scheduler...")

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id,  # 默认为 `stabilityai/stable-diffusion-2-1-base`
        subfolder="scheduler",
        local_files_only=True,
    )

    log_and_print("💬 Loading model...")

    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
        local_files_only=True,
    )
    assert isinstance(pipe, InversableStableDiffusionPipeline)

    log_and_print("💬 Moving model to GPU...")

    pipe = pipe.to(device)

    log_and_print("💬 Loading dataset...")

    dataset, prompt_key = optim_utils.get_dataset(args)

    # assume at the detection time, the original prompt is unknown
    prompt_at_test_time = ""

    log_and_print("💬 Getting text_embeddings...")

    text_embeddings = pipe.get_text_embedding(prompt_at_test_time)

    log_and_print("💬 Getting watermarking_pattern...")

    watermarked_fourier_latents = optim_utils.get_watermarked_fourier_latents(
        pipe, BATCH_SIZE, args, device
    )

    watermarking_masks = optim_utils.get_watermarking_masks(
        watermarked_fourier_latents.shape, args, device
    )

    metrics_of_clean_images = []
    metrics_of_watermarked_images = []

    log_and_print("💬 Iteration begin...")

    start_time: float = time.time()

    for current_image_index in tqdm(range(args.start, args.end)):
        # `args.gen_seed` 默认为 0
        seed = args.gen_seed + current_image_index

        optim_utils.set_random_seed(seed)

        current_prompt = dataset[current_image_index][prompt_key]

        # 获取初始高斯噪声
        clean_initial_latents = pipe.get_random_latents(batch_size=BATCH_SIZE)

        assert isinstance(clean_initial_latents, torch.Tensor)

        # 逆向去噪 + VAE 解码得到无水印图像的集合
        output_on_clean_initial_latents = pipe(
            current_prompt,
            num_images_per_prompt=BATCH_SIZE,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=clean_initial_latents,
        )

        assert isinstance(
            output_on_clean_initial_latents, ModifiedStableDiffusionPipelineOutput
        )

        clean_image = output_on_clean_initial_latents.images[0]

        watermarked_initial_latents = optim_utils.inject_watermark(
            copy.deepcopy(clean_initial_latents),
            watermarking_masks,
            watermarked_fourier_latents,
            args,
        )

        # 得到嵌入水印的图像的集合
        output_on_watermarked_initial_latents = pipe(
            current_prompt,
            num_images_per_prompt=BATCH_SIZE,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=watermarked_initial_latents,
        )

        assert isinstance(
            output_on_watermarked_initial_latents, ModifiedStableDiffusionPipelineOutput
        )

        watermarked_image = output_on_watermarked_initial_latents.images[0]

        # 进行攻击
        attacked_clean_image, attacked_watermarked_image = optim_utils.distort_image(
            clean_image, watermarked_image, seed, args
        )

        attacked_clean_images = (
            optim_utils.transform_image(attacked_clean_image)
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to(device)
        )

        # 使用 VAE 进行编码
        attacked_clean_final_latents = pipe.get_image_latents(
            attacked_clean_images, sample=False
        )

        # DDIM inversion
        reversed_attacked_clean_initial_latents = pipe.forward_diffusion(
            latents=attacked_clean_final_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # 用同样的方法处理带水印的生成图像
        attacked_watermarked_images = (
            optim_utils.transform_image(attacked_watermarked_image)
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to(device)
        )

        attacked_watermarked_final_latents = pipe.get_image_latents(
            attacked_watermarked_images, sample=False
        )

        reversed_attacked_watermarked_initial_latents = pipe.forward_diffusion(
            latents=attacked_watermarked_final_latents,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # 计算水印区域的 L1 误差
        metric_of_clean_images, metric_of_watermarked_images = optim_utils.get_metrics(
            reversed_attacked_clean_initial_latents,
            reversed_attacked_watermarked_initial_latents,
            watermarking_masks,
            watermarked_fourier_latents,
            args,
        )

        metrics_of_clean_images.append(-metric_of_clean_images)
        metrics_of_watermarked_images.append(-metric_of_watermarked_images)

    end_time: float = time.time()
    elapsed_time: float = end_time - start_time

    # roc
    predicted_logits = metrics_of_clean_images + metrics_of_watermarked_images
    actual_labels = [0] * len(metrics_of_clean_images) + [1] * len(
        metrics_of_watermarked_images
    )

    fprs, tprs, thresholds = metrics.roc_curve(
        actual_labels, predicted_logits, pos_label=1
    )
    auc = metrics.auc(fprs, tprs)
    max_accuracy = np.max((1 - fprs + tprs) / 2)
    low = tprs[np.where(fprs < 0.01)[0][-1]]

    log_and_print(f"auc: {auc}, max_accuracy: {max_accuracy}, TPR@1%FPR: {low}")
    log_and_print("Speed: {:.2f} s/it".format(elapsed_time / (args.end - args.start)))
    log_and_print(f"Mission {args.run_name} completed ✔️")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion watermark")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=10, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images", default=1, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--test_num_inference_steps", default=None, type=int)
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default=None)
    parser.add_argument("--max_num_log_image", default=100, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)

    # watermark
    parser.add_argument("--w_seed", default=999999, type=int)
    parser.add_argument("--w_channel", default=0, type=int)
    parser.add_argument("--w_pattern", default="rand")
    parser.add_argument("--w_mask_shape", default="circle")
    parser.add_argument("--w_radius", default=10, type=int)
    parser.add_argument("--w_measurement", default="l1_complex")
    parser.add_argument("--w_injection", default="complex")
    parser.add_argument("--w_pattern_const", default=0, type=float)

    # for image distortion
    parser.add_argument("--r_degree", default=None, type=float)
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--crop_scale", default=None, type=float)
    parser.add_argument("--crop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rand_aug", default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    main(args)
