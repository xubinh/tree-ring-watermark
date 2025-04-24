import argparse
import copy
import time
from statistics import mean, stdev

import torch
import wandb
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from sklearn import metrics
from tqdm import tqdm

import open_clip
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from io_utils import *
from modified_stable_diffusion import ModifiedStableDiffusionPipelineOutput
from optim_utils import *


def my_print(
    msg: str,
    log_file_path: str = os.path.join(os.getcwd(), "log.txt"),
) -> None:
    with open(log_file_path, "a", encoding="utf8") as file:
        file.write(msg + "\n")

    print(msg, flush=True)


def main(args):
    my_print(f"💬 Mission {args.run_name} started...")

    # if args.with_tracking:
    #     wandb.init(
    #         project="diffusion_watermark",
    #         name=args.run_name,
    #         tags=["tree_ring_watermark"],
    #     )
    #     wandb.config.update(args)
    #     table = wandb.Table(
    #         columns=[
    #             "gen_no_w",
    #             "no_w_clip_score",
    #             "gen_w",
    #             "w_clip_score",
    #             "prompt",
    #             "no_w_metric",
    #             "w_metric",
    #         ]
    #     )

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    my_print("💬 Getting scheduler...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    my_print("💬 Getting pipeline...")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    assert isinstance(pipe, InversableStableDiffusionPipeline)
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        my_print("💬 Getting ref_model & ref_clip_preprocess...")
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        my_print("💬 Getting ref_tokenizer...")
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    my_print("💬 Getting dataset...")
    dataset, prompt_key = get_dataset(args)

    tester_prompt = ""  # assume at the detection time, the original prompt is unknown
    my_print("💬 Getting text_embeddings...")
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # 固定的水印底板
    my_print("💬 Getting watermarking_pattern...")
    gt_patch = get_watermarked_fourier_latents(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    my_print("💬 Iteration begin...")
    start_time: float = time.time()
    for i in tqdm(range(args.start, args.end)):
        # my_print("")

        seed = i + args.gen_seed  # `args.gen_seed` 默认为 0
        set_random_seed(seed)

        current_prompt = dataset[i][prompt_key]

        # 获取初始高斯噪声
        init_latents_no_w = pipe.get_random_latents()  # `batch_size` 默认为 1

        # 逆向去噪 + VAE 解码得到无水印图像的集合
        # my_print("\n\n1\n\n")
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,  # `num_images` 默认为 1
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
        )

        assert isinstance(outputs_no_w, ModifiedStableDiffusionPipelineOutput)

        # 由于 `num_images` 默认为 1, 因此集合中只有一幅图像
        orig_image_no_w = outputs_no_w.images[0]

        # generation with watermarking
        if init_latents_no_w is None:
            raise RuntimeError("never reaches here")
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_masks(init_latents_w, args, device)

        # 将水印嵌入初始高斯噪声
        init_latents_w = inject_watermark(
            init_latents_w, watermarking_mask, gt_patch, args
        )

        # 得到嵌入水印的图像的集合
        # my_print("\n\n2\n\n")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )

        assert isinstance(outputs_w, ModifiedStableDiffusionPipelineOutput)

        orig_image_w = outputs_w.images[0]

        # 对无水印/带水印的生成图像进行攻击
        orig_image_no_w_auged, orig_image_w_auged = distort_image(
            orig_image_no_w, orig_image_w, seed, args
        )

        # 使用 VAE 对攻击后的无水印生成图像进行编码得到其潜在表示
        img_no_w = (
            transform_image(orig_image_no_w_auged)
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to(device)
        )
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        # 使用 DDIM inversion 求出初始噪声
        # my_print("\n\n3\n\n")
        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # 用同样的方法处理带水印的生成图像
        img_w = (
            transform_image(orig_image_w_auged)
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to(device)
        )
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        # 用同样的方法处理带水印的生成图像
        # my_print("\n\n4\n\n")
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )
        # my_print("\n\n5\n\n")

        # 计算水印区域的 L1 误差
        no_w_metric, w_metric = get_metrics(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )

        w_no_sim = 0
        w_sim = 0

        if args.reference_model is not None:
            # sims 为一个仅含两个元素的一维向量, 两个元素分别为无水印图像和带水印图像与提示词之间的余弦相似度
            sims = measure_similarity(
                [orig_image_no_w, orig_image_w],
                current_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device,
            )

            w_no_sim = sims[0].item()
            w_sim = sims[1].item()

        results.append(
            {
                "no_w_metric": no_w_metric,
                "w_metric": w_metric,
                "w_no_sim": w_no_sim,
                "w_sim": w_sim,
            }
        )

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.with_tracking:
            # if (args.reference_model is not None) and (i < args.max_num_log_image):
            #     # log images when we use reference_model
            #     table.add_data(
            #         wandb.Image(orig_image_no_w),
            #         w_no_sim,
            #         wandb.Image(orig_image_w),
            #         w_sim,
            #         current_prompt,
            #         no_w_metric,
            #         w_metric,
            #     )
            # else:
            #     table.add_data(
            #         None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric
            #     )

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)

    end_time: float = time.time()
    elapsed_time: float = end_time - start_time

    # roc
    preds = no_w_metrics + w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.01)[0][-1]]

    # if args.with_tracking:
    #     wandb.log({"Table": table})
    #     wandb.log(
    #         {
    #             "clip_score_mean": mean(clip_scores),
    #             "clip_score_std": stdev(clip_scores),
    #             "w_clip_score_mean": mean(clip_scores_w),
    #             "w_clip_score_std": stdev(clip_scores_w),
    #             "auc": auc,
    #             "acc": acc,
    #             "TPR@1%FPR": low,
    #         }
    #     )

    my_print(f"clip_score_mean: {mean(clip_scores)}")
    my_print(f"w_clip_score_mean: {mean(clip_scores_w)}")
    my_print(f"auc: {auc}, acc: {acc}, TPR@1%FPR: {low}")
    my_print("Speed: {:.2f} s/it".format(elapsed_time / (args.end - args.start)))
    my_print(f"Mission {args.run_name} completed ✔️")


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
