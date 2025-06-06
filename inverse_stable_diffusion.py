from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from modified_stable_diffusion import ModifiedStableDiffusionPipeline

### credit to: https://github.com/cccntu/efficient-prompt-to-prompt


def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """from noise to image

    给定噪声 `eps_xt` 和当前步的图像 `x_t`, 计算前一步的图像 `x_{t - 1}`.

    """
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    def get_random_latents(
        self, batch_size=1, latents=None, height=512, width=512, generator=None
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore
        width = width or self.unet.config.sample_size * self.vae_scale_factor  # type: ignore

        device = self._execution_device

        num_channels_latents = self.unet.in_channels  # type: ignore

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,  # type: ignore
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(  # type: ignore
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,  # type: ignore
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]  # type: ignore
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist  # type: ignore
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = False,
        **kwargs,
    ):
        """Generate image from text prompt and latents"""
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)  # type: ignore
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)  # type: ignore
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma  # type: ignore

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
            # self.progress_bar(
            #     reversed(timesteps_tensor) if reverse_process else timesteps_tensor
            # )
            reversed(timesteps_tensor)
            if reverse_process
            else timesteps_tensor
        ):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # type: ignore
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

            # predict the noise residual
            noise_pred = self.unet(  # type: ignore
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            previous_t = t - (
                self.scheduler.config.num_train_timesteps  # type: ignore
                // self.scheduler.num_inference_steps  # type: ignore
            )

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:  # type: ignore
                callback(i, t, latents)  # type: ignore

            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]  # type: ignore
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[previous_t]  # type: ignore
                if previous_t >= 0
                else self.scheduler.final_alpha_cumprod  # type: ignore
            )

            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample  # type: ignore
            for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
