import math
from dataclasses import dataclass, field
from typing import Callable, Tuple

import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.joy_image import JoyImageDiTConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def joy_image_postprocess_text(
    outputs,
    _text_inputs,
    drop_idx=34,
    max_sequence_length=4096,
):
    last_hidden_states = outputs.hidden_states[-1]
    prompt_embeds = last_hidden_states[:, drop_idx:]
    if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
        prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
    return prompt_embeds


@dataclass
class JoyImageEditPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2I

    dit_config: DiTConfig = field(default_factory=JoyImageDiTConfig)

    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    flow_shift: float = 1.5

    # Text encoding stage (Qwen3-VL for both text and image understanding)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3VLConfig(),)
    )

    enable_torch_compile: bool = False

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (joy_image_postprocess_text,)
    )
    prioritize_frame_matching: bool = True
    bucket_configs: list[tuple[int, int, int, int, int]] = field(init=False)

    def __post_init__(self):
        self.bucket_configs = self.generate_video_image_bucket(
            basesize=1024,
            min_temporal=1,
            max_temporal=1,
            bs_img=8,
            bs_vid=4,
            bs_mimg=8,
            min_items=1,
            max_items=6,
        )

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1)]
        return noise

    def _generate_hw_buckets(
        self,
        base_height=256,
        base_width=256,
        step_width=16,
        step_height=16,
        max_ratio=4.0,
    ) -> list[tuple[int, int, int, int, int]]:
        """Generate dimension buckets based on aspect ratios"""
        buckets = []
        target_pixels = base_height * base_width

        height = target_pixels // step_width
        width = step_width

        while height >= step_height:
            if max(height, width) / min(height, width) <= max_ratio:
                ratio = height / width
                buckets.append((1, 1, 1, height, width))
            # Try to increase width or decrease height
            if height * (width + step_width) <= target_pixels:
                width += step_width
            else:
                height -= step_height

        return buckets

    def generate_video_image_bucket(
        self,
        basesize=256,
        min_temporal=65,
        max_temporal=129,
        bs_img=8,
        bs_vid=1,
        bs_mimg=4,
        min_items=1,
        max_items=1,
    ):
        # (batch_size, num_items, num_frames, height, width)
        assert basesize in [
            256,
            512,
            768,
            1024,
        ], f"[generate_video_image_bucket] wrong basesize {basesize}"
        bucket_list = []

        base_bucket_list = self._generate_hw_buckets()
        # image
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bucket[0] = bs_img
            bucket_list.append(bucket)
        # video
        for temporal in range(min_temporal, max_temporal + 1, 8):
            for _bucket in base_bucket_list:
                bucket = list(_bucket)
                bs = (max_temporal + 1) // temporal * bs_vid
                bucket[0] = bs
                bucket[2] = temporal
                bucket_list.append(bucket)
        # multiple images
        for num_items in range(min_items, max_items + 1):
            for _bucket in base_bucket_list:
                bucket = list(_bucket)
                bucket[0] = bs_mimg
                bucket[1] = num_items
                bucket_list.append(bucket)
        # spatial resize
        if basesize > 256:
            ratio = basesize // 256

            def resize(bucket, r):
                bucket[-2] *= r
                bucket[-1] *= r
                return bucket

            bucket_list = [resize(bucket, ratio) for bucket in bucket_list]
        return bucket_list

    def find_best_bucket(
        self, media_shape: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int, int]:
        """
        Find the best matching bucket for given media dimensions.

        Args:
            media_shape: (num_items, num_frames, height, width) of input media

        Returns:
            Best matching bucket as (batch_size, num_items, num_frames, height, width)
        """
        num_items, num_frames, height, width = media_shape
        target_aspect_ratio = height / width

        if num_frames == 1:
            valid_buckets = []
            for bucket in self.bucket_configs:
                if bucket[1] == num_items and bucket[2] == 1:
                    valid_buckets.append(bucket)

            if len(valid_buckets) == 0:
                raise ValueError(f"No image buckets found for shape {media_shape}")

            return min(
                valid_buckets,
                key=lambda bucket: abs((bucket[3] / bucket[4]) - target_aspect_ratio),
            )
        else:
            valid_buckets = []
            for bucket in self.bucket_configs:
                if bucket[1] == num_items and bucket[2] > 1 and bucket[2] <= num_frames:
                    valid_buckets.append(bucket)

            if len(valid_buckets) == 0:
                raise ValueError(f"No video buckets found for shape {media_shape}")

            if self.prioritize_frame_matching:
                max_frame_count = max(bucket[2] for bucket in valid_buckets)
                max_frame_buckets = [
                    bucket for bucket in valid_buckets if bucket[2] == max_frame_count
                ]

                return min(
                    max_frame_buckets,
                    key=lambda bucket: abs(
                        (bucket[3] / bucket[4]) - target_aspect_ratio
                    ),
                )
            else:
                min_ratio_difference = min(
                    abs((bucket[3] / bucket[4]) - target_aspect_ratio)
                    for bucket in valid_buckets
                )
                best_ratio_buckets = [
                    bucket
                    for bucket in valid_buckets
                    if abs((bucket[3] / bucket[4]) - target_aspect_ratio)
                    == min_ratio_difference
                ]

                return max(best_ratio_buckets, key=lambda bucket: bucket[2])

    def resize_center_crop(
        self, img: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        if isinstance(img, list):
            img = img[0]
        w, h = img.size  # PIL (width, height)
        bh, bw = target_size
        if w == bw and h == bh:
            return img

        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

        img = TF.resize(
            img,
            (resize_h, resize_w),
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        img = TF.center_crop(img, target_size)
        return img

    def preprocess_condition_image(
        self, img, width, height, _vae_image_processor
    ) -> None:
        target_w, target_h = self.prepare_calculated_size(img)
        return self.resize_center_crop(img, (target_h, target_w)), (target_w, target_h)

    def get_decode_scale_and_shift(
        self, device, dtype, vae
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get VAE denormalization scale and shift.

        Args:
            device: Target device
            dtype: Target dtype
            vae: VAE model

        Returns:
            Tuple of (scaling_factor, shift_factor)
        """
        vae_arch_config = self.vae_config.arch_config

        # Create scale factor: 1.0 / std
        scaling_factor = 1.0 / torch.tensor(
            vae_arch_config.latents_std, device=device
        ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(device, dtype)

        # Create shift factor: mean
        shift_factor = (
            torch.tensor(vae_arch_config.latents_mean)
            .view(1, vae_arch_config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )

        return scaling_factor, shift_factor

    def prepare_calculated_size(self, img: Image.Image) -> Tuple[int, int]:
        img_h, img_w = img.size[1], img.size[0]  # PIL (w,h)
        bucket = self.find_best_bucket((1, 1, img_h, img_w))
        return bucket[-1], bucket[-2]  # (width, height)

    def prepare_image_processor_kwargs(self, batch, neg=False) -> dict:
        prompt = batch.prompt if not neg else batch.negative_prompt
        if prompt is None:
            return {}
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        image_list = batch.condition_image
        if image_list is None:
            image_list = []
        elif not isinstance(image_list, list):
            image_list = [image_list]

        if len(prompt_list) <= 1:
            per_prompt_images = [image_list]
        elif len(image_list) <= 1:
            per_prompt_images = [list(image_list) for _ in prompt_list]
        elif len(image_list) == len(prompt_list):
            per_prompt_images = [[image] for image in image_list]
        else:
            raise ValueError(
                "JoyImageEdit expects either one shared condition image or "
                "the same number of condition images and prompts."
            )

        prompt_template_encode = (
            "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size,"
            " texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        img_prompt_template = "<|vision_start|><|image_pad|><|vision_end|>"
        txt = []
        for p, prompt_images in zip(prompt_list, per_prompt_images):
            base_img_prompt = img_prompt_template * len(prompt_images)
            txt.append(prompt_template_encode.format(base_img_prompt + p))
        return dict(text=txt, padding=True, per_prompt_images=per_prompt_images)

    def prepare_latent_shape(self, batch, batch_size: int, num_frames: int) -> Tuple:
        """Prepare latent shape for I2I generation with multi-item support.

        Args:
            batch: The request batch
            batch_size: Batch size
            num_frames: Number of frames (1 for image)

        Returns:
            Tuple representing latent shape
        """

        shape = (
            batch_size,
            self.vae_config.arch_config.z_dim,  # 16 for WanxVAE
            1,
            int(batch.height) // self.vae_config.arch_config.scale_factor_spatial,
            int(batch.width) // self.vae_config.arch_config.scale_factor_spatial,
        )

        return shape

    def postprocess_image_latent(self, latent_condition, batch):
        if latent_condition.dim() == 4:
            latent_condition = latent_condition.unsqueeze(0)
        elif latent_condition.dim() != 5:
            raise ValueError(
                f"Expected 4D/5D condition latents, but got shape {latent_condition.shape}"
            )

        batch_size = int(batch.batch_size)
        cond_batch = int(latent_condition.shape[0])
        if batch_size > cond_batch:
            if batch_size % cond_batch != 0:
                raise ValueError(
                    f"Cannot duplicate condition image latents from batch size {cond_batch} "
                    f"to target batch size {batch_size}."
                )
            repeat_factor = batch_size // cond_batch
            latent_condition = latent_condition.repeat(repeat_factor, 1, 1, 1, 1)
        elif batch_size < cond_batch:
            raise ValueError(
                f"Condition image latents batch size {cond_batch} exceeds target batch size {batch_size}."
            )
        _, _, t, h, w = latent_condition.shape
        pt, ph, pw = self.dit_config.arch_config.patch_size
        condition_size = (t // pt, h // ph, w // pw)

        if batch.vae_image_sizes is None:
            batch.vae_image_sizes = [condition_size]
        else:
            # ImageVAEEncodingStage iterates condition images in input order.
            # Keep the same order in vae_image_sizes for RoPE range construction.
            batch.vae_image_sizes = batch.vae_image_sizes + [condition_size]

        latents = rearrange(
            latent_condition,
            "b c (t pt) (h ph) (w pw) -> b (t h w) c pt ph pw",
            pt=pt,
            ph=ph,
            pw=pw,
        )
        return latents

    def maybe_pack_latents(self, latents, batch_size, batch):
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        elif latents.dim() != 5:
            raise ValueError(f"Expected 4D/5D latents, but got shape {latents.shape}")

        _, _, t, h, w = latents.shape
        pt, ph, pw = self.dit_config.arch_config.patch_size
        if batch.vae_image_sizes is None:
            batch.vae_image_sizes = [(t // pt, h // ph, w // pw)]
        else:
            # LatentPreparationStage packs noisy latents after condition latents were packed
            # in ImageVAEEncodingStage. Denoising concatenates as [noisy, condition...],
            # so keep noisy size at index 0.
            batch.vae_image_sizes = [
                (t // pt, h // ph, w // pw)
            ] + batch.vae_image_sizes
        latents = rearrange(
            latents,
            "b c (t pt) (h ph) (w pw) -> b (t h w) c pt ph pw",
            pt=pt,
            ph=ph,
            pw=pw,
        )

        return latents

    def post_denoising_loop(self, latents, batch):
        lt, lh, lw = batch.vae_image_sizes[0]
        target_len = lt * lh * lw
        target_patches = latents[:, :target_len]
        return rearrange(
            target_patches,
            "b (t h w) c pt ph pw -> b c (t pt) (h ph) (w pw)",
            t=lt,
            h=lh,
            w=lw,
        )

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        cond_norm = torch.norm(noise_pred_cond, dim=2, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=2, keepdim=True).clamp_min(1e-12)
        return noise_pred * (cond_norm / noise_norm)
