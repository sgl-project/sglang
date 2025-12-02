# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.models.vision_utils import resize
from sglang.multimodal_gen.utils import calculate_dimensions


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def qwen_image_preprocess_text(prompt):
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    template = prompt_template_encode
    txt = template.format(prompt)
    return txt


def qwen_image_postprocess_text(outputs, _text_inputs, drop_idx=34):
    # squeeze the batch dim
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(
        hidden_states, _text_inputs.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ]
    )
    return prompt_embeds


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


@dataclass
class QwenImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the QwenImage pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=QwenImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=QwenImageVAEConfig)

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5VLConfig(),)
    )

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_preprocess_text,)
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (qwen_image_postprocess_text,)
    )
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                padding=True,
                truncation=True,
            ),
            None,
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def prepare_image_processor_kwargs(self, batch):
        if batch.prompt:
            prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            txt = prompt_template_encode.format(batch.prompt)
            return dict(text=[txt], padding=True)
        else:
            return {}

    def get_vae_scale_factor(self):
        return self.vae_config.arch_config.vae_scale_factor

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (batch.height // (vae_scale_factor * 2))
        width = 2 * (batch.width // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, 1, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        vae_arch_config = self.vae_config.arch_config
        scaling_factor = 1.0 / torch.tensor(
            vae_arch_config.latents_std, device=device
        ).view(1, vae_arch_config.z_dim, 1, 1, 1).to(device, dtype)
        shift_factor = (
            torch.tensor(vae_arch_config.latents_mean)
            .view(1, vae_arch_config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        return scaling_factor, shift_factor

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):
        # img_shapes: for global entire image
        img_freqs, txt_freqs = rotary_emb(img_shapes, txt_seq_lens, device=device)

        img_cos, img_sin = (
            img_freqs.real.to(dtype=dtype),
            img_freqs.imag.to(dtype=dtype),
        )
        txt_cos, txt_sin = (
            txt_freqs.real.to(dtype=dtype),
            txt_freqs.imag.to(dtype=dtype),
        )

        return (img_cos, img_sin), (txt_cos, txt_sin)

    def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
        batch_size = prompt_embeds[0].shape[0]
        height = batch.height
        width = batch.width
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]

        (img_cos, img_sin), (txt_cos, txt_sin) = self.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def post_denoising_loop(self, latents, batch):
        # unpack latents for qwen-image
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents


@dataclass
class QwenImageEditPipelineConfig(QwenImagePipelineConfig):
    """Configuration for the QwenImageEdit pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I

    def _prepare_edit_cond_kwargs(
        self, batch, prompt_embeds, rotary_emb, device, dtype
    ):
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width
        image_size = batch.original_condition_image_size
        edit_width, edit_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        vae_scale_factor = self.get_vae_scale_factor()

        img_shapes = [
            [
                (
                    1,
                    height // vae_scale_factor // 2,
                    width // vae_scale_factor // 2,
                ),
                (
                    1,
                    edit_height // vae_scale_factor // 2,
                    edit_width // vae_scale_factor // 2,
                ),
            ],
        ] * batch_size
        txt_seq_lens = [prompt_embeds[0].shape[1]]
        (img_cos, img_sin), (txt_cos, txt_sin) = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )

        # perform sp shard on noisy image tokens
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )

        noisy_img_cos = shard_rotary_emb_for_sp(img_cos[:noisy_img_seq_len, :])
        noisy_img_sin = shard_rotary_emb_for_sp(img_sin[:noisy_img_seq_len, :])

        # concat back the img_cos for input image (since it is not sp-shared later)
        img_cos = torch.cat([noisy_img_cos, img_cos[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )
        img_sin = torch.cat([noisy_img_sin, img_sin[noisy_img_seq_len:, :]], dim=0).to(
            device=device
        )

        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def preprocess_condition_image(
        self, image, target_width, target_height, _vae_image_processor
    ):
        return resize(image, target_height, target_width, resize_mode="default"), (
            target_width,
            target_height,
        )

    def postprocess_image_latent(self, latent_condition, batch):
        batch_size = batch.batch_size
        if batch_size > latent_condition.shape[0]:
            if batch_size % latent_condition.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // latent_condition.shape[0]
                image_latents = latent_condition.repeat(
                    additional_image_per_prompt, 1, 1, 1
                )
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latent_condition.shape[0]} to {batch_size} text prompts."
                )
        else:
            image_latents = latent_condition
        image_latent_height, image_latent_width = image_latents.shape[3:]
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        image_latents = _pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

        return image_latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.prompt_embeds, rotary_emb, device, dtype
        )

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_edit_cond_kwargs(
            batch, batch.negative_prompt_embeds, rotary_emb, device, dtype
        )

    def calculate_condition_image_size(self, image, width, height) -> tuple[int, int]:
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, width / height
        )
        return calculated_width, calculated_height

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1)]
        return noise
