# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from dataclasses import dataclass, field
from typing import Callable

import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions
from einops import rearrange

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.pipelines.base import (
    ModelTaskType,
    PipelineConfig,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)


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
class QwenImagePipelineConfig(PipelineConfig):
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

    def shard_latents_for_sp(self, batch, latents):
        # Image latents packed as [B, S, D] -> shard S across SP
        sp_world_size, rank_in_sp_group = get_sp_world_size(), get_sp_parallel_rank()

        seq_len = latents.shape[1]
        # Pad to next multiple of SP degree if needed
        if seq_len % sp_world_size != 0:
            pad_len = sp_world_size - (seq_len % sp_world_size)
            pad = torch.zeros(
                (latents.shape[0], pad_len, latents.shape[2]),
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, pad], dim=1)
            # Record padding length for later unpad
            batch.sp_seq_pad = int(getattr(batch, "sp_seq_pad", 0)) + pad_len
        sharded_tensor = rearrange(
            latents, "b (n s) d -> b n s d", n=sp_world_size
        ).contiguous()
        sharded_tensor = sharded_tensor[:, rank_in_sp_group, :, :]
        return sharded_tensor, True

    def gather_latents_for_sp(self, latents):
        # For image latents [B, S_local, D], gather along sequence dim=1
        latents = sequence_model_parallel_all_gather(latents, dim=1)
        return latents

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

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        batch_size = batch.latents.shape[0]
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    batch.height // vae_scale_factor // 2,
                    batch.width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size
        txt_seq_lens = [batch.prompt_embeds[0].shape[1]]

        (img_cos, img_sin), (txt_cos, txt_sin) = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )
        # Pad image RoPE to make total tokens divisible by SP degree (if enabled)
        try:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_world_size,
            )

            sp_world_size = get_sp_world_size()
        except Exception:
            sp_world_size = 1
        if sp_world_size and sp_world_size > 1:
            total_tokens = img_cos.shape[0]
            pad_len = (sp_world_size - (total_tokens % sp_world_size)) % sp_world_size
            if pad_len > 0:
                pad_cos = img_cos[-1:, :].repeat(pad_len, 1)
                pad_sin = img_sin[-1:, :].repeat(pad_len, 1)
                img_cos = torch.cat([img_cos, pad_cos], dim=0)
                img_sin = torch.cat([img_sin, pad_sin], dim=0)

        # FIXME: video models handle SP internally in `_forward_cached_from_grid`, while for image models we do this manually
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        batch_size = batch.latents.shape[0]
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor

        img_shapes = [
            [
                (
                    1,
                    batch.height // vae_scale_factor // 2,
                    batch.width // vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        txt_seq_lens = [batch.negative_prompt_embeds[0].shape[1]]

        (img_cos, img_sin), (txt_cos, txt_sin) = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        }

    def post_denoising_loop(self, latents, batch):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        batch_size = latents.shape[0]
        channels = latents.shape[-1]
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (int(batch.height) // (vae_scale_factor * 2))
        width = 2 * (int(batch.width) // (vae_scale_factor * 2))

        # If SP padding was applied, remove extra tokens before reshaping
        target_tokens = (height // 2) * (width // 2)
        if latents.shape[1] != target_tokens:
            latents = latents[:, :target_tokens, :]

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents


class QwenImageEditPipelineConfig(QwenImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2I

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # TODO: lots of duplications here
        batch_size = batch.latents.shape[0]
        assert batch_size == 1
        height = batch.height
        width = batch.width
        image = batch.pil_image
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        vae_scale_factor = self.get_vae_scale_factor()
        img_shapes = [
            [
                (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
                (
                    1,
                    calculated_height // vae_scale_factor // 2,
                    calculated_width // vae_scale_factor // 2,
                ),
            ]
        ] * batch_size
        txt_seq_lens = [batch.prompt_embeds[0].shape[1]]
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

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        batch_size = batch.latents.shape[0]
        height = batch.height
        width = batch.width
        image = batch.pil_image
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        vae_scale_factor = self.get_vae_scale_factor()
        img_shapes = [
            [
                (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
                (
                    1,
                    calculated_height // vae_scale_factor // 2,
                    calculated_width // vae_scale_factor // 2,
                ),
            ]
        ] * batch_size

        txt_seq_lens = [batch.negative_prompt_embeds[0].shape[1]]

        (img_cos, img_sin), (txt_cos, txt_sin) = QwenImagePipelineConfig.get_freqs_cis(
            img_shapes, txt_seq_lens, rotary_emb, device, dtype
        )
        noisy_img_seq_len = (
            1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
        )
        noisy_img_cos = shard_rotary_emb_for_sp(img_cos[:noisy_img_seq_len, :])
        noisy_img_sin = shard_rotary_emb_for_sp(img_sin[:noisy_img_seq_len, :])

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

    def preprocess_image(self, image, image_processor):
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        image = image_processor.resize(image, calculated_height, calculated_width)
        return image

    def set_width_and_height(self, width, height, image):
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.get_vae_scale_factor() * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        return width, height

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1)]
        return noise
