from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.api.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.api.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.api.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.api.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.api.configs.pipelines.base import PipelineConfig


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


def qwen_image_postprocess_text(outputs, _text_inputs):
    drop_idx = prompt_template_encode_start_idx = 34
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


@dataclass
class QwenImagePipelineConfig(PipelineConfig):
    # embedded_cfg_scale: float = 3.5

    should_use_guidance: bool = False

    is_image_gen: bool = True

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

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, num_channels_latents, height, width)
        return shape

    def pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents

    @staticmethod
    def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device):
        freqs_cis = rotary_emb(img_shapes, txt_seq_lens, device=device)
        return freqs_cis

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb):
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
        return {
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": QwenImagePipelineConfig.get_freqs_cis(
                img_shapes, txt_seq_lens, rotary_emb, device
            ),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb):
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
        return {
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": QwenImagePipelineConfig.get_freqs_cis(
                img_shapes, txt_seq_lens, rotary_emb, device
            ),
        }

    def post_denoising_loop(self, latents, batch):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        batch_size = latents.shape[0]
        channels = latents.shape[-1]
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = 2 * (int(batch.height) // (vae_scale_factor * 2))
        width = 2 * (int(batch.width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents
