# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)


def zimage_preprocess_text(prompt: str):
    messages = [
        {"role": "user", "content": prompt},
    ]
    return messages


def zimage_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    device = outputs.hidden_states[-2].device
    prompt_mask = _text_inputs.attention_mask.to(device).bool()
    return outputs.hidden_states[-2][0][prompt_mask[0]]


class TransformersModelConfig(EncoderConfig):
    tokenizer_kwargs: dict = field(default_factory=lambda: {})


@dataclass
class ZImagePipelineConfig(ImagePipelineConfig):
    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I
    dit_config: DiTConfig = field(default_factory=ZImageDitConfig)
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3TextConfig(),)
    )

    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (zimage_postprocess_text,)
    )

    SEQ_LEN_MULTIPLE: int = 32
    PATCH_SIZE: int = 2
    F_PATCH_SIZE: int = 1

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        rendered_prompts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt in prompts
        ]
        return tokenizer(
            rendered_prompts,
            padding="max_length",
            max_length=512,  # TODO (yhyang201): set max length according to config
            truncation=True,
            return_tensors="pt",
        )

    @staticmethod
    def _ceil_to_multiple(x: int, m: int) -> int:
        if m <= 0:
            return x
        return int(math.ceil(x / m) * m)

    def _build_zimage_sp_plan(self, batch) -> dict:
        """Build a minimal SP plan on batch for zimage spatial sharding."""
        sp_size = get_sp_world_size()
        rank = get_sp_parallel_rank()

        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and len(raw_latent_shape) >= 5:
            H = int(raw_latent_shape[3])
            W = int(raw_latent_shape[4])
        else:
            H = int(
                batch.height // self.vae_config.arch_config.spatial_compression_ratio
            )
            W = int(
                batch.width // self.vae_config.arch_config.spatial_compression_ratio
            )

        # Rule: shard along the larger spatial dimension (W/H), implemented via optional H/W transpose.
        # Choose the larger of H and W for sharding, so H_eff = max(H, W).
        swap_hw = W > H
        H_eff = W if swap_hw else H
        W_eff = H if swap_hw else W

        # ZImage uses PATCH_SIZE=2 for spatial patchify; shard in token space and convert back to latent rows.
        H_tok = H_eff // self.PATCH_SIZE
        W_tok = W_eff // self.PATCH_SIZE
        H_tok_pad = self._ceil_to_multiple(H_tok, sp_size)
        H_tok_local = H_tok_pad // sp_size
        h0_tok = rank * H_tok_local

        plan = {
            "sp_size": sp_size,
            "rank": rank,
            "swap_hw": swap_hw,
            "H": H,
            "W": W,
            "H_eff": H_eff,
            "W_eff": W_eff,
            "H_tok": H_tok,
            "W_tok": W_tok,
            "H_tok_pad": H_tok_pad,
            "H_tok_local": H_tok_local,
            "h0_tok": h0_tok,
        }
        batch._zimage_sp_plan = plan
        return plan

    def _get_zimage_sp_plan(self, batch) -> dict:
        plan = getattr(batch, "_zimage_sp_plan", None)
        sp_size = get_sp_world_size()
        if plan is None or plan.get("sp_size") != sp_size:
            plan = self._build_zimage_sp_plan(batch)
        return plan

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # Match the official diffusers Z-Image pipeline, which samples latents in fp32
        # and keeps scheduler state in fp32.
        return torch.float32

    def shard_latents_for_sp(self, batch, latents):
        sp_size = get_sp_world_size()
        if sp_size <= 1 or latents.dim() != 5:
            return latents, False

        plan = self._get_zimage_sp_plan(batch)

        # Layout: [B, C, T, H, W]. Always shard on dim=3 by optionally swapping H/W.
        if plan["swap_hw"]:
            latents = latents.transpose(3, 4).contiguous()

        # Pad on effective-H so that H_tok is divisible by sp.
        H_eff = latents.size(3)

        H_tok = H_eff // self.PATCH_SIZE
        pad_tok = plan["H_tok_pad"] - H_tok
        pad_lat = pad_tok * self.PATCH_SIZE
        if pad_lat > 0:
            pad = latents[:, :, :, -1:, :].repeat(1, 1, 1, pad_lat, 1)
            latents = torch.cat([latents, pad], dim=3)
        h0 = plan["h0_tok"] * self.PATCH_SIZE
        h1 = (plan["h0_tok"] + plan["H_tok_local"]) * self.PATCH_SIZE
        latents = latents[:, :, :, h0:h1, :]

        batch._zimage_sp_swap_hw = plan["swap_hw"]
        return latents, True

    def gather_latents_for_sp(self, latents):
        # Gather on effective-H dim=3 (matches shard_latents_for_sp); swap-back is handled in post_denoising_loop.
        latents = latents.contiguous()
        if get_sp_world_size() <= 1 or latents.dim() != 5:
            return latents
        return sequence_model_parallel_all_gather(latents, dim=3)

    def gather_noise_pred_for_sp(self, batch, noise_pred):
        # Z-Image shards 5D latents on the effective-H axis, but ComfyUI noise_pred is 4D [B, C, H_local, W].
        noise_pred = self.gather_latents_for_sp(noise_pred)
        if noise_pred.dim() == 4:
            # reconstruct the full spatial tensor
            noise_pred = sequence_model_parallel_all_gather(
                noise_pred.contiguous(), dim=2
            )
            # restore the original H/W orientation
            if getattr(batch, "_zimage_sp_swap_hw", False):
                noise_pred = noise_pred.transpose(2, 3).contiguous()
        return noise_pred

    def post_denoising_loop(self, latents, batch):
        # Restore swapped H/W and crop padded spatial dims before final reshape.
        if latents.dim() == 5 and getattr(batch, "_zimage_sp_swap_hw", False):
            latents = latents.transpose(3, 4).contiguous()
        raw_latent_shape = getattr(batch, "raw_latent_shape", None)
        if raw_latent_shape is not None and latents.dim() == 5:
            latents = latents[:, :, :, : raw_latent_shape[3], : raw_latent_shape[4]]

        bs, channels, num_frames, height, width = latents.shape
        if raw_latent_shape is not None and num_frames > raw_latent_shape[2]:
            latents = latents[:, :, : raw_latent_shape[2], :, :]
            num_frames = raw_latent_shape[2]
        if num_frames != 1:
            return latents[:, :, 0, :, :]
        return latents.view(bs, channels, height, width)

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        def create_coordinate_grid(size, start=None, device=None):
            if start is None:
                start = (0 for _ in size)

            axes = [
                torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
                for x0, span in zip(start, size)
            ]
            grids = torch.meshgrid(axes, indexing="ij")
            return torch.stack(grids, dim=-1)

        sp_size = get_sp_world_size()
        if sp_size > 1:
            # SP path: keep caption replicated on every rank and build local-only
            # image freqs_cis matching the spatial shard.
            plan = self._get_zimage_sp_plan(batch)
            cap_ori_len = prompt_embeds.size(0)
            cap_padding_len = (-cap_ori_len) % self.SEQ_LEN_MULTIPLE

            # caption (replicated prefix)
            cap_pos_ids = create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            cap_freqs_cis = rotary_emb(cap_pos_ids)

            # image (local, effective H-shard), offset after the full caption.
            F_tokens = 1
            H_tokens_local = plan["H_tok_local"]
            W_tokens = plan["W_tok"]
            img_pos_ids = create_coordinate_grid(
                size=(F_tokens, H_tokens_local, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, plan["h0_tok"], 0),
                device=device,
            ).flatten(0, 2)
            img_pad_len = (-img_pos_ids.shape[0]) % self.SEQ_LEN_MULTIPLE
            if img_pad_len:
                pad_ids = create_coordinate_grid(
                    size=(1, 1, 1), start=(0, 0, 0), device=device
                ).flatten(0, 2)
                img_pos_ids = torch.cat(
                    [img_pos_ids, pad_ids.repeat(img_pad_len, 1)], dim=0
                )
            x_freqs_cis = rotary_emb(img_pos_ids)
            return (cap_freqs_cis, x_freqs_cis)

        cap_ori_len = prompt_embeds.size(0)
        cap_padding_len = (-cap_ori_len) % self.SEQ_LEN_MULTIPLE
        cap_padded_pos_ids = create_coordinate_grid(
            size=(cap_ori_len + cap_padding_len, 1, 1),
            start=(1, 0, 0),
            device=device,
        ).flatten(0, 2)

        F = 1
        H = height // self.vae_config.arch_config.spatial_compression_ratio
        W = width // self.vae_config.arch_config.spatial_compression_ratio

        pH, pW = self.PATCH_SIZE, self.PATCH_SIZE
        pF = self.F_PATCH_SIZE
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image_ori_len = F_tokens * H_tokens * W_tokens
        image_padding_len = (-image_ori_len) % self.SEQ_LEN_MULTIPLE

        image_ori_pos_ids = create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(cap_ori_len + cap_padding_len + 1, 0, 0),
            device=device,
        ).flatten(0, 2)
        image_padding_pos_ids = (
            create_coordinate_grid(
                size=(1, 1, 1),
                start=(0, 0, 0),
                device=device,
            )
            .flatten(0, 2)
            .repeat(image_padding_len, 1)
        )
        image_padded_pos_ids = torch.cat(
            [image_ori_pos_ids, image_padding_pos_ids], dim=0
        )
        cap_freqs_cis = rotary_emb(cap_padded_pos_ids)
        x_freqs_cis = rotary_emb(image_padded_pos_ids)
        return (cap_freqs_cis, x_freqs_cis)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
        }
