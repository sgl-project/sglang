# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.configs.post_training.pipeline_configs import (
    ZImageRolloutPipelineMixin,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
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
class ZImagePipelineConfig(ZImageRolloutPipelineMixin, ImagePipelineConfig):
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

    @staticmethod
    def _split_evenly(total: int, parts: int) -> list[int]:
        base, remainder = divmod(total, parts)
        return [base + int(rank < remainder) for rank in range(parts)]

    def _build_zimage_sp_plan(self, batch) -> dict:
        """Build an SP plan that preserves native spatial layout for Z-Image."""
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

        # ZImage patchifies [C, F, H, W] latents in native F/H/W order, so shard
        # native H or W directly.
        H_tok = H // self.PATCH_SIZE
        W_tok = W // self.PATCH_SIZE

        shard_options = []
        for shard_axis, axis_tok, other_tok, tie_break in (
            ("h", H_tok, W_tok, 0),
            ("w", W_tok, H_tok, 1),
        ):
            axis_sizes = self._split_evenly(axis_tok, sp_size)
            local_seq_lens = [axis_size * other_tok for axis_size in axis_sizes]
            img_seq_target = self._ceil_to_multiple(
                max(local_seq_lens), self.SEQ_LEN_MULTIPLE
            )
            total_pad_tokens = img_seq_target * sp_size - (H_tok * W_tok)
            shard_options.append(
                (
                    total_pad_tokens,
                    -axis_tok,
                    tie_break,
                    shard_axis,
                    axis_sizes,
                    img_seq_target,
                )
            )

        _, _, _, shard_axis, axis_sizes, img_seq_target = min(shard_options)
        axis_start_tok = sum(axis_sizes[:rank])
        axis_local_tok = axis_sizes[rank]

        if shard_axis == "h":
            h0_tok = axis_start_tok
            w0_tok = 0
            local_h_tok = axis_local_tok
            local_w_tok = W_tok
        else:
            h0_tok = 0
            w0_tok = axis_start_tok
            local_h_tok = H_tok
            local_w_tok = axis_local_tok

        plan = {
            "sp_size": sp_size,
            "rank": rank,
            "H": H,
            "W": W,
            "H_tok": H_tok,
            "W_tok": W_tok,
            "shard_axis": shard_axis,
            "shard_sizes_tok": axis_sizes,
            "h0_tok": h0_tok,
            "w0_tok": w0_tok,
            "local_h_tok": local_h_tok,
            "local_w_tok": local_w_tok,
            "img_seq_target": img_seq_target,
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
        if plan["shard_axis"] == "h":
            h0 = plan["h0_tok"] * self.PATCH_SIZE
            h1 = (plan["h0_tok"] + plan["local_h_tok"]) * self.PATCH_SIZE
            return latents[:, :, :, h0:h1, :].contiguous(), True

        w0 = plan["w0_tok"] * self.PATCH_SIZE
        w1 = (plan["w0_tok"] + plan["local_w_tok"]) * self.PATCH_SIZE
        return latents[:, :, :, :, w0:w1].contiguous(), True

    def gather_latents_for_sp(self, latents, batch):
        # Gather native H/W shards by padding to a common collective shape, then crop.
        latents = latents.contiguous()
        if get_sp_world_size() <= 1 or latents.dim() not in (4, 5, 6):
            return latents

        assert batch is not None
        plan = self._get_zimage_sp_plan(batch)
        if latents.dim() == 4:
            shard_dim = 2 if plan["shard_axis"] == "h" else 3
        elif latents.dim() == 5:
            shard_dim = 3 if plan["shard_axis"] == "h" else 4
        else:
            shard_dim = 4 if plan["shard_axis"] == "h" else 5
        max_axis_tok = max(plan["shard_sizes_tok"])
        max_axis_lat = max_axis_tok * self.PATCH_SIZE

        pad_shape = list(latents.shape)
        pad_shape[shard_dim] = max_axis_lat
        padded = latents.new_zeros(pad_shape)
        axis_len = latents.shape[shard_dim]
        padded_slices = [slice(None)] * latents.dim()
        padded_slices[shard_dim] = slice(axis_len)
        padded[tuple(padded_slices)] = latents

        gathered = [torch.empty_like(padded) for _ in range(plan["sp_size"])]
        dist.all_gather(gathered, padded, group=get_sp_group().device_group)

        pieces = []
        for rank, tensor in enumerate(gathered):
            axis_lat = plan["shard_sizes_tok"][rank] * self.PATCH_SIZE
            gather_slices = [slice(None)] * latents.dim()
            gather_slices[shard_dim] = slice(axis_lat)
            pieces.append(tensor[tuple(gather_slices)])
        return torch.cat(pieces, dim=shard_dim)

    def gather_noise_pred_for_sp(self, batch, noise_pred):
        return self.gather_latents_for_sp(noise_pred, batch=batch)

    def post_denoising_loop(self, latents, batch):
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

            # Build image positions for the local native shard.
            F_tokens = 1
            H_tokens_local = plan["local_h_tok"]
            W_tokens_local = plan["local_w_tok"]
            img_pos_ids = create_coordinate_grid(
                size=(F_tokens, H_tokens_local, W_tokens_local),
                start=(
                    cap_ori_len + cap_padding_len + 1,
                    plan["h0_tok"],
                    plan["w0_tok"],
                ),
                device=device,
            ).flatten(0, 2)
            img_pad_len = plan["img_seq_target"] - img_pos_ids.shape[0]
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
            "image_seq_len_target": (
                self._get_zimage_sp_plan(batch)["img_seq_target"]
                if get_sp_world_size() > 1
                else None
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
            "image_seq_len_target": (
                self._get_zimage_sp_plan(batch)["img_seq_target"]
                if get_sp_world_size() > 1
                else None
            ),
        }
