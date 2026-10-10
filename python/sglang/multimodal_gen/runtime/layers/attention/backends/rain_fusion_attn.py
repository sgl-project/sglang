import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# ``blocks.<idx>.attn`` is a DiT layer; ``token_refiner.blocks.<idx>.attn`` and
# anything else is not and stays dense.
_DIT_LAYER_PREFIX = re.compile(r"^blocks\.(\d+)\.")


class RainFusionAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.RAIN_FUSION_ATTN

    @staticmethod
    def get_impl_cls() -> type["RainFusionAttentionImpl"]:
        return RainFusionAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["RainFusionAttentionMetadata"]:
        return RainFusionAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["RainFusionAttentionMetadataBuilder"]:
        return RainFusionAttentionMetadataBuilder


@dataclass
class RainFusionAttentionMetadata(AttentionMetadata):
    current_timestep: int
    skip_first_steps: int
    sparsity: float
    latent_shape: list[int]


class RainFusionAttentionMetadataBuilder(AttentionMetadataBuilder):
    def build(
        self,
        current_timestep: int,
        skip_first_steps: int,
        sparsity: float,
        raw_latent_shape: list[int],
        patch_size: tuple[int, int, int],
        **kwargs: dict[str, Any],
    ) -> RainFusionAttentionMetadata:
        if not (skip_first_steps >= 0 and 0.0 <= sparsity < 1.0):
            raise ValueError(
                (
                    "Invalid attention metadata values."
                    f"Sparsity should be in [0, 1), skip_first_steps should be non-negative."
                    f"Got sparsity={sparsity}, skip_first_steps={skip_first_steps}"
                )
            )

        if sparsity == 0.0:
            logger.warning(
                (
                    "Sparsity is set to 0.0, which means no tokens will be dropped."
                    "For better performance use Laser Attention or increase sparsity."
                )
            )

        latent_shape = raw_latent_shape[-3:]
        latent_shape = [latent_shape[i] // patch_size[i] for i in range(3)]

        return RainFusionAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            latent_shape=latent_shape,
        )


class RainFusionAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("Rain Fusion Attention does not support causal masks")

        self.softmax_scale = softmax_scale
        self.block_size = 128
        self.tile_size = 8
        self._is_a5 = torch_npu.npu.get_soc_version() == 260

        sparse_config = get_global_server_args().attention_backend_config or {}
        self.skip_first_steps = int(sparse_config.get("skip_first_steps", 10))
        self.sparsity = float(sparse_config.get("sparsity", 0.2))

        if self.skip_first_steps < 0 or not 0.0 <= self.sparsity < 1.0:
            raise ValueError(
                "Invalid Rain Fusion attention config: "
                f"skip_first_steps={self.skip_first_steps}, "
                f"sparsity={self.sparsity}"
            )
        # A layer outside the DiT stack (MiniMax-H3's token refiner) runs before
        # the denoise loop, outside the forward context, and its text rows are
        # shorter than one sparse block. It stays dense.
        self.layer_enabled = _DIT_LAYER_PREFIX.match(prefix) is not None
        if self._is_a5:
            from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
                AscendFABackend,
            )

            dense_backend = AscendFABackend
        else:
            from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (
                LaserAttentionBackend,
            )

            dense_backend = LaserAttentionBackend
        self.dense_attn_impl = dense_backend.get_impl_cls()(
            num_heads,
            head_size,
            causal,
            softmax_scale,
            num_kv_heads,
            prefix,
            **extra_impl_args,
        )

    def _pool_token_blocks(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = input_tensor.shape
        full_blocks, tail_tokens = divmod(seq_len, self.block_size)

        if full_blocks == 0:
            if tail_tokens == 0:
                return input_tensor.new_empty((batch_size, 0, num_heads, head_dim))
            return input_tensor.mean(dim=1, keepdim=True)

        full_block_tokens = full_blocks * self.block_size
        full_blocks_pool = (
            input_tensor[:, :full_block_tokens]
            .reshape(batch_size, full_blocks, self.block_size, num_heads, head_dim)
            .mean(dim=2)
        )

        if tail_tokens == 0:
            return full_blocks_pool

        tail_pool = input_tensor[:, full_block_tokens:].mean(dim=1, keepdim=True)
        return torch.cat((full_blocks_pool, tail_pool), dim=1)

    def _rearrange_spatial_tokens(
        self, tensor: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        num_frames, height, width = latent_shape
        batch_size, seq_len, num_heads, head_dim = tensor.shape

        if num_frames == 1 or height < self.tile_size or width < self.tile_size:
            return tensor

        height_block_count, height_remainder = divmod(height, self.tile_size)
        width_block_count, width_remainder = divmod(width, self.tile_size)

        if height_remainder == 0 and width_remainder == 0:
            return (
                tensor.reshape(
                    batch_size,
                    num_frames,
                    height_block_count,
                    self.tile_size,
                    width_block_count,
                    self.tile_size,
                    num_heads,
                    head_dim,
                )
                .permute(0, 1, 2, 4, 3, 5, 6, 7)
                .contiguous()
                .reshape(batch_size, seq_len, num_heads, head_dim)
            )

        first_frame_token_count = height * width
        first_frame = tensor[:, :first_frame_token_count]
        tail_frames = tensor[:, first_frame_token_count:].reshape(
            batch_size,
            num_frames - 1,
            height,
            width,
            num_heads,
            head_dim,
        )
        aligned_height = height_block_count * self.tile_size
        aligned_width = width_block_count * self.tile_size

        tiled_tokens = (
            tail_frames[:, :, :aligned_height, :aligned_width]
            .reshape(
                batch_size,
                num_frames - 1,
                height_block_count,
                self.tile_size,
                width_block_count,
                self.tile_size,
                num_heads,
                head_dim,
            )
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .contiguous()
            .reshape(batch_size, num_frames - 1, -1, num_heads, head_dim)
        )
        rearranged_parts = [tiled_tokens]
        if height_remainder:
            rearranged_parts.append(
                tail_frames[:, :, aligned_height:].reshape(
                    batch_size, num_frames - 1, -1, num_heads, head_dim
                )
            )
        if width_remainder:
            rearranged_parts.append(
                tail_frames[:, :, :aligned_height, aligned_width:].reshape(
                    batch_size, num_frames - 1, -1, num_heads, head_dim
                )
            )

        remaining_tokens = torch.cat(rearranged_parts, dim=2).reshape(
            batch_size, -1, num_heads, head_dim
        )
        return torch.cat((first_frame, remaining_tokens), dim=1)

    def _prepare_rearranged_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rearranged_qkv = self._rearrange_spatial_tokens(
            torch.cat((query, key, value), dim=0), latent_shape
        )
        qkv_pool = self._pool_token_blocks(rearranged_qkv)
        rearranged_query, rearranged_key, rearranged_value = torch.chunk(
            rearranged_qkv, 3, dim=0
        )
        return (
            rearranged_query,
            rearranged_key,
            rearranged_value,
            qkv_pool,
        )

    def _build_sparse_mask(
        self,
        qkv_pool: torch.Tensor,
        sparsity: float,
        latent_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
        return self._build_sparse_mask_from_pools(
            query_pool, key_pool, sparsity, latent_shape
        )

    def _build_sparse_mask_from_pools(
        self,
        query_pool: torch.Tensor,
        key_pool: torch.Tensor,
        sparsity: float,
        latent_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        attention_scores = (
            query_pool.permute(0, 2, 1, 3)
            @ key_pool.permute(0, 2, 3, 1)
            * self.softmax_scale
        )
        probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        retained_block_count = math.ceil(probs.shape[-1] * (1.0 - sparsity))
        retained_probabilities = torch.topk(
            probs, k=retained_block_count, dim=-1
        ).values
        block_sparse_mask = probs >= retained_probabilities[..., -1:]

        if latent_shape is not None:
            first_frame_tokens = latent_shape[1] * latent_shape[2]
            first_frame_block_count = math.ceil(first_frame_tokens / self.block_size)
            block_sparse_mask[:, :, :first_frame_block_count, :] = True
            block_sparse_mask[:, :, :, :first_frame_block_count] = True
        return block_sparse_mask.to(torch.int8)

    def _build_varlen_sparse_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        boundaries: tuple[int, ...],
        max_seqlen: int,
    ) -> tuple[torch.Tensor, list[int]]:
        segment_lengths = [
            stop - start
            for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start
        ]
        if not segment_lengths:
            return query.new_empty((0, query.shape[1], 0, 0), dtype=torch.int8), []

        actual_max_seqlen = max(segment_lengths)
        if max_seqlen < actual_max_seqlen:
            raise ValueError(
                f"max_seqlen={max_seqlen} is smaller than the longest packed "
                f"sequence ({actual_max_seqlen})"
            )
        max_block_count = math.ceil(actual_max_seqlen / self.block_size)

        masks = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop == start:
                continue
            query_pool = self._pool_token_blocks(query[start:stop].unsqueeze(0))
            key_pool = self._pool_token_blocks(key[start:stop].unsqueeze(0))
            mask = self._build_sparse_mask_from_pools(
                query_pool,
                key_pool,
                self.sparsity,
                None,
            )
            block_count = mask.shape[-1]
            masks.append(
                torch.nn.functional.pad(
                    mask,
                    (
                        0,
                        max_block_count - block_count,
                        0,
                        max_block_count - block_count,
                    ),
                )
            )

        # npu_block_sparse_attention validates sum(actual_seq_lengths) == T: it
        # takes per-batch lengths, not the cumulative TND offsets that
        # npu_fused_infer_attention_score takes.
        return torch.cat(masks, dim=0), segment_lengths

    def _rain_fusion_sparse_attention_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        boundaries: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        block_sparse_mask, actual_seq_lengths = self._build_varlen_sparse_mask(
            query, key, boundaries, max_seqlen
        )
        if not actual_seq_lengths:
            return torch.empty_like(query)

        output, _ = torch_npu.npu_block_sparse_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            block_sparse_mask,
            [self.block_size, self.block_size],
            q_input_layout="TND",
            kv_input_layout="TND",
            num_key_value_heads=key.shape[1],
            scale_value=self.softmax_scale,
            inner_precise=4 if self._is_a5 else 0,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths,
            softmax_lse_flag=0,
        )
        return output

    def _block_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Block sparse attention kernel. Input layout BSND, output layout BNSD"""
        # TODO A5 supports BSND layout.
        actual_seq_lengths = [query.shape[1]] * query.shape[0]
        actual_seq_lengths_kv = [key.shape[1]] * key.shape[0]

        output, _ = torch_npu.npu_block_sparse_attention(
            query.permute(0, 2, 1, 3).contiguous(),
            key.permute(0, 2, 1, 3).contiguous(),
            value.permute(0, 2, 1, 3).contiguous(),
            block_sparse_mask,
            [self.block_size, self.block_size],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=key.shape[2],
            scale_value=self.softmax_scale,
            inner_precise=4 if self._is_a5 else 0,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            softmax_lse_flag=0,
        )
        return output

    def _restore_spatial_order(
        self, output: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        """Restore original BSND order directly from the BNSD kernel output."""
        num_frames, height, width = latent_shape
        batch_size, num_heads, seq_len, head_dim = output.shape

        if num_frames == 1 or height < self.tile_size or width < self.tile_size:
            return output.permute(0, 2, 1, 3).contiguous()

        height_block_count, height_remainder = divmod(height, self.tile_size)
        width_block_count, width_remainder = divmod(width, self.tile_size)

        if height_remainder == 0 and width_remainder == 0:
            return (
                output.reshape(
                    batch_size,
                    num_heads,
                    num_frames,
                    height_block_count,
                    width_block_count,
                    self.tile_size,
                    self.tile_size,
                    head_dim,
                )
                .permute(0, 2, 3, 5, 4, 6, 1, 7)
                .contiguous()
                .reshape(batch_size, seq_len, num_heads, head_dim)
            )

        first_frame_token_count = height * width
        first_frame = output[:, :, :first_frame_token_count].permute(0, 2, 1, 3)
        tail_frames = output[:, :, first_frame_token_count:].reshape(
            batch_size,
            num_heads,
            num_frames - 1,
            height * width,
            head_dim,
        )

        aligned_height = height_block_count * self.tile_size
        aligned_width = width_block_count * self.tile_size
        tiled_token_count = (
            height_block_count * width_block_count * self.tile_size * self.tile_size
        )
        height_tail_tokens = height_remainder * width

        restored_frames = (
            tail_frames[:, :, :, :tiled_token_count]
            .reshape(
                batch_size,
                num_heads,
                num_frames - 1,
                height_block_count,
                width_block_count,
                self.tile_size,
                self.tile_size,
                head_dim,
            )
            .permute(0, 2, 3, 5, 4, 6, 1, 7)
            .contiguous()
            .reshape(
                batch_size,
                num_frames - 1,
                aligned_height,
                aligned_width,
                num_heads,
                head_dim,
            )
        )

        if width_remainder:
            width_tail_tokens = tail_frames[
                :, :, :, tiled_token_count + height_tail_tokens :
            ]
            width_tail_tokens = width_tail_tokens.reshape(
                batch_size,
                num_heads,
                num_frames - 1,
                aligned_height,
                width_remainder,
                head_dim,
            ).permute(0, 2, 3, 4, 1, 5)
            restored_frames = torch.cat((restored_frames, width_tail_tokens), dim=3)

        if height_remainder:
            height_remainder_tokens = tail_frames[
                :,
                :,
                :,
                tiled_token_count : (tiled_token_count + height_tail_tokens),
            ]
            height_remainder_tokens = height_remainder_tokens.reshape(
                batch_size,
                num_heads,
                num_frames - 1,
                height_remainder,
                width,
                head_dim,
            ).permute(0, 2, 3, 4, 1, 5)
            restored_frames = torch.cat(
                (restored_frames, height_remainder_tokens), dim=2
            )

        remaining_tokens = restored_frames.reshape(
            batch_size,
            (num_frames - 1) * height * width,
            num_heads,
            head_dim,
        )
        return torch.cat((first_frame, remaining_tokens), dim=1)

    def _rain_fusion_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
        sparsity: float,
    ) -> torch.Tensor:
        query, key, value, qkv_pool = self._prepare_rearranged_qkv(
            query,
            key,
            value,
            latent_shape,
        )
        block_sparse_mask = self._build_sparse_mask(qkv_pool, sparsity, latent_shape)
        output = self._block_sparse_attention(query, key, value, block_sparse_mask)
        return self._restore_spatial_order(output, latent_shape)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: RainFusionAttentionMetadata,
    ) -> torch.Tensor:

        if attn_metadata.current_timestep < attn_metadata.skip_first_steps:
            return self.dense_attn_impl.forward(query, key, value, attn_metadata)

        return self._rain_fusion_sparse_attention(
            query,
            key,
            value,
            attn_metadata.latent_shape,
            attn_metadata.sparsity,
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run packed self-attention for TND ``[total_tokens, heads, dim]`` input."""
        boundaries = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != query.shape[0]
            or any(stop < start for start, stop in zip(boundaries[:-1], boundaries[1:]))
        ):
            raise ValueError(
                "cu_seqlens must start at 0, be non-decreasing, and end at "
                f"the packed token count {query.shape[0]}"
            )
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError(
                "Rain Fusion TND self-attention requires matching Q/K/V token "
                "counts and matching K/V head counts"
            )

        if (
            not self.layer_enabled
            or get_forward_context().current_timestep < self.skip_first_steps
        ):
            return self.dense_attn_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=boundaries,
            )
        return self._rain_fusion_sparse_attention_varlen(
            query,
            key,
            value,
            boundaries=boundaries,
            max_seqlen=max_seqlen,
        )
