import math
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.platforms.npu import NPUPlatformEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Rain Fusion groups spatial tokens into 8x8 tiles before 128-token blocks.
RAIN_FUSION_TILE_SIZE = 8


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
                    "For better performance use dense attention or increase sparsity."
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
        self.tile_size = RAIN_FUSION_TILE_SIZE
        self._is_a5 = current_platform.return_platform() == NPUPlatformEnum.ASCEND_950
        # CANN: 4 selects mixed precision on A5; 0 selects high precision,
        # which is required for BF16 on A2/A3. Keep the established A5 mode.
        self.inner_precise = 4 if self._is_a5 else 0

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

    def _avgpool(
        self, input_tensor: torch.Tensor, pool_size: int = 128
    ) -> torch.Tensor:
        batch, seqlen, heads, dim = input_tensor.shape
        full_blocks, tail_tokens = divmod(seqlen, pool_size)

        if full_blocks == 0:
            if tail_tokens == 0:
                return input_tensor.new_empty((batch, 0, heads, dim))
            return input_tensor.mean(dim=1, keepdim=True)

        full_block_tokens = full_blocks * pool_size
        full_blocks_pool = (
            input_tensor[:, :full_block_tokens]
            .reshape(batch, full_blocks, pool_size, heads, dim)
            .mean(dim=2)
        )

        if tail_tokens == 0:
            return full_blocks_pool

        tail_pool = input_tensor[:, full_block_tokens:].mean(dim=1, keepdim=True)
        return torch.cat((full_blocks_pool, tail_pool), dim=1)

    def _get_blockwise_mask(
        self,
        qkv_pool: torch.Tensor,
        sparsity: float,
        scale: float,
        pool_size: int,
        latent_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
        return self._get_blockwise_mask_from_pools(
            query_pool, key_pool, sparsity, scale, pool_size, latent_shape
        )

    def _get_blockwise_mask_from_pools(
        self,
        query_pool: torch.Tensor,
        key_pool: torch.Tensor,
        sparsity: float,
        scale: float,
        pool_size: int,
        latent_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        attn_scores = (
            query_pool.permute(0, 2, 1, 3) @ key_pool.permute(0, 2, 3, 1) * scale
        )
        # Retain probability-based selection: ranking logits directly changes
        # low-precision ties and has different observed generation quality.
        probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        keep_len = math.ceil(probs.shape[-1] * (1 - sparsity))
        topk_values, _ = torch.topk(probs, k=keep_len, dim=-1)
        mask = probs >= topk_values[..., -1:]

        if latent_shape is not None:
            first_frame_len = latent_shape[1] * latent_shape[2]
            firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
            if firstframe_block_num > 0:
                mask[:, :, :firstframe_block_num, :] = True
                mask[:, :, :, :firstframe_block_num] = True
        return mask.to(torch.int8)

    def _rearrange_with_remaining(
        self, tensor: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        """Pack each frame as spatial tiles, bottom rows, then right columns."""
        tq, hq, wq = latent_shape
        b, s, n, d = tensor.shape
        if tq == 1 or hq < self.tile_size or wq < self.tile_size:
            return tensor

        hn, h_r = divmod(hq, self.tile_size)
        wn, w_r = divmod(wq, self.tile_size)
        # Keep the first frame intact when the spatial grid has partial tiles.
        first_frame_len = hq * wq if h_r or w_r else 0
        frame_num = tq - bool(first_frame_len)
        tensor_first = tensor[:, :first_frame_len]
        tensor_hwt = tensor[:, first_frame_len:].reshape(b, frame_num, hq, wq, n, d)
        tensor_h_r = tensor_hwt[:, :, hn * self.tile_size :].reshape(
            b, frame_num, -1, n, d
        )
        tensor_w_r = tensor_hwt[
            :, :, : hn * self.tile_size, wn * self.tile_size :
        ].reshape(b, frame_num, -1, n, d)
        tensor_hwt = (
            tensor_hwt[:, :, : hn * self.tile_size, : wn * self.tile_size]
            .reshape(b, frame_num, hn, self.tile_size, wn, self.tile_size, n, d)
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .reshape(b, frame_num, -1, n, d)
        )
        if h_r or w_r:
            tensor_hwt = torch.cat((tensor_hwt, tensor_h_r, tensor_w_r), dim=2)
            return torch.cat((tensor_first, tensor_hwt.reshape(b, -1, n, d)), dim=1)
        return tensor_hwt.reshape(b, s, n, d)

    def _inv_rearrange_with_remaining(
        self, tensor: torch.Tensor, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        tq, hq, wq = latent_shape
        b, s, n, d = tensor.shape
        if tq == 1 or hq < self.tile_size or wq < self.tile_size:
            return tensor

        hn, h_r = divmod(hq, self.tile_size)
        wn, w_r = divmod(wq, self.tile_size)
        first_frame_len = hq * wq if h_r or w_r else 0
        frame_num = tq - bool(first_frame_len)
        tensor_first = tensor[:, :first_frame_len]
        tensor = tensor[:, first_frame_len:].reshape(b, frame_num, -1, n, d)
        tiled_len = hn * wn * self.tile_size**2
        height_tail_len = h_r * wq
        tensor_hwt = (
            tensor[:, :, :tiled_len]
            .reshape(b, frame_num, hn, wn, self.tile_size, self.tile_size, n, d)
            .permute(0, 1, 2, 4, 3, 5, 6, 7)
            .reshape(b, frame_num, hn * self.tile_size, wn * self.tile_size, n, d)
        )
        if h_r or w_r:
            tensor_h_r = tensor[:, :, tiled_len : tiled_len + height_tail_len].reshape(
                b, frame_num, h_r, wq, n, d
            )
            tensor_w_r = tensor[:, :, tiled_len + height_tail_len :].reshape(
                b, frame_num, hn * self.tile_size, w_r, n, d
            )
            tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)
            tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            return torch.cat((tensor_first, tensor_hwt.reshape(b, -1, n, d)), dim=1)
        return tensor_hwt.reshape(b, s, n, d)

    def _do_tensor_rearrange_pooling(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pool_size: int,
        latent_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tensor block rearrangement + pooling operation
        """
        tensor = torch.cat((query, key, value), dim=0)

        tensor = self._rearrange_with_remaining(tensor, latent_shape)
        tensor_pool = self._avgpool(tensor, pool_size)

        query_, key_, value_ = torch.chunk(tensor, 3, dim=0)
        return query_, key_, value_, tensor_pool

    def _pad_head_dim(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CANN BSA accepts D=64/128. Zero padding preserves the original scale.
        head_dim = query.shape[-1]
        if head_dim not in (32, 64, 96, 128):
            raise ValueError(f"Unsupported Rain Fusion head dimension: {head_dim}")
        padded_dim = 64 if head_dim <= 64 else 128
        if head_dim == padded_dim:
            return query, key, value
        return tuple(
            torch.nn.functional.pad(t, (0, padded_dim - head_dim))
            for t in (query, key, value)
        )

    def _rain_fusion_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """CANN block sparse attention with BSND input and output."""
        # TODO A5 supports BSND layout.
        actual_seq_lengths = [query.shape[1]] * query.shape[0]
        actual_seq_lengths_kv = [key.shape[1]] * key.shape[0]

        head_dim = query.shape[-1]
        query, key, value = self._pad_head_dim(query, key, value)
        output, _ = torch.ops.npu.npu_block_sparse_attention(
            query.permute(0, 2, 1, 3).contiguous(),
            key.permute(0, 2, 1, 3).contiguous(),
            value.permute(0, 2, 1, 3).contiguous(),
            block_sparse_mask,
            [self.block_size, self.block_size],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=key.shape[2],
            scale_value=self.softmax_scale,
            inner_precise=self.inner_precise,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            softmax_lse_flag=0,
        )
        return output[..., :head_dim].permute(0, 2, 1, 3).contiguous()

    def _rain_fusion_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
        sparsity: float,
    ) -> torch.Tensor:
        q, k, v, qkv_pool = self._do_tensor_rearrange_pooling(
            query, key, value, self.block_size, latent_shape
        )
        mask = self._get_blockwise_mask(
            qkv_pool, sparsity, self.softmax_scale, self.block_size, latent_shape
        )
        out = self._rain_fusion_attention(q, k, v, mask)
        out = self._inv_rearrange_with_remaining(out, latent_shape)
        return out

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

    def _build_varlen_sparse_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        boundaries: tuple[int, ...],
        max_seqlen: int,
        sparsity: float,
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
            query_pool = self._avgpool(query[start:stop].unsqueeze(0))
            key_pool = self._avgpool(key[start:stop].unsqueeze(0))
            mask = self._get_blockwise_mask_from_pools(
                query_pool,
                key_pool,
                sparsity,
                self.softmax_scale,
                self.block_size,
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
        sparsity: float,
    ) -> torch.Tensor:
        block_sparse_mask, actual_seq_lengths = self._build_varlen_sparse_mask(
            query, key, boundaries, max_seqlen, sparsity
        )
        if not actual_seq_lengths:
            return torch.empty_like(query)

        head_dim = query.shape[-1]
        query, key, value = self._pad_head_dim(query, key, value)
        output, _ = torch.ops.npu.npu_block_sparse_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            block_sparse_mask,
            [self.block_size, self.block_size],
            q_input_layout="TND",
            kv_input_layout="TND",
            num_key_value_heads=key.shape[1],
            scale_value=self.softmax_scale,
            inner_precise=self.inner_precise,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths,
            softmax_lse_flag=0,
        )
        return output[..., :head_dim].contiguous()

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
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            raise ValueError(
                "Rain Fusion TND self-attention requires matching Q/K/V token "
                "counts and matching K/V head counts"
            )

        # Refiner attention can run before the denoising forward context exists.
        context = get_forward_context_or_none()
        attn_metadata = context.attn_metadata if context is not None else None
        if attn_metadata is not None and not isinstance(
            attn_metadata, RainFusionAttentionMetadata
        ):
            raise ValueError(
                "Rain Fusion requires RainFusionAttentionMetadata "
                "in the active forward context"
            )
        if (
            attn_metadata is None
            or attn_metadata.current_timestep < attn_metadata.skip_first_steps
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
            sparsity=attn_metadata.sparsity,
        )
