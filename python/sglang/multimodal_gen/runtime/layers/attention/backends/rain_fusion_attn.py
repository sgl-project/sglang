import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Any, Optional

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Rain Fusion groups spatial tokens into 8x8 tiles before 128-token blocks.
RAIN_FUSION_TILE_SIZE = 8


class RainFusionAttentionBackend(AttentionBackend):
    """Rain Fusion attention adapted from MindIE-SD's RF-v3 implementation."""

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
    video_spans: list[dict[str, Any]] | None = None


class RainFusionAttentionMetadataBuilder(AttentionMetadataBuilder):
    def build(
        self,
        current_timestep: int,
        skip_first_steps: int,
        sparsity: float,
        raw_latent_shape: list[int],
        patch_size: tuple[int, int, int],
        video_spans: Optional[list[dict[str, Any]]] = None,
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

        previous_end = 0
        for span in video_spans or []:
            start, shape = span["start"], span["latent_shape"]
            if (
                not isinstance(start, int)
                or start < previous_end
                or len(shape) != 3
                or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            ):
                raise ValueError(
                    "video_spans must be ordered, nonoverlapping [T, H, W] grids"
                )
            previous_end = start + math.prod(shape)

        return RainFusionAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            latent_shape=latent_shape,
            video_spans=video_spans,
        )


class RainFusionAttentionImpl(AttentionImpl):
    """MindIE-SD RF-v3 adapted for SGLang"""

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
        from sglang.multimodal_gen.runtime.platforms.npu import NPUGenerationEnum

        if causal:
            raise ValueError("Rain Fusion Attention does not support causal masks")

        self.softmax_scale = softmax_scale
        # MiniMax-H3 declares bounds=(0, used, total), with max_seqlen=used.
        # This flag permits skipping the padding tail and zeroing its output.
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
        self.block_size = 128
        self.tile_size = RAIN_FUSION_TILE_SIZE
        self._is_a5 = (
            current_platform.platform_generation() == NPUGenerationEnum.ASCEND_950
        )
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

    @staticmethod
    def _rearrange_with_remaining(
        tensor: torch.Tensor,
        latent_shape: tuple[int, int, int],
        tile_size: int = RAIN_FUSION_TILE_SIZE,
    ) -> torch.Tensor:
        """Pack each frame as spatial tiles, bottom rows, then right columns."""
        tq, hq, wq = latent_shape
        b, s, n, d = tensor.shape
        if tq == 1 or hq < tile_size or wq < tile_size:
            return tensor

        hn, h_r = divmod(hq, tile_size)
        wn, w_r = divmod(wq, tile_size)
        # Keep the first frame intact when the spatial grid has partial tiles.
        first_frame_len = hq * wq if h_r or w_r else 0
        frame_num = tq - bool(first_frame_len)
        tensor_first = tensor[:, :first_frame_len]
        tensor_hwt = tensor[:, first_frame_len:].reshape(b, frame_num, hq, wq, n, d)
        tensor_h_r = tensor_hwt[:, :, hn * tile_size :].reshape(b, frame_num, -1, n, d)
        tensor_w_r = tensor_hwt[:, :, : hn * tile_size, wn * tile_size :].reshape(
            b, frame_num, -1, n, d
        )
        tensor_hwt = (
            tensor_hwt[:, :, : hn * tile_size, : wn * tile_size]
            .reshape(b, frame_num, hn, tile_size, wn, tile_size, n, d)
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

    def _prepare_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
        sparsity: float,
        video_spans: list[dict[str, Any]] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        if video_spans == []:
            block_count = math.ceil(query.shape[1] / self.block_size)
            mask = query.new_ones(
                (query.shape[0], query.shape[2], block_count, block_count),
                dtype=torch.int8,
            )
            return query, key, value, mask, None
        inverse = None
        if video_spans is not None:
            q, k, v, qk_pool, inverse, retained = (
                self._do_multi_span_tensor_rearrange_pooling(
                    query, key, value, video_spans
                )
            )
            query_pool, key_pool = torch.chunk(qk_pool, 2, dim=0)
            mask = self._get_blockwise_mask_from_pools(
                query_pool,
                key_pool,
                sparsity,
                self.softmax_scale,
                self.block_size,
                None,
            )
            if retained.numel():
                mask[:, :, retained, :] = 1
                mask[:, :, :, retained] = 1
        else:
            if query.shape[1] != math.prod(latent_shape):
                raise ValueError(
                    "Without video_spans, each sequence must match latent_shape; "
                    "provide explicit spans for mixed video/context sequences"
                )
            q, k, v, qkv_pool = self._do_tensor_rearrange_pooling(
                query, key, value, self.block_size, latent_shape
            )
            mask = self._get_blockwise_mask(
                qkv_pool, sparsity, self.softmax_scale, self.block_size, latent_shape
            )
        return q, k, v, mask, inverse

    def _rain_fusion_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        latent_shape: tuple[int, int, int],
        sparsity: float,
        video_spans: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        if video_spans == []:
            return self.dense_attn_impl.forward(query, key, value, None)
        q, k, v, mask, inverse = self._prepare_sparse_attention(
            query, key, value, latent_shape, sparsity, video_spans
        )
        out = self._rain_fusion_attention(q, k, v, mask)
        if inverse is not None:
            return out.index_select(1, inverse)
        return self._inv_rearrange_with_remaining(out, latent_shape)

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
            attn_metadata.video_spans,
        )

    def _video_span_permutation(
        self, query: torch.Tensor, video_spans: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._cached_video_span_permutation(
            query.device,
            query.shape[1],
            self.block_size,
            self.tile_size,
            tuple((span["start"], tuple(span["latent_shape"])) for span in video_spans),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_video_span_permutation(
        device: torch.device,
        sequence_length: int,
        block_size: int,
        tile_size: int,
        video_spans: tuple[tuple[int, tuple[int, int, int]], ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device)
        dense_parts = []
        previous_end = 0
        for start, shape in video_spans:
            end = start + math.prod(shape)
            if start < previous_end or end > sequence_length:
                raise ValueError("video_spans must be ordered and within the sequence")
            dense_parts.append(positions[previous_end:start])
            previous_end = end
        dense_parts.append(positions[previous_end:])
        dense_indices = torch.cat(dense_parts)

        parts = []
        retained: set[int] = set()
        position = dense_cursor = 0
        for index, (start, shape) in enumerate(video_spans):
            length = math.prod(shape)
            clip_block = position // block_size
            retained.update(
                range(
                    clip_block,
                    clip_block + math.ceil(shape[1] * shape[2] / block_size),
                )
            )
            parts.append(
                RainFusionAttentionImpl._rearrange_with_remaining(
                    positions[start : start + length].reshape(1, -1, 1, 1),
                    shape,
                    tile_size,
                ).reshape(-1)
            )
            position += length
            if index != len(video_spans) - 1 and position % block_size:
                needed = -position % block_size
                if dense_cursor + needed > dense_indices.numel():
                    raise ValueError(
                        "Rain Fusion needs dense context rows to align video spans; "
                        "use dense attention for this layout"
                    )
                retained.add(position // block_size)
                parts.append(dense_indices[dense_cursor : dense_cursor + needed])
                dense_cursor += needed
                position += needed

        if dense_cursor < dense_indices.numel():
            retained.update(
                range(
                    position // block_size,
                    math.ceil(sequence_length / block_size),
                )
            )
            parts.append(dense_indices[dense_cursor:])
        permutation = torch.cat(parts)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = positions
        retained_indices = torch.tensor(
            sorted(retained), device=device, dtype=torch.long
        )
        return permutation, inverse, retained_indices

    def _do_multi_span_tensor_rearrange_pooling(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        video_spans: list[dict[str, Any]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if query.shape != key.shape or key.shape != value.shape:
            raise ValueError(
                "Rain Fusion multi-span self-attention requires equal Q/K/V shapes"
            )
        permutation, inverse, retained = self._video_span_permutation(
            query, video_spans
        )
        q, k, v = (
            tensor.index_select(1, permutation) for tensor in (query, key, value)
        )
        qk_pool = torch.cat(
            (self._avgpool(q, self.block_size), self._avgpool(k, self.block_size)),
            dim=0,
        )
        return q, k, v, qk_pool, inverse, retained

    def _rain_fusion_sparse_attention_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        boundaries: tuple[int, ...],
        max_seqlen: int,
        sparsity: float,
        latent_shape: tuple[int, int, int],
        video_spans: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        segments = [(a, b) for a, b in pairwise(boundaries) if b > a]
        if not segments:
            return torch.empty_like(query)
        used = (
            trailing_padding_used_len(query.shape[0], max_seqlen, boundaries)
            if self.packed_trailing_padding
            else None
        )
        previous_end = 0
        for span in video_spans or []:
            start, shape = span["start"], span["latent_shape"]
            if len(shape) != 3 or any(dim <= 0 for dim in shape):
                raise ValueError("video_spans require positive [T, H, W] grids")
            end = start + math.prod(shape)
            if (
                start < previous_end
                or (used is not None and end > used)
                or not any(a <= start < end <= b for a, b in segments)
            ):
                raise ValueError(
                    "video_spans must be ordered, nonoverlapping, and contained "
                    "within a live packed sequence"
                )
            previous_end = end
        if used is not None:
            segments = [(0, used)] if used else []
        if not segments:
            return torch.zeros_like(query)
        lengths = [b - a for a, b in segments]

        if len(segments) == 1:
            start, stop = segments[0]
            output = self._rain_fusion_sparse_attention(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                latent_shape,
                sparsity,
                video_spans,
            )[0]
            if used is None:
                return output.contiguous()
            padded_output = torch.zeros_like(query)
            padded_output[:used] = output
            return padded_output

        block_count = math.ceil(max(lengths) / self.block_size)
        queries, keys, values, masks, layouts = [], [], [], [], []
        for start, stop in segments:
            local_spans = (
                None
                if video_spans is None
                else [
                    {
                        "start": span["start"] - start,
                        "latent_shape": span["latent_shape"],
                    }
                    for span in video_spans
                    if start <= span["start"] < stop
                ]
            )
            q, k, v, mask, inverse = self._prepare_sparse_attention(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                latent_shape,
                sparsity,
                local_spans,
            )
            queries.append(q[0])
            keys.append(k[0])
            values.append(v[0])
            masks.append(
                torch.nn.functional.pad(
                    mask,
                    (0, block_count - mask.shape[-1], 0, block_count - mask.shape[-2]),
                )
            )
            layouts.append((inverse, local_spans))

        head_dim = query.shape[-1]
        q, k, v = self._pad_head_dim(
            torch.cat(queries), torch.cat(keys), torch.cat(values)
        )
        # BSA takes segment lengths, not cumulative cu_seqlens offsets.
        output, _ = torch.ops.npu.npu_block_sparse_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            torch.cat(masks),
            [self.block_size, self.block_size],
            q_input_layout="TND",
            kv_input_layout="TND",
            num_key_value_heads=k.shape[1],
            scale_value=self.softmax_scale,
            inner_precise=self.inner_precise,
            actual_seq_lengths=lengths,
            actual_seq_lengths_kv=lengths,
            softmax_lse_flag=0,
        )
        output = output[..., :head_dim]
        outputs = []
        for (start, stop), (inverse, local_spans) in zip(segments, layouts):
            segment = output[start:stop]
            if inverse is not None:
                segment = segment.index_select(0, inverse)
            elif local_spans is None:
                segment = self._inv_rearrange_with_remaining(
                    segment[None], latent_shape
                )[0]
            outputs.append(segment)
        return torch.cat(outputs).contiguous()

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
        boundaries = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != query.shape[0]
            or any(stop < start for start, stop in pairwise(boundaries))
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

        context = get_forward_context_or_none()
        attn_metadata = context.attn_metadata if context is not None else None
        if not isinstance(attn_metadata, RainFusionAttentionMetadata):
            raise ValueError(
                "Rain Fusion requires RainFusionAttentionMetadata "
                "in the active forward context"
            )
        if attn_metadata.current_timestep < attn_metadata.skip_first_steps:
            used = (
                trailing_padding_used_len(query.shape[0], max_seqlen, boundaries)
                if self.packed_trailing_padding
                else None
            )
            if used is not None:
                output = torch.zeros_like(query)
                if used:
                    output[:used] = self.dense_attn_impl.forward_varlen(
                        query[:used],
                        key[:used],
                        value[:used],
                        cu_seqlens=cu_seqlens[:2],
                        max_seqlen=used,
                        cu_seqlens_host=boundaries[:2],
                    )
                return output
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
            latent_shape=attn_metadata.latent_shape,
            video_spans=attn_metadata.video_spans,
        )
