# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD block-FP8 attention through its public quant_attention API."""

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

try:
    from mindiesd import quant_attention
except ImportError as error:
    raise ImportError(
        "FIA Attention requires MindIE-SD with quant_attention and the "
        "mindiesd.fused_infer_attention_score_v2 operator."
    ) from error


class FIAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FIA_ATTN

    @staticmethod
    def get_impl_cls() -> type["FIAAttentionImpl"]:
        return FIAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None


class FIAAttentionImpl(AttentionImpl):
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
            raise ValueError(
                "FIA Attention's FP8 HIGH_PRECISION path does not support causal "
                "attention; select a backend with causal mask support."
            )
        self.softmax_scale = softmax_scale
        self.packed_trailing_padding = bool(
            extra_impl_args.get("packed_trailing_padding", False)
        )
        if extra_impl_args.get("dropout_p", 0.0):
            raise ValueError("FIA Attention does not support attention dropout.")

    def _forward_single(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # These options select MindIE-SD's block-FP8 path, whose final kernel is
        # torch.ops.mindiesd.fused_infer_attention_score_v2. Keep native errors
        # visible: selecting FIA must never silently switch attention kernels.
        return quant_attention(
            query,
            key,
            value,
            precision="fp8",
            layout="BSND",
            scale=self.softmax_scale,
            fp8_fa_mode="HIGH_PRECISION",
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            raise ValueError("FIA Attention requires Q/K/V in BSND layout.")
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            raise ValueError("Q/K/V batch sizes and K/V shapes must match.")
        if query.shape[0] == 0:
            return torch.empty_like(query)
        if query.shape[0] == 1:
            return self._forward_single(query, key, value)

        # MindIE-SD's block-FP8 implementation accepts batch size one. Execute
        # batches independently to retain the dense attention batch semantics.
        output = torch.empty_like(query)
        for index in range(query.shape[0]):
            output[index : index + 1].copy_(
                self._forward_single(
                    query[index : index + 1],
                    key[index : index + 1],
                    value[index : index + 1],
                )
            )
        return output

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
        if any(tensor.ndim != 3 for tensor in (query, key, value)):
            raise ValueError("Packed FIA Attention requires Q/K/V in TND layout.")
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            raise ValueError("Packed Q/K/V token counts and K/V shapes must match.")
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        if (
            len(bounds) < 2
            or bounds[0] != 0
            or bounds[-1] != query.shape[0]
            or any(start > stop for start, stop in zip(bounds[:-1], bounds[1:]))
        ):
            raise ValueError("cu_seqlens must monotonically cover all packed tokens.")

        # MiniMax-H3 marks this implementation with packed_trailing_padding=True.
        # In that contract [0, used, padded] contains one real sequence followed
        # by alignment rows, and max_seqlen == used. Do not execute attention on
        # those padding rows. The helper deliberately does not classify a generic
        # [0, a, b] two-sequence packing as padding unless max_seqlen == a.
        used = (
            trailing_padding_used_len(
                total_tokens=query.shape[0],
                max_seqlen=max_seqlen,
                bounds=bounds,
            )
            if self.packed_trailing_padding
            else None
        )
        if used is not None:
            output = torch.empty_like(query)
            if used > 0:
                output[:used].copy_(
                    self._forward_single(
                        query[:used].unsqueeze(0),
                        key[:used].unsqueeze(0),
                        value[:used].unsqueeze(0),
                    )[0]
                )
            output[used:].zero_()
            return output

        # Generic packed segments are independent. In particular, [0, a, b]
        # can describe two real sequences and must not be treated as padding.
        segments = [
            (start, stop)
            for start, stop in zip(bounds[:-1], bounds[1:])
            if start < stop
        ]
        if len(segments) == 1:
            return self._forward_single(
                query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
            )[0]
        output = torch.empty_like(query)
        for start, stop in segments:
            output[start:stop].copy_(
                self._forward_single(
                    query[start:stop].unsqueeze(0),
                    key[start:stop].unsqueeze(0),
                    value[start:stop].unsqueeze(0),
                )[0]
            )
        return output
