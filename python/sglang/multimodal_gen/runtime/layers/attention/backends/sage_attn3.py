# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from sageattn3 import sageattn3_blackwell

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _trailing_padding_used_len(
    *,
    total_tokens: int,
    max_seqlen: int,
    bounds: tuple[int, ...],
) -> int | None:
    """Return live token count for H3-style [0, used, total] trailing padding."""
    if len(bounds) != 3:
        return None
    start, used, total = bounds
    if start != 0 or used >= total or total != total_tokens or used != max_seqlen:
        return None
    return used


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN_3

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError


class SageAttention3Impl(AttentionImpl):
    _warned_gqa_fallback_global: bool = False

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
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # SageAttention3's Blackwell kernel assumes MHA (Hq == Hkv). For GQA/MQA
        # (Hq != Hkv), fall back to torch SDPA which supports GQA.
        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            if not type(self)._warned_gqa_fallback_global:
                logger.warning(
                    "SageAttention3 does not support GQA/MQA (Hq != Hkv); falling back to torch SDPA."
                )
                type(self)._warned_gqa_fallback_global = True
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self.causal,
                dropout_p=self.dropout,
                scale=self.softmax_scale,
                enable_gqa=True,
            )
        else:
            # SageAttention3 centers K in place during preprocessing. Materialize
            # the transposed view so the backend does not mutate its caller's K.
            output = sageattn3_blackwell(
                query, key.contiguous(), value, is_causal=self.causal
            )
        output = output.transpose(1, 2)
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
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        return self._sage_packed(
            query,
            key,
            value,
            bounds=bounds,
            max_seqlen=max_seqlen,
        )

    def _sage_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        used = _trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        if used is not None:
            live_out = self.forward(
                query[:used].unsqueeze(0),
                key[:used].unsqueeze(0),
                value[:used].unsqueeze(0),
                None,
            )[0]
            output = torch.empty_like(query)
            output[:used].copy_(live_out)
            output[used:].zero_()
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )[0]
        return output
