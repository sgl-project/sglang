# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func


@lru_cache(maxsize=128)
def _get_cu_seqlens(bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32)


class AMXAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AMX_ATTN

    @staticmethod
    def get_impl_cls() -> type["AMXATTNImpl"]:
        return AMXATTNImpl


class AMXATTNImpl(AttentionImpl):
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        bsz, seqlen_q, nheads_q, head_size = query.shape
        _, seqlen_k, nheads_k, _ = key.shape
        out = flash_attn_varlen_func(
            query.reshape(bsz * seqlen_q, nheads_q, head_size),
            key.reshape(bsz * seqlen_k, nheads_k, head_size),
            value.reshape(bsz * seqlen_k, nheads_k, value.shape[-1]),
            _get_cu_seqlens(bsz, seqlen_q),
            _get_cu_seqlens(bsz, seqlen_k),
            seqlen_q,
            seqlen_k,
            self.causal,
            self.softmax_scale,
        )
        return out.view(bsz, seqlen_q, nheads_q, -1)
