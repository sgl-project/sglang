# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from flash_attn_interface import flash_attn_varlen_func
except ImportError as e:
    raise ImportError(
        "flash-attention library is required. Please install it with: "
        "pip install flash-attn --no-build-isolation"
    ) from e

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    _get_cu_seqlens,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FlashAttention2Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA2

    @staticmethod
    def get_impl_cls() -> type["FlashAttention2Impl"]:
        return FlashAttention2Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class FlashAttention2Impl(AttentionImpl):

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
    ):
        bsz, seqlen, nheads_q, d = query.shape
        bsz_k, seqlen_k, nheads_k, d_k = key.shape

        q_ = query.contiguous().reshape(bsz * seqlen, nheads_q, d)
        k_ = key.contiguous().reshape(bsz * seqlen_k, nheads_k, d_k)
        v_ = value.contiguous().reshape(bsz * seqlen_k, nheads_k, value.shape[-1])

        cu_seqlens_q = _get_cu_seqlens(q_.device.index, bsz, seqlen)
        cu_seqlens_k = _get_cu_seqlens(k_.device.index, bsz, seqlen_k)

        out = flash_attn_varlen_func(
            q_,
            k_,
            v_,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen,
            seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )

        return out.reshape(bsz, seqlen, nheads_q, -1)
