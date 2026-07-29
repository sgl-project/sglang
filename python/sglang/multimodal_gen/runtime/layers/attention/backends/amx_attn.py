# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

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
        max_seqlen_q = query.shape[1]
        max_seqlen_k = key.shape[1]
        return flash_attn_varlen_func(
            query[0],
            key[0],
            value[0],
            torch.tensor([0, max_seqlen_q]).to(torch.int),
            torch.tensor([0, max_seqlen_k]).to(torch.int),
            max_seqlen_q,
            max_seqlen_k,
            self.causal,
            self.softmax_scale,
        ).unsqueeze(0)
