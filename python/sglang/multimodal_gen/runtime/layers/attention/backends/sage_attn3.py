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
            output = sageattn3_blackwell(query, key, value, is_causal=self.causal)
        output = output.transpose(1, 2)
        return output
