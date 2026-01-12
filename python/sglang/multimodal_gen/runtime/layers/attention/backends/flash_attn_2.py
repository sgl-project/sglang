# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

try:
    from flash_attn import flash_attn_func  # Use the flash attention v2 function
except ImportError:

    def _unsupported(*args, **kwargs):
        raise ImportError(
            "flash-attn is not installed. Please install it, e.g., `pip install flash-attn`."
        )

    flash_attn_func = _unsupported

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
        return None


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
        output = flash_attn_func(
            q=query,  # type: ignore[no-untyped-call]
            k=key,
            v=value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output
