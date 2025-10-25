# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    # from a import b
    # flash_attn 3 no longer have a different API, see following commit:
    # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e
    # flash_attn_func = flash_attn_2_func

from sgl_diffusion.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sgl_diffusion.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


class FlashAttentionImpl(AttentionImpl):

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
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output
