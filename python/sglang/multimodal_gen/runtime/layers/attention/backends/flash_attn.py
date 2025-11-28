# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    # flash_attn 3 no longer have a different API, see following commit:
    # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

fa_ver = 3


def set_fa_ver(ver: int):
    global fa_ver
    fa_ver = ver


@dataclass
class FlashAttentionMetadata:
    # Sequence lengths for the forward batch
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        raw_latent_shape=list,
        **kwargs: dict[str, Any],
    ) -> FlashAttentionMetadata:
        # TODO: put empty values here to be set at first-run, since the q_len calculation can be complicated
        return FlashAttentionMetadata(max_seqlen_q=None, max_seqlen_k=None)


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
        return FlashAttentionMetadataBuilder


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
        self.attention_metadata = FlashAttentionMetadata()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        *,
        return_softmax_lse: bool = False,
    ):
        attn_metadata: FlashAttentionMetadata = get_forward_context().attn_metadata
        if attn_metadata is not None and attn_metadata.max_seqlen_q is None:
            attn_metadata.max_seqlen_q = query.shape[1]
            attn_metadata.max_seqlen_k = key.shape[1]
            max_seqlen_q = attn_metadata.max_seqlen_q
            max_seqlen_k = attn_metadata.max_seqlen_k
        else:
            max_seqlen_q = query.shape[1]
            max_seqlen_k = key.shape[1]
        output = flash_attn_func(
            q=query,  # type: ignore[no-untyped-call]
            k=key,
            v=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            return_softmax_lse=return_softmax_lse,
            ver=fa_ver,
        )
        return output
