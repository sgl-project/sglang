# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=128)
def _get_cu_seqlens(device_index: int, bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * seqlen,
        step=seqlen,
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )


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
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
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
            return_softmax_lse=return_softmax_lse,
        )

        if return_softmax_lse:
            out, softmax_lse = out
            return out.reshape(bsz, seqlen, nheads_q, -1), softmax_lse
        return out.reshape(bsz, seqlen, nheads_q, -1)
