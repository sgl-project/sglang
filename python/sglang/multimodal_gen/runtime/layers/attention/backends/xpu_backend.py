# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
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


class XPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.XPU_FA

    @staticmethod
    def get_impl_cls() -> type["XPUAttentionImpl"]:
        return XPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder


@lru_cache(maxsize=128)
def _get_cu_seqlens(device_index: int, bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * seqlen,
        step=seqlen,
        device=torch.device("xpu", device_index),
        dtype=torch.int32,
    )


class XPUAttentionImpl(AttentionImpl):

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
        attn_metadata: FlashAttentionMetadata = get_forward_context().attn_metadata

        bsz, seqlen_q, nheads_q, d = tuple(query.shape)
        _, seqlen_k, nheads_k, _ = tuple(key.shape)

        if attn_metadata is not None and attn_metadata.max_seqlen_q is None:
            attn_metadata.max_seqlen_q = seqlen_q
            attn_metadata.max_seqlen_k = seqlen_k
            max_seqlen_q = seqlen_q
            max_seqlen_k = seqlen_k
        else:
            max_seqlen_q = seqlen_q
            max_seqlen_k = seqlen_k

        q_ = query.contiguous().reshape(bsz * seqlen_q, nheads_q, d)
        k_ = key.contiguous().reshape(bsz * seqlen_k, nheads_k, d)
        v_ = value.contiguous().reshape(bsz * seqlen_k, nheads_k, d)
        cu_q = _get_cu_seqlens(q_.device.index, bsz, seqlen_q)
        cu_k = _get_cu_seqlens(q_.device.index, bsz, seqlen_k)

        out = flash_attn_func(
            q=q_,
            k=k_,
            v=v_,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            return_softmax_lse=return_softmax_lse,
        )

        if return_softmax_lse:
            out_tensor, softmax_lse = out[:2]
            result = out_tensor.reshape(bsz, seqlen_q, nheads_q, d)
            return result, softmax_lse

        result = out.reshape(bsz, seqlen_q, nheads_q, d)
        return result
