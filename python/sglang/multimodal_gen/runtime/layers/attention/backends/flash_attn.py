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

    # flash_attn 3 no longer have a different API, see following commit:
    # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
    flash_attn_func = flash_attn_varlen_func
except ImportError as e:
    raise e


try:
    from flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_upstream,
    )
except Exception:
    flash_attn_varlen_func_upstream = None

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

fa_ver = 3


@lru_cache(maxsize=128)
def _get_cu_seqlens(device_index: int, bsz: int, seqlen: int) -> torch.Tensor:
    return torch.arange(
        0,
        (bsz + 1) * seqlen,
        step=seqlen,
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )


@lru_cache(maxsize=256)
def _should_use_upstream_flash_attention(
    upstream_available: bool,
    upstream_heads_ok: bool,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> bool:
    if not upstream_available or not upstream_heads_ok:
        return False

    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return False

    bsz, seqlen, nheads_q, d = q_shape
    bsz_k, seqlen_k, nheads_k, d_k = k_shape
    bsz_v, seqlen_v, nheads_v, d_v = v_shape

    if (
        bsz != bsz_k
        or bsz != bsz_v
        or seqlen != seqlen_k
        or seqlen != seqlen_v
        or d != d_k
        or d != d_v
    ):
        return False
    if nheads_k != nheads_v:
        return False
    if nheads_k == 0 or (nheads_q % nheads_k) != 0:
        return False
    return True


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
        if self.num_kv_heads is None:
            self._upstream_heads_ok = True
        else:
            # For gqa, the num_heads must be a multiple of num_kv_heads
            self._upstream_heads_ok = (
                self.num_kv_heads > 0 and (self.num_heads % self.num_kv_heads) == 0
            )

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
        q_shape = tuple(query.shape)
        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)
        use_upstream = _should_use_upstream_flash_attention(
            flash_attn_varlen_func_upstream is not None,
            self._upstream_heads_ok,
            q_shape,
            k_shape,
            v_shape,
        )

        if use_upstream:
            bsz, seqlen, nheads_q, d = q_shape
            bsz_k, seqlen_k, nheads_k, d_k = k_shape
            bsz_v, seqlen_v, nheads_v, d_v = v_shape
            q_ = query.contiguous().reshape(bsz * seqlen, nheads_q, d)
            k_ = key.contiguous().reshape(bsz * seqlen, nheads_k, d)
            v_ = value.contiguous().reshape(bsz * seqlen, nheads_v, d)
            cu = _get_cu_seqlens(q_.device.index, bsz, seqlen)
            out = flash_attn_varlen_func_upstream(
                q_,
                k_,
                v_,
                cu,
                cu,
                seqlen,
                seqlen,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
            return out.reshape(bsz, seqlen, nheads_q, d)

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
