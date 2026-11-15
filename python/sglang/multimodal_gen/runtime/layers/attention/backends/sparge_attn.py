# SPDX-License-Identifier: Apache-2.0

from numbers import Real
from typing import Any

import torch
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

_SUPPORTED_HEAD_SIZES = (64, 128)
_MIN_SEQUENCE_LENGTH = 128
_DEFAULT_TOPK = 0.5


def _validate_topk(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("SpargeAttention 'topk' must be a number in (0, 1].")
    topk = float(value)
    if not 0.0 < topk <= 1.0:
        raise ValueError("SpargeAttention 'topk' must be in (0, 1].")
    return topk


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    return (
        torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        .transpose(1, 2)
        .contiguous()
    )


class SpargeAttentionBackend(AttentionBackend):
    """Training-free sparse, quantized attention backed by SpargeAttention."""

    @staticmethod
    def get_supported_head_sizes() -> tuple[int, ...]:
        return _SUPPORTED_HEAD_SIZES

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl


class SpargeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(
                "SpargeAttention supports head sizes 64 and 128, "
                f"but received {head_size}."
            )
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError(
                "SpargeAttention does not support grouped-query attention."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        config = get_global_server_args().attention_backend_config or {}
        self.topk = _validate_topk(config.get("topk", _DEFAULT_TOPK))
        self._attention_op = spas_sage2_attn_meansim_topk_cuda

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if query.device.type != "cuda":
            raise ValueError("SpargeAttention requires CUDA tensors.")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("SpargeAttention requires FP16 or BF16 inputs.")
        if key.device != query.device or value.device != query.device:
            raise ValueError("SpargeAttention requires Q, K, and V on one device.")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("SpargeAttention requires Q, K, and V with one dtype.")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "SpargeAttention expects Q, K, and V in "
                "[batch, sequence, heads, head_dim] layout."
            )
        if (
            query.shape[0] != key.shape[0]
            or query.shape[0] != value.shape[0]
            or query.shape[2:] != key.shape[2:]
            or query.shape[2:] != value.shape[2:]
            or key.shape[1] != value.shape[1]
            or query.shape[-1] != self.head_size
        ):
            raise ValueError(
                "SpargeAttention requires compatible Q, K, and V batch, head, "
                f"and head-dim shapes with head_dim={self.head_size}."
            )

        # The upstream sparse kernel is square self-attention only and requires
        # at least one 128-token block. LTX can produce shorter audio sequences
        # and asymmetric Q/KV in its replicated-audio SP path.
        if (
            query.shape != key.shape
            or query.shape != value.shape
            or query.shape[1] < _MIN_SEQUENCE_LENGTH
        ):
            return _dense_attention(
                query,
                key,
                value,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

        output = self._attention_op(
            query,
            key,
            value,
            is_causal=self.causal,
            scale=self.softmax_scale,
            tensor_layout="NHD",
            topk=self.topk,
        )
        return output.contiguous()
