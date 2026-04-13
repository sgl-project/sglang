# SPDX-License-Identifier: Apache-2.0

"""ROCm AITER FP8 FMHA using ``aiter.flash_attn_fp8_pertensor_func``."""

from __future__ import annotations

import os

import torch

import aiter
from aiter import dtypes

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DISABLE_ENV = "SGLANG_ROCM_DISABLE_FMHA_FP8"


class AITerFP8Backend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_FP8

    @staticmethod
    def get_impl_cls() -> type["AITerFP8Impl"]:
        return AITerFP8Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER FP8 backend does not have a metadata builder."
        )


class AITerFP8Impl(AttentionImpl):
    """
    FP8 attention via AITER CK FMHA with per-tensor Q/K/V scales.

    Expects ``query``/``key``/``value`` in **[batch, seq, heads, head_dim]** — the
    same layout as :class:`USPAttention` and AITER's ``flash_attn_func`` docstring.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "AITER FP8 backend does not support grouped-query attention yet."
            )
        self.causal = causal
        self.dropout_p = dropout_p
        self._softmax_scale = softmax_scale

        fp8_fn = getattr(aiter, "flash_attn_fp8_pertensor_func", None)
        if fp8_fn is None:
            raise ImportError(
                "aiter.flash_attn_fp8_pertensor_func is not available. "
                "Install a ROCm AITER build that exports this API (see ROCm/aiter "
                "``aiter/ops/mha.py``)."
            )
        self._flash_attn_fp8 = fp8_fn

        try:
            from aiter.ops.quant import per_tensor_quant
        except ImportError as e:
            raise ImportError(
                "aiter.ops.quant.per_tensor_quant is required for AITER FP8 attention."
            ) from e
        self._per_tensor_quant = per_tensor_quant

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        out_dtype = query.dtype
        disable = os.environ.get(_DISABLE_ENV, "").lower() in ("1", "true", "yes")

        if disable:
            logger.info(
                "%s set: using bf16/fp16 aiter.flash_attn_func instead of FP8 FMHA.",
                _DISABLE_ENV,
            )
            output, _ = aiter.flash_attn_func(
                query,
                key,
                value,
                dropout_p=self.dropout_p,
                causal=self.causal,
                return_attn_probs=False,
                return_lse=True,
            )
            return output

        q8, q_descale = self._per_tensor_quant(query, quant_dtype=dtypes.fp8)
        k8, k_descale = self._per_tensor_quant(key, quant_dtype=dtypes.fp8)
        v8, v_descale = self._per_tensor_quant(value, quant_dtype=dtypes.fp8)

        out = self._flash_attn_fp8(
            q8,
            k8,
            v8,
            q_descale,
            k_descale,
            v_descale,
            causal=self.causal,
            window_size=(-1, -1, 0),
            softmax_scale=self._softmax_scale,
            sink_ptr=None,
        )
        return out.to(out_dtype)
