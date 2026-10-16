# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 LM head (shared with the DSpark draft) on FlashInfer's CuTe DSL
low-M BF16 GEMM for 1-16 rows instead of cuBLAS split-K plus its reduce: about 48 us
against 55 us on the TP4 shard (N = 32320, K = 5120) at 6 rows. Same BF16 output; only
the K summation order differs."""

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models.dsv41_low_latency.low_m_gemm import (
    cutedsl_low_m_mm,
    cutedsl_low_m_supported,
    fits,
)
from sglang.srt.models.dsv41_low_latency.switch import LM_HEAD_MAX_ROWS

logger = logging.getLogger(__name__)

# One call per FlashInfer token bucket (powers of two) tunes 1-16 rows.
_TUNING_ROWS = tuple(1 << i for i in range(LM_HEAD_MAX_ROWS.bit_length()))


class LowLatencyLmHeadMethod(UnquantizedEmbeddingMethod):
    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is None and fits(x, LM_HEAD_MAX_ROWS):
            return cutedsl_low_m_mm(x, layer.weight)
        return super().apply(layer, x.to(layer.weight.dtype), bias)


def install(lm_head: nn.Module) -> None:
    """Route an unquantized BF16 head's logits through `LowLatencyLmHeadMethod`."""
    if not (
        type(lm_head.quant_method) is UnquantizedEmbeddingMethod
        and lm_head.weight.dtype == torch.bfloat16
        and cutedsl_low_m_supported()
    ):
        logger.warning(
            "DSV4.1 low-latency LM head not installed: needs an unquantized BF16 "
            "head and FlashInfer's CuTe DSL mm_bf16."
        )
        return
    lm_head.quant_method = LowLatencyLmHeadMethod()


def autotune(lm_head: nn.Module) -> None:
    """Inside the FlashInfer autotune context: the same calls on every rank."""
    if not isinstance(lm_head.quant_method, LowLatencyLmHeadMethod):
        return
    weight = lm_head.weight
    for rows in _TUNING_ROWS:
        x = torch.zeros(
            (rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        cutedsl_low_m_mm(x, weight)
