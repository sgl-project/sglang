# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-ratio layers at decode: the indexer query on its own side stream, and the
compressor / indexer projections on low-M BF16 GEMMs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.models.dsv41_low_latency.low_m_gemm import (
    cutedsl_low_m_mm,
    cutedsl_low_m_supported,
    fits,
    ll_bf16_supported,
)
from sglang.srt.models.dsv41_low_latency.switch import LOW_M_BF16_MAX_ROWS


def _ll_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from sglang.srt.models.dsv41_low_latency.ll_bf16 import ll_bf16_gemm

    return ll_bf16_gemm(x, weight)


def _bf16_rounded(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    # FP32 accumulation rounded to the layer's BF16 output.
    if not fits(x, LOW_M_BF16_MAX_ROWS):
        return None
    return _ll_bf16_gemm(x, weight).to(torch.bfloat16)


def _fp32(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    return _ll_bf16_gemm(x, weight) if fits(x, LOW_M_BF16_MAX_ROWS) else None


def _bf16(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    return cutedsl_low_m_mm(x, weight) if fits(x, LOW_M_BF16_MAX_ROWS) else None


def install_projections(attn: nn.Module) -> None:
    """The compressor's ratio-1 wkv (BF16 out) and ratio-2 wkv_gate (FP32 out), the
    indexer's weights_proj and wk: split-K GEMMs at 1-8 rows instead of cuBLAS or
    the JIT small GEMMs, which give few output tiles to these skinny shapes."""
    if not (ll_bf16_supported() and cutedsl_low_m_supported()):
        return
    if attn.compressor is not None:
        attn.compressor.low_m_bf16_gemm = _bf16
        attn.compressor.low_m_fp32_gemm = _fp32
    if attn.indexer is not None:
        attn.indexer.low_m_gemm = _bf16_rounded


class IndexQ:
    """One decode step's index query, built on a side stream beside wqkv_a, wq_b
    and the compressor; the scoring on the sources stream waits for it."""

    def __init__(self, stream: torch.cuda.Stream, head_weights_raw: torch.Tensor):
        self.stream = stream
        self._head_weights_raw = head_weights_raw

    def finish(
        self,
        *,
        layer: nn.Module,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        attn_backend,
        stream_sources: torch.cuda.Stream,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            q, _ = layer.indexer.wq_b(q_lora)
            prepared = attn_backend.low_ratio_decode_index_q(
                layer=layer,
                q=q,
                head_weights_raw=self._head_weights_raw,
                positions=positions,
                forward_batch=forward_batch,
            )
        stream_sources.wait_stream(self.stream)
        return prepared


class IndexQSideStream:
    def __init__(self, stream: torch.cuda.Stream):
        self._stream = stream

    def begin(
        self, layer: nn.Module, x: torch.Tensor, attn_backend
    ) -> Optional[IndexQ]:
        """Start weights_proj, which reads only x; None keeps the stock order."""
        if layer.indexer is None or not attn_backend.low_ratio_decode_reads_prepared_q:
            return None
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            head_weights_raw = layer.indexer.head_weights_raw(x)
        return IndexQ(self._stream, head_weights_raw)


def install_index_q_side_stream(attn: nn.Module) -> None:
    if attn.indexer is not None and attn.alt_streams is not None:
        # The low-ratio prepare leaves this one of the layer's three streams idle.
        attn.low_latency_index_q = IndexQSideStream(attn.alt_streams[1])
