# SPDX-License-Identifier: Apache-2.0
"""Shared forward metadata and quantization helpers for MiniMax sparse attention."""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import torch


def _quant_q_fp8(q: torch.Tensor, q_scale: Optional[float]) -> torch.Tensor:
    # Same convention as the KV pools: the fp8 tensor stores value/scale and
    # the attention kernels multiply the logits back by the scale (None = unit).
    if q_scale is not None:
        q = q / q_scale
    return q.to(torch.float8_e4m3fn)


class MiniMaxSparseForwardMetadata(msgspec.Struct):
    """Per-forward plan container for MiniMax sparse attention backends."""

    msa_v1_decode_plan: Any = None
    msa_v1_attention_plan: Any = None
    msa_v1_prefill_plan: Any = None
