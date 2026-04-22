# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon extend-attention kernel sources for MI350X (gfx950 / CDNA 4).

Drop-in replacement for the Triton ``extend_attention_fwd`` kernel in
``sglang.srt.layers.attention.triton_ops.extend_attention`` when running
on gfx950. Supports symmetric head dims (D=64, 128, 256) with BF16 or
FP8 KV caches, causal + sliding-window, attention sinks, logit capping,
and GQA up to 32:8.

Public entry point consumed by
``sglang.srt.layers.attention.gluon_extend_attention``:

* ``gluon_extend_attention_fwd`` -- matches the Triton reference
  signature.
"""

from .extend_attention_gfx950 import _get_num_CUs  # noqa: F401
from .extend_attention_gfx950 import gluon_extend_attention_fwd  # noqa: F401
