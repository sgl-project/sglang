# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon extend-attention kernels for MI350/MI355 (gfx950 / CDNA 4).

Symmetric-head extend attention (Lq == Lv) with BF16 or FP8 KV cache,
covering the Triton-extend feature set: causal + sliding-window,
attention sinks (GPT-OSS), logit capping (Grok / Gemma 2), GQA up to
32:8, and D in {64, 128, 256}.

This package is the standalone Gluon-kernels home for the extend
kernel; the SGLang fork vendors a mirror of these files at
``python/sglang/srt/layers/attention/gluon_ops/cdna4/extend_attention/``
and consumes them through ``TritonAttnBackend`` behind
``SGLANG_ENABLE_GLUON_EXTEND_ATTENTION=1``.

Public entry point:

* :func:`gluon_extend_attention_fwd` -- drop-in replacement for the
  Triton extend-attention reference, symmetric heads.

All launches go through Triton's ``kernel[grid](...)`` JIT entry
point; Triton's compile cache handles first-call JIT. A shape-keyed
tile-config cache and per-forward WCA metadata reuse (split-K
workspace, WCA-grid size) keep warm-launch
CPU overhead at the Triton-launcher floor.
"""

from .extend_attention_gfx950 import _get_num_CUs  # noqa: F401
from .extend_attention_gfx950 import gluon_extend_attention_fwd  # noqa: F401
