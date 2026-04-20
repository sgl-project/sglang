# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon extend-attention kernel sources for MI350X (gfx950 / CDNA 4).

Drop-in replacement for the Triton ``extend_attention_fwd`` kernel in
``sglang.srt.layers.attention.triton_ops.extend_attention`` when running
on gfx950. Supports symmetric head dims (D=64, 128, 256) with BF16 or
FP8 KV caches, causal + sliding-window, attention sinks, logit capping,
and GQA up to 32:8.

Public re-exports consumed by
``sglang.srt.layers.attention.gluon_extend_attention``:

* ``gluon_extend_attention_fwd`` -- main entry point, matches the
  Triton reference signature.
* ``prewarm_extend_attention`` / ``prewarm_for_model`` -- JIT-free
  serving prewarm by ``(head_dim, num_q_heads, num_kv_heads)`` or by
  HuggingFace config.
* ``MODEL_PRESETS`` + ``spec_*`` -- known-good layer specs for
  GPT-OSS, Gemma 3, Qwen 3, Llama 3 / 4, etc.
"""

from ._prewarm import (  # noqa: F401
    MODEL_PRESETS,
    enumerate_basic_configs,
    enumerate_layer_patterns,
    enumerate_persistent_configs,
    prewarm_extend_attention,
    prewarm_for_model,
    prewarm_preset,
    spec_cohere2,
    spec_gemma2,
    spec_gemma3,
    spec_gpt_oss,
    spec_grok,
    spec_llama3,
    spec_llama4,
    spec_qwen3,
)
from .extend_attention_gfx950 import _get_num_CUs  # noqa: F401
from .extend_attention_gfx950 import gluon_extend_attention_fwd  # noqa: F401
