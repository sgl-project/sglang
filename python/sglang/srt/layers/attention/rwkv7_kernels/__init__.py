# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Self-contained RWKV-7 Triton kernels (no FLA dependency).

wkv_recurrent -> DECODE (T==1 fast path) + recurrent varlen (cu_seqlens) WKV.
"""

from sglang.srt.layers.attention.rwkv7_kernels.wkv_recurrent import wkv_recurrent

__all__ = ["wkv_recurrent"]
