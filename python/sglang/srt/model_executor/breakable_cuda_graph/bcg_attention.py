# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Attention entry point for the breakable CUDA graph (BCG) runner.

:func:`bcg_unified_attention_with_output` mirrors
:func:`sglang.srt.layers.radix_attention.unified_attention_with_output` exactly
at the call site; internally it installs a graph break so the attention call
runs eagerly at every replay.
"""

from __future__ import annotations

from typing import Optional

import torch


__all__ = ["bcg_unified_attention_with_output"]


def bcg_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    """BCG counterpart of :func:`unified_attention_with_output`."""
    global _breakable_attention_fn
    if _breakable_attention_fn is None:
        from sglang.srt.layers.radix_attention import unified_attention_with_output
        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import eager_on_graph

        _breakable_attention_fn = eager_on_graph(True)(unified_attention_with_output)
    _breakable_attention_fn(
        query, key, value, output, save_kv_cache, layer_id,
        q_rope=q_rope, k_rope=k_rope, sinks=sinks,
    )


# Lazy-init: wraps unified_attention_with_output with eager_on_graph(True) on
# first use to avoid import-time dependency on breakable_cuda_graph (which
# requires cuda.bindings) and to break the radix_attention <-> bcg_attention
# import cycle.
_breakable_attention_fn = None
