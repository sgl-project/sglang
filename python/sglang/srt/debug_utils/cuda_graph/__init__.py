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
"""CUDA-graph-compatible tensor dumping.

`dumper`'s non-intrusive mode registers `nn.Module` forward hooks that call
`dumper.dump(...)` inline.  That works in eager mode, but a CUDA graph replays
only *recorded device kernels* -- the Python hook body never runs again, so with
graphs enabled the hooked modules simply stop producing frames.  Nothing warns;
the dump directory is just quietly missing every module that lives inside a
captured region.

This package closes that hole by splitting "take the value" from "write the
file":

* **take the value** -- during capture the hook issues
  `buffer.copy_(tensor)`.  That is a real device kernel, so it is *recorded*,
  with both `data_ptr()`s baked in.  Every later `graph.replay()` re-runs the
  copy for free, leaving `buffer` holding this replay's value.
* **write the file** -- after `replay()` returns, host-side code reads the
  buffers and calls `dumper.dump(...)` outside the graph.

Values therefore come from the capture side and metadata from the replay-side
`ForwardBatch`; they meet at one collect point per replay.

Three graph backends, three taps:

===============  ==========================================================
`full`           T1: the existing forward hook, whose `copy_` is recorded.
`breakable`      T1 inside captured segments; module outputs inside
                 `eager_on_graph` break bodies fall through to the ordinary
                 eager `dumper.dump(...)` path, because capture is off there
                 on both the capture and the replay side.
`tc_piecewise`   T3: dynamo inlines `Module.__call__`, so the hook's `copy_`
                 would be traced away.  Instead the hook emits one
                 dynamo-opaque custom op (`sglang.dumper_tap`) that survives
                 tracing as an FX node, becomes a piece boundary, and does
                 the `copy_` from inside the piece's graph.
===============  ==========================================================

See `sglang.srt.debug_utils.cuda_graph.state` for the dispatch ladder and
`seams.py` for the two `__init_subclass__` seams that arm it.
"""

from sglang.srt.debug_utils.cuda_graph.config import CudaGraphDumpConfig
from sglang.srt.debug_utils.cuda_graph.registry import (
    BufferKey,
    BufferRegistry,
    DumpBudgetExceeded,
)
from sglang.srt.debug_utils.cuda_graph.state import cuda_graph_dump

__all__ = [
    "BufferKey",
    "BufferRegistry",
    "CudaGraphDumpConfig",
    "DumpBudgetExceeded",
    "cuda_graph_dump",
]
