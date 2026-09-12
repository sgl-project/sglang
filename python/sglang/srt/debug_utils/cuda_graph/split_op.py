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
"""The T3 tap: one dynamo-opaque custom op, for `tc_piecewise`.

Under `tc_piecewise` the model is `torch.compile`-d with `fullgraph=True`, and
the forward hook cannot reach the eager writer from there: `dumper.dump(...)`
opens a file, which dynamo refuses to trace, and under `fullgraph=True` a graph
break is a hard error rather than a fallback.  A custom op is the one construct
that survives tracing intact: it stays an FX node, `add_split_op` turns it into
a piece boundary, and its body runs inside the piece's graph where the `copy_`
is recorded exactly as in the T1 case.

Two declarations carry weight:

* `mutates_args=["x"]` is a deliberate over-statement -- the body only reads
  `x`.  It is what stops inductor from dead-code-eliminating a call whose
  return value nobody uses.  Over-constraining inductor costs fusion; under-
  constraining it costs the whole frame, silently.
* `eager=True` is required: `register_custom_op`'s own NOTE says lazy
  registration does not work with `torch.compile`.

Keeping the Python bookkeeping (name interning, registry lookups, occurrence
counting) on the near side of the op boundary is the other half of the reason.
A bare `buffer.copy_(x)` in a hook body does in fact survive dynamo -- measured
on CPU under both the `eager` and `inductor` backends, the buffer refreshes on
every call -- but `_record`'s dict mutations and logging would be traced too,
and whether the resulting copy lands inside a *captured piece* is a property of
the piecewise splitter, not of dynamo.  The op boundary makes that placement
explicit instead of incidental.

This module is imported lazily (from `state._emit_compiled_tap` and from the
`build_compilation_config` patch) so that `import sglang` does not register a
torch custom op as a side effect, and only when the T3 tap is armed.
"""

from __future__ import annotations

import torch

from sglang.srt.debug_utils.cuda_graph.state import cuda_graph_dump
from sglang.srt.utils.custom_op import register_custom_op

# Must match the name handed to `CompilationConfig.add_split_op`.
DUMPER_TAP_OP_NAME = "sglang.dumper_tap"


@register_custom_op(op_name="dumper_tap", mutates_args=["x"], eager=True)
def dumper_tap(x: torch.Tensor, tag_id: int) -> None:
    """Copy `x` into the graph-resident buffer interned as `tag_id`."""
    cuda_graph_dump.tap_by_tag(tag_id, x)
