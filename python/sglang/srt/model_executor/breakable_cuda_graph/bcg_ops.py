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
"""Factory for turning a PCG custom op into a BCG graph-break point.

Each model declares its break points next to its PCG ``@register_split_op``
definitions, e.g.::

    bcg_unified_attention_with_output = make_bcg_break_point(unified_attention_with_output)

The wrap is lazy so importing this module does not drag in ``cuda.bindings``.
"""

from __future__ import annotations


def make_bcg_break_point(fn):
    """Return a callable that runs ``fn`` as a BCG eager break point.

    The underlying ``eager_on_graph(True)(fn)`` wrap is deferred to the first
    call: that avoids an import-time dependency on ``cuda.bindings`` (required
    by ``breakable_cuda_graph``) and sidesteps import cycles between BCG
    infrastructure and the model files that declare their own break points.
    """
    wrapped = None

    def call(*args, **kwargs):
        nonlocal wrapped
        if wrapped is None:
            from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
                eager_on_graph,
            )

            wrapped = eager_on_graph(True)(fn)
        return wrapped(*args, **kwargs)

    call.__name__ = f"bcg_{getattr(fn, '__name__', 'break_point')}"
    return call
