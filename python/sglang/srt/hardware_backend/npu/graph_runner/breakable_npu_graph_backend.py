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
"""Breakable NPU graph backend: one torch.npu.NPUGraph per segment,
attention (and other host-input ops) eager between segments.

The parent capture context, capture loop and replay path are used
unchanged; only the per-segment graph class is device-specific
(_device_graph_cls in breakable_cuda_graph.py). Segment graphs are
captured with bare capture_begin/capture_end — the graph.update channel
is not used because every op that consumes host-side values runs at
the eager break points, outside the captured segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_ascend_ops_config
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.runtime_context import get_flags, get_forward

if TYPE_CHECKING:
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BreakableNPUGraphBackend(BreakableCudaGraphBackend):
    """NPU prefill graphs split at eager_on_graph break points."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        super().__init__(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            debug_eager=debug_eager,
        )
        # torch.compile cannot contain the eager break points this backend
        # depends on (NPU decode still compiles through NPUGraphRunner).
        assert not get_flags().capture.enable_torch_compile, (
            "breakable NPU prefill graph does not support torch.compile; "
            "disable --enable-torch-compile or prefill graph"
        )
        # AllToAllV sizes its buffers from live per-expert counts read on the
        # host, which a captured segment cannot do.  Reject at construction so
        # the failure lands at startup, before any capture.
        if (
            get_moe_a2a_backend().is_ascend_ops()
            and get_moe_ascend_ops_config().prefill != "allgather"
        ):
            raise ValueError(
                "breakable NPU prefill graph requires ascend_ops prefill=allgather; "
                "alltoallv reads live expert counts from the host, which capture "
                "cannot do"
            )

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Every TP rank runs the capture bucket at the same width; publish that
        # layout so ascend_ops skips the live row-count exchange, a host sync
        # the captured graph cannot contain.
        with get_forward().scoped(
            ascend_allgather_row_sizes=[shape_key.size] * self._tp_group.world_size
        ):
            super().capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
