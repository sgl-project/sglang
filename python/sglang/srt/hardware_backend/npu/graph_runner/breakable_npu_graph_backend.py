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

from typing import TYPE_CHECKING

from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.runtime_context import get_flags

if TYPE_CHECKING:
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


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
