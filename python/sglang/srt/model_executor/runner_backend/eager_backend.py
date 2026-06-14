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
"""EagerBackend — the no-capture ExecutionBackend.

The dual of :class:`FullCudaGraphBackend`: where the cuda-graph backend
*records* a ``torch.cuda.CUDAGraph`` per shape and *replays* it, the eager
backend records nothing and simply runs the model forward live each
iteration. This lets the unified Runner serve eager forwards through the
same ``prepare()`` / ``reserve_batch()`` / ``load_batch()`` / ``execute()``
lifecycle as the cuda-graph runners — the only difference is the execution
strategy plugged in here.

Because the model forward (and the in-graph attention metadata op) is run
live rather than baked at record time, ``run`` invokes a ``forward_fn``
supplied per iteration by the runner, against the current static batch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.model_executor.runner_backend.base_execution_backend import (
    ExecutionBackend,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_runner import BaseRunner
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class EagerBackend(ExecutionBackend):
    """Run the model forward eagerly — no capture, no graph.

    Symmetric with :class:`FullCudaGraphBackend` so the Runner ABC does
    not branch on eager-vs-graph: ``record_session`` / ``run_session`` are
    no-ops, ``record`` captures nothing, and ``run`` executes the live
    ``forward_fn`` against the per-iteration static batch.
    """

    def __init__(self, cuda_graph_runner: Optional[BaseRunner] = None) -> None:
        # No per-shape artifacts to hold; kept for parity with the
        # resolve_*_backend(runner) construction contract.
        self._runner = cuda_graph_runner

    @contextmanager
    def record_session(self, stream: torch.cuda.Stream):
        # Nothing to record; eager has no capture phase.
        yield

    def record(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # No artifact is recorded for eager; the forward is run live in run().
        return None

    def can_run_graph(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        # Eager has no captured-shape constraint — it serves any batch.
        return True

    @contextmanager
    def run_session(self):
        yield

    def run(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        *,
        forward_fn: Optional[Callable[[ForwardBatch], Any]] = None,
        **kwargs,
    ) -> Any:
        # Eager runs the model forward live against the current static batch
        # (which load_batch() has already filled + had its metadata refreshed).
        assert (
            forward_fn is not None
        ), "EagerBackend.run requires a forward_fn to execute live"
        return forward_fn(static_forward_batch)

    def cleanup(self) -> None:
        return None
