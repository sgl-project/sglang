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
"""Shared scaffolding for the prefill and decode CUDA graph runners."""

from __future__ import annotations

import gc
import logging
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from sglang.srt.model_executor.cuda_graph_config import (
    filter_capture_sizes,
    pad_to_capture_size,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
)
from sglang.srt.utils import (
    get_cuda_graph_batch_size_alignment,
    get_cuda_graph_max_batch_size,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
        BaseCudaGraphBackend,
    )

logger = logging.getLogger(__name__)


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """Optimize garbage collection during CUDA graph capture.

    Clean up first, then freeze remaining objects from being included in
    future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def get_batch_sizes_to_capture(
    model_runner: ModelRunner, captured_req_width: int = 1
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters cuda_graph_config[decode].bs by attention-tp/cp alignment
    constraints and clamps to req_to_token_pool.size.
    """

    capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = get_cuda_graph_batch_size_alignment()
    # TBO splits each request's rows across two micro-batches, so the
    # alignment constraint applies per request rather than per token row.
    alignment_width = captured_req_width
    if get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1

    # pad `num_max_requests` to avoid being filtered out
    num_max_requests = get_cuda_graph_max_batch_size(num_max_requests)
    capture_bs = filter_capture_sizes(
        capture_bs,
        max_size=num_max_requests,
        alignment=mul_base,
        request_width=alignment_width,
    )
    compile_bs = (
        [bs for bs in capture_bs if bs <= get_exec().graph.torch_compile_max_bs]
        if get_flags().capture.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


class BaseCudaGraphRunner(BaseRunner):
    """Abstract base for phase-specific cuda-graph runners.

    A subclass (DecodeCudaGraphRunner / PrefillCudaGraphRunner) owns one
    phase and plugs in a BaseCudaGraphBackend that handles the
    capture / replay mechanics. The runner orchestrates bucket
    selection, static buffer population, attention metadata init,
    replay dispatch, and output slicing.

    Adds the capture/shape machinery on top of BaseRunner:
      - capture_prepare(size, ...) — build the dummy ForwardBatch and
        per-shape local state needed by capture_one_shape.
      - capture() — one-time setup; iterates over shapes and calls
        capture_one_shape for each.
      - capture_one_shape(size, ...) — drive one model forward at this
        shape into the backend's captured artifact.
      - _pad_to_bucket(...) — round a raw shape up to the nearest captured
        bucket.

    Inherits from BaseRunner: __init__ and the abstract
    can_run_graph / load_batch / execute.

    Notes:
      - buffers and backend are populated by the subclass before
        capture(); the base only declares them.
    """

    # Subclasses populate before calling capture().
    buffers: ForwardInputBuffers
    backend: BaseCudaGraphBackend

    def cuda_graph_output_rows(self, output: Any) -> Optional[int]:
        """Rows of graph output that must be preserved for post-replay work.

        The default graph key is a request count, which is also the output row
        count for ordinary decode. A graph that returns per-token hidden states
        for an eager tail can instead produce ``requests * tokens_per_request``
        rows. Such a runner must return that actual row count here. ``None``
        keeps the backend's default request-count behavior.
        """
        return None

    def cuda_graph_output_capacity_rows(self, output: Any) -> Optional[int]:
        """Capacity required by the output buffer shared across graph keys.

        The breakable backend allocates this buffer once, while capturing its
        first shape. A runner whose output uses token rows rather than request
        rows must return the largest possible output here so later graph shapes
        fit. ``None`` uses the current graph key as the capacity.
        """
        return None

    _pad_to_bucket = staticmethod(pad_to_capture_size)

    @abstractmethod
    def capture_prepare(self, size: int, *args, **kwargs) -> Any: ...

    @abstractmethod
    def capture(self) -> None: ...

    @abstractmethod
    def capture_one_shape(self, size: int, *args, **kwargs) -> Any: ...
