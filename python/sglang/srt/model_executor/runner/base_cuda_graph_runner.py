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

import bisect
import gc
import logging
import math
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
)
from sglang.srt.utils import (
    ceil_align,
    get_cuda_graph_batch_size_alignment,
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
    model_runner: ModelRunner,
    captured_req_width: int = 1,
    *,
    gathered_buffer_required: Optional[bool] = None,
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters cuda_graph_config[decode].bs by attention-tp/cp alignment and caps
    it at req_to_token_pool.size. With padding, the top bucket is padded up to
    the alignment, so it can exceed both the pool and the configured max_bs;
    graph-static buffers size from runtime_context.decode_cuda_graph_max_bs().
    """

    capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size
    dllm_config = DllmConfig.from_server_args(model_runner.server_args)
    if dllm_config is not None and dllm_config.requires_separate_context_encoding:
        num_max_requests = min(num_max_requests, dllm_config.max_running_requests)

    mul_base = get_cuda_graph_batch_size_alignment(
        gathered_buffer_required=gathered_buffer_required
    )
    # TBO splits each request's rows across two micro-batches, so the
    # alignment constraint applies per request rather than per token row.
    alignment_width = captured_req_width
    if get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1

    # Model input rows = bs * alignment_width must be a multiple of mul_base,
    # so a request count only needs mul_base / gcd(mul_base, width) alignment.
    request_alignment = mul_base // math.gcd(mul_base, alignment_width)
    num_max_requests = min(max(capture_bs), num_max_requests)
    if not get_exec().graph.disable_cuda_graph_padding:
        # Pad the largest serviceable batch up to the next aligned bucket and keep
        # it, so an unaligned --cuda-graph-max-bs or request pool still leaves a
        # graph covering every batch size instead of an eager tail.
        num_max_requests = ceil_align(num_max_requests, request_alignment)
        capture_bs.append(num_max_requests)

    capture_bs = [
        bs
        for bs in capture_bs
        if bs % request_alignment == 0 and bs <= num_max_requests
    ]
    capture_bs = list(sorted(set(capture_bs)))
    if not capture_bs:
        # Without padding no configured size may be aligned below the pool;
        # keep the aligned endpoint so the runner still has a valid shape.
        capture_bs = [ceil_align(num_max_requests, request_alignment)]
        logger.warning(
            f"No configured decode CUDA graph batch size is a multiple of "
            f"{request_alignment} within {num_max_requests} requests; capturing "
            f"{capture_bs} only."
        )

    assert capture_bs[0] > 0, f"{capture_bs=}"
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

    @staticmethod
    def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
        """Return the smallest buckets[i] >= raw_size.

        Caller's can_run_graph must reject raw_size > max(buckets) before
        reaching load_batch; this assertion makes the contract
        explicit (bisect_left returns len(buckets) when the value
        exceeds all buckets, which would otherwise IndexError below
        with no diagnostic).
        """
        assert raw_size <= buckets[-1], (
            f"size {raw_size} exceeds max captured bucket {buckets[-1]}; "
            f"can_run_graph should have rejected this batch"
        )
        index = bisect.bisect_left(buckets, raw_size)
        return buckets[index]

    @abstractmethod
    def capture_prepare(self, size: int, *args, **kwargs) -> Any: ...

    @abstractmethod
    def capture(self) -> None: ...

    @abstractmethod
    def capture_one_shape(self, size: int, *args, **kwargs) -> Any: ...
