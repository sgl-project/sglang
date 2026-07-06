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
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import require_gathered_buffer

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
    model_runner: ModelRunner, num_tokens_per_bs: int = 1
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters cuda_graph_config[decode].bs by attention-tp/cp alignment
    constraints and clamps to req_to_token_pool.size.
    """

    server_args = model_runner.server_args
    capture_bs = list(server_args.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1

    if require_gathered_buffer(server_args):
        mul_base *= get_parallel().attn_tp_size

    if mul_base % get_parallel().attn_cp_size != 0:
        mul_base *= get_parallel().attn_cp_size

    # pad `num_max_requests` to avoid being filtered out
    num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
    if max(capture_bs) > num_max_requests:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [num_max_requests]

    # Model input token count = bs * num_tokens_per_bs; must be a multiple of attn_tp_size.
    capture_bs = [bs for bs in capture_bs if bs * num_tokens_per_bs % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    # A capture bs above the DeepEP low_latency dispatch cap trips the deep_ep assert.
    from sglang.srt.model_executor.deepep_capacity import is_deepep_low_latency

    if is_deepep_low_latency(server_args):
        deepep_cap = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        # Clamp on the max draft tokens, not the startup value: adaptive spec can
        # grow it at runtime, and each request dispatches num_tokens_per_bs tokens.
        spec_mult = max(
            num_tokens_per_bs, server_args.max_speculative_num_draft_tokens or 0
        )
        if max(capture_bs) * spec_mult > deepep_cap:
            capture_bs = [bs for bs in capture_bs if bs * spec_mult <= deepep_cap]

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
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
