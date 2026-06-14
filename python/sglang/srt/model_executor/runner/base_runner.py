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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.layers.dp_attention import (
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.utils import require_gathered_buffer

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.base_execution_backend import (
        ExecutionBackend,
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
    capture_bs = server_args.cuda_graph_config.decode.bs
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    if mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()

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

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


class BaseRunner(ABC):
    """Abstract base for phase-specific cuda-graph runners.

    A subclass (DecodeRunner / PrefillRunner) owns one
    phase and plugs in a ExecutionBackend that handles the
    capture / replay mechanics. The runner orchestrates bucket
    selection, static buffer population, attention metadata init,
    replay dispatch, and output slicing.

    Methods:
      - can_run_graph(forward_batch) — should forward_batch go through cuda
        graph replay (vs eager fallback)?
      - reserve_batch(size, ...) — build the dummy ForwardBatch and
        per-shape local state needed by _prepare_one.
      - prepare() — one-time setup; iterates over shapes and calls
        _prepare_one for each.
      - _prepare_one(size, ...) — drive one model forward at this
        shape into the backend's recorded artifact.
      - load_batch(forward_batch, ...) — pad to the nearest captured
        bucket, populate static input buffers, init attention metadata.
      - execute(forward_batch, ...) — dispatch one batch through the
        backend (graph replay for cuda graph; model.forward for eager).

    Notes:
      - buffers and backend are populated by the subclass before
        prepare(); the base only declares them.
    """

    # Subclasses populate before calling prepare().
    buffers: ForwardInputBuffers
    backend: ExecutionBackend

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    def warmup(self) -> None:
        """Warm up + autotune kernels once, before this runner captures (graph)
        or reserves (eager) — part of the Runner lifecycle, called from
        prepare().

        Run-once across the decode and prefill runners via a flag on the shared
        ModelRunner: whichever runner prepares first does the warmup; the other
        is a no-op. (Replaces the unconditional ModelRunner.kernel_warmup call.)
        """
        model_runner = self.model_runner
        if getattr(model_runner, "_kernel_warmed_up", False):
            return
        model_runner._kernel_warmed_up = True

        if model_runner.device != "cuda":
            return

        if model_runner._should_run_flashinfer_autotune():
            model_runner._flashinfer_autotune()

        from sglang.srt.environ import envs
        from sglang.srt.layers import deep_gemm_wrapper

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and self.pp_size > 1
            and not model_runner.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(model_runner)

    @staticmethod
    def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
        """Return the smallest buckets[i] >= raw_size.

        Caller's can_run must reject raw_size > max(buckets) before
        reaching load_batch; this assertion makes the contract
        explicit (bisect_left returns len(buckets) when the value
        exceeds all buckets, which would otherwise IndexError below
        with no diagnostic).
        """
        assert raw_size <= buckets[-1], (
            f"size {raw_size} exceeds max captured bucket {buckets[-1]}; "
            f"can_run should have rejected this batch"
        )
        index = bisect.bisect_left(buckets, raw_size)
        return buckets[index]

    @abstractmethod
    def can_run_graph(self, forward_batch: ForwardBatch) -> bool: ...

    @abstractmethod
    def reserve_batch(self, size: int, *args, **kwargs) -> Any: ...

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def _prepare_one(self, size: int, *args, **kwargs) -> Any: ...

    @abstractmethod
    def load_batch(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def execute(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...
