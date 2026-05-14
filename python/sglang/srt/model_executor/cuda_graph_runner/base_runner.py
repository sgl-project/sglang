"""Shared scaffolding for the prefill and decode CUDA graph runners.

The phase-specific subclasses (``DecodeCudaGraphRunner``,
``PrefillCudaGraphRunner``) own their own buffer dataclasses, capture
forward-mode, and ``can_run`` logic. This base contributes:

- Shared ``__init__`` fields (model_runner, device, parallel sizes,
  attn-tp coordinates, tbo plugin).
- ``freeze_gc`` — gc-freeze context used during capture.
- ``get_batch_sizes_to_capture`` — bucket-sizing helper for decode.
- ``_pad_to_bucket`` — shared bisect-bucket lookup with a clear-fail
  assertion (the runner's ``can_run`` is responsible for ensuring the
  raw size fits within the bucket list).
- abstract methods describing the contract a phase runner must fulfil.
"""

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
    from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
        BaseCudaGraphBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
    from sglang.srt.model_executor.model_runner import ModelRunner

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
    model_runner: "ModelRunner", num_tokens_per_bs: int = 1
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters server_args.cuda_graph_bs by attention-tp/cp alignment
    constraints and clamps to req_to_token_pool.size.
    """
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    if mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()

    num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
    if max(capture_bs) > num_max_requests:
        capture_bs += [num_max_requests]

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


class BaseCudaGraphRunner(ABC):
    """Abstract base for phase-specific cuda-graph runners.

    A subclass implements one of the two phases (``DecodeCudaGraphRunner``
    or ``PrefillCudaGraphRunner``) and plugs in a backend that handles
    capture/replay mechanics. The runner orchestrates: bucket selection,
    static buffer population, attention metadata init, replay dispatch,
    and output slicing. The backend handles only "given a populated
    forward_batch, run the captured artifact for this shape".

    Concrete state populated here (subclasses extend):
      - ``self.model_runner`` — back-reference to ModelRunner.
      - ``self.device``, ``self.device_module`` — device handle.
      - ``self.tp_size``, ``self.dp_size``, ``self.pp_size``,
        ``self.attn_tp_size``, ``self.attn_tp_rank`` — parallelism.
      - ``self.tbo_plugin`` — two-batch-overlap plugin.
      - ``self.buffers`` — phase-specific input buffers (assigned by
        subclass before ``capture()``).
      - ``self.backend`` — pluggable ``BaseCudaGraphBackend`` (assigned
        by subclass).
    """

    # Subclasses populate before calling ``capture``.
    buffers: "ForwardInputBuffers"
    backend: "BaseCudaGraphBackend"

    def __init__(self, model_runner: "ModelRunner") -> None:
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tbo_plugin = TboCudaGraphRunnerPlugin()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
        """Return the smallest ``buckets[i] >= raw_size``.

        Caller's ``can_run`` must reject ``raw_size > max(buckets)``
        before reaching ``replay_prepare`` — this assertion makes the
        contract explicit. ``bisect_left`` returns ``len(buckets)``
        when the value exceeds all buckets, which would otherwise
        IndexError below with no diagnostic.
        """
        assert raw_size <= buckets[-1], (
            f"size {raw_size} exceeds max captured bucket {buckets[-1]}; "
            f"can_run should have rejected this batch"
        )
        index = bisect.bisect_left(buckets, raw_size)
        return buckets[index]

    # -----------------------------------------------------------------
    # Abstract contract
    # -----------------------------------------------------------------
    @abstractmethod
    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        """Decide whether ``forward_batch`` should go through cuda graph
        replay (vs falling back to eager forward). Subclasses should AND
        their phase-level checks with ``self.backend.can_run(fb)``.
        """

    @abstractmethod
    def capture(self) -> None:
        """Outer capture loop. Iterates over shapes, calls
        ``self.capture_one_shape`` for each.
        """

    @abstractmethod
    def capture_one_shape(self, size: int, *args, **kwargs) -> Any:
        """Per-shape capture: build a dummy ForwardBatch, run model
        forward once into the backend's captured artifact for ``size``.
        Decode passes the patched-model forward + stream/variant info;
        prefill takes ``size`` only. Subclasses define the full signature.
        """

    @abstractmethod
    def replay_prepare(
        self,
        forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        """Replay-time setup: pad to nearest captured bucket, populate
        static input buffers from ``forward_batch``, init attention
        metadata. Decode mutates state on ``self`` (no return); prefill
        returns the static ``ForwardBatch`` model code reads during
        replay. Caller (``replay``) consumes whatever is returned.
        """

    @abstractmethod
    def replay(
        self,
        forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        """Dispatch one batch through cuda graph replay."""
