"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (``eager_on_graph`` decorators on attention / mamba layers).

Backend owns its own ``_graphs[shape] -> BreakableCUDAGraph`` and
``_outputs[shape] -> LogitsProcessorOutput`` tables, plus the memory
pool handle. When used as a prefill backend, also owns the static
prefill input buffers that captured segments read from.

Uses ``BreakableCUDAGraph`` / ``BreakableCUDAGraphCapture`` from
``cuda_graph_backend_utils.breakable_cuda_graph``. No torch.compile.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# Names of the static prefill input tensors Breakable owns. Each is a
# 1-D int64 tensor of length max_bs; captured segments read from these
# stable addresses.
_PREFILL_STATIC_FIELDS = (
    "seq_lens",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
    "req_pool_indices",
    "orig_seq_lens",
)


class BreakableCudaGraphBackend(BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention/mamba boundaries;
    attention metadata is recomputed at replay (outside captured segments).
    """

    def __init__(
        self,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        if is_hip():
            raise RuntimeError("Breakable CUDA graph is not supported on ROCm/HIP")
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = None
        self._tp_group = None
        self._memory_saver_adapter: Optional[Any] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._enable_memory_saver = enable_memory_saver
        self._debug_eager = debug_eager
        # Set by ``setup_prefill_state`` when the runner is prefill-phase.
        self._prefill_static: Optional[Dict[str, torch.Tensor]] = None

    def prepare(self, runner) -> None:
        self._device_module = runner.device_module
        self._tp_group = runner.model_runner.tp_group
        self._memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self._enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    # -----------------------------------------------------------------
    # Prefill hooks (Breakable needs stable addresses for captured segments)
    # -----------------------------------------------------------------
    def setup_prefill_state(self, runner) -> None:
        """Allocate static prefill input buffers. Called by
        ``PrefillCudaGraphRunner.__init__`` before ``prepare()``.
        """
        max_bs = runner.max_bs
        device = runner.device
        with torch.device(device):
            self._prefill_static = {
                name: torch.zeros((max_bs,), dtype=torch.int64)
                for name in _PREFILL_STATIC_FIELDS
            }

    def populate_prefill_dummy_inputs(
        self,
        kwargs: dict,
        *,
        bs: int,
        num_tokens: int,
    ) -> None:
        """Swap fresh literals in ``kwargs`` for Breakable's static buffers
        (sliced to ``bs``) so captured segments read from fixed
        pointers.
        """
        assert (
            self._prefill_static is not None
        ), "setup_prefill_state must run before capture"
        s = self._prefill_static
        s["seq_lens"][:bs].fill_(num_tokens)
        s["extend_seq_lens"][:bs].fill_(num_tokens)
        s["extend_prefix_lens"][:bs].zero_()
        s["extend_start_loc"][:bs].zero_()
        s["req_pool_indices"][:bs].copy_(
            torch.arange(bs, device=s["req_pool_indices"].device)
        )
        s["orig_seq_lens"][:bs].fill_(num_tokens)
        kwargs["req_pool_indices"] = s["req_pool_indices"][:bs]
        kwargs["seq_lens"] = s["seq_lens"][:bs]
        kwargs["orig_seq_lens"] = s["orig_seq_lens"][:bs]
        kwargs["extend_seq_lens"] = s["extend_seq_lens"][:bs]
        kwargs["extend_prefix_lens"] = s["extend_prefix_lens"][:bs]
        kwargs["extend_start_loc"] = s["extend_start_loc"][:bs]

    def commit_prefill_serving_inputs(self, forward_batch: "ForwardBatch") -> None:
        """Replay-time: copy serving values into the static buffers so
        the addresses captured segments hold stay live with current data.
        """
        assert self._prefill_static is not None
        bs = forward_batch.batch_size
        s = self._prefill_static
        s["seq_lens"][:bs].copy_(forward_batch.seq_lens)
        s["extend_seq_lens"][:bs].copy_(forward_batch.extend_seq_lens)
        s["extend_prefix_lens"][:bs].copy_(forward_batch.extend_prefix_lens)
        s["extend_start_loc"][:bs].copy_(forward_batch.extend_start_loc)
        s["req_pool_indices"][:bs].copy_(forward_batch.req_pool_indices)
        if forward_batch.orig_seq_lens is not None:
            s["orig_seq_lens"][:bs].copy_(forward_batch.orig_seq_lens)

    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        # Breakable-prefill captures bs=1 only — multi-req would silently return
        # wrong-shaped logits, corrupting downstream output_ids. The
        # presence of ``_prefill_static`` marks "this Breakable instance is
        # being used as a prefill backend".
        if self._prefill_static is not None and forward_batch.batch_size > 1:
            return False
        return True

    @contextmanager
    def runtime_session(self):
        with enable_breakable_cuda_graph():
            yield

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            with self.runtime_session():
                yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
    ) -> None:
        # Two jit warmups, then capture under BreakableCUDAGraphCapture.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()

        graph = BreakableCUDAGraph()
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            out = captured_fn()
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def has_shape(self, shape_key: Any) -> bool:
        return shape_key in self._graphs

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        # static_forward_batch / kwargs are unused — Breakable replays
        # against static buffers populated by the runner.
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
