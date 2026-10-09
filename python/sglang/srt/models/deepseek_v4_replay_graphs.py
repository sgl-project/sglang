"""CUDA graphs of DeepSeek-V4's trimmed late layers on eager prefill steps.

Under decoder SWA bounded replay the layers after the last kv_source layer run on
each request's last SWA_WINDOW extend rows. On an eager prefill step those rows
replay a breakable graph captured for their row bucket (a multiple of 128): the
KV store, the attention and the low-ratio sources are eager breaks that read the
step's live tail metadata; everything else reads static row buffers. Graphs are
captured on first use of a bucket, on every TP rank in the same step.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
    get_tc_piecewise_forward_context,
    set_tc_piecewise_forward_context,
)
from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Tail rows come in per-request blocks of at most SWA_WINDOW = 128.
ROW_STEP = 128

_in_replay_graph = False


def in_decoder_replay_graph() -> bool:
    """True while the late layers capture or replay a decoder replay graph."""
    return _in_replay_graph


@contextmanager
def _replay_graph_scope():
    global _in_replay_graph
    _in_replay_graph = True
    try:
        yield
    finally:
        _in_replay_graph = False


def _late_kv_store(attention, x, positions, qkv_a) -> None:
    # The SWA store writes the step's live tail slots, so a replay graph breaks here.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_rows = forward_batch.global_num_token_non_padded_cpu
    attention._compute_kv_to_cache(
        x[:num_rows],
        positions[:num_rows],
        forward_batch,
        get_attn_backend(),
        qkv_a=None if qkv_a is None else qkv_a[:num_rows],
    )


bcg_late_kv_store = eager_on_graph(True)(_late_kv_store)


def _replay_checked(graph: BreakableCUDAGraph, rows: int) -> None:
    for i, seg in enumerate(graph._segments):
        for what, step in (("segment", seg.replay), ("break", None)):
            if what == "break":
                if i >= len(graph._break_fns):
                    continue
                step = graph._break_fns[i]
            try:
                step()
                torch.cuda.synchronize()
            except Exception:
                logger.error(
                    "Decoder replay graph %d rows: %s %d of %d segments faulted",
                    rows,
                    what,
                    i,
                    len(graph._segments),
                )
                raise


def _flatten_state(state: HcState):
    """The state's row tensors, a key for their structure, and the inverse."""
    pending = isinstance(state.streams, HcPending)
    leaves = list(state.streams) if pending else [state.streams]
    num_streams = len(leaves)
    if state.pre is not None:
        leaves.append(state.pre)
    has_pre = state.pre is not None

    def rebuild(tensors: list[torch.Tensor]) -> HcState:
        streams = HcPending(*tensors[:num_streams]) if pending else tensors[0]
        return HcState(streams, tensors[num_streams] if has_pre else None)

    return leaves, (pending, has_pre), rebuild


class _ReplayGraph:
    __slots__ = ("graph", "inputs", "num_token_non_padded", "residual", "pre", "batch")

    def __init__(self, graph, inputs, num_token_non_padded, residual, pre, batch):
        self.graph = graph
        self.inputs = inputs
        # MoE top-k masks rows at or past this count; captured by address.
        self.num_token_non_padded = num_token_non_padded
        self.residual = residual
        self.pre = pre
        # The capture batch; kept alive in case a captured kernel reads its tensors.
        self.batch = batch


class DecoderReplayGraphs:
    """Breakable CUDA graphs of the late layers, keyed by padded tail rows."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        run_layers: Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]],
        max_rows: int,
    ) -> None:
        self._model = model
        self._run_layers = run_layers
        self.max_rows = max_rows
        self._graphs: dict[tuple, _ReplayGraph] = {}
        self._layers = None
        self._pool = None
        self._stream = None
        self.capture_seconds = 0.0
        self.capture_bytes = 0
        self._warm = False

    def ready(self, num_tokens: int) -> bool:
        """Whether graphs may run from this step on. Some kernels size their cached
        workspaces lazily by row count and reallocate them when a bigger step
        comes, which would leave an earlier capture reading freed memory; a step
        of a full prefill chunk has grown them all, so capture waits for one."""
        if not self._warm:
            chunk = get_schedule().chunked_prefill_size
            self._warm = num_tokens >= (chunk if chunk and chunk > 0 else 8192)
        return self._warm

    def bucket_rows(self, num_rows: int) -> Optional[int]:
        if num_rows == 0 or num_rows > self.max_rows:
            return None
        return -(-num_rows // ROW_STEP) * ROW_STEP

    def run(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the late layers on the tail rows; returns the materialized residual
        and pre of those rows, which stay valid until the next replay."""
        num_rows = positions.shape[0]
        rows = self.bucket_rows(num_rows)
        assert rows is not None, num_rows
        leaves, structure, rebuild = _flatten_state(state)
        live = [*leaves, positions, input_ids, input_ids_global]
        key = (rows, structure, tuple((t.dtype, t.shape[1:]) for t in live))
        # The eager breaks read the step's real row count off this view.
        tail_batch = copy.copy(forward_batch)
        tail_batch.global_num_token_non_padded_cpu = num_rows
        with self._break_context(tail_batch), _replay_graph_scope():
            graph = self._graphs.get(key)
            if graph is None:
                graph = self._graphs[key] = self._capture(
                    key, live, rebuild, forward_batch
                )
            else:
                for buf, t in zip(graph.inputs, live):
                    buf[:num_rows].copy_(t)
                if graph.num_token_non_padded is not None:
                    graph.num_token_non_padded.fill_(num_rows)
            if envs.SGLANG_DSV4_DECODER_REPLAY_GRAPH_DEBUG.get():
                _replay_checked(graph.graph, rows)
            else:
                graph.graph.replay()
        pre = None if graph.pre is None else graph.pre[:num_rows]
        return graph.residual[:num_rows], pre

    @contextmanager
    def _break_context(self, tail_batch: ForwardBatch):
        if self._layers is None:
            self._layers = compute_attention_and_moe_layers(self._model)
        layers = self._layers
        with (
            set_tc_piecewise_forward_context(
                tail_batch,
                layers.attention_layers,
                None,
                layers.moe_layers,
                layers.moe_fusions,
                dsa_indexers=layers.dsa_indexers,
                mha_companion_layers=layers.mha_companion_layers,
            ),
            enable_breakable_cuda_graph(),
        ):
            yield

    def _capture(self, key, live, rebuild, forward_batch) -> _ReplayGraph:
        rows = key[0]
        num_rows = live[0].shape[0]
        start = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        inputs = [t.new_zeros((rows, *t.shape[1:])) for t in live]
        for buf, t in zip(inputs, live):
            buf[:num_rows].copy_(t)
        num_state = len(live) - 3
        # The layers read the step's batch; the copy's count tensor is the graph's.
        capture_batch = copy.copy(forward_batch)
        capture_batch.global_num_token_non_padded_cpu = rows
        num_token_non_padded = None
        if forward_batch.num_token_non_padded is not None:
            num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, num_rows
            )
            capture_batch.num_token_non_padded = num_token_non_padded

        def body():
            return self._run_layers(
                rebuild(inputs[:num_state]),
                positions=inputs[num_state],
                input_ids=inputs[num_state + 1],
                input_ids_global=inputs[num_state + 2],
                forward_batch=capture_batch,
            )

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream()
        tp_group = get_parallel().tp_group
        graph = BreakableCUDAGraph()
        with graph_capture(stream=self._stream) as context:
            # The warmups run eagerly; their KV writes repeat the replay's.
            for _ in range(2):
                torch.cuda.synchronize()
                tp_group.barrier()
                body()
            with BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=context.stream,
                barrier_fn=tp_group.barrier,
            ):
                residual, pre = body()
        torch.cuda.current_stream().wait_stream(self._stream)
        torch.cuda.synchronize()
        replay_graph = _ReplayGraph(
            graph, inputs, num_token_non_padded, residual, pre, capture_batch
        )
        seconds = time.perf_counter() - start
        used = free_before - torch.cuda.mem_get_info()[0]
        self.capture_seconds += seconds
        self.capture_bytes += used
        if get_parallel().tp_rank == 0:
            logger.info(
                "Decoder replay graph: captured %d rows (%d segments) in %.2f s, "
                "%.3f GiB; bank %d graphs, %.2f s, %.3f GiB",
                rows,
                len(graph._segments),
                seconds,
                used / 2**30,
                len(self._graphs) + 1,
                self.capture_seconds,
                self.capture_bytes / 2**30,
            )
        return replay_graph
