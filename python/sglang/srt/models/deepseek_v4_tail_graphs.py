"""CUDA graphs of DeepSeek-V4.1's decoder-replay tail on eager prefill steps.

Under decoder SWA bounded replay the layers from ``late_layer_start`` on run over
each request's last SWA_WINDOW extend rows. An eager prefill step pays one host
launch per kernel for those few rows, so it replays a breakable graph captured for
the tail's row bucket (a multiple of SWA_WINDOW) instead. Inside the graph the SWA
KV store, the attention and the low-ratio sources are eager breaks that read the
step's live tail metadata and its real row count; everything else reads static
row buffers shared by every bucket. A bucket is captured on its first use, on
every TP rank in the same step, into the global graph pool on the global capture
stream, so it reuses memory the other graphs already hold.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
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
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_in_tail_graph = False


def in_decoder_tail_graph() -> bool:
    """True while the tail layers capture or replay a decoder tail graph."""
    return _in_tail_graph


@contextmanager
def _tail_graph_scope():
    global _in_tail_graph
    _in_tail_graph = True
    try:
        yield
    finally:
        _in_tail_graph = False


def _tail_kv_store(attention, x, positions, qkv_a) -> None:
    # The SWA store writes the step's live tail slots, so a tail graph breaks here.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_rows = forward_batch.global_num_token_non_padded_cpu
    if num_rows == 0:
        return
    attention._compute_kv_to_cache(
        x[:num_rows],
        positions[:num_rows],
        forward_batch,
        get_attn_backend(),
        qkv_a=None if qkv_a is None else qkv_a[:num_rows],
    )


bcg_tail_kv_store = eager_on_graph(True)(_tail_kv_store)


def _flatten_state(state: HcState):
    """The state's row tensors, a key for their structure, and the inverse. A
    normed input and the mix stats are dropped, as `HcState.take_rows` drops a
    quantized input: the first tail layer rebuilds them, so every step keys the
    same graphs whatever the previous post left behind."""
    assert isinstance(state.streams, torch.Tensor), "tail graphs are CUDA only"
    has_pre = state.pre is not None
    leaves = [state.streams, state.pre] if has_pre else [state.streams]

    def rebuild(tensors: List[torch.Tensor]) -> HcState:
        return HcState(tensors[0], tensors[1] if has_pre else None)

    return leaves, has_pre, rebuild


class _TailGraph:
    __slots__ = ("graph", "num_token_non_padded", "residual", "pre", "aux", "batch")

    def __init__(self, graph, num_token_non_padded, residual, pre, aux, batch):
        self.graph = graph
        # MoE top-k masks rows at or past this count; captured by address.
        self.num_token_non_padded = num_token_non_padded
        self.residual = residual
        self.pre = pre
        self.aux = aux
        # The capture step's batch and tail attention metadata: kernels write
        # buffers cached on them, so the graph keeps them alive.
        self.batch = batch


class DecoderTailGraphs:
    """Breakable CUDA graphs of the decoder-replay tail layers, keyed by padded
    tail rows; the static input buffers are sized once for the largest bucket and
    shared by every bucket."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        layer_ids: range,
        row_step: int,
        max_rows: int,
    ) -> None:
        self._model = model
        self._layer_ids = layer_ids
        self.row_step = row_step
        self.max_rows = max_rows - max_rows % row_step
        self._graphs: dict[tuple, _TailGraph] = {}
        self._static: dict[tuple, List[torch.Tensor]] = {}
        self._layers = None
        self._warm = False
        self.capture_seconds = 0.0
        self.capture_bytes = 0

    def ready(self, num_tokens: int) -> bool:
        """Whether graphs may run from this step on; every extend forward reports
        its token count here. Some kernels size cached workspaces lazily by row
        count and reallocate them when a bigger step comes, which would leave an
        earlier capture reading freed memory; a forward of a full prefill chunk
        (the prefill graphs' own capture warmups included) has grown them all."""
        if not self._warm:
            chunk = get_schedule().chunked_prefill_size
            self._warm = num_tokens >= (chunk if chunk and chunk > 0 else 8192)
        return self._warm

    @property
    def warm(self) -> bool:
        return self._warm

    def bucket_rows(self, num_rows: int) -> Optional[int]:
        if num_rows == 0 or num_rows > self.max_rows:
            return None
        return -(-num_rows // self.row_step) * self.row_step

    def run(
        self,
        *,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        forward_batch: ForwardBatch,
        capture_dspark: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """Run the tail layers on the tail rows. Returns the materialized residual
        and pre of those rows and the DSpark auxiliary states, all views of graph
        buffers that stay valid until the next replay."""
        num_rows = positions.shape[0]
        rows = self.bucket_rows(num_rows)
        assert rows is not None, num_rows
        leaves, structure, rebuild = _flatten_state(state)
        live = [*leaves, positions, input_ids, input_ids_global]
        shape_key = (structure, tuple((t.dtype, t.shape[1:]) for t in live))
        static = self._static_inputs(shape_key, live)
        for buf, t in zip(static, live):
            buf[:num_rows].copy_(t)
        key = (rows, capture_dspark, shape_key)
        # The eager breaks read the step's real row count off this view.
        tail_batch = copy.copy(forward_batch)
        tail_batch.global_num_token_non_padded_cpu = num_rows
        with self._break_context(tail_batch), _tail_graph_scope():
            if key not in self._graphs:
                # Largest first, so smaller buckets reuse the pool memory the
                # larger ones took; a bucket below this step's rows cannot be
                # captured here (the breaks write every live row).
                for r in range(self.max_rows, rows - 1, -self.row_step):
                    if (r, *key[1:]) not in self._graphs:
                        self._graphs[(r, *key[1:])] = self._capture(
                            r, num_rows, static, rebuild, forward_batch, capture_dspark
                        )
            graph = self._graphs[key]
            if graph.num_token_non_padded is not None:
                graph.num_token_non_padded.fill_(num_rows)
            with graph_pool_replay_scope():
                graph.graph.replay()
        pre = None if graph.pre is None else graph.pre[:num_rows]
        aux = [t[:num_rows] for t in graph.aux]
        return graph.residual[:num_rows], pre, aux

    def _static_inputs(self, shape_key, live) -> List[torch.Tensor]:
        static = self._static.get(shape_key)
        if static is None:
            static = self._static[shape_key] = [
                t.new_zeros((self.max_rows, *t.shape[1:])) for t in live
            ]
        return static

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

    def _layer_stack(
        self,
        state: HcState,
        *,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        forward_batch: ForwardBatch,
        capture_dspark: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """The tail layers on the static rows: the graphs' body, the same stages
        as the model loop past the tail switch."""
        model = self._model
        aux: List[torch.Tensor] = []
        residual_batch.start(forward_batch)
        hidden_states = residual_batch.set_written(state, forward_batch)
        try:
            for i in self._layer_ids:
                layer = model.layers[i]
                assert layer.engram is None
                if capture_dspark and i in model.dspark_layers_to_capture:
                    call = hc_boundary.LayerCall(input_ids, input_ids_global)
                    next_boundary = (
                        hc_boundary.HcRead(layer, True)
                        .bind(forward_batch, call)
                        .next_boundary
                    )
                    hidden_states = state = hc_boundary.materialize(
                        hidden_states, forward_batch, next=next_boundary
                    )
                    # The draft head reads the attention input of its target layers.
                    aux.append(state.residual.mean(dim=1))
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    hidden_states = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                        input_ids_global=input_ids_global,
                    )
            state = residual_batch.read_output(
                hidden_states, forward_batch, hc_boundary.FinalRead(), None
            )
        finally:
            residual_batch.clear(forward_batch)
        state = state.materialized(model.layers[self._layer_ids[-1]].hc_cfg)
        return state.residual, state.pre, aux

    def _capture(
        self,
        rows: int,
        num_rows: int,
        static: List[torch.Tensor],
        rebuild: Callable[[List[torch.Tensor]], HcState],
        forward_batch: ForwardBatch,
        capture_dspark: bool,
    ) -> _TailGraph:
        start = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        inputs = [buf[:rows] for buf in static]
        # The state leaves, then positions, IDs and global IDs.
        num_state = len(inputs) - 3
        # The layers read the step's batch; the copy's count tensor is the graph's.
        # Host-side row counts read during capture cover the whole bucket.
        capture_batch = copy.copy(forward_batch)
        capture_batch.global_num_token_non_padded_cpu = rows
        num_token_non_padded = None
        if forward_batch.num_token_non_padded is not None:
            num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, num_rows
            )
            capture_batch.num_token_non_padded = num_token_non_padded

        def body():
            return self._layer_stack(
                rebuild(inputs[:num_state]),
                positions=inputs[num_state],
                input_ids=inputs[num_state + 1],
                input_ids_global=inputs[num_state + 2],
                forward_batch=capture_batch,
                capture_dspark=capture_dspark,
            )

        return self._record(
            body,
            rows,
            capture_dspark,
            num_token_non_padded,
            capture_batch,
            start,
            free_before,
        )

    def _record(
        self,
        body,
        rows,
        capture_dspark,
        num_token_non_padded,
        capture_batch,
        start,
        free_before,
    ) -> _TailGraph:
        pool = get_or_create_global_graph_memory_pool(torch.cuda)
        set_graph_pool_id(pool)
        tp_group = get_parallel().tp_group
        graph = BreakableCUDAGraph()
        with graph_capture(stream=get_or_create_global_graph_capture_stream()) as ctx:
            # The warmups run eagerly; their KV writes repeat the replay's.
            for _ in range(2):
                torch.cuda.synchronize()
                tp_group.barrier()
                body()
            with (
                graph_pool_capture_scope(),
                BreakableCUDAGraphCapture(
                    cuda_graph=graph,
                    pool=pool,
                    stream=ctx.stream,
                    barrier_fn=tp_group.barrier,
                ),
            ):
                residual, pre, aux = body()
        torch.cuda.current_stream().wait_stream(ctx.stream)
        torch.cuda.synchronize()
        tail_graph = _TailGraph(
            graph,
            num_token_non_padded,
            residual,
            pre,
            aux,
            (capture_batch, get_attn_backend().forward_metadata),
        )
        seconds = time.perf_counter() - start
        used = free_before - torch.cuda.mem_get_info()[0]
        self.capture_seconds += seconds
        self.capture_bytes += used
        if get_parallel().tp_rank == 0:
            logger.info(
                "Decoder tail graph: captured %d rows (dspark=%s, %d segments) in "
                "%.2f s, %.3f GiB; %d graphs, %.2f s, %.3f GiB",
                rows,
                capture_dspark,
                len(graph._segments),
                seconds,
                used / 2**30,
                len(self._graphs) + 1,
                self.capture_seconds,
                self.capture_bytes / 2**30,
            )
        return tail_graph
