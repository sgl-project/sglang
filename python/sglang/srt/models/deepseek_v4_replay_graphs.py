"""Breakable CUDA graphs of DeepSeek-V4 layer ranges on eager prefill steps.

Two ranges use them: the full-width layers before the last kv_source layer, keyed
by the step's token count, and (under decoder SWA bounded replay) the trimmed late
layers, keyed by the tail's row count. In both, the KV store, the attention and
the low-ratio sources are eager breaks that read the step's live metadata;
everything else reads static row buffers. Every bucket is captured at startup
(``capture_at_startup``), after a full-chunk eager forward has grown every
row-sized workspace; a bucket missing at serving time runs eagerly. Each replay
asserts that the buffers its kernels read by address have not moved.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
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
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Tail rows come in per-request blocks of at most SWA_WINDOW = 128.
TAIL_ROW_STEP = 128
# Arbitrary; at most 3% padding on a full 8K chunk.
TOKEN_STEP = 256

_in_replay_graph = False

# One private pool and capture stream for every range: the ranges never run
# concurrently, so their graphs reuse each other's free blocks.
_pool = None
_stream = None


def in_decoder_replay_graph() -> bool:
    """True while a layer range captures or replays an eager replay graph."""
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
    # The SWA store writes the step's live slots, so a replay graph breaks here.
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


# ---------------------------------------------------------------------------
# Pointer guard: buffers a graph reads by address must not move after capture
# ---------------------------------------------------------------------------

_pointer_probes: dict[str, Callable[[], dict[str, int]]] = {}


def register_pointer_probe(name: str, probe: Callable[[], dict[str, int]]) -> None:
    """``probe`` returns {buffer name: data_ptr} for process-wide buffers that
    captured kernels may read; a recorded pointer that changes fails the replay."""
    _pointer_probes[name] = probe


def _request_window_probe() -> dict[str, int]:
    window = getattr(get_token_to_kv_pool(), "request_window", None)
    if window is None or window.workspace is None:
        return {}
    return {"workspace": window.workspace.kv_buffer[0].data_ptr()}


def _flashinfer_cache_probe() -> dict[str, int]:
    try:
        from flashinfer import utils as flashinfer_utils
    except ImportError:
        return {}
    cache = getattr(flashinfer_utils, "_cache_buf", None) or {}
    return {str(k): v.data_ptr() for k, v in cache.items() if torch.is_tensor(v)}


register_pointer_probe("request_window", _request_window_probe)
register_pointer_probe("flashinfer_cache", _flashinfer_cache_probe)


def _pointer_snapshot(owned: dict[str, Optional[torch.Tensor]]) -> dict[str, int]:
    pointers = {f"owned.{k}": t.data_ptr() for k, t in owned.items() if t is not None}
    for name, probe in _pointer_probes.items():
        pointers.update({f"{name}.{k}": p for k, p in probe().items()})
    return pointers


def check_pointers(
    recorded: dict[str, int], owned: dict[str, Optional[torch.Tensor]], label: str
) -> None:
    current = _pointer_snapshot(owned)
    moved = [
        f"{k}: {p:#x} -> {current.get(k, 0):#x}"
        for k, p in recorded.items()
        if current.get(k) != p
    ]
    if moved:
        raise AssertionError(
            f"{label}: buffers read by address moved after capture: " + "; ".join(moved)
        )


# ---------------------------------------------------------------------------
# State flattening and debug replay
# ---------------------------------------------------------------------------


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


def _replay_checked(graph: BreakableCUDAGraph, label: str) -> None:
    for i, seg in enumerate(graph._segments):
        steps = [("segment", seg.replay)]
        if i < len(graph._break_fns):
            steps.append(("break", graph._break_fns[i]))
        for what, step in steps:
            try:
                step()
                torch.cuda.synchronize()
            except Exception:
                logger.error(
                    "%s: %s %d of %d segments faulted",
                    label,
                    what,
                    i,
                    len(graph._segments),
                )
                raise


# ---------------------------------------------------------------------------
# The graphs
# ---------------------------------------------------------------------------


class _ReplayGraph:
    __slots__ = ("graph", "rows", "out_rebuild", "pointers", "owned", "keepalive")

    def __init__(self, graph, rows, out_rebuild, pointers, owned, keepalive):
        self.graph = graph
        self.rows = rows
        self.out_rebuild = out_rebuild
        self.pointers = pointers
        # Buffers the graph owns and its kernels read by address.
        self.owned = owned
        # The capture step's batch and attention metadata: kernels write buffers
        # cached on them (the TP-padded query heads), so the graph keeps them.
        self.keepalive = keepalive


class EagerReplayGraphs:
    """Breakable CUDA graphs of one layer range, keyed by padded rows.

    ``run_layers(state, forward_batch=..., **inputs) -> HcState`` is the range's
    body; ``inputs`` are row tensors (or None) handed to it from static buffers.
    """

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        run_layers: Callable[..., HcState],
        buckets: list[int],
        tail_rows: bool,
    ) -> None:
        self.name = name
        # Keyed by the late-layer tail's rows rather than the step's tokens.
        self.tail_rows = tail_rows
        self._model = model
        self._run_layers = run_layers
        self.buckets = sorted(buckets)
        self._graphs: dict[tuple, _ReplayGraph] = {}
        # Max-size static buffers shared by every bucket of one input structure.
        self._static: dict[tuple, list[torch.Tensor]] = {}
        self._static_out: dict[tuple, list[torch.Tensor]] = {}
        self._layers = None
        self._capture_open = False
        self.capture_seconds = 0.0
        self.capture_bytes = 0

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    def bucket_rows(self, num_rows: int) -> Optional[int]:
        if num_rows == 0:
            return None
        return next((b for b in self.buckets if b >= num_rows), None)

    @contextmanager
    def capture_scope(self):
        self._capture_open = True
        try:
            yield
        finally:
            self._capture_open = False

    def run(
        self, *, state: HcState, forward_batch: ForwardBatch, **inputs
    ) -> Optional[HcState]:
        """Run the range on ``state``'s rows from its graph; None (run eagerly) when
        no graph exists for the bucket and capture is closed."""
        leaves, structure, rebuild = _flatten_state(state)
        num_rows = leaves[0].shape[0]
        rows = self.bucket_rows(num_rows)
        if rows is None:
            return None
        names = tuple(k for k, v in inputs.items() if v is not None)
        live = leaves + [inputs[k] for k in names]
        shape_key = (structure, names, tuple((t.dtype, t.shape[1:]) for t in live))
        key = (rows, shape_key)
        graph = self._graphs.get(key)
        if graph is None and not self._capture_open:
            return None
        # The eager breaks read the step's real row count off this view.
        break_batch = copy.copy(forward_batch)
        break_batch.global_num_token_non_padded_cpu = num_rows
        label = f"{self.name} graph {rows} rows"
        with self._break_context(break_batch), _replay_graph_scope():
            static = self._static_inputs(shape_key, live)
            for buf, t in zip(static, live):
                buf[:num_rows].copy_(t)
            if graph is None:
                graph = self._graphs[key] = self._capture(
                    rows=rows,
                    num_rows=num_rows,
                    shape_key=shape_key,
                    static=static,
                    num_leaves=len(leaves),
                    rebuild=rebuild,
                    names=names,
                    forward_batch=forward_batch,
                )
            else:
                check_pointers(graph.pointers, graph.owned, label)
            ntnp = graph.owned["num_token_non_padded"]
            if ntnp is not None:
                ntnp.fill_(num_rows)
            if envs.SGLANG_DSV4_DECODER_REPLAY_GRAPH_DEBUG.get():
                _replay_checked(graph.graph, label)
            else:
                graph.graph.replay()
        return graph.out_rebuild(num_rows)

    def _static_inputs(self, shape_key, live) -> list[torch.Tensor]:
        static = self._static.get(shape_key)
        if static is None:
            top = self.buckets[-1]
            static = self._static[shape_key] = [
                t.new_zeros((top, *t.shape[1:])) for t in live
            ]
        return static

    @contextmanager
    def _break_context(self, break_batch: ForwardBatch):
        if self._layers is None:
            self._layers = compute_attention_and_moe_layers(self._model)
        layers = self._layers
        with (
            set_tc_piecewise_forward_context(
                break_batch,
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

    def _capture(
        self,
        *,
        rows,
        num_rows,
        shape_key,
        static,
        num_leaves,
        rebuild,
        names,
        forward_batch,
    ) -> _ReplayGraph:
        start = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        inputs = [buf[:rows] for buf in static]
        # The layers read the step's batch; the copy's count tensor is the graph's.
        capture_batch = copy.copy(forward_batch)
        capture_batch.global_num_token_non_padded_cpu = rows
        num_token_non_padded = None
        if forward_batch.num_token_non_padded is not None:
            num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, num_rows
            )
            capture_batch.num_token_non_padded = num_token_non_padded
        out_static: list = []

        def body():
            out = self._run_layers(
                rebuild(inputs[:num_leaves]),
                forward_batch=capture_batch,
                **dict(zip(names, inputs[num_leaves:])),
            )
            out_leaves, out_structure, out_rebuild = _flatten_state(out)
            if not out_static:
                # Shared max-size output buffers per output structure.
                okey = (shape_key, out_structure)
                buffers = self._static_out.get(okey)
                if buffers is None:
                    top = self.buckets[-1]
                    buffers = self._static_out[okey] = [
                        t.new_empty((top, *t.shape[1:])) for t in out_leaves
                    ]
                out_static.extend([buffers, out_rebuild])
            for buf, t in zip(out_static[0], out_leaves):
                buf[: t.shape[0]].copy_(t)

        global _pool, _stream
        if _pool is None:
            _pool = torch.cuda.graph_pool_handle()
            _stream = torch.cuda.Stream()
        tp_group = get_parallel().tp_group
        graph = BreakableCUDAGraph()
        with graph_capture(stream=_stream) as context:
            # The warmups run eagerly; their KV writes repeat the replay's.
            for _ in range(2):
                torch.cuda.synchronize()
                tp_group.barrier()
                body()
            with BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=_pool,
                stream=context.stream,
                barrier_fn=tp_group.barrier,
            ):
                body()
        torch.cuda.current_stream().wait_stream(_stream)
        torch.cuda.synchronize()
        metadata = get_attn_backend().forward_metadata
        owned = {
            "num_token_non_padded": num_token_non_padded,
            "q_pad_buffer": getattr(metadata, "q_pad_buffer", None),
        }
        buffers, out_rebuild = out_static

        def rebuild_rows(n: int) -> HcState:
            return out_rebuild([b[:n] for b in buffers])

        replay_graph = _ReplayGraph(
            graph,
            rows,
            rebuild_rows,
            _pointer_snapshot(owned),
            owned,
            (capture_batch, metadata),
        )
        seconds = time.perf_counter() - start
        used = free_before - torch.cuda.mem_get_info()[0]
        self.capture_seconds += seconds
        self.capture_bytes += used
        if get_parallel().tp_rank == 0:
            logger.info(
                "%s graph: captured %d rows (%d segments) in %.2f s, %.3f GiB; "
                "%d graphs, %.2f s, %.3f GiB",
                self.name,
                rows,
                len(graph._segments),
                seconds,
                used / 2**30,
                len(self._graphs) + 1,
                self.capture_seconds,
                self.capture_bytes / 2**30,
            )
        return replay_graph


# ---------------------------------------------------------------------------
# Startup capture
# ---------------------------------------------------------------------------


def _dummy_extend(eager_runner, buffers, *, batch_size: int, tokens_per_req: int):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    n = batch_size * tokens_per_req
    # Real positions: rotary tables and the request-window layout read them.
    buffers.positions[:n].copy_(
        torch.arange(tokens_per_req, device=buffers.positions.device).repeat(batch_size)
    )
    eager_runner._dummy_run(
        batch_size,
        forward_mode_override=ForwardMode.EXTEND,
        buffers=buffers,
        extend_num_tokens_per_req=tokens_per_req,
    )


def capture_at_startup(
    *, eager_runner, request_window, graphs: list[EagerReplayGraphs]
) -> None:
    """Capture every bucket of ``graphs`` from dummy eager forwards, after one
    full-chunk forward with graphs closed has grown every row-sized workspace."""
    graphs = [g for g in graphs if g is not None]
    if not graphs:
        return
    chunk = max(g.buckets[-1] for g in graphs)
    # A tail bucket of B rows is B / 128 requests of 128 tokens.
    max_bs = max([g.buckets[-1] // TAIL_ROW_STEP for g in graphs if g.tail_rows] + [1])
    buffers = eager_runner._alloc_dummy_decode_buffers(
        max_bs, num_tokens_per_req=-(-chunk // max_bs)
    )
    start = time.perf_counter()
    _dummy_extend(eager_runner, buffers, batch_size=1, tokens_per_req=chunk)
    plan = []
    for g in graphs:
        if g.tail_rows:
            # One 128-token request per tail block.
            plan += [(b // TAIL_ROW_STEP, TAIL_ROW_STEP) for b in g.buckets]
        else:
            plan += [(1, b) for b in g.buckets]
    # Largest shapes first, so later captures reuse the pool's free blocks.
    plan = sorted(set(plan), key=lambda p: -p[0] * p[1])
    with ExitStack() as stack:
        for g in graphs:
            stack.enter_context(g.capture_scope())
        for batch_size, tokens_per_req in plan:
            _dummy_extend(
                eager_runner,
                buffers,
                batch_size=batch_size,
                tokens_per_req=tokens_per_req,
            )
    if request_window is not None:
        # Dummy requests used slots [0, max_bs); real requests start clean.
        request_window.reset(torch.arange(max_bs, device=buffers.positions.device))
    del buffers
    # The warmups cached activation blocks of every bucket size; serving reuses none.
    torch.cuda.empty_cache()
    if get_parallel().tp_rank == 0:
        free, _ = torch.cuda.mem_get_info()
        logger.info(
            "Eager replay graphs captured at startup in %.1f s (free %.2f GiB, "
            "allocated %.2f GiB, reserved %.2f GiB after capture): %s",
            time.perf_counter() - start,
            free / 2**30,
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
            ", ".join(
                f"{g.name} {g.num_graphs} graphs {g.capture_bytes / 2**30:.2f} GiB"
                for g in graphs
            ),
        )
