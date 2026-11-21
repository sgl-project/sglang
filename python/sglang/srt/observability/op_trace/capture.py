"""Capture-time op map for CUDA graphs (experimental, SGLANG_ENABLE_OP_TRACE_MAP).

Module forward hooks read the node count of the graph under capture at every
module entry and exit, so each node gets the innermost module whose forward
created it. Nothing runs at replay: CUPTI tags each replayed kernel with
``graph node id`` whose low 32 bits are the node's creation index, and the
analyzer joins on that. Outside a capture the same hooks emit
``op::<module path>`` profiler spans on eager forwards, but only while a torch
profiler is active.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.observability.op_trace.schema import (
    NODE_KERNEL,
    OP_MAP_VERSION,
    CapturedGraph,
    OpMapFile,
    OpScope,
)

logger = logging.getLogger(__name__)

EAGER_SPAN_PREFIX = "op::"
ROOT_PATH = "<root>"


class _NodeCounter:
    """Reads the node count of one graph while its stream is capturing."""

    def __init__(self, stream: torch.cuda.Stream):
        from cuda.bindings import runtime as cudart

        self._cudart = cudart
        info = cudart.cudaStreamGetCaptureInfo(stream.cuda_stream)
        _check(err=info[0], what="cudaStreamGetCaptureInfo")
        if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            raise RuntimeError("op map capture scope entered outside a capture")
        self.graph = info[3]

    def count(self) -> int:
        res = self._cudart.cudaGraphGetNodes(self.graph, numNodes=0)
        _check(err=res[0], what="cudaGraphGetNodes")
        return res[2]


class _OpMapState:
    def __init__(self):
        self.ops: list[OpScope] = []
        self.op_index: dict[str, int] = {}
        self.symbols: list[str] = []
        self.symbol_index: dict[str, int] = {}
        self.graphs: list[CapturedGraph] = []
        self.registered_models: set[int] = set()
        # (node count, op index, is_enter) while a capture scope is open.
        self.counter: Optional[_NodeCounter] = None
        self.events: list[tuple[int, int, bool]] = []
        # Open module forwards: (module id, op index, eager profiler span or None).
        self.stack: list[tuple[int, int, Optional[Any]]] = []
        # None follows torch.autograd._profiler_enabled(); a profiler owner that
        # knows its activities sets False when CPU activity is off.
        self.eager_spans: Optional[bool] = None

    def intern_op(self, *, path: str, module_class: str) -> int:
        idx = self.op_index.get(path)
        if idx is None:
            idx = len(self.ops)
            self.ops.append(OpScope(path=path, module_class=module_class))
            self.op_index[path] = idx
        return idx

    def intern_child(self, *, local: str, module_class: str) -> int:
        parent = self.ops[self.stack[-1][1]].path if self.stack else ROOT_PATH
        return self.intern_op(path=f"{parent}.{local}", module_class=module_class)

    def intern_symbol(self, symbol: str) -> int:
        idx = self.symbol_index.get(symbol)
        if idx is None:
            idx = len(self.symbols)
            self.symbols.append(symbol)
            self.symbol_index[symbol] = idx
        return idx


_state: Optional[_OpMapState] = None


def enabled() -> bool:
    return envs.SGLANG_ENABLE_OP_TRACE_MAP.get()


def _get_state() -> _OpMapState:
    global _state
    if _state is None:
        _state = _OpMapState()
    return _state


def register_model(model: torch.nn.Module) -> None:
    """Install the op-scope hooks on every submodule of ``model`` once."""
    state = _get_state()
    if id(model) in state.registered_models:
        return
    state.registered_models.add(id(model))
    names: dict[int, list[str]] = {}
    modules: dict[int, torch.nn.Module] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        names.setdefault(id(module), []).append(name or ROOT_PATH)
        modules[id(module)] = module
    for key, module in modules.items():
        paths = names[key]
        module_class = type(module).__name__
        if len(paths) == 1:
            op = state.intern_op(path=paths[0], module_class=module_class)
            module.register_forward_pre_hook(_make_pre_hook(lambda op=op: op))
        else:
            # A module shared by several parents (e.g. one rotary embedding for
            # all layers) takes its path from whichever parent is calling it.
            local = paths[0].rsplit(".", 1)[-1]
            module.register_forward_pre_hook(
                _make_pre_hook(
                    lambda local=local, cls=module_class: state.intern_child(
                        local=local, module_class=cls
                    )
                )
            )
        # always_call keeps the stack balanced when a forward raises.
        module.register_forward_hook(_post_hook, always_call=True)


def _make_pre_hook(resolve_op):
    def pre_hook(module, args):
        state = _state
        op = resolve_op()
        span = None
        if state.counter is not None:
            state.events.append((state.counter.count(), op, True))
        elif (
            state.eager_spans
            if state.eager_spans is not None
            else torch.autograd._profiler_enabled()
        ):
            from torch._C._profiler import _RecordFunctionFast

            span = _RecordFunctionFast(EAGER_SPAN_PREFIX + state.ops[op].path)
            span.__enter__()
        state.stack.append((id(module), op, span))

    return pre_hook


def _post_hook(module, args, output):
    state = _state
    if not state.stack:
        return
    module_id, op, span = state.stack.pop()
    assert module_id == id(module), (
        f"op_trace hook stack out of sync at {state.ops[op].path}"
    )
    if state.counter is not None:
        state.events.append((state.counter.count(), op, False))
    elif span is not None:
        span.__exit__(None, None, None)


def set_eager_spans(enabled: Optional[bool]) -> None:
    """Override whether eager forwards emit ``op::`` spans while profiling;
    torch has no query for the active profiler's activities, so the code that
    starts the profiler says whether CPU activity (which reads them) is on."""
    _get_state().eager_spans = enabled


@contextmanager
def capture_scope(
    *, runner: str, shape: str, forward_mode: str, size: int
) -> Iterator[None]:
    """Wrap the body of one CUDA graph capture; must be entered after the
    capture began and exit before it ends, so the graph is still readable."""
    state = _get_state()
    state.counter = _NodeCounter(torch.cuda.current_stream())
    state.events = []
    state.stack = []
    try:
        yield
        state.graphs.append(
            _snapshot_graph(
                state,
                runner=runner,
                shape=shape,
                forward_mode=forward_mode,
                size=size,
            )
        )
    finally:
        state.counter = None
        state.events = []
        state.stack = []


def _snapshot_graph(
    state: _OpMapState, *, runner: str, shape: str, forward_mode: str, size: int
) -> CapturedGraph:
    from cuda.bindings import runtime as cudart

    graph = state.counter.graph
    num_nodes = state.counter.count()
    res = cudart.cudaGraphGetNodes(graph, numNodes=num_nodes)
    _check(err=res[0], what="cudaGraphGetNodes")
    node_type, node_symbol, node_grid, node_block = [], [], [], []
    for node in res[1]:
        kind, symbol, grid, block = _describe_node(node)
        node_type.append(kind)
        node_symbol.append(state.intern_symbol(symbol) if symbol else -1)
        node_grid.append(grid)
        node_block.append(block)
    return CapturedGraph(
        runner=runner,
        shape=shape,
        forward_mode=forward_mode,
        size=size,
        capture_seq=len(state.graphs),
        graph_handle=int(graph),
        node_type=node_type,
        node_symbol=node_symbol,
        node_grid=node_grid,
        node_block=node_block,
        node_op=_assign_ops(events=state.events, num_nodes=num_nodes),
    )


def _assign_ops(*, events: list[tuple[int, int, bool]], num_nodes: int) -> list[int]:
    """Each node gets the innermost module that was open when it was created."""
    node_op = [-1] * num_nodes
    stack: list[int] = []
    prev = 0
    for count, op, is_enter in events:
        top = stack[-1] if stack else -1
        node_op[prev:count] = [top] * (count - prev)
        prev = count
        if is_enter:
            stack.append(op)
        elif stack and stack[-1] == op:
            stack.pop()
    return node_op


def _describe_node(node) -> tuple[str, str, tuple, tuple]:
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings import runtime as cudart

    res = cudart.cudaGraphNodeGetType(node)
    _check(err=res[0], what="cudaGraphNodeGetType")
    if res[1] != cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel:
        return (
            res[1].name.removeprefix("cudaGraphNodeType").lower(),
            "",
            (0, 0, 0),
            (0, 0, 0),
        )
    params = cuda_driver.cuGraphKernelNodeGetParams(cuda_driver.CUgraphNode(int(node)))
    if int(params[0]) != 0:
        return NODE_KERNEL, "", (0, 0, 0), (0, 0, 0)
    p = params[1]
    grid = (p.gridDimX, p.gridDimY, p.gridDimZ)
    block = (p.blockDimX, p.blockDimY, p.blockDimZ)
    name = cuda_driver.cuFuncGetName(p.func)
    if int(name[0]) != 0:
        name = cuda_driver.cuKernelGetName(p.kern)
    symbol = name[1].decode() if int(name[0]) == 0 else ""
    return NODE_KERNEL, symbol, grid, block


def flush(*, tp_rank: int) -> Optional[str]:
    """Rewrite this process's op map file with every graph captured so far."""
    state = _state
    if state is None or not state.graphs:
        return None
    out_dir = envs.SGLANG_TORCH_PROFILER_DIR.get()
    os.makedirs(out_dir, exist_ok=True)
    pid = os.getpid()
    path = os.path.join(out_dir, f"op_map-TP{tp_rank}-pid{pid}.json")
    payload = OpMapFile(
        version=OP_MAP_VERSION,
        pid=pid,
        tp_rank=tp_rank,
        symbols=state.symbols,
        ops=state.ops,
        graphs=state.graphs,
    )
    handles = [g.graph_handle for g in state.graphs]
    if len(set(handles)) != len(handles):
        # torch frees the captured graph after instantiation, so its address
        # can be reused; joins must pair by capture order, not by handle.
        logger.info("op map: CUgraph handles reused across captures")
    with open(path, "wb") as f:
        f.write(msgspec.json.encode(payload))
    logger.info(
        "op map: %d graphs, %d ops written to %s",
        len(state.graphs),
        len(state.ops),
        path,
    )
    return path


def _check(*, err, what: str) -> None:
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
