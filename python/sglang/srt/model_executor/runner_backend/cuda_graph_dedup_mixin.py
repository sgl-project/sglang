"""Shared CUDA graph executable-dedup plumbing for CUDA graph backends."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

import torch

try:
    from cuda.bindings import driver as cuda_drv
    from cuda.bindings import runtime as cuda_rt
except ImportError:
    cuda_drv = None
    cuda_rt = None

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


def dedup_update(graph_exec: int, raw_graph: int) -> tuple[bool, str]:
    assert cuda_rt is not None
    err, info = cuda_rt.cudaGraphExecUpdate(graph_exec, raw_graph)
    if info is None:
        return False, f"err={int(err)}"
    result = info.result
    ok = (
        err == cuda_rt.cudaError_t.cudaSuccess
        and result == cuda_rt.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess
    )
    return ok, "" if ok else f"err={int(err)} result={result}"


def maybe_cuda_result(result):
    return None if int(result[0]) != 0 else checkCudaErrors(result)


def kernel_name(params) -> str:
    assert cuda_drv is not None
    for handle, getter in (
        (getattr(params, "kern", None), cuda_drv.cuKernelGetName),
        (getattr(params, "func", None), cuda_drv.cuFuncGetName),
    ):
        if handle is None or int(handle) == 0:
            continue
        name = maybe_cuda_result(getter(handle))
        if name is not None:
            return name.decode("utf-8", "replace")
    return f"func:{int(getattr(params, 'func', 0))}"


def kernel_attrs(node) -> tuple[tuple[str, object], ...]:
    assert cuda_drv is not None
    attrs = []
    for name, attr_name, get_value in (
        (
            "cooperative",
            "CU_LAUNCH_ATTRIBUTE_COOPERATIVE",
            lambda v: int(v.cooperative),
        ),
        (
            "clusterDim",
            "CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION",
            lambda v: (
                int(v.clusterDim.x),
                int(v.clusterDim.y),
                int(v.clusterDim.z),
            ),
        ),
        (
            "clusterSchedulingPolicyPreference",
            "CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
            lambda v: int(v.clusterSchedulingPolicyPreference),
        ),
        (
            "preferredClusterDim",
            "CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION",
            lambda v: (
                int(v.preferredClusterDim.x),
                int(v.preferredClusterDim.y),
                int(v.preferredClusterDim.z),
            ),
        ),
        (
            "sharedMemCarveout",
            "CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
            lambda v: int(v.sharedMemCarveout),
        ),
    ):
        attr = getattr(cuda_drv.CUkernelNodeAttrID, attr_name, None)
        if attr is None:
            continue
        value = maybe_cuda_result(cuda_drv.cuGraphKernelNodeGetAttribute(node, attr))
        if value is not None:
            attrs.append((name, get_value(value)))
    return tuple(attrs)


def kernel_node_payload(node):
    assert cuda_drv is not None
    params = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(node))
    return (
        kernel_name(params),
        (int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ)),
        (int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ)),
        int(params.sharedMemBytes),
        kernel_attrs(node),
    )


def graph_node_payload(node):
    assert cuda_drv is not None
    node_type = checkCudaErrors(cuda_drv.cuGraphNodeGetType(node))
    match node_type:
        case cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            payload = kernel_node_payload(node)
        case cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY:
            params = checkCudaErrors(cuda_drv.cuGraphMemcpyNodeGetParams(node))
            payload = (int(params.srcMemoryType), int(params.dstMemoryType))
        case cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET:
            params = checkCudaErrors(cuda_drv.cuGraphMemsetNodeGetParams(node))
            payload = (int(params.elementSize),)
        case cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
            child_graph = checkCudaErrors(cuda_drv.cuGraphChildGraphNodeGetGraph(node))
            payload = graph_signature(child_graph)
        case cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY:
            payload = ()
        case _:
            payload = ()
    return (node_type.name, payload)


def graph_signature(raw_graph: int):
    assert cuda_drv is not None
    _, num_nodes = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw_graph, num_nodes))
    node_indices = {int(node): i for i, node in enumerate(nodes)}

    _, _, _, num_edges = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw_graph, 0))
    from_nodes, to_nodes, _, _ = checkCudaErrors(
        cuda_drv.cuGraphGetEdges(raw_graph, num_edges)
    )
    edges = [
        (node_indices[int(src)], node_indices[int(dst)])
        for src, dst in zip(from_nodes, to_nodes)
    ]

    children = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for src, dst in edges:
        children[src].append(dst)
        indegree[dst] += 1

    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node_idx = heapq.heappop(ready)
        order.append(node_idx)
        for child_idx in sorted(children[node_idx]):
            indegree[child_idx] -= 1
            if indegree[child_idx] == 0:
                heapq.heappush(ready, child_idx)
    assert len(order) == len(nodes), "CUDA graph contains a dependency cycle"

    topo_indices = {node_idx: i for i, node_idx in enumerate(order)}
    topo_edges = tuple(
        sorted((topo_indices[src], topo_indices[dst]) for src, dst in edges)
    )
    return (
        tuple(graph_node_payload(nodes[node_idx]) for node_idx in order),
        topo_edges,
    )


@dataclass(slots=True)
class GraphExecGroup:
    graph_exec: int
    current_raw_graph: int
    compat_exec: int | None
    graphs: list[DedupedCudaGraph] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class DedupedCudaGraph:
    raw_graph: int
    original_graph: object | None
    registry: DedupedCudaGraphRegistry
    group: GraphExecGroup | None = None

    def replay(self, stream: int | None = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        self.registry.replay(self, stream)


class DedupedCudaGraphRegistry:
    def __init__(self):
        self.groups: dict[tuple, GraphExecGroup] = {}
        self.sealed = False

    def instantiate(self, raw_graph: int) -> int:
        assert cuda_rt is not None
        graph_exec = checkCudaErrors(
            cuda_rt.cudaGraphInstantiateWithFlags(raw_graph, 0)
        )
        return graph_exec

    def destroy_exec(self, graph_exec: int) -> None:
        assert cuda_rt is not None
        checkCudaErrors(cuda_rt.cudaGraphExecDestroy(graph_exec))

    def register(self, captured_graph) -> DedupedCudaGraph:
        assert not self.sealed
        raw_graph = captured_graph.raw_cuda_graph()
        signature = graph_signature(raw_graph)
        graph = DedupedCudaGraph(raw_graph, captured_graph, self)

        group = self.groups.get(signature)
        if group is not None:
            assert group.compat_exec is not None
            ok, detail = dedup_update(group.compat_exec, graph.raw_graph)
            assert ok, f"CUDA graph dedup register update failed ({detail})"
            graph.group = group
            group.graphs.append(graph)
            return graph

        group = GraphExecGroup(
            graph_exec=self.instantiate(graph.raw_graph),
            current_raw_graph=graph.raw_graph,
            compat_exec=self.instantiate(graph.raw_graph),
            graphs=[graph],
        )
        graph.group = group
        self.groups[signature] = group
        return graph

    def seal(self) -> None:
        if self.sealed:
            return
        self.sealed = True
        for group in self.groups.values():
            if group.compat_exec is not None:
                self.destroy_exec(group.compat_exec)
                group.compat_exec = None

    def stats(self) -> tuple[int, int]:
        return sum(len(group.graphs) for group in self.groups.values()), len(
            self.groups
        )

    def replay(self, graph: DedupedCudaGraph, stream: int) -> None:
        assert cuda_rt is not None
        group = graph.group
        assert (
            group is not None
        ), "captured CUDA graph does not belong to this dedup state"

        raw_graph = graph.raw_graph
        graph_exec = group.graph_exec
        if group.current_raw_graph != raw_graph:
            ok, detail = dedup_update(graph_exec, raw_graph)
            assert ok, (
                "CUDA graph dedup replay update failed "
                f"({detail}); captured graph is not compatible with its dedup group"
            )
            group.current_raw_graph = raw_graph

        checkCudaErrors(cuda_rt.cudaGraphLaunch(graph_exec, stream))

    def close(self) -> None:
        self.sealed = True

        for group in self.groups.values():
            if group.compat_exec is not None:
                self.destroy_exec(group.compat_exec)
                group.compat_exec = None
            self.destroy_exec(group.graph_exec)
            for graph in group.graphs:
                if graph.original_graph is not None:
                    graph.original_graph.reset()
                graph.original_graph = None
                graph.group = None
            group.graphs.clear()

        self.groups.clear()


class DedupedCudaGraphMixin:
    deduped_cuda_graph: DedupedCudaGraphRegistry | None = None

    def _dedup_registries(self) -> list[DedupedCudaGraphRegistry]:
        registries = getattr(self, "_deduped_cuda_graph_registries", None)
        if registries is None:
            registries = []
            self._deduped_cuda_graph_registries = registries
        return registries

    def _memory_saver_cuda_graph_enabled(self) -> bool:
        adapter = getattr(self, "_memory_saver_adapter", None)
        if adapter is not None and getattr(adapter, "enabled", False):
            return True

        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            model_runner = getattr(self, "_model_runner", None)
        server_args = getattr(model_runner, "server_args", None)
        return bool(
            server_args is not None
            and getattr(server_args, "enable_memory_saver", False)
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    def build_deduped_cuda_graph(self):
        if not envs.SGLANG_ENABLE_CUDA_GRAPH_DEDUP.get():
            return None
        if cuda_drv is None or cuda_rt is None:
            return None
        try:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            if not hasattr(graph, "raw_cuda_graph"):
                return None
            return DedupedCudaGraphRegistry()
        except TypeError:
            return None
        except Exception as e:
            logger.warning(
                "[CudaGraph][dedup] %s init failed (%s); using plain executables.",
                type(self).__name__,
                e,
            )
            return None

    def begin_cuda_graph_capture(self) -> None:
        if self.deduped_cuda_graph is not None:
            self.end_cuda_graph_capture()

        if self._memory_saver_cuda_graph_enabled():
            self.deduped_cuda_graph = None
            return

        self.deduped_cuda_graph = self.build_deduped_cuda_graph()
        if self.deduped_cuda_graph is not None:
            self._dedup_registries().append(self.deduped_cuda_graph)

    def end_cuda_graph_capture(self) -> None:
        dedup = self.deduped_cuda_graph
        self.deduped_cuda_graph = None
        if dedup is not None:
            captured, execs = dedup.stats()
            dedup.seal()
            logger.info("captured %d CUDA graphs, deduped to %d execs", captured, execs)

    def close(self) -> None:
        registries = self._dedup_registries()
        seen: set[int] = set()
        for registry in [self.deduped_cuda_graph, *registries]:
            if registry is None or id(registry) in seen:
                continue
            seen.add(id(registry))
            registry.close()
        registries.clear()
        self.deduped_cuda_graph = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
