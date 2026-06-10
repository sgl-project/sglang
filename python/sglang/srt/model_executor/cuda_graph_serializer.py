# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/LICENSE.org/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA Graph Serializer: save and restore CUDA graph metadata for fast restart.

On first server start, captures CUDA graphs normally and serializes node
metadata (kernel functions, parameters, topology) to disk. On subsequent
starts, reconstructs the graph from metadata without running the full
warmup + capture forward passes.

The key insight is that CUDA graph nodes contain absolute GPU addresses
that change across process restarts. This module records "symbolic" references
(e.g., "buffers.input_ids + offset", "model.layers.0.q_proj.weight + offset")
instead of raw addresses, then patches in the new addresses on restart.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.model_executor.cuda_graph_inspector import (
    BufferAddressRegistry,
    CUGraphNodeType,
    CUFunctionRegistry,
    GraphMetadata,
    KernelNodeInfo,
    KernelParamInfo,
    MemcpyNodeInfo,
    MemsetNodeInfo,
    PointerCategory,
    checkCudaErrorsDriver,
    get_global_registry,
    inspect_cuda_graph,
)

logger = logging.getLogger(__name__)

try:
    from cuda.bindings import driver as cu
except ImportError:
    cu = None


# ---------------------------------------------------------------------------
# Serialization format
# ---------------------------------------------------------------------------
CACHE_VERSION = 1

SERIALIZABLE_CATEGORIES = {
    PointerCategory.INPUT_BUFFER,
    PointerCategory.MODEL_WEIGHT,
    PointerCategory.INTERMEDIATE,
    PointerCategory.ATTENTION_METADATA,
    PointerCategory.UNKNOWN,
}


@dataclass
class GraphCacheKey:
    """Cache key for a CUDA graph. Includes all factors that affect the
    graph structure or kernel selection."""
    model_hash: str
    batch_size: int
    num_tokens_per_bs: int
    capture_hidden_mode: str
    device_name: str
    cuda_driver_version: int
    compute_capability: Tuple[int, int]
    sglang_version: str = ""

    def to_string(self) -> str:
        cc = f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return (
            f"{self.model_hash}_bs{self.batch_size}"
            f"_tpb{self.num_tokens_per_bs}"
            f"_chm{self.capture_hidden_mode}"
            f"_{self.device_name.replace(' ', '_')}"
            f"_drv{self.cuda_driver_version}"
            f"_cc{cc}"
            f"_v{self.sglang_version}"
        )


@dataclass
class SerializableKernelParam:
    """JSON-serializable kernel parameter."""
    index: int
    is_device_pointer: bool
    raw_value: int
    category: str
    symbolic_name: str
    offset: int
    base_addr: int
    alloc_size: int
    scalar_type: str = ""
    scalar_value: Any = None


@dataclass
class SerializableKernelNode:
    """JSON-serializable kernel node."""
    node_index: int
    func_ptr: int
    kernel_name: str
    module_hash: str
    grid_dim: List[int]
    block_dim: List[int]
    shared_mem_bytes: int
    params: List[SerializableKernelParam]
    dependency_indices: List[int]


@dataclass
class SerializableMemcpyNode:
    """JSON-serializable memcpy node."""
    node_index: int
    src_ptr: int
    dst_ptr: int
    copy_size: int
    dependency_indices: List[int]


@dataclass
class SerializableMemsetNode:
    """JSON-serializable memset node."""
    node_index: int
    dst_ptr: int
    value: int
    size: int
    dependency_indices: List[int]


@dataclass
class SerializableGraphMetadata:
    """JSON-serializable graph metadata."""
    version: int = CACHE_VERSION
    cache_key: str = ""
    num_nodes: int = 0
    nodes: List[Dict] = field(default_factory=list)
    # Address maps at capture time (for debugging / validation)
    buffer_map: Dict[str, str] = field(default_factory=dict)   # ptr_hex -> "name:size"
    weight_map: Dict[str, str] = field(default_factory=dict)   # ptr_hex -> "fqn"
    # Pool base address for intermediate tensor address translation.
    # All intermediate tensors from the same pool shift by a single constant
    # offset across process restarts. Recording the pool base enables computing
    # this offset: pool_offset = new_pool_base - pool_base_addr.
    pool_base_addr: int = 0
    # Capture context
    device_name: str = ""
    cuda_driver_version: int = 0
    compute_capability: List[int] = field(default_factory=list)
    capture_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize_graph_metadata(
    metadata: GraphMetadata,
    cache_key: GraphCacheKey,
) -> SerializableGraphMetadata:
    """Convert GraphMetadata to a JSON-serializable format."""
    serializable = SerializableGraphMetadata(
        version=CACHE_VERSION,
        cache_key=cache_key.to_string(),
        num_nodes=metadata.num_nodes,
        pool_base_addr=metadata.pool_base_addr,
        device_name=metadata.device_name,
        cuda_driver_version=metadata.cuda_driver_version,
        compute_capability=list(metadata.compute_capability),
        capture_timestamp=time.time(),
    )

    # Convert buffer_map: int -> str for JSON
    serializable.buffer_map = {
        hex(ptr): f"{name}:{size}" for ptr, (name, size) in metadata.buffer_map.items()
    }
    serializable.weight_map = {
        hex(ptr): fqn for ptr, fqn in metadata.weight_map.items()
    }

    # Convert nodes
    for node in metadata.nodes:
        if isinstance(node, KernelNodeInfo):
            s_node = SerializableKernelNode(
                node_index=node.node_index,
                func_ptr=node.func_ptr,
                kernel_name=node.kernel_name,
                module_hash=node.module_hash,
                grid_dim=list(node.grid_dim),
                block_dim=list(node.block_dim),
                shared_mem_bytes=node.shared_mem_bytes,
                params=[
                    SerializableKernelParam(
                        index=p.index,
                        is_device_pointer=p.is_device_pointer,
                        raw_value=p.raw_value,
                        category=p.category.value if isinstance(p.category, PointerCategory) else str(p.category),
                        symbolic_name=p.symbolic_name,
                        offset=p.offset,
                        base_addr=p.base_addr,
                        alloc_size=p.alloc_size,
                        scalar_type=p.scalar_type,
                        scalar_value=p.scalar_value,
                    )
                    for p in node.params
                ],
                dependency_indices=node.dependency_indices,
            )
            serializable.nodes.append(asdict(s_node))
        elif isinstance(node, MemcpyNodeInfo):
            s_node = SerializableMemcpyNode(
                node_index=node.node_index,
                src_ptr=node.src_ptr,
                dst_ptr=node.dst_ptr,
                copy_size=node.copy_size,
                dependency_indices=node.dependency_indices,
            )
            serializable.nodes.append(asdict(s_node))
        elif isinstance(node, MemsetNodeInfo):
            s_node = SerializableMemsetNode(
                node_index=node.node_index,
                dst_ptr=node.dst_ptr,
                value=node.value,
                size=node.size,
                dependency_indices=node.dependency_indices,
            )
            serializable.nodes.append(asdict(s_node))

    return serializable


def save_graph_cache(
    metadata: GraphMetadata,
    cache_key: GraphCacheKey,
    cache_dir: str,
) -> str:
    """Save graph metadata to a cache file.

    Returns:
        Path to the saved cache file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"graph_cache_{cache_key.to_string()}.json"
    filepath = os.path.join(cache_dir, filename)

    serializable = serialize_graph_metadata(metadata, cache_key)

    with open(filepath, "w") as f:
        json.dump(asdict(serializable), f, indent=2)

    logger.info(f"Saved CUDA graph cache to {filepath} ({len(metadata.nodes)} nodes)")
    return filepath


def load_graph_cache(
    cache_key: GraphCacheKey,
    cache_dir: str,
) -> Optional[SerializableGraphMetadata]:
    """Load graph metadata from a cache file.

    Returns:
        SerializableGraphMetadata if cache exists and is valid, None otherwise.
    """
    filename = f"graph_cache_{cache_key.to_string()}.json"
    filepath = os.path.join(cache_dir, filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        metadata = SerializableGraphMetadata(**data)

        # Version check
        if metadata.version != CACHE_VERSION:
            logger.info(
                f"Graph cache version mismatch: expected {CACHE_VERSION}, "
                f"got {metadata.version}. Invalidating cache."
            )
            return None

        # Context validation
        current_cc = torch.cuda.get_device_capability()
        if list(current_cc) != metadata.compute_capability:
            logger.info(
                f"Compute capability changed: cache has {metadata.compute_capability}, "
                f"current is {list(current_cc)}. Invalidating cache."
            )
            return None

        if torch.cuda.driver_version() != metadata.cuda_driver_version:
            logger.info(
                f"CUDA driver version changed: cache has {metadata.cuda_driver_version}, "
                f"current is {torch.cuda.driver_version()}. Invalidating cache."
            )
            return None

        logger.info(
            f"Loaded CUDA graph cache from {filepath} "
            f"({metadata.num_nodes} nodes, "
            f"captured at {time.ctime(metadata.capture_timestamp)})"
        )
        return metadata

    except Exception as e:
        logger.warning(f"Failed to load graph cache from {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Deserialization and Graph Reconstruction
# ---------------------------------------------------------------------------
def reconstruct_cuda_graph(
    cached: SerializableGraphMetadata,
    address_registry: BufferAddressRegistry,
    func_registry: CUFunctionRegistry,
    pool_offset: int = 0,
) -> Optional[torch.cuda.CUDAGraph]:
    """Reconstruct a CUDA graph from cached metadata and current addresses.

    This creates a new cudaGraph_t by programmatically adding nodes using
    cuGraphAddKernelNode, with all device pointers patched to the current
    addresses.

    Args:
        cached: Serialized graph metadata from a previous capture.
        address_registry: Current buffer/weight addresses for pointer patching.
        func_registry: Registry for resolving kernel names to CUfunction handles.
        pool_offset: Offset to apply to intermediate tensor addresses.
            Computed as new_pool_base - old_pool_base. All intermediate
            (pool-allocated) tensor addresses are shifted by this constant.

    Returns:
        A torch.cuda.CUDAGraph ready for replay, or None if reconstruction fails.
    """
    if cu is None:
        raise ImportError(
            "cuda.bindings.driver is required for CUDA graph reconstruction. "
            "Install it with: pip install cuda-python"
        )

    try:
        # 1. Build address translation table
        # We need the OLD address registry (from cache) to know what to translate.
        # The cache stores raw addresses, so we build the translation from
        # symbolic names in the cache to current addresses.
        old_buffers = {}
        for ptr_hex, name_size in cached.buffer_map.items():
            ptr = int(ptr_hex, 16)
            name, size_str = name_size.rsplit(":", 1)
            old_buffers[ptr] = (name, int(size_str))

        old_weights = {}
        for ptr_hex, fqn in cached.weight_map.items():
            ptr = int(ptr_hex, 16)
            old_weights[ptr] = fqn

        old_registry = BufferAddressRegistry()
        old_registry._buffers = old_buffers
        old_registry._weights = old_weights

        translation = address_registry.build_address_translation(old_registry)

        logger.info(
            f"Address translation: {len(translation)}/{len(old_buffers) + len(old_weights)} "
            f"addresses resolved"
        )

        # 2. Create a new cudaGraph_t
        result = cu.cuGraphCreate(0)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphCreate failed: {result[0]}")
        graph_handle = int(result[1])
        graph = cu.CUgraph(graph_handle)

        # 3. Add nodes in topological order
        node_handles = []  # CUgraphNode handles, indexed by node_index
        param_memory = []  # Keep parameter memory alive during graph construction

        for node_data in cached.nodes:
            node_type = _get_node_type(node_data)

            if node_type == "kernel":
                handle = _add_kernel_node(
                    graph, node_data, translation, func_registry,
                    node_handles, param_memory, pool_offset
                )
                if handle is None:
                    # Failed to add a kernel node - cannot reconstruct
                    cu.cuGraphDestroy(graph)
                    return None
                node_handles.append(handle)

            elif node_type == "memcpy":
                handle = _add_memcpy_node(graph, node_data, translation, node_handles, pool_offset)
                if handle is None:
                    cu.cuGraphDestroy(graph)
                    return None
                node_handles.append(handle)

            elif node_type == "memset":
                handle = _add_memset_node(graph, node_data, translation, node_handles, pool_offset)
                if handle is None:
                    cu.cuGraphDestroy(graph)
                    return None
                node_handles.append(handle)

            else:
                logger.warning(f"Skipping unsupported node type: {node_type}")
                node_handles.append(None)

        # 4. Instantiate the graph
        instantiate_params = cu.CUDA_GRAPH_INSTANTIATE_PARAMS()
        instantiate_params.flags = cu.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
        instantiate_params.hUploadStream = 0  # NULL stream
        result = cu.cuGraphInstantiateWithParams(graph, instantiate_params)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            cu.cuGraphDestroy(graph)
            raise RuntimeError(f"cuGraphInstantiateWithParams failed: {result[0]}")

        graph_exec_handle = int(result[1])

        # Parameter memory is no longer needed after instantiation
        param_memory.clear()

        logger.info(
            f"Successfully reconstructed CUDA graph with {len(node_handles)} nodes"
        )

        # Return a wrapper that can replay the graph via cuGraphLaunch
        return _ReconstructedCUDAGraph(graph_exec_handle, graph_handle)

    except Exception as e:
        logger.warning(f"CUDA graph reconstruction failed: {e}", exc_info=True)
        return None


class _ReconstructedCUDAGraph:
    """A CUDA graph that was reconstructed from serialized metadata.

    Unlike torch.cuda.CUDAGraph, this holds raw CUDA handles and replays
    via cuGraphLaunch directly.
    """

    def __init__(self, graph_exec_handle: int, graph_handle: int):
        if cu is None:
            raise ImportError("cuda.bindings.driver required")
        self._graph_exec = cu.CUgraphExec(graph_exec_handle)
        self._graph = cu.CUgraph(graph_handle)
        self._replay_count = 0
        # Keep references to parameter memory so it stays alive
        self._param_memory = []

    def replay(self):
        """Replay the reconstructed CUDA graph."""
        stream = torch.cuda.current_stream()
        stream_ptr = stream.stream if hasattr(stream, 'stream') else stream.cuda_stream
        result = cu.cuGraphLaunch(self._graph_exec, cu.CUstream(stream_ptr))
        # cuda.bindings returns a tuple (CUresult,) for some functions
        if isinstance(result, tuple):
            err = result[0]
        else:
            err = result
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphLaunch failed: {err}")
        self._replay_count += 1

    def __del__(self):
        try:
            if self._graph_exec is not None:
                cu.cuGraphExecDestroy(self._graph_exec)
            if self._graph is not None:
                cu.cuGraphDestroy(self._graph)
        except Exception:
            pass


def _get_node_type(node_data: Dict) -> str:
    """Determine the type of a serialized node."""
    if "kernel_name" in node_data:
        return "kernel"
    elif "copy_size" in node_data:
        return "memcpy"
    elif "value" in node_data and "dst_ptr" in node_data:
        return "memset"
    return "unknown"


def _add_kernel_node(
    graph,
    node_data: Dict,
    translation: Dict[int, int],
    func_registry: CUFunctionRegistry,
    existing_nodes: list,
    param_memory: list,
    pool_offset: int = 0,
):
    """Add a kernel node to the graph being reconstructed.

    Args:
        param_memory: List to hold references to parameter memory so it stays
            alive until after cuGraphAddKernelNode is called. The CUDA driver
            copies the kernelParams data during the call, so the memory only
            needs to survive until then, but we keep references to be safe.
    """
    import ctypes

    kernel_name = node_data.get("kernel_name", "")
    module_hash = node_data.get("module_hash", "")

    # Resolve CUfunction from (module_hash, kernel_name)
    func_ptr = func_registry.resolve_function_on_restart(module_hash, kernel_name)
    if func_ptr is None:
        logger.warning(
            f"Cannot resolve kernel function: module={module_hash}, "
            f"name={kernel_name}. Falling back to normal capture."
        )
        return None

    func = cu.CUfunction(func_ptr)

    # Patch kernel parameters
    params_data = node_data.get("params", [])

    # Build the kernelParams array (void**)
    # Each element is a void* pointing to the argument value.
    # We store all values as 8-byte integers (c_uint64) since CUDA kernel
    # parameters are 8-byte aligned on 64-bit systems.
    param_values = []
    for p in params_data:
        raw_value = p["raw_value"]
        is_device_ptr = p["is_device_pointer"]
        category = p.get("category", "unknown")

        if is_device_ptr:
            # Translate the address
            new_value = translation.get(raw_value, raw_value)
            if new_value == raw_value:
                # No exact match in translation table.
                # For intermediate/unknown pool-allocated tensors,
                # apply the pool offset (all such tensors shift by a
                # single constant across process restarts).
                if category in (
                    PointerCategory.INTERMEDIATE.value,
                    PointerCategory.UNKNOWN.value,
                ) and pool_offset != 0:
                    new_value = raw_value + pool_offset
                elif category not in (
                    PointerCategory.UNKNOWN.value,
                    PointerCategory.INTERMEDIATE.value,
                ):
                    logger.warning(
                        f"Cannot translate pointer 0x{raw_value:x} "
                        f"(category={category}, name={p.get('symbolic_name', '')}). "
                        f"Falling back to normal capture."
                    )
                    return None
            param_values.append(new_value)
        else:
            param_values.append(raw_value)

    num_params = len(param_values)
    if num_params == 0:
        logger.warning(f"Kernel node {node_data['node_index']} has no params, skipping")
        return None

    # Create a ctypes array for the parameter values.
    # This must stay alive until after cuGraphAddKernelNode returns.
    c_uint64_array = ctypes.c_uint64 * num_params
    values = c_uint64_array(*param_values)

    # Create the void** array: each element is the address of the i-th value.
    param_addrs = [
        ctypes.addressof(values) + i * ctypes.sizeof(ctypes.c_uint64)
        for i in range(num_params)
    ]
    c_void_p_array = ctypes.c_void_p * num_params
    kernel_params = c_void_p_array(*param_addrs)

    # Keep references alive until after the cuGraphAddKernelNode call
    param_memory.append((values, kernel_params))

    # Build dependency array
    dep_indices = node_data.get("dependency_indices", [])
    deps = [
        existing_nodes[i]
        for i in dep_indices
        if i < len(existing_nodes) and existing_nodes[i] is not None
    ]

    # Add the kernel node
    grid = node_data["grid_dim"]
    block = node_data["block_dim"]

    node_params = cu.CUDA_KERNEL_NODE_PARAMS()
    node_params.func = func
    node_params.gridDimX = grid[0]
    node_params.gridDimY = grid[1]
    node_params.gridDimZ = grid[2]
    node_params.blockDimX = block[0]
    node_params.blockDimY = block[1]
    node_params.blockDimZ = block[2]
    node_params.sharedMemBytes = node_data["shared_mem_bytes"]
    node_params.kernelParams = ctypes.cast(kernel_params, ctypes.POINTER(ctypes.c_void_p))
    node_params.extra = 0

    if deps:
        dep_array = (cu.CUgraphNode * len(deps))(*deps)
        result = cu.cuGraphAddKernelNode(
            graph, dep_array, len(deps), node_params
        )
    else:
        result = cu.cuGraphAddKernelNode(graph, None, 0, node_params)

    if result[0] != cu.CUresult.CUDA_SUCCESS:
        logger.warning(f"cuGraphAddKernelNode failed: {result[0]}")
        return None

    return result[1]


def _add_memcpy_node(graph, node_data: Dict, translation: Dict[int, int], existing_nodes: list, pool_offset: int = 0):
    """Add a memcpy node to the graph being reconstructed."""
    import ctypes

    src_ptr = node_data.get("src_ptr", 0)
    dst_ptr = node_data.get("dst_ptr", 0)
    copy_size = node_data.get("copy_size", 0)

    if copy_size == 0:
        logger.warning(f"Memcpy node {node_data['node_index']} has zero copy size, skipping")
        return None

    # Translate addresses (apply pool_offset for untranslated pointers)
    new_src = translation.get(src_ptr, src_ptr + pool_offset if pool_offset else src_ptr)
    new_dst = translation.get(dst_ptr, dst_ptr + pool_offset if pool_offset else dst_ptr)

    # Build CUDA_MEMCPY3D structure for the copy parameters
    copy_params = cu.CUDA_MEMCPY3D(
        srcXInBytes=0,
        srcY=0,
        srcZ=0,
        srcLOD=0,
        srcMemoryType=cu.CUmemorytype.CU_MEMORYTYPE_DEVICE,  # Assume device-to-device
        srcHost=None,
        srcDevice=cu.CUdeviceptr(new_src),
        srcArray=None,
        srcPitch=0,
        dstXInBytes=0,
        dstY=0,
        dstZ=0,
        dstLOD=0,
        dstMemoryType=cu.CUmemorytype.CU_MEMORYTYPE_DEVICE,
        dstHost=None,
        dstDevice=cu.CUdeviceptr(new_dst),
        dstArray=None,
        dstPitch=0,
        WidthInBytes=copy_size,
        Height=1,
        Depth=1,
    )

    # Build CUDA_MEMCPY_NODE_PARAMS
    memcpy_params = cu.CUDA_MEMCPY_NODE_PARAMS(
        copyParams=copy_params,
        flags=0,
    )

    # Build dependency array
    dep_indices = node_data.get("dependency_indices", [])
    deps = [
        existing_nodes[i]
        for i in dep_indices
        if i < len(existing_nodes) and existing_nodes[i] is not None
    ]

    if deps:
        dep_array = (cu.CUgraphNode * len(deps))(*deps)
        result = cu.cuGraphAddMemcpyNode(graph, dep_array, len(deps), memcpy_params, None)
    else:
        result = cu.cuGraphAddMemcpyNode(graph, None, 0, memcpy_params, None)

    if result[0] != cu.CUresult.CUDA_SUCCESS:
        logger.warning(f"cuGraphAddMemcpyNode failed: {result[0]}")
        return None

    return result[1]


def _add_memset_node(graph, node_data: Dict, translation: Dict[int, int], existing_nodes: list, pool_offset: int = 0):
    """Add a memset node to the graph being reconstructed."""
    import ctypes

    dst_ptr = node_data.get("dst_ptr", 0)
    value = node_data.get("value", 0)
    size = node_data.get("size", 0)

    if size == 0:
        logger.warning(f"Memset node {node_data['node_index']} has zero size, skipping")
        return None

    # Translate address (apply pool_offset for untranslated pointers)
    new_dst = translation.get(dst_ptr, dst_ptr + pool_offset if pool_offset else dst_ptr)

    # Build CUDA_MEMSET_NODE_PARAMS structure
    memset_params = cu.CUDA_MEMSET_NODE_PARAMS(
        dst=cu.CUdeviceptr(new_dst),
        pitch=size,  # pitch = width in bytes for 1D memset
        value=value,
        elementSize=1,  # Assume byte-level memset
        width=size,
        height=1,
    )

    # Build dependency array
    dep_indices = node_data.get("dependency_indices", [])
    deps = [
        existing_nodes[i]
        for i in dep_indices
        if i < len(existing_nodes) and existing_nodes[i] is not None
    ]

    if deps:
        dep_array = (cu.CUgraphNode * len(deps))(*deps)
        result = cu.cuGraphAddMemsetNode(graph, dep_array, len(deps), memset_params, None)
    else:
        result = cu.cuGraphAddMemsetNode(graph, None, 0, memset_params, None)

    if result[0] != cu.CUresult.CUDA_SUCCESS:
        logger.warning(f"cuGraphAddMemsetNode failed: {result[0]}")
        return None

    return result[1]


# ---------------------------------------------------------------------------
# Pool offset computation
# ---------------------------------------------------------------------------
def compute_pool_offset(
    cached: SerializableGraphMetadata,
    current_pool_base: int,
) -> int:
    """Compute the pool offset for translating intermediate tensor addresses.

    All intermediate tensors from the same CUDA graph memory pool shift by a
    single constant offset across process restarts. This function computes that
    offset from the cached pool base and the current pool base.

    Args:
        cached: Serialized graph metadata from a previous capture.
        current_pool_base: The base address of the current pool (from
            inspecting a newly captured graph's intermediate tensors).

    Returns:
        The offset to add to old intermediate addresses to get new addresses.
        Returns 0 if the cached metadata has no pool base recorded.
    """
    if cached.pool_base_addr == 0:
        logger.warning(
            "Cached metadata has no pool_base_addr; cannot compute pool offset. "
            "Intermediate tensor addresses will not be translated."
        )
        return 0
    offset = current_pool_base - cached.pool_base_addr
    logger.info(
        f"Pool offset: 0x{offset:x} "
        f"(cached_base=0x{cached.pool_base_addr:x}, "
        f"current_base=0x{current_pool_base:x})"
    )
    return offset


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------
def invalidate_cache(cache_dir: str, cache_key: Optional[GraphCacheKey] = None):
    """Remove cached graph metadata.

    If cache_key is specified, only that specific cache file is removed.
    Otherwise, all cache files in the directory are removed.
    """
    if not os.path.exists(cache_dir):
        return

    if cache_key is not None:
        filename = f"graph_cache_{cache_key.to_string()}.json"
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Invalidated graph cache: {filepath}")
    else:
        for f in os.listdir(cache_dir):
            if f.startswith("graph_cache_") and f.endswith(".json"):
                os.remove(os.path.join(cache_dir, f))
                logger.info(f"Invalidated graph cache: {f}")
