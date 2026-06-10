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
"""CUDA Graph Node Inspector and CUFunction Registry.

Provides utilities to inspect CUDA graph nodes (kernel params, function
pointers, dependencies) and resolve CUfunction pointers to human-readable
kernel names. Used by the CUDA graph serializer to record and reconstruct
captured graphs across process restarts.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for cuda.bindings – may not be available on all systems.
# ---------------------------------------------------------------------------
try:
    from cuda.bindings import driver as cu
    from cuda.bindings import runtime as rt
except ImportError:
    cu = None
    rt = None


def _check_cuda_bindings():
    if cu is None:
        raise ImportError(
            "cuda.bindings.driver is required for CUDA graph inspection. "
            "Install it with: pip install cuda-python"
        )


# ---------------------------------------------------------------------------
# CUDA Driver API Constants (not all exposed by cuda.bindings)
# ---------------------------------------------------------------------------
class CUGraphNodeType(IntEnum):
    """cudaGraphNodeType enum values."""

    KERNEL = 0
    MEMCPY = 1
    MEMSET = 2  # cudaGraphNodeTypeMemset
    HOST = 3
    MEMCPY_FROM_SYMBOL = 4
    MEMCPY_TO_SYMBOL = 5
    EVENT_RECORD = 6
    EVENT_WAIT = 7
    EXT_SEMAS_SIGNAL = 8
    EXT_SEMAS_WAIT = 9
    COND = 10  # cudaGraphNodeTypeConditional
    WHILE_LOOP = 11  # cudaGraphNodeTypeWhileLoop
    CHILD_GRAPH = 12
    MEMORY_ALLOC = 13
    MEMORY_FREE = 14


class CUPointerAttribute(IntEnum):
    """CU_POINTER_ATTRIBUTE enum values for cuPointerGetAttribute."""

    MEMORY_TYPE = 0
    DEVICE_POINTER = 1
    HOST_POINTER = 2
    IS_MANAGED = 3
    IS_LEGACY_CUDA_IPC_CAPABLE = 4
    RANGE_START_ADDR = 5
    RANGE_SIZE = 6
    MAPPED = 7
    CONTEXT = 8
    MEMORY_POOL = 9
    IS_GPU_DIRECT_RDMA_CAPABLE = 10
    ALLOW_ACCESS = 11
    IS_UNIFIED_MEMORY = 12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class PointerCategory(str, Enum):
    """Category of a device pointer found in kernel arguments."""

    INPUT_BUFFER = "input_buffer"
    MODEL_WEIGHT = "model_weight"
    INTERMEDIATE = "intermediate"  # Pool-allocated scratch tensor
    ATTENTION_METADATA = "attn_metadata"
    UNKNOWN = "unknown"


@dataclass
class KernelParamInfo:
    """Metadata for a single kernel parameter."""

    index: int
    is_device_pointer: bool
    raw_value: int  # The 8-byte value as uint64
    # For device pointers:
    category: PointerCategory = PointerCategory.UNKNOWN
    symbolic_name: str = (
        ""  # e.g. "buffers.input_ids" or "model.layers.0.q_proj.weight"
    )
    offset: int = 0  # Offset within the named buffer
    base_addr: int = 0  # Base address of the containing allocation
    alloc_size: int = 0  # Size of the containing allocation
    # For scalars:
    scalar_type: str = ""  # e.g. "int32", "float32"
    scalar_value: Any = None


@dataclass
class KernelNodeInfo:
    """Metadata for a single kernel node in a CUDA graph."""

    node_index: int
    func_ptr: int  # CUfunction as int
    kernel_name: str = ""  # Resolved kernel name (may be empty)
    module_hash: str = ""  # Hash of the module containing this kernel
    grid_dim: Tuple[int, int, int] = (0, 0, 0)
    block_dim: Tuple[int, int, int] = (0, 0, 0)
    shared_mem_bytes: int = 0
    params: List[KernelParamInfo] = field(default_factory=list)
    dependency_indices: List[int] = field(default_factory=list)


@dataclass
class MemcpyNodeInfo:
    """Metadata for a memcpy node."""

    node_index: int
    src_ptr: int
    dst_ptr: int
    copy_size: int
    dependency_indices: List[int] = field(default_factory=list)


@dataclass
class MemsetNodeInfo:
    """Metadata for a memset node."""

    node_index: int
    dst_ptr: int
    value: int
    size: int
    dependency_indices: List[int] = field(default_factory=list)


@dataclass
class GraphMetadata:
    """Complete metadata for a CUDA graph, sufficient for reconstruction."""

    nodes: List[Any] = field(
        default_factory=list
    )  # KernelNodeInfo | MemcpyNodeInfo | MemsetNodeInfo
    num_nodes: int = 0
    # Buffer address mapping at capture time: data_ptr -> (symbolic_name, offset)
    buffer_map: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    # Model weight mapping: data_ptr -> parameter FQN
    weight_map: Dict[int, str] = field(default_factory=dict)
    # Pool-allocated intermediate tensor mapping: data_ptr -> (alloc_index, size)
    intermediate_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # Pool base address: the base allocation address of the first intermediate
    # tensor found in the graph. Used to compute pool offset for address
    # translation across process restarts (all intermediate tensors from the
    # same pool shift by a single constant offset).
    pool_base_addr: int = 0
    # Capture context
    device_name: str = ""
    cuda_driver_version: int = 0
    compute_capability: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# CUFunction Registry
# ---------------------------------------------------------------------------
class CUFunctionRegistry:
    """Tracks loaded CUDA modules and their functions to resolve CUfunction
    pointers to (module_hash, function_name) tuples.

    Works by hooking into cuModuleLoadData/cuModuleLoadDataEx to record
    every module load and enumerating its functions via cuModuleEnumerate
    (not available) or by tracking known kernel names.
    """

    def __init__(self):
        self._func_to_name: Dict[int, Tuple[str, str]] = (
            {}
        )  # CUfunction -> (module_id, func_name)
        self._module_to_funcs: Dict[int, List[Tuple[str, int]]] = (
            {}
        )  # CUmodule -> [(name, CUfunction)]
        self._module_hash_to_module: Dict[str, int] = {}  # module_hash -> CUmodule
        self._hooked: bool = False

    def hook_module_loading(self):
        """Install hooks to track CUDA module loading.

        Intercepts cuModuleLoadData/cuModuleLoadDataEx to record every
        loaded module and enumerate its functions.
        """
        if self._hooked:
            return
        _check_cuda_bindings()

        # We cannot easily hook cuModuleLoadData at the Python level because
        # PyTorch calls it through the C++ NVRTC stub. Instead, we build the
        # registry on-demand by scanning loaded modules after the fact.
        # See resolve_function() for the on-demand approach.
        self._hooked = True

    def register_module(self, module_ptr: int, module_id: str = ""):
        """Register a loaded CUDA module and enumerate its functions.

        Args:
            module_ptr: CUmodule handle as int.
            module_id: Optional identifier for this module (e.g., cubin hash).
        """
        _check_cuda_bindings()
        module = cu.CUmodule(module_ptr)
        funcs = []

        # Enumerate functions in the module using cuModuleEnumerate
        # Not available in public API; instead, we rely on known kernel names
        # or on-demand resolution via resolve_function().
        # For now, store the module for later lookup.
        self._module_to_funcs[module_ptr] = funcs
        if module_id:
            self._module_hash_to_module[module_id] = module_ptr

    def register_function(self, func_ptr: int, module_id: str, func_name: str):
        """Manually register a CUfunction -> (module_id, func_name) mapping."""
        self._func_to_name[func_ptr] = (module_id, func_name)

    def resolve_function(self, func_ptr: int) -> Optional[Tuple[str, str]]:
        """Resolve a CUfunction pointer to (module_id, function_name).

        Returns None if the function cannot be resolved.
        """
        if func_ptr in self._func_to_name:
            return self._func_to_name[func_ptr]
        return None

    def resolve_function_on_restart(
        self, module_id: str, func_name: str
    ) -> Optional[int]:
        """Look up a CUfunction by (module_id, func_name) after restart.

        This requires the module to have been re-loaded (e.g., from Triton cache).
        Returns the CUfunction pointer as int, or None if not found.
        """
        _check_cuda_bindings()

        # Strategy 1: Look up in the module hash table
        module_ptr = self._module_hash_to_module.get(module_id)
        if module_ptr is not None:
            try:
                # Try the exact name first
                result = cu.cuModuleGetFunction(
                    cu.CUmodule(module_ptr), func_name.encode()
                )
                if result[0] == cu.CUresult.CUDA_SUCCESS:
                    return int(result[1])
            except Exception:
                pass

        # Strategy 2: Search all known modules for the function
        for mod_ptr, funcs in self._module_to_funcs.items():
            # Check if we already have this function registered
            for known_name, known_func in funcs:
                if known_name == func_name:
                    return known_func

            # Try cuModuleGetFunction on this module
            try:
                # The kernel name from cuFuncGetName is typically the mangled name.
                # cuModuleGetFunction also expects the mangled name.
                result = cu.cuModuleGetFunction(
                    cu.CUmodule(mod_ptr), func_name.encode()
                )
                if result[0] == cu.CUresult.CUDA_SUCCESS:
                    func_ptr = int(result[1])
                    # Cache for future lookups
                    self._func_to_name[func_ptr] = (module_id, func_name)
                    self._module_to_funcs.setdefault(mod_ptr, []).append(
                        (func_name, func_ptr)
                    )
                    return func_ptr
            except Exception:
                pass

        # Strategy 3: Try cuModuleEnumerateFunctions on all known modules
        # (available since CUDA 12.4)
        if hasattr(cu, "cuModuleGetFunctionCount") and hasattr(
            cu, "cuModuleEnumerateFunctions"
        ):
            for mod_ptr in list(self._module_to_funcs.keys()):
                try:
                    # Get function count
                    count_result = cu.cuModuleGetFunctionCount(cu.CUmodule(mod_ptr))
                    if count_result[0] != cu.CUresult.CUDA_SUCCESS:
                        continue
                    func_count = int(count_result[1])

                    # Enumerate functions
                    enum_result = cu.cuModuleEnumerateFunctions(
                        cu.CUmodule(mod_ptr), func_count
                    )
                    if enum_result[0] != cu.CUresult.CUDA_SUCCESS:
                        continue

                    for func_handle in enum_result[1]:
                        if func_handle is None:
                            continue
                        func_ptr_new = int(func_handle)
                        # Get the function name
                        name_result = cu.cuFuncGetName(cu.CUfunction(func_handle))
                        if name_result[0] == cu.CUresult.CUDA_SUCCESS:
                            name = name_result[1]
                            if isinstance(name, bytes):
                                name = name.decode("utf-8", "replace")
                            # Register this function
                            if func_ptr_new not in self._func_to_name:
                                self._func_to_name[func_ptr_new] = (module_id, name)
                                self._module_to_funcs.setdefault(mod_ptr, []).append(
                                    (name, func_ptr_new)
                                )
                            # Check if this is the function we're looking for
                            if name == func_name:
                                return func_ptr_new
                except Exception:
                    pass

        # Strategy 4: Search by function name in all registered functions
        for func_ptr, (mid, fname) in self._func_to_name.items():
            if fname == func_name:
                return func_ptr

        return None

    def scan_known_modules(self):
        """Scan for loaded CUDA modules and build the CUfunction->name mapping.

        Uses multiple strategies:
        1. Find Triton CompiledKernel instances via gc.get_objects()
        2. Find PyTorch-loaded CUDA modules via torch.cuda internals
        3. Walk CUDA context for loaded modules (if supported)
        """

        _check_cuda_bindings()

        # Strategy 1: Find Triton CompiledKernel instances
        self._scan_triton_kernels()

        # Strategy 2: Find PyTorch-compiled kernels
        self._scan_pytorch_kernels()

        logger.info(
            f"CUFunctionRegistry: {len(self._func_to_name)} functions registered "
            f"after scan"
        )

    def _scan_triton_kernels(self):
        """Find Triton CompiledKernel instances and register their CUfunction handles.

        Triton stores compiled kernels as CompiledKernel objects (in triton.compiler.compiler).
        Each CompiledKernel has:
          - .function: CUfunction handle (int)
          - .module: CUmodule handle (int)
          - .name: kernel name (str, unmangled for Triton kernels)
          - .hash: unique hash of the compiled kernel
        """
        import gc

        # Try multiple import paths for CompiledKernel across Triton versions
        CompiledKernel = None
        for module_path in [
            "triton.compiler.compiler",
            "triton.runtime.compiler",
            "triton.compiler",
        ]:
            try:
                mod = __import__(module_path, fromlist=["CompiledKernel"])
                CompiledKernel = getattr(mod, "CompiledKernel", None)
                if CompiledKernel is not None:
                    break
            except (ImportError, AttributeError):
                continue

        if CompiledKernel is None:
            logger.debug("CompiledKernel class not found, skipping Triton kernel scan")
            return

        count = 0
        seen_modules = set()
        for obj in gc.get_objects():
            if not isinstance(obj, CompiledKernel):
                continue
            try:
                # CompiledKernel stores CUfunction and CUmodule handles
                # as int attributes after the kernel is loaded.
                func_handle = None
                module_handle = None
                func_name = None

                # Get function handle
                func_val = getattr(obj, "function", None)
                if func_val is not None:
                    try:
                        func_handle = int(func_val)
                    except (TypeError, ValueError):
                        pass

                # Get module handle
                mod_val = getattr(obj, "module", None)
                if mod_val is not None:
                    try:
                        module_handle = int(mod_val)
                    except (TypeError, ValueError):
                        pass

                # Get kernel name
                name_val = getattr(obj, "name", None)
                if name_val is not None and isinstance(name_val, str):
                    func_name = name_val

                # Validate the function handle using cuFuncGetName
                if func_handle and func_name:
                    # Verify this is a valid CUfunction
                    try:
                        name_result = cu.cuFuncGetName(cu.CUfunction(func_handle))
                        if name_result[0] == cu.CUresult.CUDA_SUCCESS:
                            verified_name = name_result[1]
                            if isinstance(verified_name, bytes):
                                verified_name = verified_name.decode("utf-8", "replace")
                            # Use the verified name from CUDA driver
                            func_name = verified_name
                        else:
                            # Function handle is not valid (kernel not loaded yet?)
                            continue
                    except Exception:
                        continue

                    module_id = (
                        f"triton_{module_handle:x}" if module_handle else "triton"
                    )
                    self._func_to_name[func_handle] = (module_id, func_name)
                    if module_handle and module_handle not in seen_modules:
                        seen_modules.add(module_handle)
                        self._module_to_funcs.setdefault(module_handle, []).append(
                            (func_name, func_handle)
                        )
                        self._module_hash_to_module[module_id] = module_handle
                    count += 1
            except Exception:
                continue

        if count > 0:
            logger.debug(
                f"Scanned {count} Triton kernel functions from CompiledKernel instances"
            )

    def _scan_pytorch_kernels(self):
        """Find PyTorch-compiled kernels and register their CUfunction handles.

        PyTorch stores compiled kernel caches internally. This method
        attempts to find them via gc.get_objects().
        """

        # PyTorch CUDAGraph and compiled functions store kernel info
        # but not in a way that's easily accessible from Python.
        # For now, this is a placeholder for future PyTorch-specific scanning.
        pass


# Global singleton registry
_global_registry: Optional[CUFunctionRegistry] = None


def get_global_registry() -> CUFunctionRegistry:
    """Get the global CUFunctionRegistry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CUFunctionRegistry()
    return _global_registry


# ---------------------------------------------------------------------------
# Kernel Name Extraction
# ---------------------------------------------------------------------------
def get_kernel_name(func_ptr: int) -> Optional[str]:
    """Get the name of a CUDA kernel function from its CUfunction pointer.

    Uses cuFuncGetName (available since CUDA 12.4) to get the mangled
    kernel name directly. Falls back to cuGraphDebugDotPrint if unavailable.

    Args:
        func_ptr: CUfunction handle as int.

    Returns:
        The mangled kernel name, or None if the name cannot be resolved.
    """
    _check_cuda_bindings()

    # Try cuFuncGetName first (available since CUDA 12.4)
    if hasattr(cu, "cuFuncGetName"):
        try:
            result = cu.cuFuncGetName(cu.CUfunction(func_ptr))
            if result[0] == cu.CUresult.CUDA_SUCCESS:
                name = result[1]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                return name
        except Exception:
            pass

    return None


def get_kernel_module(func_ptr: int) -> Optional[int]:
    """Get the CUmodule handle for a CUfunction pointer.

    Uses cuFuncGetModule (available since CUDA 12.4).

    Args:
        func_ptr: CUfunction handle as int.

    Returns:
        The CUmodule handle as int, or None if unavailable.
    """
    _check_cuda_bindings()

    if hasattr(cu, "cuFuncGetModule"):
        try:
            result = cu.cuFuncGetModule(cu.CUfunction(func_ptr))
            if result[0] == cu.CUresult.CUDA_SUCCESS:
                return int(result[1])
        except Exception:
            pass

    return None


def get_kernel_names_from_graph(
    cuda_graph: torch.cuda.CUDAGraph,
) -> Dict[int, str]:
    """Extract kernel names from a captured CUDA graph.

    Uses cuFuncGetName for each kernel node (available since CUDA 12.4).
    Falls back to cuGraphDebugDotPrint if cuFuncGetName is unavailable.

    Args:
        cuda_graph: A torch.cuda.CUDAGraph captured with keep_graph=True.

    Returns:
        Mapping from node index to kernel name (mangled). Only kernel nodes
        are included; memcpy/memset/other nodes are skipped.
    """
    _check_cuda_bindings()

    graph_handle = cuda_graph.raw_cuda_graph()
    graph = cu.CUgraph(graph_handle)

    # Enumerate nodes
    result = cu.cuGraphGetNodes(graph, 0)
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        return {}

    num_nodes = result[2] if len(result) > 2 else 0
    if num_nodes == 0:
        nodes_list = result[1] if len(result) > 1 else []
        node_array = [n for n in nodes_list if n is not None]
        num_nodes = len(node_array)
    else:
        result = cu.cuGraphGetNodes(graph, num_nodes)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            return {}
        nodes_list = result[1]
        node_array = [n for n in nodes_list if n is not None]
        num_nodes = len(node_array)

    kernel_names = {}

    # Method 1: Use cuFuncGetName for each kernel node
    for i in range(num_nodes):
        node = node_array[i]
        type_result = cu.cuGraphNodeGetType(node)
        if type_result[0] != cu.CUresult.CUDA_SUCCESS:
            continue
        node_type_val = int(type_result[1])

        if node_type_val == CUGraphNodeType.KERNEL:
            try:
                params_result = cu.cuGraphKernelNodeGetParams(node)
                if params_result[0] == cu.CUresult.CUDA_SUCCESS:
                    func_ptr = int(params_result[1].func)
                    name = get_kernel_name(func_ptr)
                    if name:
                        kernel_names[i] = name
            except Exception:
                pass

    if kernel_names:
        return kernel_names

    # Method 2: Fall back to cuGraphDebugDotPrint
    fd, dot_path = tempfile.mkstemp(suffix=".dot")
    os.close(fd)

    try:
        result = cu.cuGraphDebugDotPrint(graph, dot_path.encode(), 0x01)
        if result != cu.CUresult.CUDA_SUCCESS:
            return {}

        with open(dot_path, "r") as f:
            dot_content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read DOT output: {e}")
        return {}
    finally:
        try:
            os.unlink(dot_path)
        except OSError:
            pass

    # Parse kernel names from DOT output
    pattern = re.compile(r'N(?:ode)?(\d+)\s*\[label="([^"\\]+)')
    for match in pattern.finditer(dot_content):
        node_idx = int(match.group(1))
        label = match.group(2).strip()
        if label and label not in (
            "Memcpy",
            "Memset",
            "Host",
            "EventRecord",
            "EventWait",
            "MemoryAlloc",
            "MemoryFree",
        ):
            kernel_names[node_idx] = label

    return kernel_names


# ---------------------------------------------------------------------------
# Graph Node Inspector
# ---------------------------------------------------------------------------
def inspect_cuda_graph(
    cuda_graph: torch.cuda.CUDAGraph,
    known_buffers: Optional[Dict[int, Tuple[str, int]]] = None,
    known_weights: Optional[Dict[int, str]] = None,
    registry: Optional[CUFunctionRegistry] = None,
) -> GraphMetadata:
    """Inspect all nodes in a captured CUDA graph and return structured metadata.

    Args:
        cuda_graph: A torch.cuda.CUDAGraph that has been captured with
            keep_graph=True. Must have capture_end() called already.
        known_buffers: Mapping of data_ptr -> (name, size) for input/output
            buffers that appear in kernel arguments.
        known_weights: Mapping of data_ptr -> parameter FQN for model weights.
        registry: CUFunctionRegistry for resolving kernel function names.

    Returns:
        GraphMetadata containing all node information needed for serialization.
    """
    _check_cuda_bindings()

    if registry is None:
        registry = get_global_registry()

    known_buffers = known_buffers or {}
    known_weights = known_weights or {}

    # Get the raw cudaGraph_t handle
    graph_handle = cuda_graph.raw_cuda_graph()
    graph = cu.CUgraph(graph_handle)

    # Enumerate nodes using cuda.bindings Pythonic API
    # cuGraphGetNodes(graph, numNodes) returns (CUresult, nodes_list, numNodes)
    result = cu.cuGraphGetNodes(graph, 0)
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuGraphGetNodes failed: {result[0]}")

    # First call with numNodes=0 gives us the count
    # The returned list may be empty but the third element has the count
    num_nodes = result[2] if len(result) > 2 else 0
    if num_nodes == 0:
        # Try getting nodes with the count
        nodes_list = result[1] if len(result) > 1 else []
        num_nodes = len([n for n in nodes_list if n is not None]) if nodes_list else 0
        if num_nodes == 0:
            return GraphMetadata(num_nodes=0)
        node_array = [n for n in nodes_list if n is not None]
    else:
        # Second call with the actual count
        result = cu.cuGraphGetNodes(graph, num_nodes)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphGetNodes failed: {result[0]}")
        nodes_list = result[1]
        node_array = [n for n in nodes_list if n is not None]
        num_nodes = len(node_array)

    metadata = GraphMetadata(
        num_nodes=num_nodes,
        buffer_map=known_buffers,
        weight_map=known_weights,
        device_name=torch.cuda.get_device_name(),
        cuda_driver_version=torch.cuda.driver_version(),
        compute_capability=torch.cuda.get_device_capability(),
    )

    # Extract kernel names from DOT dump for more reliable identification
    kernel_names = get_kernel_names_from_graph(cuda_graph)

    for i in range(num_nodes):
        node = node_array[i]

        # Get node type
        type_result = cu.cuGraphNodeGetType(node)
        if type_result[0] != cu.CUresult.CUDA_SUCCESS:
            logger.debug(f"Node {i}: failed to get type: {type_result[0]}")
            continue
        node_type_val = int(type_result[1])

        if node_type_val == CUGraphNodeType.KERNEL:
            node_info = _inspect_kernel_node(
                i, node, known_buffers, known_weights, registry
            )
            # Override kernel name with DOT dump if available (more reliable)
            if i in kernel_names:
                node_info.kernel_name = kernel_names[i]
                # Register the kernel name in the registry for future lookups
                if node_info.func_ptr and not registry.resolve_function(
                    node_info.func_ptr
                ):
                    registry.register_function(
                        node_info.func_ptr, "graph_kernel", kernel_names[i]
                    )
            metadata.nodes.append(node_info)
        elif node_type_val == CUGraphNodeType.MEMCPY:
            node_info = _inspect_memcpy_node(i, node)
            metadata.nodes.append(node_info)
        elif node_type_val == CUGraphNodeType.MEMSET:
            node_info = _inspect_memset_node(i, node)
            metadata.nodes.append(node_info)
        elif node_type_val == CUGraphNodeType.MEMORY_ALLOC:
            logger.debug(f"Node {i}: MEMORY_ALLOC (skipped)")
        elif node_type_val == CUGraphNodeType.MEMORY_FREE:
            logger.debug(f"Node {i}: MEMORY_FREE (skipped)")
        else:
            logger.debug(f"Node {i}: type={node_type_val} (skipped)")

    # Build dependency information
    for i in range(num_nodes):
        node = node_array[i]
        deps = _get_node_dependencies(node, node_array, num_nodes)
        if i < len(metadata.nodes) and hasattr(metadata.nodes[i], "dependency_indices"):
            metadata.nodes[i].dependency_indices = deps

    # Detect pool base address from the first intermediate tensor found.
    # All intermediate tensors from the same CUDA graph memory pool share a
    # common base offset that shifts uniformly across process restarts.
    # Recording the pool base enables address translation on restart.
    for node in metadata.nodes:
        if isinstance(node, KernelNodeInfo):
            for p in node.params:
                if p.is_device_pointer and p.category == PointerCategory.INTERMEDIATE:
                    if p.base_addr != 0:
                        metadata.pool_base_addr = p.base_addr
                        break
            if metadata.pool_base_addr != 0:
                break

    if metadata.pool_base_addr != 0:
        logger.debug(
            f"Detected pool base address: 0x{metadata.pool_base_addr:x} "
            f"(from first intermediate tensor allocation)"
        )

    return metadata


def _inspect_kernel_node(
    index: int,
    node,
    known_buffers: Dict[int, Tuple[str, int]],
    known_weights: Dict[int, str],
    registry: CUFunctionRegistry,
) -> KernelNodeInfo:
    """Inspect a kernel node and extract its parameters."""
    # Get kernel node parameters using cuda.bindings Pythonic API
    params_result = cu.cuGraphKernelNodeGetParams(node)
    if params_result[0] != cu.CUresult.CUDA_SUCCESS:
        logger.warning(f"Failed to get kernel node params for node {index}")
        return KernelNodeInfo(node_index=index, func_ptr=0)

    params = params_result[1]
    func_ptr = int(params.func)
    grid_dim = (params.gridDimX, params.gridDimY, params.gridDimZ)
    block_dim = (params.blockDimX, params.blockDimY, params.blockDimZ)
    shared_mem = params.sharedMemBytes

    # Resolve kernel name
    resolved = registry.resolve_function(func_ptr)
    kernel_name = resolved[1] if resolved else ""
    module_hash = resolved[0] if resolved else ""

    # Try cuFuncGetName for direct name resolution (CUDA 12.4+)
    if not kernel_name:
        direct_name = get_kernel_name(func_ptr)
        if direct_name:
            kernel_name = direct_name

    # Try cuFuncGetModule for module handle (CUDA 12.4+)
    if not module_hash:
        module_handle = get_kernel_module(func_ptr)
        if module_handle:
            module_hash = f"module_{module_handle:x}"

    # Register in the registry if we have both name and module
    if kernel_name and func_ptr not in registry._func_to_name:
        registry.register_function(func_ptr, module_hash or "unknown", kernel_name)

    info = KernelNodeInfo(
        node_index=index,
        func_ptr=func_ptr,
        kernel_name=kernel_name,
        module_hash=module_hash,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=shared_mem,
    )

    # Parse kernel parameters
    # kernelParams is a void** where each element points to an argument value.
    # We need to determine the number of arguments and which are pointers.
    # Strategy: read 8-byte values and check if they look like device pointers.
    kernel_params = params.kernelParams
    if kernel_params is not None:
        info.params = _parse_kernel_params(
            kernel_params, func_ptr, known_buffers, known_weights
        )

    return info


def _parse_kernel_params(
    kernel_params_ptr: int,
    func_ptr: int,
    known_buffers: Dict[int, Tuple[str, int]],
    known_weights: Dict[int, str],
    max_params: int = 64,
) -> List[KernelParamInfo]:
    """Parse kernel parameters from a void** pointer.

    Uses /proc/self/mem for safe reading (avoids segfaults from reading
    past the end of the kernelParams array). Uses cuPointerGetAttribute
    to detect device pointers and cuMemGetAddressRange for base allocations.

    Args:
        kernel_params_ptr: The void** kernelParams pointer from CUkernelNodeParams.
        func_ptr: The CUfunction handle (for logging).
        known_buffers: Mapping of data_ptr -> (name, size) for known buffers.
        known_weights: Mapping of data_ptr -> parameter FQN for model weights.
        max_params: Maximum number of parameters to scan.
    """
    import ctypes
    import struct

    params = []
    ptr_size = ctypes.sizeof(ctypes.c_void_p)

    # Use /proc/self/mem for safe reading to avoid segfaults when
    # reading past the end of the kernelParams array.
    try:
        mem_fd = open("/proc/self/mem", "rb")
    except OSError:
        # Fallback to ctypes (less safe but works in most cases)
        mem_fd = None

    try:
        for i in range(max_params):
            try:
                if mem_fd is not None:
                    # Safe read via /proc/self/mem
                    mem_fd.seek(kernel_params_ptr + i * ptr_size)
                    ptr_bytes = mem_fd.read(ptr_size)
                    if len(ptr_bytes) < ptr_size:
                        break
                    param_ptr = struct.unpack("<Q", ptr_bytes)[0]
                    if param_ptr == 0:
                        break

                    mem_fd.seek(param_ptr)
                    val_bytes = mem_fd.read(8)
                    if len(val_bytes) < 8:
                        break
                    value = struct.unpack("<Q", val_bytes)[0]
                else:
                    # Fallback: use ctypes (may segfault if we read past the end)
                    param_ptr_ptr = kernel_params_ptr + i * ptr_size
                    param_ptr = ctypes.c_void_p.from_address(param_ptr_ptr).value
                    if param_ptr is None or param_ptr == 0:
                        break
                    value = ctypes.c_uint64.from_address(param_ptr).value

            except (OSError, OverflowError, ValueError):
                # Past the end of the parameter array
                break

            # Check if this value is a device pointer
            is_device_ptr = _is_device_pointer(value)
            category = PointerCategory.UNKNOWN
            symbolic_name = ""
            offset = 0
            base_addr = 0
            alloc_size = 0

            if is_device_ptr:
                # Try to find the base allocation
                base_addr, alloc_size = _get_allocation_range(value)

                # Check against known buffers
                if value in known_buffers:
                    name, size = known_buffers[value]
                    category = PointerCategory.INPUT_BUFFER
                    symbolic_name = name
                    offset = 0
                elif base_addr in known_buffers:
                    name, size = known_buffers[base_addr]
                    category = PointerCategory.INPUT_BUFFER
                    symbolic_name = name
                    offset = value - base_addr
                elif value in known_weights:
                    category = PointerCategory.MODEL_WEIGHT
                    symbolic_name = known_weights[value]
                    offset = 0
                elif base_addr in known_weights:
                    category = PointerCategory.MODEL_WEIGHT
                    symbolic_name = known_weights[base_addr]
                    offset = value - base_addr
                else:
                    category = PointerCategory.INTERMEDIATE

            params.append(
                KernelParamInfo(
                    index=i,
                    is_device_pointer=is_device_ptr,
                    raw_value=value,
                    category=category,
                    symbolic_name=symbolic_name,
                    offset=offset,
                    base_addr=base_addr,
                    alloc_size=alloc_size,
                )
            )

    finally:
        if mem_fd is not None:
            mem_fd.close()

    return params


def _is_device_pointer(value: int) -> bool:
    """Check if a value is a valid CUDA device pointer using cuPointerGetAttribute."""
    if value == 0:
        return False

    try:
        # CU_POINTER_ATTRIBUTE_MEMORY_TYPE returns CU_MEMORYTYPE_DEVICE (1)
        # for device pointers and CU_MEMORYTYPE_HOST (2) for host pointers.
        # For non-CUDA pointers, it returns CUDA_ERROR_INVALID_VALUE.
        attr = cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
        result = cu.cuPointerGetAttribute(attr, value)
        if result[0] == cu.CUresult.CUDA_SUCCESS:
            memory_type = result[1]
            # CUmemorytype enum: 0=HOST, 1=DEVICE, 2=ARRAY, 3=UNIFIED
            return memory_type in (1, 2, 3)
        else:
            # Not a CUDA-managed pointer
            return False
    except Exception:
        return False


def _get_allocation_range(ptr: int) -> Tuple[int, int]:
    """Get the base address and size of the allocation containing a pointer.

    Uses cuMemGetAddressRange to find the base address and size.
    Returns (base_addr, size). If the query fails, returns (0, 0).
    """
    try:
        result = cu.cuMemGetAddressRange(ptr)
        if result[0] == cu.CUresult.CUDA_SUCCESS:
            base = int(result[1])
            size = int(result[2])
            return (base, size)
    except Exception:
        pass
    return (0, 0)


def _inspect_memcpy_node(index: int, node) -> MemcpyNodeInfo:
    """Inspect a memcpy node."""
    try:
        result = cu.cuGraphMemcpyNodeGetParams(node)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            return MemcpyNodeInfo(node_index=index, src_ptr=0, dst_ptr=0, copy_size=0)
        params = result[1]
        # CUmemcpy3DParms has srcPos, dstPos, extent, srcPtr, dstPtr
        # Extract source and destination pointers and copy size
        src_ptr = int(params.srcPtr) if hasattr(params, "srcPtr") else 0
        dst_ptr = int(params.dstPtr) if hasattr(params, "dstPtr") else 0
        copy_size = int(params.extent.width) if hasattr(params, "extent") else 0
        return MemcpyNodeInfo(
            node_index=index,
            src_ptr=src_ptr,
            dst_ptr=dst_ptr,
            copy_size=copy_size,
        )
    except Exception as e:
        logger.debug(f"Failed to inspect memcpy node {index}: {e}")
        return MemcpyNodeInfo(node_index=index, src_ptr=0, dst_ptr=0, copy_size=0)


def _inspect_memset_node(index: int, node) -> MemsetNodeInfo:
    """Inspect a memset node."""
    try:
        result = cu.cuGraphMemsetNodeGetParams(node)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            return MemsetNodeInfo(node_index=index, dst_ptr=0, value=0, size=0)
        params = result[1]
        dst_ptr = int(params.dst) if hasattr(params, "dst") else 0
        value = int(params.value) if hasattr(params, "value") else 0
        size = int(params.pitch) if hasattr(params, "pitch") else 0
        return MemsetNodeInfo(
            node_index=index,
            dst_ptr=dst_ptr,
            value=value,
            size=size,
        )
    except Exception as e:
        logger.debug(f"Failed to inspect memset node {index}: {e}")
        return MemsetNodeInfo(node_index=index, dst_ptr=0, value=0, size=0)


def _get_node_dependencies(
    node,
    all_nodes,
    num_nodes: int,
) -> List[int]:
    """Get the indices of nodes that this node depends on."""
    try:
        # cuGraphNodeGetDependencies returns (CUresult, deps_list, numDeps)
        result = cu.cuGraphNodeGetDependencies(node, 0)
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            return []

        deps_list = result[1] if len(result) > 1 else []
        if not deps_list:
            return []

        # Map node handles to indices
        node_to_idx = {}
        for i, n in enumerate(all_nodes):
            if n is not None:
                node_to_idx[int(n)] = i

        deps = []
        for dep in deps_list:
            if dep is not None and int(dep) in node_to_idx:
                deps.append(node_to_idx[int(dep)])

        return deps
    except Exception as e:
        logger.debug(f"Failed to get dependencies: {e}")
        return []


# ---------------------------------------------------------------------------
# Convenience: checkCudaErrors for driver API
# ---------------------------------------------------------------------------
def checkCudaErrorsDriver(result):
    """Check CUDA driver API errors (cu* functions return CUresult)."""
    _check_cuda_bindings()
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        err_code = int(result[0])
        try:
            _, err_str = cu.cuGetErrorString(err_code)
            if isinstance(err_str, bytes):
                err_str = err_str.decode("utf-8", "replace")
        except Exception:
            err_str = f"error code {err_code}"
        raise RuntimeError(f"CUDA driver error: {err_str}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# Alias for use in this module
checkCudaErrors = checkCudaErrorsDriver


# ---------------------------------------------------------------------------
# Buffer address registry for SGLang integration
# ---------------------------------------------------------------------------
class BufferAddressRegistry:
    """Tracks the addresses of known tensors (input buffers, model weights,
    attention metadata) so that kernel parameters can be mapped to symbolic
    names during inspection and address-patched during reconstruction.
    """

    def __init__(self):
        # data_ptr -> (name, size)
        self._buffers: Dict[int, Tuple[str, int]] = {}
        # data_ptr -> parameter FQN
        self._weights: Dict[int, str] = {}

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register an input/output buffer tensor."""
        ptr = tensor.data_ptr()
        self._buffers[ptr] = (name, tensor.numel() * tensor.element_size())

    def register_weight(self, fqn: str, tensor: torch.Tensor):
        """Register a model weight parameter."""
        ptr = tensor.data_ptr()
        self._weights[ptr] = fqn

    def register_model_weights(self, model: torch.nn.Module):
        """Register all parameters of a model."""
        for name, param in model.named_parameters():
            if param.data_ptr() != 0:
                self._weights[param.data_ptr()] = name
        for name, buf in model.named_buffers():
            if buf.data_ptr() != 0:
                self._weights[buf.data_ptr()] = f"buffer:{name}"

    def get_known_buffers(self) -> Dict[int, Tuple[str, int]]:
        return dict(self._buffers)

    def get_known_weights(self) -> Dict[int, str]:
        return dict(self._weights)

    def build_address_translation(
        self, old_registry: "BufferAddressRegistry"
    ) -> Dict[int, int]:
        """Build an address translation table from old addresses to new addresses.

        Compares the symbolic names in old_registry with this registry and
        returns a mapping of old_data_ptr -> new_data_ptr for matching entries.
        """
        translation = {}

        # Translate input buffers
        old_by_name = {name: ptr for ptr, (name, _) in old_registry._buffers.items()}
        new_by_name = {name: ptr for ptr, (name, _) in self._buffers.items()}
        for name, old_ptr in old_by_name.items():
            if name in new_by_name:
                translation[old_ptr] = new_by_name[name]

        # Translate model weights
        old_weights_by_name = {name: ptr for ptr, name in old_registry._weights.items()}
        new_weights_by_name = {name: ptr for ptr, name in self._weights.items()}
        for name, old_ptr in old_weights_by_name.items():
            if name in new_weights_by_name:
                translation[old_ptr] = new_weights_by_name[name]

        return translation
