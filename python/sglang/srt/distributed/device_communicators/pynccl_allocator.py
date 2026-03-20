import ctypes
import os
import tempfile
from contextlib import nullcontext
from typing import Dict, Set

import torch
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import torch_release

after_2_8_0 = torch_release >= (2, 8)

# C++ source for the NCCL allocator plugin
# Key design:
# 1. nccl_alloc_plug: Allocates memory via ncclMemAlloc and TRACKS the segment
#    (ptr, size). Does NOT register with any comm at allocation time.
# 2. nccl_free_plug: Frees memory via ncclMemFree and UNTRACKS the segment.
#    Each segment is tracked only during its lifetime (from alloc to free).
# 3. Segment tracking uses a thread-safe map keyed by ptr.
# 4. Python layer handles registration at context exit time using pynccl API.
nccl_allocator_source = """

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>

extern "C" {

// copy from https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;

ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t  ncclMemFree(void *ptr);
const char*  ncclGetErrorString(ncclResult_t result);

#define NCCLCHECK(cmd) do {                                               \
  ncclResult_t res = cmd;                                                 \
  if (res != ncclSuccess) {                                               \
    fprintf(stderr, "ERROR: NCCL symmetric memory allocation failed. Most likely out of device memory. '%s'\\n", \
           ncclGetErrorString(res));                       \
    return NULL;                                                        \
  }                                                                       \
} while(0)

// Thread-safe segment tracking using unordered_map (keyed by ptr)
// Segments are tracked during their lifetime (from alloc to free).
static std::unordered_map<void*, size_t> g_segments;
static std::mutex g_segment_mutex;

// Add or update a segment in the tracking map
static void track_segment(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    g_segments[ptr] = size;
}

// Remove a segment from the tracking map
static void untrack_segment(void* ptr) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    g_segments.erase(ptr);
}

void* nccl_alloc_plug(size_t size, int device, void* stream) {
    void* ptr;
    NCCLCHECK(ncclMemAlloc(&ptr, size));

    // Track the segment but do NOT register with any comm
    // Registration will be done at context exit in Python
    track_segment(ptr, size);

    return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
    // Untrack the segment before freeing
    untrack_segment(ptr);
    ncclResult_t err = ncclMemFree(ptr);
}

// C API for Python to query tracked segments
// out_ptrs: output array for pointers (must have max_segments elements)
// out_sizes: output array for sizes (must have max_segments elements)
// max_segments: maximum number of segments to return
// Returns: actual number of segments returned
extern "C" int nccl_allocator_get_segments(void** out_ptrs, size_t* out_sizes, int max_segments) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    int count = 0;
    for (const auto& seg : g_segments) {
        if (count >= max_segments) break;
        out_ptrs[count] = seg.first;
        out_sizes[count] = seg.second;
        count++;
    }
    return count;
}

extern "C" int nccl_allocator_get_segment_count() {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    return (int)g_segments.size();
}

}
"""

_allocator = None
_shared_mem_pool = None
_graph_pool_id = None
_cur_device = None
_active_symmetric_memory_context = None

# Reference to the loaded library
_nccl_allocator_lib = None

# Global registry for tracking registrations
# Key: ptr (int), Value: set of comm_ptr (int) that have registered this ptr
# This allows the same memory to be registered with multiple communicators
_ptr_to_registered_comms: Dict[int, Set[int]] = {}


def is_symmetric_memory_enabled():
    try:
        return get_global_server_args().enable_symm_mem
    except ValueError:
        return False


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def disable_symmetric_memory_context():
    if _active_symmetric_memory_context is None:
        return None
    saved_context = _active_symmetric_memory_context
    saved_context.__exit__(None, None, None)
    return saved_context


def restore_symmetric_memory_context(saved_context):
    if saved_context is not None:
        saved_context.__enter__()


def get_nccl_mem_pool() -> torch.cuda.MemPool:
    """
    Get the shared MemPool for all groups.

    All groups share the same pool to avoid memory fragmentation.
    Comm registration is handled at context exit time.
    """
    global _allocator, _shared_mem_pool, _cur_device, _nccl_allocator_lib
    if _allocator is None:
        import torch.utils.cpp_extension

        out_dir = os.path.join(tempfile.gettempdir(), "symm_allocator")
        os.makedirs(out_dir, exist_ok=True)
        # Make sure to clean up leftover pytorch lock files
        # from previous runs and synchronize across processes
        # right after
        try:
            os.remove(os.path.join(out_dir, "lock"))
        except FileNotFoundError:
            pass
        torch.distributed.barrier()

        nccl_allocator_libname = "nccl_allocator"
        lib_path = torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )
        _nccl_allocator_lib = ctypes.CDLL(lib_path)
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _shared_mem_pool = torch.cuda.MemPool(_allocator)
        _cur_device = torch.cuda.current_device()

    return _shared_mem_pool


def _get_tracked_segments() -> list:
    """
    Get all tracked segments from C++ as a list of (ptr, size) tuples.

    Returns:
        List of (ptr_int, size) tuples representing all tracked segments.
    """
    # Get segment count first
    count_func = _nccl_allocator_lib.nccl_allocator_get_segment_count
    count_func.restype = ctypes.c_int
    count_func.argtypes = []

    max_segments = count_func()
    if max_segments == 0:
        return []

    # Allocate buffers for results
    ptrs = (ctypes.c_uint64 * max_segments)()
    sizes = (ctypes.c_size_t * max_segments)()

    # Get segments
    get_segments_func = _nccl_allocator_lib.nccl_allocator_get_segments
    get_segments_func.restype = ctypes.c_int
    get_segments_func.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]

    actual_count = get_segments_func(ptrs, sizes, max_segments)

    # Convert to Python list of tuples
    segments = []
    for i in range(actual_count):
        ptr_int = ptrs[i]
        size = sizes[i]
        if ptr_int != 0:
            segments.append((ptr_int, size))

    return segments


class SymmetricMemoryContext:
    """
    Context manager for using symmetric memory with pynccl.

    To Utilize the symmetric memory feature in NCCL, the buffers need to be allocated
    by `ncclMemAlloc` and registered by `ncclCommWindowRegister`. Due to this, we introduce
    this context manager. All tensors created under this context will be correctly
    allocated and registered with a custom allocator.

    Key design:
    - All groups share a single MemPool to avoid memory fragmentation.
    - At allocation time, ptrs are tracked but NOT registered with any comm.
    - At context exit time, we iterate over segments and register unregistered ptrs
      with the current comm using pynccl.register_comm_window_raw.
      This handles both:
      1. Newly allocated memory (never registered)
      2. Memory reused from pool (may need registration for different comm)
    """

    def __init__(
        self,
        group_coordinator: GroupCoordinator,
    ):
        self.group_coordinator = group_coordinator
        self.group_name = group_coordinator.unique_name
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.exited = False

        # Get comm ptr for tracking registrations
        # Use the comm pointer value as unique identifier
        self._comm_ptr = self.group_coordinator.pynccl_comm.comm.value

    def __enter__(self):
        assert (
            self.group_coordinator.pynccl_comm is not None
        ), f"Symmetric memory requires pynccl to be enabled in group '{self.group_name}'"

        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            # Pause graph memory pool to use symmetric memory with cuda graph
            if after_2_8_0:
                torch._C._cuda_endAllocateToPool(_cur_device, _graph_pool_id)
            else:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    _cur_device, _graph_pool_id
                )

        if self.exited:
            # mempool ctx (@contextlib.contextmanager) is not re-entrant
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.exited = False
        self._mem_pool_ctx.__enter__()

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        # Register all unregistered segments
        # with the current comm
        self._register_segments_for_comm()

        if self.is_graph_capture:
            if after_2_8_0:
                torch._C._cuda_beginAllocateCurrentThreadToPool(
                    _cur_device, _graph_pool_id
                )
            else:
                torch._C._cuda_beginAllocateToPool(_cur_device, _graph_pool_id)

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = None

        self.exited = True

    def _register_segments_for_comm(self):
        """
        Register all tracked segments with the current comm if not already registered.

        This handles two scenarios:
        1. Newly allocated memory: needs registration
        2. Memory reused from pool: may need registration for different comm
        """
        global _ptr_to_registered_comms

        comm_ptr = self._comm_ptr
        pynccl_comm = self.group_coordinator.pynccl_comm

        # Get all tracked segments from C++
        segments = _get_tracked_segments()
        if not segments:
            return

        # Register segments that are not yet registered with this comm
        for ptr_int, size in segments:
            # Check if this ptr is already registered with this comm
            if ptr_int not in _ptr_to_registered_comms:
                _ptr_to_registered_comms[ptr_int] = set()

            if comm_ptr in _ptr_to_registered_comms[ptr_int]:
                # Already registered with this comm, skip
                continue

            # Register this ptr with the current comm using pynccl API
            pynccl_comm.register_comm_window_raw(ptr_int, size)

            # Track the registration
            _ptr_to_registered_comms[ptr_int].add(comm_ptr)


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()
