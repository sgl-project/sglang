import ctypes
import os
import tempfile
from contextlib import nullcontext

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
# 3. Segment tracking uses thread-safe std::vector + unordered_map for O(1) operations.
# 4. Registration via nccl_allocator_register_segments_with_comm: Registers all
#    tracked segments with a given comm, using index-based tracking to avoid
#    re-registration. Registration state is maintained per-communicator in C++.
nccl_allocator_source = """

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <utility>

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

// NCCL symmetric memory window flags
#define NCCL_WIN_COLL_SYMMETRIC 0x01

typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow_vidmem* ncclWindow_t;

ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t  ncclMemFree(void *ptr);
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
const char*  ncclGetErrorString(ncclResult_t result);

#define NCCLCHECK(cmd) do {                                               \
  ncclResult_t res = cmd;                                                 \
  if (res != ncclSuccess) {                                               \
    fprintf(stderr, "ERROR: NCCL symmetric memory allocation failed. Most likely out of device memory. '%s'\\n", \
           ncclGetErrorString(res));                       \
    return NULL;                                                        \
  }                                                                       \
} while(0)

// Segment information structure
struct Segment {
    void* ptr;
    size_t size;
    Segment(void* p, size_t s) : ptr(p), size(s) {}
};

// Thread-safe segment tracking
// Segment tracking using std::vector for FIFO order.
// g_segments is maintained in insertion order (oldest first).
// g_ptr_to_index provides O(1) lookup for untrack operations.
static std::vector<Segment> g_segments;
static std::unordered_map<void*, size_t> g_ptr_to_index;
static std::mutex g_segment_mutex;

// Track which segments have been registered with each communicator.
// Key: comm_ptr, Value: the next segment index to register for this comm.
static std::unordered_map<uintptr_t, size_t> g_comm_registration_index;

// Add a segment to the tracking (appends to end, maintaining FIFO order)
static void track_segment(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    size_t index = g_segments.size();
    g_segments.emplace_back(ptr, size);
    g_ptr_to_index[ptr] = index;
}

// Remove a segment from the tracking (O(1) using swap-and-pop)
static void untrack_segment(void* ptr) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    auto it = g_ptr_to_index.find(ptr);
    if (it == g_ptr_to_index.end()) {
        return;
    }
    size_t index = it->second;
    size_t last_index = g_segments.size() - 1;

    // If not the last element, swap with the last element
    if (index != last_index) {
        // Swap the segment at index with the last segment
        g_segments[index] = g_segments[last_index];
        // Update the index map for the swapped segment's pointer
        g_ptr_to_index[g_segments[index].ptr] = index;
    }

    // Remove the last element
    g_segments.pop_back();
    g_ptr_to_index.erase(it);
}

void* nccl_alloc_plug(size_t size, int device, void* stream) {
    void* ptr;
    NCCLCHECK(ncclMemAlloc(&ptr, size));

    // Track the segment but do NOT register with any comm
    // Registration will be done at context exit via register_segments_with_comm
    track_segment(ptr, size);

    return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
    // Untrack the segment before freeing
    untrack_segment(ptr);
    ncclResult_t err = ncclMemFree(ptr);
}

// Register all tracked segments with a communicator.
// Uses an index-based approach to avoid re-registering already-registered segments.
// Returns 0 on success, non-zero on failure.
int nccl_allocator_register_segments_with_comm(uintptr_t comm_ptr) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);

    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_ptr);

    // Get the starting index for this communicator
    size_t start_index = 0;
    auto it = g_comm_registration_index.find(comm_ptr);
    if (it != g_comm_registration_index.end()) {
        start_index = it->second;
    }

    // Register all segments from start_index to the current end
    for (size_t i = start_index; i < g_segments.size(); ++i) {
        const Segment& seg = g_segments[i];
        ncclWindow_t win;
        ncclResult_t res = ncclCommWindowRegister(comm, seg.ptr, seg.size, &win, NCCL_WIN_COLL_SYMMETRIC);
        if (res != ncclSuccess) {
            return res;
        }
    }

    // Update the registration index for this communicator
    g_comm_registration_index[comm_ptr] = g_segments.size();

    return ncclSuccess;
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
    - At context exit time, nccl_allocator_register_segments_with_comm is called
      to register all tracked segments with the current comm. The C++ layer
      tracks per-comm registration state using index-based tracking to avoid
      re-registration of already-registered segments.
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
        Register all tracked segments with the current comm.

        Delegates to C++ layer which handles:
        1. Tracking which segments have been registered with each comm
        2. Only registering new segments (avoiding re-registration)
        3. Thread-safe access to the segment registry
        """
        global _nccl_allocator_lib

        # Setup the C function signature
        register_func = _nccl_allocator_lib.nccl_allocator_register_segments_with_comm
        register_func.restype = ctypes.c_int
        register_func.argtypes = [ctypes.c_uint64]

        # Call C++ API to register all segments with this comm
        # C++ layer tracks per-comm registration state internally
        result = register_func(self._comm_ptr)
        if result != 0:
            raise RuntimeError(f"ncclCommWindowRegister failed with error code {result}")


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()
