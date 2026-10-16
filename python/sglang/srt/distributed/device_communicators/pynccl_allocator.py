import ctypes
import faulthandler
import logging
import os
import signal
import tempfile
import threading
import traceback
from contextlib import nullcontext
from typing import Optional

import psutil
import torch

# The private _cuda_* pool APIs are absent before torch 2.8; the call sites below
# reach them via torch._C.<name> so torch 2.7 (Ascend NPU) can still import this.
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
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
static std::vector<Segment> g_segments;
static std::mutex g_segment_mutex;

// Track which segments have been registered with each communicator.
// Key: comm_ptr, Value: the next segment index to register for this comm.
static std::unordered_map<uintptr_t, size_t> g_comm_registration_index;

// Add a segment to the tracking (appends to end, maintaining FIFO order)
static void track_segment(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    g_segments.emplace_back(ptr, size);
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
    ncclResult_t err = ncclMemFree(ptr);
    // NOTE: We assume that no individual allocation will be freed until the
    // entire memory pool is destroyed. If this assumption does not hold,
    // we will encounter asymmetry issues between GPUs. For now, we clear
    // all tracking state when the pool is destroyed.
    std::lock_guard<std::mutex> lock(g_segment_mutex);
    g_segments = std::vector<Segment>();
    g_comm_registration_index = std::unordered_map<uintptr_t, size_t>();
}

// Register all tracked segments with a communicator.
// Uses an index-based approach to avoid re-registering already-registered segments.
// Returns 0 on success, non-zero on failure.
int nccl_allocator_register_segments_with_comm(uintptr_t comm_ptr) {
    std::lock_guard<std::mutex> lock(g_segment_mutex);

    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_ptr);

    // Get the starting index for this communicator
    size_t start_index = g_comm_registration_index[comm_ptr];

    // Register all segments from start_index to the current end
    for (size_t i = start_index; i < g_segments.size(); ++i) {
        const Segment& seg = g_segments[i];
        ncclWindow_t win;
        ncclResult_t res = ncclCommWindowRegister(comm, seg.ptr, seg.size, &win, NCCL_WIN_COLL_SYMMETRIC);
        if (res != ncclSuccess) {
            fprintf(stderr, "ERROR: NCCL symmetric memory registration failed. '%s'\\n", ncclGetErrorString(res));
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
_mem_pool = None
_graph_pool_id = None
_cur_device = None
_active_symmetric_memory_context = None

# Reference to the C registration function (with arg types set)
_register_func = None


def is_symmetric_memory_enabled():
    try:
        return get_exec().comm.enable_symm_mem
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
    assert after_2_8_0, (
        "--enable-symm-mem requires torch>=2.8 "
        "(torch._C._cuda_beginAllocateCurrentThreadToPool was added there)."
    )

    global _allocator, _mem_pool, _cur_device, _register_func
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
        nccl_allocator_lib = ctypes.CDLL(lib_path)
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
        _cur_device = torch.cuda.current_device()

        # Setup the C function for registration with correct arg types
        _register_func = nccl_allocator_lib.nccl_allocator_register_segments_with_comm
        _register_func.restype = ctypes.c_int
        _register_func.argtypes = [ctypes.c_uint64]

    return _mem_pool


# NCCL's window registration blocks in a Unix-domain-socket fd-exchange
# rendezvous that has no timeout of its own. When that rendezvous loses a
# message, every rank in the group sits in recvmsg with no error and no log
# line: the engine stays up, `/health` keeps returning 200, and no batch can
# make progress. Bound the wait (SGLANG_SYMM_MEM_REGISTRATION_TIMEOUT_S) so a
# wedge is reported instead of hidden, and keep the in-progress state readable
# so the scheduler watchdog can name it rather than reporting a bare stall.

_registration_in_progress: Optional[str] = None


def describe_registration_in_progress() -> Optional[str]:
    """What the symmetric-memory registration is doing, or ``None`` if idle.

    The scheduler watchdog reads this for its dump, so a stall inside the
    registration is named instead of showing up as a generic watchdog timeout.
    """
    return _registration_in_progress


def _run_with_timeout(fn, timeout_s: float):
    """Run ``fn()`` on a worker thread and return its result.

    Raises ``TimeoutError`` if it has not returned within ``timeout_s``; any
    other exception raised by ``fn`` is re-raised. The worker is a daemon, so a
    call that never returns cannot keep the process alive by itself.
    """

    outcome: dict = {}

    def _target():
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
            outcome["exc"] = exc

    worker = threading.Thread(target=_target, name="symm-mem-register", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"did not return within {timeout_s:.0f}s")
    if "exc" in outcome:
        raise outcome["exc"]
    return outcome["result"]


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
        self._pool_id = get_nccl_mem_pool().id
        self._device_index = torch.cuda.current_device()
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()

        # Get comm ptr for tracking registrations
        # Use the comm pointer value as unique identifier
        self._comm_ptr = self.group_coordinator.pynccl_comm.comm.value

    def __enter__(self):
        assert self.group_coordinator.pynccl_comm is not None, (
            f"Symmetric memory requires pynccl to be enabled in group '{self.group_coordinator.unique_name}'"
        )

        if self.is_graph_capture:
            assert _graph_pool_id is not None, (
                "graph_pool_id is not set under graph capture"
            )
            # Pause graph memory pool to use symmetric memory with cuda graph
            if after_2_8_0:
                torch._C._cuda_endAllocateToPool(_cur_device, _graph_pool_id)
            else:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    _cur_device, _graph_pool_id
                )

        torch._C._cuda_beginAllocateCurrentThreadToPool(
            self._device_index, self._pool_id
        )

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._cuda_endAllocateToPool(self._device_index, self._pool_id)
        torch._C._cuda_releasePool(self._device_index, self._pool_id)
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

    def _register_segments_for_comm(self):
        """
        Register all tracked segments with the current comm.

        Delegates to C++ layer which handles:
        1. Tracking which segments have been registered with each comm
        2. Only registering new segments (avoiding re-registration)
        3. Thread-safe access to the segment registry

        The call is bounded: see the module comment above.
        """

        global _registration_in_progress

        group = self.group_coordinator.unique_name
        timeout_s = envs.SGLANG_SYMM_MEM_REGISTRATION_TIMEOUT_S.get()

        # Call C++ API to register all segments with this comm
        # C++ layer tracks per-comm registration state internally
        _registration_in_progress = f"comm={self._comm_ptr} group={group}"
        logger.debug("Registering symmetric-memory segments with %s", group)
        try:
            result = _run_with_timeout(
                lambda: _register_func(self._comm_ptr),
                timeout_s,
            )
        except TimeoutError:
            self._report_registration_timeout(group, timeout_s)
            raise  # unreachable: the report exits the process
        finally:
            _registration_in_progress = None
        assert result == 0, (
            f"nccl_allocator_register_segments_with_comm failed with return code: {result}"
        )

    def _report_registration_timeout(self, group: str, timeout_s: float) -> None:
        """Give up loudly: the group is wedged and no batch can make progress.

        NCCL's registration has no timeout of its own, so without this the
        engine would sit here indefinitely with `/health` still answering 200.
        Take the same exit path as any other scheduler failure: dump what a
        maintainer needs, then SIGQUIT the parent so the normal crash handling
        runs, and stop this process because the wedged thread still holds the
        C++ segment mutex -- it could never register again.
        """
        logger.error(
            "Symmetric-memory registration for comm %s (group %s) did not return "
            "within %.0f s. Every rank in the group is wedged in NCCL's "
            "fd-exchange rendezvous: the engine is up and /health answers 200, "
            "but no batch can make progress. Dumping stacks and aborting.",
            self._comm_ptr,
            group,
            timeout_s,
        )
        try:
            from sglang.srt.utils.cudacore_pyspy_dump_utils import (
                pyspy_dump_schedulers,
            )

            pyspy_dump_schedulers()
        except Exception:
            logger.exception("Failed to dump scheduler stacks")
        faulthandler.dump_traceback()
        psutil.Process().parent().send_signal(signal.SIGQUIT)
        os._exit(1)


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()


# --- Debug mode for symmetric memory validation ---

_symm_mem_logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_debug_seen_traces: set = set()


def is_tensor_in_symmetric_mempool(tensor: torch.Tensor) -> bool:
    """Check if a tensor's storage is allocated in the NCCL symmetric memory pool."""

    if _mem_pool is None:
        return False  # Pool not initialized

    data_ptr = tensor.untyped_storage().data_ptr()

    for segment in _mem_pool.snapshot():
        for block in segment["blocks"]:
            if block["address"] == data_ptr:
                return True
    return False


def debug_check_symmetric_mempool(
    group_coordinator: GroupCoordinator,
    tensors: dict,
    op_name: str,
) -> None:
    """
    Debug check: verify that tensors passed to communication ops are allocated
    in the NCCL symmetric memory pool.

    Enabled by setting SGLANG_DEBUG_SYMM_MEM=1.
    Only prints warnings on rank 0 and deduplicates identical stack traces.

    Args:
        tensors: dict mapping argument name to tensor
                 (e.g. {"input": t1, "output": t2})
        op_name: name of the communication operation being checked
    """
    if not envs.SGLANG_DEBUG_SYMM_MEM.get() or not is_symmetric_memory_enabled():
        return

    # Only print on rank 0
    if not group_coordinator.is_first_rank:
        return

    bad_names = []
    bad_details = []
    for name, tensor in tensors.items():
        if not is_tensor_in_symmetric_mempool(tensor):
            bad_names.append(name)
            bad_details.append(
                f"  - '{name}' (data_ptr=0x{tensor.storage().data_ptr():x}, "
                f"shape={list(tensor.shape)}, dtype={tensor.dtype})"
            )

    if bad_names:
        traces = traceback.format_stack()
        # Skip autotune stack traces
        if any("_flashinfer_autotune" in trace for trace in traces):
            return
        stack = "".join(traces[:-1])
        trace_key = f"{op_name}:{','.join(bad_names)}:{stack}"
        if trace_key not in _debug_seen_traces:
            _debug_seen_traces.add(trace_key)
            _symm_mem_logger.warning(
                "[SymmMem Debug] %s: %d tensor(s) are NOT in the "
                "NCCL symmetric memory pool:\n%s\n"
                "Stack trace:\n%s",
                op_name,
                len(bad_names),
                "\n".join(bad_details),
                stack,
            )


def prealloc_symmetric_memory_pool(
    *,
    is_draft_worker: bool,
    enable_symm_mem: bool,
    device: str,
    forward_stream: torch.cuda.Stream,
):
    # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
    if (
        is_draft_worker
        or not enable_symm_mem
        or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
    ):
        return

    from sglang.srt.distributed.parallel_state import get_tp_group

    # Memory allocation is tied to a cuda stream, use the forward stream
    with torch.get_device_module(device).stream(forward_stream):
        logger.info(
            f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
        )
        with use_symmetric_memory(get_tp_group()):
            torch.empty(
                (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                dtype=torch.uint8,
                device=device,
            )
