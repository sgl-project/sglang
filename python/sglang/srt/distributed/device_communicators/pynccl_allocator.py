import logging
import os
import tempfile
import traceback
from contextlib import nullcontext

import torch
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import torch_release

after_2_8_0 = torch_release >= (2, 8)

nccl_allocator_source = """

#include <cuda_runtime.h>

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
typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow_vidmem* ncclWindow_t;
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
#define NCCL_WIN_COLL_SYMMETRIC 0x01

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

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  NCCLCHECK(ncclMemAlloc(&ptr, size));

  const char *str_val = getenv("SGLANG_TMP_NCCL_COMM_VALUE");
  char *endptr;
  void* int_val = (void *)strtoull(str_val, &endptr, 0);

  ncclComm_t comm = (ncclComm_t)(int_val);
  ncclWindow_t win;
  NCCLCHECK(ncclCommWindowRegister(comm, ptr, size, &win, NCCL_WIN_COLL_SYMMETRIC));

  return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""

_allocator = None
_mem_pool = None
_graph_pool_id = None
_cur_device = None
_active_symmetric_memory_context = None


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


def get_nccl_mem_pool():
    global _allocator, _mem_pool, _cur_device
    if _mem_pool is None:
        import torch.utils.cpp_extension

        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
        _cur_device = torch.cuda.current_device()
    return _mem_pool


class SymmetricMemoryContext:
    """
    Context manager for using symmetric memory with pynccl.

    To Utilize the symmetric memory feature in NCCL, the buffers need to be allocated
    by `ncclMemAlloc` and registered by `ncclCommWindowRegister`. Due to this, we introduce
    this context manager. All tensors created under this context will be correctly
    allocated and registered with a custom allocator.
    """

    def __init__(
        self,
        group_coordinator: GroupCoordinator,
    ):
        self.group_coordinator = group_coordinator
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.exited = False

    def __enter__(self):
        assert (
            self.group_coordinator.pynccl_comm is not None
        ), f"Symmetric memory requires pynccl to be enabled in group '{self.group_coordinator.group_name}'"

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

        # Set the env var to pass this argument to the C functions.
        os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = str(
            self.group_coordinator.pynccl_comm.comm.value
        )

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)

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


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()


# --- Debug mode for symmetric memory validation ---

_symm_mem_logger = logging.getLogger(__name__)
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
