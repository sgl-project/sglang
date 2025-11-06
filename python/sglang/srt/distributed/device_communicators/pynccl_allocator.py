import os
import tempfile
from contextlib import nullcontext

import torch
import torch.utils.cpp_extension
from packaging import version
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.server_args import get_global_server_args

after_2_8_0 = version.parse(torch.__version__) >= version.parse("2.8.0")

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

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  ncclResult_t err = ncclMemAlloc(&ptr, size);

  const char *str_val = getenv("SGLANG_TMP_NCCL_COMM_VALUE");
  char *endptr;
  void* int_val = (void *)strtoull(str_val, &endptr, 0);

  ncclComm_t comm = (ncclComm_t)(int_val);
  ncclWindow_t win;
  ncclResult_t err2 = ncclCommWindowRegister(comm, ptr, size, &win, NCCL_WIN_COLL_SYMMETRIC);

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


def is_symmetric_memory_enabled():
    return get_global_server_args().enable_symm_mem


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _allocator, _mem_pool, _cur_device
    if _mem_pool is None:
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

        self._mem_pool_ctx.__enter__()

        # Set the env var to pass this argument to the C functions.
        os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = str(
            self.group_coordinator.pynccl_comm.comm.value
        )
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


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()
