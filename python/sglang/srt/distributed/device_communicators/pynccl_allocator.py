import tempfile
from sglang.srt.distributed.parallel_state import GroupCoordinator
import torch
from torch.cuda.memory import CUDAPluggableAllocator


nccl_allocator_source = """
#include <nccl.h>
#include <c10/cuda/CUDAGuard.h>
extern "C" {

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemAlloc(&ptr, size);
  return ptr;

}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  ncclResult_t err = ncclMemFree(ptr);
}

}
"""

_allocator = None
_mem_pool = None
_registered_base_addrs = set()
_graph_pool_id = None


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def get_nccl_mem_pool():
    global _allocator, _mem_pool
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
            f"{out_dir}/{nccl_allocator_libname}.so", "nccl_alloc_plug", "nccl_free_plug"
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
        # with torch.cuda.use_mem_pool(_mem_pool):
        #     # Prelallocate
        #     a = torch.empty(111010816, dtype=torch.bfloat16, device='cuda')
    return _mem_pool


class use_symmetric_memory:
    def __init__(self, group_coordinator: GroupCoordinator):
        self.group_coordinator = group_coordinator
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()

    def __enter__(self):
        if self.is_graph_capture:
            assert _graph_pool_id is not None, "graph_pool_id is not set under graph capture"
            torch._C._cuda_endAllocateCurrentStreamToPool(self.device, _graph_pool_id)
        self._mem_pool_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _registered_base_addrs
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)
        for segment in get_nccl_mem_pool().snapshot():
            if segment['address'] not in _registered_base_addrs:
                # Check symmetric is maintained across all ranks
                # TODO
                self.group_coordinator.pynccl_comm.register_comm_window_raw(segment['address'], segment['total_size'])
                _registered_base_addrs.add(segment['address'])

        if self.is_graph_capture:
            assert _graph_pool_id is not None, "graph_pool_id is not set under graph capture"
            torch._C._cuda_beginAllocateToPool(self.device, _graph_pool_id)


class SymmMemoryTensor:
    def __init__(self, group_coordinator: GroupCoordinator):
        self.tensor = None
        #self.window = None
        self.group_coordinator = group_coordinator

    def is_supported(self) -> bool:
        return (
            self.group_coordinator.pynccl_comm is not None
            and self.group_coordinator.pynccl_comm.nccl_version >= 22703
        )

    def get_tensor(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        assert self.is_supported(), "Symmetric memory is not supported"

        if (self.tensor is not None and
            self.tensor.dtype == dtype and
            self.tensor.numel() >= shape.numel()):
            view = self.tensor.view(-1)[:shape.numel()].view(shape)
            return view
        else:
            # if self.window is not None:
            #     self.group_coordinator.pynccl_comm.deregister_comm_window(self.window)
            #     self.window = None
            with torch.cuda.use_mem_pool(get_nccl_mem_pool()):
                self.tensor = torch.empty(shape, dtype=dtype, device='cuda')
            #self.window =
            self.group_coordinator.pynccl_comm.register_comm_window(self.tensor)
            return self.tensor
