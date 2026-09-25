"""Decode-sized all-reduce for two GPUs without peer access (SGLANG_LL_ALLREDUCE=1).

When the two GPUs have no P2P path (`nvidia-smi topo` shows SYS, e.g. two
sockets or consumer cards), the custom all-reduce cannot run and NCCL falls
back to its SHM transport, which costs ~22-25 us for a 4096-element bf16
decode all-reduce. This is one kernel per call, capturable in a CUDA graph:
each rank writes its vector into pinned host memory that both GPUs map, in
NCCL's LL form (every 8-byte store carries two bf16 and the call's sequence
number, so data and flag arrive together), and spins on the peer's words
until they carry the same number. The sum is a + b rounded once, which is
what NCCL returns for two ranks.

Only bf16 tensors of at most SGLANG_LL_ALLREDUCE_MAX_BYTES (default 64 KiB)
take this path; everything else keeps the existing one.
"""
import ctypes
import logging
import mmap
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HEADER = 4096

_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

// asm volatile: a plain load in the spin below was hoisted out of the loop
// as pure, and the loop then assumed to exit at once.
__device__ __forceinline__ unsigned long long ld_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Every 8-byte word is (seq << 32 | two bf16): the reader needs no separate
// flag, which over PCIe arrived ahead of the data it guarded. Slots are
// [rank][seq parity]; the peer can be at most one call ahead, and its next
// call writes the other parity.
__global__ void ll_ar_kernel(const unsigned* in, unsigned* out, int nw, char* base, int rank,
                             long long slot_bytes, unsigned* counter) {
  __shared__ unsigned seq;
  if (threadIdx.x == 0) seq = *counter + 1;
  __syncthreads();
  const unsigned s = seq;
  unsigned long long* mine = (unsigned long long*)(base + 4096 + (rank * 2 + (s & 1)) * slot_bytes);
  const unsigned long long* theirs =
      (const unsigned long long*)(base + 4096 + ((1 - rank) * 2 + (s & 1)) * slot_bytes);
  for (int i = threadIdx.x; i < nw; i += blockDim.x) st_sys(&mine[i], ((unsigned long long)s << 32) | in[i]);
  for (int i = threadIdx.x; i < nw; i += blockDim.x) {
    unsigned long long v;
    do { v = ld_sys(&theirs[i]); } while ((int)((unsigned)(v >> 32) - s) < 0);
    unsigned a = in[i], b = (unsigned)v;
    __nv_bfloat162 r = __hadd2(*(__nv_bfloat162*)&a, *(__nv_bfloat162*)&b);
    out[i] = *(unsigned*)&r;
  }
  __syncthreads();
  if (threadIdx.x == 0) *counter = s;
}

void ll_ar(torch::Tensor t, int64_t base, int64_t rank, int64_t slot_bytes, torch::Tensor counter) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  ll_ar_kernel<<<1, 512, 0, stream>>>((const unsigned*)t.data_ptr(), (unsigned*)t.data_ptr(),
                                      (int)(t.numel() / 2), (char*)base, (int)rank, slot_bytes,
                                      (unsigned*)counter.data_ptr());
}
"""
_CPP = "void ll_ar(torch::Tensor t, int64_t base, int64_t rank, int64_t slot_bytes, torch::Tensor counter);"


class LLAllReduce:
    def __init__(self, group, rank_in_group: int, world_size: int, device: torch.device):
        assert world_size == 2, "LL all-reduce is written for two ranks"
        self.rank = rank_in_group
        self.max_bytes = int(os.environ.get("SGLANG_LL_ALLREDUCE_MAX_BYTES", str(64 << 10)))
        # an LL word carries 4 data bytes in 8
        self.slot_bytes = 2 * self.max_bytes
        total = _HEADER + 4 * self.slot_bytes

        from torch.utils.cpp_extension import load_inline

        def build():
            return load_inline("sgl_ll_allreduce", _CPP, cuda_sources=_SRC, functions=["ll_ar"],
                               extra_cuda_cflags=["-O3"], verbose=False)

        # one builder, the other loads the cached module
        self.ext = build() if rank_in_group == 0 else None
        torch.distributed.barrier(group=group)
        if self.ext is None:
            self.ext = build()

        name = [f"/dev/shm/sgl_ll_ar_{os.getpid()}_{time.time_ns()}" if rank_in_group == 0 else None]
        src = torch.distributed.get_process_group_ranks(group)[0]
        torch.distributed.broadcast_object_list(name, src=src, group=group)
        self.path = name[0]
        if rank_in_group == 0:
            with open(self.path, "wb") as f:
                f.truncate(total)
        torch.distributed.barrier(group=group)
        fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(fd, total)
        os.close(fd)
        buf = np.frombuffer(self.mm, dtype=np.uint8)
        # the reader first-touches the peer's slots
        peer = 1 - rank_in_group
        buf[_HEADER + peer * 2 * self.slot_bytes: _HEADER + (peer + 1) * 2 * self.slot_bytes] = 0
        torch.distributed.barrier(group=group)
        # one rank at a time: both registering the same pages at once failed
        # now and then with cudaErrorInvalidValue
        for r in range(world_size):
            if r == rank_in_group:
                with torch.cuda.device(device):
                    err = torch.cuda.cudart().cudaHostRegister(buf.ctypes.data, total, 3)  # Portable | Mapped
                    if int(err) != 0:
                        raise RuntimeError(f"cudaHostRegister failed: {int(err)}")
            torch.distributed.barrier(group=group)
        dptr = ctypes.c_uint64()
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        with torch.cuda.device(device):
            rc = cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), buf.ctypes.data, 0)
        if rc != 0:
            raise RuntimeError(f"cuMemHostGetDevicePointer failed: {rc}")
        self.base = dptr.value
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._buf = buf
        torch.distributed.barrier(group=group)
        if rank_in_group == 0:
            os.unlink(self.path)  # both ranks hold the mapping
        logger.info("[AR] LL all-reduce through host memory for bf16 up to %d KiB", self.max_bytes >> 10)

    def should_use(self, t: torch.Tensor) -> bool:
        n = t.numel() * t.element_size()
        return (t.dtype == torch.bfloat16 and t.is_contiguous() and 0 < n <= self.max_bytes
                and t.numel() % 2 == 0)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        self.ext.ll_ar(t, self.base, self.rank, self.slot_bytes, self.counter)
        return t
