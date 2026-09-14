"""All-reduce for two GPUs without peer access, staged through pinned host
memory with the copy engines and driven entirely from the GPU streams.

Consumer cards (no NVLink, no PCIe P2P) fall back to NCCL's SHM transport,
whose kernels read host memory over PCIe one line at a time: on a dual
RTX 5070 Ti host a 20 MB all-reduce ([2048, 5120] bf16, the shape a TP=2
prefill chunk produces) runs at 1.9 GB/s, 10.9 ms, whatever NCCL_PROTO says,
and 128 of them per chunk were 91% of the prefill step. NCCL_SHM_USE_CUDA_MEMCPY
fixes the bandwidth but doubles the latency of the 10 KB decode all-reduces
(39 -> 81 us, x130 per token), so it is not a free switch.

Here each rank DMAs its input into a shared pinned buffer in pieces and
publishes a monotonic piece counter with cuStreamWriteValue32; the peer's
stream waits on that counter with cuStreamWaitValue32, DMAs each piece back
and adds it. The D2H and H2D copy engines overlap and the CPU never blocks,
so kernel launches for the next layer keep running ahead. 20 MB: ~1.5 ms.
The sum is a+b on both ranks, so the result is bit-identical to NCCL's.

Used only for messages of at least SGLANG_HOST_STAGED_ALLREDUCE_MIN_BYTES
(default 1 MiB) outside CUDA graph capture; decode keeps NCCL. Enable with
SGLANG_HOST_STAGED_ALLREDUCE=1. Two ranks only.
"""
import ctypes
import logging
import mmap
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

_FLAG_BYTES = 4096  # one page of uint32 flags at the head of the mapping
_NBUF = 2  # double buffer per rank
_PIECE = int(os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE_PIECE", str(4 << 20)))

_cuda = None


def _libcuda():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        _cuda.cuStreamWriteValue32_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        _cuda.cuStreamWaitValue32_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    return _cuda


class HostStagedAllReduce:
    def __init__(self, group, rank_in_group: int, world_size: int, device: torch.device,
                 region_bytes: int = 32 << 20):
        assert world_size == 2, "host-staged all-reduce is written for two ranks"
        self.rank = rank_in_group
        self.peer = 1 - rank_in_group
        self.device = device
        self.region_bytes = region_bytes
        self.min_bytes = int(os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE_MIN_BYTES", str(1 << 20)))
        # Both ranks must agree on the file; rank 0 picks the name and the
        # gloo group carries it (init time only, before any forward).
        name = [f"/dev/shm/sgl_host_ar_{os.getpid()}_{time.time_ns()}" if rank_in_group == 0 else None]
        src = torch.distributed.get_process_group_ranks(group)[0]
        torch.distributed.broadcast_object_list(name, src=src, group=group)
        self.path = name[0]
        self.total = _FLAG_BYTES + world_size * _NBUF * region_bytes
        if rank_in_group == 0:
            with open(self.path, "wb") as f:
                f.truncate(self.total)
        torch.distributed.barrier(group=group)
        fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(fd, self.total)
        os.close(fd)
        buf = np.frombuffer(self.mm, dtype=np.uint8)
        self.flags = np.frombuffer(self.mm, dtype=np.uint32, count=_FLAG_BYTES // 4)
        with torch.cuda.device(device):
            err = torch.cuda.cudart().cudaHostRegister(buf.ctypes.data, self.total, 3)  # Portable | Mapped
            if err != 0:
                raise RuntimeError(f"cudaHostRegister failed: {err}")
            dptr = ctypes.c_uint64()
            rc = _libcuda().cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), buf.ctypes.data, 0)
            if rc != 0:
                raise RuntimeError(f"cuMemHostGetDevicePointer failed: {rc}")
        self.dflags = dptr.value  # device address of the flag page
        self.host = torch.from_numpy(buf)
        self.regions = [[self._region(r, i) for i in range(_NBUF)] for r in range(world_size)]
        self.tmp = torch.empty(region_bytes, dtype=torch.uint8, device=device)
        self.seq = 0
        self.published = 0  # pieces this rank has published, ever
        self.d2h = torch.cuda.Stream(device=device)
        self.h2d = torch.cuda.Stream(device=device)
        if rank_in_group == 0:
            self.flags[:] = 0
        torch.distributed.barrier(group=group)
        if rank_in_group == 1:
            os.unlink(self.path)  # the mappings keep it alive
        logger.info("host-staged all-reduce ready: %s, %d MB/region, %d MB pieces, min %d KB",
                    self.path, region_bytes >> 20, _PIECE >> 20, self.min_bytes >> 10)

    def _region(self, r, i):
        off = _FLAG_BYTES + (r * _NBUF + i) * self.region_bytes
        return self.host[off:off + self.region_bytes]

    # flags (uint32, monotonic): [r] = pieces rank r has landed in host
    # memory, [2 + r] = all-reduces rank r has finished reading back.
    def _write(self, stream, idx, value):
        rc = _libcuda().cuStreamWriteValue32_v2(stream.cuda_stream, self.dflags + 4 * idx, value, 0)
        if rc != 0:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {rc}")

    def _wait(self, stream, idx, value):
        rc = _libcuda().cuStreamWaitValue32_v2(stream.cuda_stream, self.dflags + 4 * idx, value, 0)  # GEQ
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {rc}")

    def should_use(self, t: torch.Tensor) -> bool:
        return (t.is_cuda and t.is_contiguous()
                and t.numel() * t.element_size() >= self.min_bytes
                and not torch.cuda.is_current_stream_capturing())

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        flat = t.view(-1).view(torch.uint8)
        nbytes = flat.numel()
        cur = torch.cuda.current_stream(self.device)
        self.d2h.wait_stream(cur)
        self.h2d.wait_stream(cur)
        r, p = self.rank, self.peer
        for off in range(0, nbytes, self.region_bytes):
            n = min(self.region_bytes, nbytes - off)
            self.seq += 1
            seq = self.seq
            slot = seq % _NBUF
            mine = self.regions[r][slot]
            theirs = self.regions[p][slot]
            src = flat[off:off + n]
            pieces = [(a, min(_PIECE, n - a)) for a in range(0, n, _PIECE)]
            # the peer must have read this slot back from two calls ago
            if seq > _NBUF:
                self._wait(self.d2h, 2 + p, seq - _NBUF)
            with torch.cuda.stream(self.d2h):
                for a, m in pieces:
                    mine[a:a + m].copy_(src[a:a + m], non_blocking=True)
                    self.published += 1
                    self._write(self.d2h, r, self.published)
            base = self.published - len(pieces)
            with torch.cuda.stream(self.h2d):
                for i, (a, m) in enumerate(pieces):
                    self._wait(self.h2d, p, base + i + 1)
                    self.tmp[a:a + m].copy_(theirs[a:a + m], non_blocking=True)
                src.view(t.dtype).add_(self.tmp[:n].view(t.dtype))
                self._write(self.h2d, 2 + r, seq)
        cur.wait_stream(self.h2d)
        return t
