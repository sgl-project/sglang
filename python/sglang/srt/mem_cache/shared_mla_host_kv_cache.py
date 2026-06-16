"""
SharedMLA: a single host (L2) KV cache slab shared across the TP ranks of one node.

For MLA models every TP rank produces an identical compressed KV cache, so the
default per-rank host pool stores TP copies of the same bytes. Here rank 0 owns a
POSIX shm slab (cudaHostRegister) and the others attach read-only; every rank runs
the same deterministic allocator, so all derive identical slot indices with no
cross-rank broadcast. Single-node only (the slab is visible on one host).
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Optional

import numpy as np
import psutil
import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


def _mbind_interleave(addr: int, length: int) -> bool:
    """Interleave a range's physical pages across all NUMA nodes via mbind(2).

    The slab is read fan-out by all GPUs; interleaving spreads reads over both
    UPI directions (measured ~212 -> ~281 GB/s for 8 GPUs). Must run before
    first-touch. Returns False on any failure (caller falls back to default
    first-touch placement, still correct).
    """
    try:
        with open("/sys/devices/system/node/online") as f:
            online = f.read().strip()
    except OSError:
        return False

    node_ids = []
    for part in online.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            node_ids.extend(range(int(lo), int(hi) + 1))
        else:
            node_ids.append(int(part))
    if len(node_ids) < 2:
        return False  # single NUMA node: interleave is a no-op

    max_node = max(node_ids)
    nodemask = 0
    for n in node_ids:
        nodemask |= 1 << n

    MPOL_INTERLEAVE = 3
    maxnode = max_node + 2

    page_size = os.sysconf("SC_PAGESIZE")
    aligned_addr = addr & ~(page_size - 1)
    aligned_len = length + (addr - aligned_addr)

    # glibc may not export mbind as a symbol, so invoke the raw syscall.
    machine = platform.machine()
    if machine == "x86_64":
        NR_mbind = 237
    elif machine in ("aarch64", "arm64"):
        NR_mbind = 235
    else:
        return False

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mask_arr = ctypes.c_ulong(nodemask)
        libc.syscall.restype = ctypes.c_long
        rc = libc.syscall(
            ctypes.c_long(NR_mbind),
            ctypes.c_void_p(aligned_addr),
            ctypes.c_ulong(aligned_len),
            ctypes.c_int(MPOL_INTERLEAVE),
            ctypes.byref(mask_arr),
            ctypes.c_ulong(maxnode),
            ctypes.c_uint(0),
        )
    except Exception as e:
        logger.warning(f"SharedMLA: mbind raised ({e}), using default placement")
        return False
    if rc != 0:
        logger.warning(
            f"SharedMLA: mbind failed (errno={ctypes.get_errno()}), "
            f"using default placement"
        )
        return False
    return True


class SharedMLATokenToKVPoolHost(MLATokenToKVPoolHost):
    """MLA shared L2 host pool: rank 0 owns a POSIX shm slab, the others read it.

    Subclasses MLATokenToKVPoolHost and only overrides where the KV bytes come
    from: rank 0 creates a NUMA-interleaved shm slab + cudaHostRegister, the other
    ranks attach read-only. All layout/transfer/data-page logic is inherited, so
    every hicache mem layout and IO backend works unchanged. Every rank runs the
    same deterministic allocator, so slot indices match with no cross-rank
    broadcast. Single-node only (the slab is visible on one host).
    """

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        tp_rank: int,
        tp_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
    ):
        # Shared-only state must be set before super().__init__, which calls
        # init_kv_buffer() / _check_host_memory() below.
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_owner = tp_rank == 0
        self._registered_ptr = None
        self._shm = None
        self._shm_name = None

        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            override_kv_cache_dim=override_kv_cache_dim,
        )

        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.mem_usage = self.get_kv_size_bytes() / (1024**3)
        logger.info(
            f"SharedMLA rank {self.tp_rank}: ready, size={self.size} tokens, "
            f"{self.mem_usage:.2f} GB "
            f"({'owns' if self.is_owner else 'reads'} the slab "
            f"shared across all {self.tp_size} ranks)"
        )

    def _check_host_memory(self):
        # Only the owner allocates the slab; followers attach to it, so the
        # base per-rank availability check would double-count and misfire.
        if not self.is_owner:
            return
        total_bytes = self.size * self.size_per_token
        host_mem = psutil.virtual_memory()
        safety_margin = 10 * (1024**3)
        if total_bytes > host_mem.available:
            raise ValueError(
                f"Not enough host memory for SharedMLA. "
                f"Need {total_bytes / 1e9:.2f} GB, "
                f"have {host_mem.available / 1e9:.2f} GB available."
            )
        if total_bytes > host_mem.available - safety_margin:
            logger.warning(
                f"SharedMLA slab ({total_bytes / 1e9:.2f} GB) leaves less "
                f"than {safety_margin / 1e9:.0f} GB of host memory free "
                f"({host_mem.available / 1e9:.2f} GB available)."
            )

    def init_kv_buffer(self):
        # Same layout dims as the parent MLATokenToKVPoolHost, but the buffer is
        # backed by a shared shm slab (owner creates, others attach) instead of a
        # per-rank pinned allocation.
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, 1, self.kv_cache_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, 1, self.kv_cache_dim)
        elif self.layout == "page_first_direct":
            dims = (self.page_num, self.layer_num, self.page_size, 1, self.kv_cache_dim)
        else:
            raise ValueError(f"SharedMLA unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        total_bytes = self.size * self.layer_num * self.token_stride_size

        owner_pid = os.getpid() if self.is_owner else 0
        pid_tensor = torch.tensor(
            [owner_pid], dtype=torch.int64, device=self.device_pool.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pid_tensor, src=0)
        self._shm_name = f"shared_mla_tp{self.tp_size}_p{pid_tensor.item()}"

        if self.is_owner:
            logger.info(
                f"SharedMLA rank 0: creating shared slab "
                f"{total_bytes / 1e9:.2f} GB (name={self._shm_name})"
            )
            # Reclaim multi-GB slabs leaked by a crashed run (dead owner PID).
            self._reclaim_stale_slabs()
            try:
                stale = shared_memory.SharedMemory(name=self._shm_name)
                stale.close()
                stale.unlink()
                logger.warning(f"SharedMLA rank 0: removed stale slab {self._shm_name}")
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=True, size=total_bytes
            )
            # Interleave pages across NUMA nodes before first-touch (mbind only
            # governs future faults), then touch them to realize the placement.
            owner_np = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
            interleaved = _mbind_interleave(
                torch.from_numpy(owner_np).data_ptr(), total_bytes
            )
            owner_np[:] = 0
            logger.info(
                f"SharedMLA rank 0: slab pages "
                f"{'interleaved across NUMA nodes' if interleaved else 'default (first-touch) placement'}"
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not self.is_owner:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            logger.info(f"SharedMLA rank {self.tp_rank}: attached to shared slab")

        np_array = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
        shm_tensor = torch.from_numpy(np_array)
        if _is_cuda:
            self._cuda_host_register(shm_tensor.data_ptr(), total_bytes)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return shm_tensor.view(self.dtype).view(*dims)

    def _cuda_host_register(self, ptr: int, total_bytes: int):
        cudart = torch.cuda.cudart()
        rc = cudart.cudaHostRegister(ptr, total_bytes, 1)  # portable
        if int(rc) == 712:
            # Already registered by a leaked slab at this address: retry once.
            cudart.cudaHostUnregister(ptr)
            rc = cudart.cudaHostRegister(ptr, total_bytes, 1)
        if int(rc) != 0:
            raise RuntimeError(
                f"cudaHostRegister failed on rank {self.tp_rank} (rc={int(rc)}). "
                f"This often means the locked-memory limit is too low; try "
                f"raising it with `ulimit -l unlimited`."
            )
        self._registered_ptr = ptr

    @staticmethod
    def _reclaim_stale_slabs():
        """Unlink shared_mla_* slabs in /dev/shm whose owner PID is dead."""
        try:
            names = os.listdir("/dev/shm")
        except OSError:
            return
        for name in names:
            if not name.startswith("shared_mla_tp") or "_p" not in name:
                continue
            try:
                pid = int(name.rsplit("_p", 1)[1])
            except ValueError:
                continue
            if psutil.pid_exists(pid):
                continue
            try:
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
                logger.warning(
                    f"SharedMLA rank 0: reclaimed stale slab {name} (dead pid {pid})"
                )
            except Exception:
                pass

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        # Only rank 0 writes the shared slab; the parent handles all layouts.
        if not self.is_owner:
            return
        return super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        # All ranks read the shared slab into their own L1. Gate on the transfer
        # counter (the parent does not) so loads observe completed backups.
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        # L3 prefetch fill; only rank 0 writes the slab (defensive: followers are
        # already short-circuited by the controller's prefetch_skip).
        if not self.is_owner:
            return
        return super().set_from_flat_data_page(index, data_page)

    def get_kv_size_bytes(self):
        return self.size * self.layer_num * self.token_stride_size

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def maybe_get_custom_mem_pool(self):
        return None

    def shutdown(self):
        # __init__ may fail before these are assigned; guard against that.
        registered_ptr = getattr(self, "_registered_ptr", None)
        if registered_ptr is not None and _is_cuda:
            try:
                torch.cuda.cudart().cudaHostUnregister(registered_ptr)
            except Exception:
                pass
            self._registered_ptr = None
        shm = getattr(self, "_shm", None)
        if shm is not None:
            shm.close()
            if self.is_owner:
                try:
                    shm.unlink()
                except Exception:
                    pass
            self._shm = None
