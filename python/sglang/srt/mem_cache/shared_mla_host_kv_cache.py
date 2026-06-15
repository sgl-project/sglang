"""
SharedMLA: a single host (L2) KV cache slab shared across the TP ranks of one node.

For MLA models (DeepSeek V2/V3, Kimi, GLM-4, etc.), every TP rank produces an
identical compressed KV cache, so the default per-rank host pool stores TP copies
of the same bytes. This module keeps a single copy instead:

- rank 0 allocates a POSIX shared-memory slab and registers it with
  cudaHostRegister (portable) in its own CUDA context
- every other rank attaches to the same slab and registers it in its own CUDA
  context (no extra host allocation)
- only rank 0 owns slot management (alloc/free) and writes L2 (backup); other
  ranks receive the slot indices from rank 0 via broadcast
- all ranks read from the slab: the backup transfer kernel reads host memory over
  PCIe (host pinned memory mapped into each GPU's address space), not over NVLink

Single-node only: the slab lives in POSIX shared memory + a per-process
cudaHostRegister, visible only on one physical host, so the whole TP group must
fit on one node (nnodes=1); server_args rejects multi-node configs.

Host_value and eviction synchronization use NCCL broadcast (the TP group is
already available). This eliminates (TP-1)/TP of the L2 DRAM footprint and the
per-rank L1->L2 DMA copies.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Optional

import numpy as np
import psutil
import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


def _mbind_interleave(addr: int, length: int) -> bool:
    """Set MPOL_INTERLEAVE on a virtual address range via the mbind(2) syscall.

    The shared slab is read fan-out by all GPUs. Placing all physical pages on
    a single NUMA node makes the cross-socket GPUs saturate one UPI direction
    (measured ~82 GB/s for 4 GPUs vs ~154 GB/s local). Interleaving pages across
    both nodes spreads the traffic over both full-duplex UPI directions, raising
    8-GPU aggregate read bandwidth from ~212 to ~281 GB/s with no extra DRAM.

    Must be called BEFORE the pages are first touched (memset), since mbind only
    governs future page faults, not already-resident pages.

    Returns True on success; on any failure the caller falls back to default
    (first-touch) placement, which is still correct, just less balanced.
    """
    # Build a node mask covering all online NUMA nodes.
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
    # maxnode counts bits; +1 for the inclusive upper bound the kernel expects.
    maxnode = max_node + 2

    page_size = os.sysconf("SC_PAGESIZE")
    aligned_addr = addr & ~(page_size - 1)
    aligned_len = length + (addr - aligned_addr)

    # glibc does not export mbind as a symbol on all builds, so invoke it via
    # the raw syscall (x86_64: __NR_mbind = 237, aarch64: 235).
    machine = platform.machine()
    if machine == "x86_64":
        NR_mbind = 237
    elif machine in ("aarch64", "arm64"):
        NR_mbind = 235
    else:
        return False

    # CDLL/syscall can raise on non-glibc systems (e.g. musl) or if libc.so.6 is
    # not on the loader path. Interleave is an optimization, so fall back to the
    # default placement instead of crashing startup.
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mask_arr = ctypes.c_ulong(nodemask)
        libc.syscall.restype = ctypes.c_long
        # mbind(addr, len, mode, nodemask*, maxnode, flags)
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
        logger.warning(
            f"SharedMLA: mbind(MPOL_INTERLEAVE) raised ({e}), "
            f"falling back to default NUMA placement"
        )
        return False
    if rc != 0:
        errno = ctypes.get_errno()
        logger.warning(
            f"SharedMLA: mbind(MPOL_INTERLEAVE) failed (errno={errno}), "
            f"falling back to default NUMA placement"
        )
        return False
    return True


class SharedMLATokenToKVPoolHost:
    """MLA shared L2 host pool: rank 0 owns the slab, the other ranks read it.

    Uses a POSIX shared-memory slab registered with cudaHostRegister (portable)
    in each rank's CUDA context, so every GPU can read the slab from host pinned
    memory over PCIe. rank 0 owns slot allocation and the backup writes; the
    other ranks attach read-only and receive slot indices via broadcast.

    This is NOT a subclass of HostKVCache — it replaces the entire
    HostKVCache for MLA models in shared mode. The interface is
    compatible with HiCacheController's expectations.
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
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_owner = tp_rank == 0
        self.page_size = page_size
        self.layout = layout
        self.device = device
        self.device_pool = device_pool
        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.start_layer = device_pool.start_layer or 0
        self.end_layer = device_pool.end_layer or self.layer_num - 1

        # Compute kv_cache_dim
        self.kv_lora_rank = device_pool.kv_lora_rank
        self.qk_rope_head_dim = device_pool.qk_rope_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Size computation
        self.size_per_token = self.kv_cache_dim * self.dtype.itemsize * self.layer_num
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        total_bytes = self.size * self.layer_num * self.token_stride_size

        if self.is_owner:
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

        # Shared memory name — include rank 0's PID to avoid collision across
        # instances. Broadcast it from rank 0 so all ranks agree on the name.
        owner_pid = os.getpid() if self.is_owner else 0
        pid_tensor = torch.tensor(
            [owner_pid], dtype=torch.int64, device=device_pool.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pid_tensor, src=0)
        self._shm_name = f"shared_mla_tp{tp_size}_p{pid_tensor.item()}"

        if self.is_owner:
            logger.info(
                f"SharedMLA rank 0: creating shared slab "
                f"{total_bytes / 1e9:.2f} GB (name={self._shm_name})"
            )
            # A previous run that crashed after create but before shutdown()
            # leaves the multi-GB POSIX slab on /dev/shm. The name embeds the
            # owner PID, so scan for shared_mla_* slabs whose PID is no longer
            # alive and reclaim them (otherwise they leak forever). This also
            # matters for the NUMA interleave below, whose page placement is
            # silently degraded when a node is already full of stale pages.
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

            # Interleave the slab's physical pages across all NUMA nodes BEFORE
            # first-touch, so cross-socket GPUs spread their reads over both
            # full-duplex UPI directions instead of saturating one. mbind only
            # affects future faults, so we set the policy then touch the pages.
            owner_np = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
            owner_tensor = torch.from_numpy(owner_np)
            interleaved = _mbind_interleave(owner_tensor.data_ptr(), total_bytes)
            owner_np[:] = 0  # first-touch: realize pages under the chosen policy
            logger.info(
                f"SharedMLA rank 0: slab pages "
                f"{'interleaved across NUMA nodes' if interleaved else 'default (first-touch) placement'}"
            )

        # All ranks barrier so the other ranks wait for rank 0 to create the shm
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not self.is_owner:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            logger.info(f"SharedMLA rank {self.tp_rank}: attached to shared slab")

        # Map as torch tensor
        np_array = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
        shm_tensor = torch.from_numpy(np_array)

        # Every process must cudaHostRegister the shared memory in its own
        # CUDA context. "Portable" means all GPUs within ONE process can access
        # it, but each process still needs its own registration.
        self._registered_ptr = None
        if _is_cuda:
            cudart = torch.cuda.cudart()
            ptr = shm_tensor.data_ptr()
            rc = cudart.cudaHostRegister(ptr, total_bytes, 1)  # portable
            if int(rc) == 712:
                # cudaErrorHostMemoryAlreadyRegistered: a previous slab at this
                # same address leaked its registration (e.g. crash without
                # shutdown). Unregister the stale mapping and retry once.
                cudart.cudaHostUnregister(ptr)
                rc = cudart.cudaHostRegister(ptr, total_bytes, 1)
            if int(rc) != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed on rank {self.tp_rank} (rc={int(rc)}). "
                    f"This often means the locked-memory limit is too low; try "
                    f"raising it with `ulimit -l unlimited`."
                )
            # Remember the pointer so shutdown() can unregister it; otherwise the
            # registration outlives the slab and a new slab mapped to the same
            # address fails with cudaErrorHostMemoryAlreadyRegistered.
            self._registered_ptr = ptr

        # Barrier after all ranks register
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # View as KV buffer: [layer_num, size, 1, kv_cache_dim] (layer first)
        self.kv_buffer = shm_tensor.view(self.dtype).view(
            self.layer_num, self.size, 1, self.kv_cache_dim
        )

        # Per-layer data_refs and data_ptrs (for transfer kernels)
        # data_refs[i] is layer i's buffer: [size, 1, kv_cache_dim]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=device_pool.device,
        )

        # Slot management. Every rank keeps its OWN free-slot bookkeeping and
        # runs the identical, deterministic allocator. Because all ranks see the
        # same lockstep alloc/free sequence, they independently arrive at the
        # same slot indices for every node -- no broadcast needed to keep the
        # host radix trees mirror-identical. Only the KV *buffer* is shared
        # (rank 0 owns the slab, the others map it read-only).
        self.lock = threading.RLock()
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

        self.mem_usage = total_bytes / (1024**3)
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        logger.info(
            f"SharedMLA rank {self.tp_rank}: ready, size={self.size} tokens, "
            f"{self.mem_usage:.2f} GB "
            f"({'owns' if self.is_owner else 'reads'} the slab "
            f"shared across all {self.tp_size} ranks)"
        )

    # ---- Slot management ----

    @staticmethod
    def _reclaim_stale_slabs():
        """Unlink shared_mla_* slabs in /dev/shm whose owner PID is dead.

        The slab name embeds the owner PID (shared_mla_tp{n}_p{pid}). A crash
        that skips shutdown() leaves a multi-GB slab behind; reclaim any whose
        PID is no longer alive. Best-effort: any failure is ignored.
        """
        shm_dir = "/dev/shm"
        try:
            names = os.listdir(shm_dir)
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

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate host slot indices. Deterministic on every rank, so all
        ranks independently return the same indices for the same call sequence.
        """
        assert need_size % self.page_size == 0
        if need_size > len(self.free_slots):
            return None
        select = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select

    def free(self, indices: torch.Tensor):
        if indices.numel() == 0:
            return
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])

    def available_size(self):
        return len(self.free_slots)

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    # ---- Transfer operations ----

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        """Only rank 0: GPU DMA → shared slab."""
        if not self.is_owner:
            return
        if io_backend != "kernel" or not _is_cuda:
            return

        from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

        element_size = self.kv_cache_dim * self.dtype.itemsize
        try:
            from sglang.jit_kernel.hicache import (
                can_use_hicache_jit_kernel,
            )
            from sglang.jit_kernel.hicache import (
                transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
            )

            use_jit = can_use_hicache_jit_kernel(element_size=element_size)
        except ImportError:
            use_jit = False

        if use_jit:
            jit_transfer_hicache_all_layer_mla(
                ptr_dst=self.data_ptrs,
                indices_dst=host_indices,
                ptr_src=device_pool.data_ptrs,
                indices_src=device_indices,
                cache_dst_stride_bytes=self.token_stride_size,
                cache_src_stride_bytes=self.token_stride_size,
                element_size=element_size,
            )
        else:
            transfer_kv_all_layer_mla(
                src_layers=device_pool.data_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.token_stride_size,
                num_layers=self.layer_num,
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        """All ranks: GPU ld.global [shared slab] → local L1."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if io_backend != "kernel" or not _is_cuda:
            return

        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla

        try:
            from sglang.jit_kernel.hicache import (
                can_use_hicache_jit_kernel,
            )
            from sglang.jit_kernel.hicache import (
                transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
            )

            use_jit = can_use_hicache_jit_kernel(
                element_size=self.kv_cache_dim * self.dtype.itemsize
            )
        except ImportError:
            use_jit = False

        if use_jit:
            jit_transfer_hicache_one_layer_mla(
                cache_dst=device_pool.kv_buffer[layer_id],
                cache_src=self.data_refs[layer_id],
                indices_dst=device_indices,
                indices_src=host_indices,
                element_dim=self.kv_cache_dim,
            )
        else:
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=device_pool.kv_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                item_size=self.token_stride_size,
            )

    def get_contiguous_buf_infos(self):
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.data_refs[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size * self.page_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_page_buffer_meta(self, indices):
        """Return (ptr_list, size_list) for RDMA/Mooncake zero-copy.
        Layout is layer_first: [layer_num, size, 1, kv_cache_dim].
        Returns per-page per-layer pointers, matching MLATokenToKVPoolHost.
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        base_ptr = self.kv_buffer.data_ptr()
        indices_list = indices.tolist()
        for i in range(0, len(indices_list), self.page_size):
            for layer_id in range(self.layer_num):
                ptr = (
                    base_ptr
                    + indices_list[i] * self.kv_cache_dim * self.dtype.itemsize
                    + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        elem_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
        return ptr_list, [elem_size] * len(ptr_list)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        # One page = the KV of `page_size` consecutive tokens across all layers.
        # Layout is layer_first [layer_num, size, 1, kv_cache_dim]; gather each
        # layer's slice and concat so the flat bytes match the L3 storage page.
        pages = [
            self.kv_buffer[layer_id, index : index + self.page_size, 0, :]
            for layer_id in range(self.layer_num)
        ]
        data_page = torch.cat(pages, dim=0)
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.layer_num * self.page_size * self.kv_cache_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        # Inverse of get_data_page: only rank 0 owns the slab and writes it; on
        # followers the slab is read-only and L3 IO never reaches this path.
        flat = data_page.view(self.dtype).reshape(
            self.layer_num, self.page_size, self.kv_cache_dim
        )
        for layer_id in range(self.layer_num):
            self.kv_buffer[layer_id, index : index + self.page_size, 0, :].copy_(
                flat[layer_id]
            )

    def get_size_per_token(self):
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def get_kv_size_bytes(self):
        return self.size * self.layer_num * self.token_stride_size

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def maybe_get_custom_mem_pool(self):
        return None

    def shutdown(self):
        # __init__ may fail before these are assigned; guard so tearDown/atexit
        # don't raise AttributeError on a half-built instance.
        registered_ptr = getattr(self, "_registered_ptr", None)
        if registered_ptr is not None and _is_cuda:
            # Pair the cudaHostRegister from __init__; without this the pinned
            # registration outlives the slab and a future slab at the same
            # address fails with cudaErrorHostMemoryAlreadyRegistered.
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
