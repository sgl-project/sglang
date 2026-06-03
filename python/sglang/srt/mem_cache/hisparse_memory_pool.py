# mapping on device memory, host memory and memory allocator

import logging
from typing import Optional

import psutil
import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    HiSparseC4DevicePool,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import HiSparseHostPoolMixin
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices].to(
            torch.int32
        )

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")


class DeepSeekV4SingleKVPoolHost(HiSparseHostPoolMixin):

    def __init__(
        self,
        device_pool: HiSparseC4DevicePool,
        host_size: int,
        page_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
    ):

        assert host_size > 0, "Host size must be specified and greater than 0"

        self.device_pool = device_pool
        self.size = host_size
        self.page_size = page_size
        self.num_pages = (self.size + self.page_size - 1) // self.page_size
        self.size = self.num_pages * self.page_size
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.kv_cache_total_dim = device_pool.kv_cache_total_dim

        self.kv_buffer = self.init_kv_buffer()
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.clear()

    def clear(self):
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device="cpu"
        )

    def init_kv_buffer(self):
        dims = (self.layer_num, self.size + self.page_size, self.kv_cache_total_dim)
        requested_bytes = (
            self.layer_num
            * (self.size + self.page_size)
            * self.kv_cache_total_dim
            * self.dtype.itemsize
        )
        host_mem = psutil.virtual_memory()
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        host_pool = torch.empty(dims, dtype=self.dtype, device=self.device)
        assert self.pin_memory, "DeepSeekV4SingleKVPoolHost requires pin_memory=True"
        if self.pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                host_pool.data_ptr(), host_pool.numel() * host_pool.element_size(), 0
            )
        return host_pool

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if io_backend != "kernel":
            raise ValueError(f"Unsupported IO backend: {io_backend}")

        from sglang.jit_kernel.dsv4 import hisparse_offload_to_host

        if host_indices.device != device_indices.device:
            host_indices = host_indices.to(device=device_indices.device)
        host_indices_i64 = (
            host_indices.to(torch.int64)
            if host_indices.dtype != torch.int64
            else host_indices
        )
        device_indices_i64 = (
            device_indices.to(torch.int64)
            if device_indices.dtype != torch.int64
            else device_indices
        )
        hisparse_offload_to_host(
            gpu_ptrs=device_pool.data_ptrs,
            cpu_ptrs=self.data_ptrs,
            gpu_indices=device_indices_i64,
            cpu_indices=host_indices_i64,
        )

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)
