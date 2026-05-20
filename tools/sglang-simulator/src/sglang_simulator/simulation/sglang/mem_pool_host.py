from typing import Optional

import numpy as np
import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_MHATokenToKVPoolHostHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPoolHost"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    KV_CACHE_BYTES: Optional[int] = None
    KV_CACHE_BYTES_PER_LAYER: Optional[int] = None
    MEMORY_READ_BANDWIDTH_BYTES: Optional[float] = None
    MEMORY_WRITE_BANDWIDTH_BYTES: Optional[float] = None

    @classmethod
    def hook(cls, target):

        def est_bandwidth_batch(size_bytes_arr: np.ndarray, cat: str):
            if cls.MEMORY_READ_BANDWIDTH_BYTES is None:
                cls.MEMORY_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().memory_read_bandwidth
                )
            if cls.MEMORY_WRITE_BANDWIDTH_BYTES is None:
                cls.MEMORY_WRITE_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().memory_write_bandwidth
                )
            x = size_bytes_arr.astype(np.float64)
            if cat == "H2D":
                eff = 0.85
                t0 = 6.67e-6
                bw = cls.MEMORY_READ_BANDWIDTH_BYTES * eff
            else:
                eff = 0.85
                t0 = 4e-6
                bw = cls.MEMORY_WRITE_BANDWIDTH_BYTES * eff
            return x * bw / (t0 * bw + x)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            # update global clock
            # Merge cache indices
            # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
            assert len(host_indices) == len(device_indices)
            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            if cls.KV_CACHE_BYTES_PER_LAYER is None:
                cls.KV_CACHE_BYTES_PER_LAYER = (
                    ConfigManager.get_kv_cache_bytes_per_layer()
                )

            size_bytes_arr = seg_len * float(cls.KV_CACHE_BYTES_PER_LAYER)
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ) -> None:
            """
            Backup KV data from the device memory pool to the host memory pool for all layers.
            """
            # update global clock
            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            if cls.KV_CACHE_BYTES is None:
                cls.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

            size_bytes_arr = seg_len * float(cls.KV_CACHE_BYTES)
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead

            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
            """
            Get a flat data page from the host memory pool.
            """
            return torch.ones(size=(1, 1)) * index

        def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
            """
            Set a flat data page to the host memory pool.
            """
            pass

        target.load_to_device_per_layer = load_to_device_per_layer
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.get_data_page = get_data_page
        target.set_from_flat_data_page = set_from_flat_data_page


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HostKVCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pip memory, which might fail on CPU platforms.
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            else:
                logger.warning(
                    "Failed to disable pip memory while initializing the hoot memory pool."
                )
            return original_init(self, *args, **kwargs)

        target.__init__ = wrapped_init
