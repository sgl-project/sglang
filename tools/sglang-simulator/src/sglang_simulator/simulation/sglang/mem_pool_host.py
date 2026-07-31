from typing import Optional

import numpy as np
import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_MHATokenToKVPoolHostHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPoolHost"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.mha)$"
    REGEX = True

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
            StateManager.inc_hicache_l2_load_stats(
                call_count=1,
                segment_count=len(size_bytes_arr),
                units=int(np.sum(seg_len)),
                bytes_=float(np.sum(size_bytes_arr)),
            )
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
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.base)$"
    REGEX = True

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


class C_DeepSeekV4SingleKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4SingleKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.deepseek_v4_memory_pool"

    @classmethod
    def hook(cls, target):
        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def override_create_buffer(self, *, num_pages: int):
            bytes_per_token = self.get_bytes_per_token()
            self.kv_cache_total_dim = bytes_per_token
            bytes_per_page_non_padded = self.page_size * bytes_per_token
            self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

            assert self.store_dtype == torch.uint8

            return torch.zeros(
                num_pages,
                self.bytes_per_page_padded,
                dtype=self.store_dtype,
                device=self.device,
            )

        target.create_buffer = override_create_buffer


class C_DeepSeekV4PagedHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4PagedHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    MEMORY_READ_BANDWIDTH_BYTES: Optional[float] = None
    MEMORY_WRITE_BANDWIDTH_BYTES: Optional[float] = None

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pip memory, which might fail on CPU platforms.
            print(1)
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 6:
                args = list(args)
                args[6] = False
            elif "pin_memory" not in kwargs:
                kwargs["pin_memory"] = False
            else:
                logger.warning(
                    "Failed to disable pip memory while initializing the DeepSeekV4PagedHostPool."
                )
            return original_init(self, *args, **kwargs)

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

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            if host_indices is None or device_indices is None:
                return
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            # print(f"[backup_from_device_all_layer DeepSeekV4PagedHostPool] {seg_len=}")

            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            assert len(host_indices) == len(device_indices)
            num_indices = len(host_indices)

            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            size_bytes_arr = seg_len * self.get_size_per_token()
            StateManager.inc_hicache_l2_load_stats(
                call_count=1,
                segment_count=len(size_bytes_arr),
                units=int(np.sum(seg_len)),
                bytes_=float(np.sum(size_bytes_arr)),
            )
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # print(f"[Paged load_to_device_per_layer] {self.pool_name=}, {seg_len=}, {self.get_size_per_token()=}, {total_time_cost=}")
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def get_size_per_token(self):
            if self.pool_name in ["swa"]:
                return self.size_per_token * 130
            elif self.pool_name in ["deepseek_v4_c4"]:
                return self.size_per_token * 65
            elif self.pool_name in ["deepseek_v4_c4_indexer"]:
                return self.size_per_token * 132
            elif self.pool_name in ["deepseek_v4_c128"]:
                return self.size_per_token * 3
            else:
                # return self.size_per_token
                raise ValueError(
                    f"[DeepSeekV4PagedHostPool] unsupported pool name: {self.pool_name}"
                )

        target.__init__ = wrapped_init
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.load_to_device_per_layer = load_to_device_per_layer
        target.get_size_per_token = get_size_per_token


class C_DeepSeekV4StateHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4StateHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    MEMORY_READ_BANDWIDTH_BYTES: Optional[float] = None
    MEMORY_WRITE_BANDWIDTH_BYTES: Optional[float] = None

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pip memory, which might fail on CPU platforms.
            print(1)
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            elif "pin_memory" not in kwargs:
                kwargs["pin_memory"] = False
            else:
                logger.warning(
                    "Failed to disable pip memory while initializing the DeepSeekV4StateHostPool."
                )
            return original_init(self, *args, **kwargs)

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

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            if host_indices is None or device_indices is None:
                return
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            # print(f"[backup_from_device_all_layer DeepSeekV4StateHostPool] {seg_len=}")

            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            assert len(host_indices) == len(device_indices)
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            size_bytes_arr = seg_len * self.get_size_per_token()
            StateManager.inc_hicache_l2_load_stats(
                call_count=1,
                segment_count=len(size_bytes_arr),
                units=int(np.sum(seg_len)),
                bytes_=float(np.sum(size_bytes_arr)),
            )
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            # print(f"[State load_to_device_per_layer] {self.pool_name=}, {seg_len=}, {self.get_size_per_token()=}, {total_time_cost=}")
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def get_size_per_token(self):
            if self.pool_name in ["deepseek_v4_c4_state", "deepseek_v4_c128_state"]:
                return self.size_per_token * 256
            elif self.pool_name in [
                "deepseek_v4_indexer_state",
                "deepseek_v4_c4_indexer_state",
            ]:
                return self.size_per_token * 128
            else:
                # return self.size_per_token
                raise ValueError(
                    f"[DeepSeekV4StateHostPool] unsupported pool name: {self.pool_name}"
                )

        target.__init__ = wrapped_init
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.load_to_device_per_layer = load_to_device_per_layer
        target.get_size_per_token = get_size_per_token
