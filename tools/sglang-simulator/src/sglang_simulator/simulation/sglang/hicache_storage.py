import os
from typing import Any, List, Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.utils.logger import get_logger

logger = get_logger("sglang-simulator")


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class MockHiCacheStorage:
    def __init__(self, *args, **kwargs):

        self.storage: set = set()
        self.storage_file_path: str = Envs.hicache_storage_keys_path()
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                line = f.readline()
                while line:
                    self.storage.add(line.strip())
                    line = f.readline()

        self.registered_pools = {}

    def register_mem_pool_host(self, mem_pool_host):
        pass

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        """Register one pool from UnifiedRadixCache's multi-pool HiCache stack."""
        self.registered_pools[host_pool_name] = host_pool

    @staticmethod
    def _pool_storage_key(key: str, pool_name) -> str:
        name = str(pool_name)
        return key if name == "kv" else f"{key}.{name}"

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a+") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info=None,  # HiCacheStorageExtraInfo
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:

        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info) -> int:
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        """Return Unified HiCache's per-pool longest-prefix result."""
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult

        kv_hit_pages = self.batch_exists(keys, extra_info)
        extra_pool_hit_pages = {}
        final_pages = kv_hit_pages
        for transfer in pool_transfers or []:

            def has_component(page_idx):
                return self.exists(
                    self._pool_storage_key(keys[page_idx], transfer.name)
                )

            hit_policy = getattr(transfer.hit_policy, "value", transfer.hit_policy)
            if hit_policy == "all_pages":
                boundary = next(
                    (i for i in range(kv_hit_pages) if not has_component(i)),
                    kv_hit_pages,
                )
            else:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_hit_pages, 0, -1):
                    if all(
                        has_component(i)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            extra_pool_hit_pages[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(
            kv_hit_pages=final_pages,
            extra_pool_hit_pages=extra_pool_hit_pages,
        )

    def batch_get_v2(self, transfers, extra_info=None):
        """Simulate loading every available pool page into registered host pools."""
        results = {}
        for transfer in transfers:
            keys = transfer.keys or []
            results[transfer.name] = [
                self.exists(self._pool_storage_key(key, transfer.name)) for key in keys
            ]
        return results

    def batch_set_v2(self, transfers, extra_info=None):
        """Persist Unified HiCache component keys without materializing payloads."""
        results = {}
        for transfer in transfers:
            keys = transfer.keys or []
            pool_results = []
            for key in keys:
                pool_results.append(
                    self.set(self._pool_storage_key(key, transfer.name))
                )
            results[transfer.name] = pool_results
        return results

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True
