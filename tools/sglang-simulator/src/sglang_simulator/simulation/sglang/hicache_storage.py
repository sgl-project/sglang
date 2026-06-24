import os
from typing import Any, List, Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils.logger import get_logger

logger = get_logger("hisim")


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class MockHiCacheStorage:
    def __init__(self, *args, **kwargs):

        self.storage: set = set()
        self.storage_file_path: str = "/tmp/sglang_simulator/hicache_storage_keys.txt"
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                line = f.readline()
                while line:
                    self.storage.add(line.strip())
                    line = f.readline()

    def register_mem_pool_host(self, mem_pool_host):
        pass

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

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True
