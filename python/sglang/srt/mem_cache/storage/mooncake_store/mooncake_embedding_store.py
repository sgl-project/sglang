import logging
from typing import Any, List

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeBaseStore

logger = logging.getLogger(__name__)


class MooncakeEmbeddingStore(MooncakeBaseStore):
    def __init__(
        self,
        storage_config: Any = None,
    ):
        super().__init__()

        MooncakeDistributedStore = self._import_mooncake_store()
        self.store = MooncakeDistributedStore()
        self.config = self._load_config(storage_config)
        ret_code = self.store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            16 * 1024 * 1024,  # Internal local buffer size
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        if ret_code != 0:
            raise RuntimeError(f"Failed to setup Mooncake Embedding Store: {ret_code}")

        logger.info("Mooncake Embedding Store initialized successfully.")

    def get_key(self, image_hash: str) -> str:
        return f"emb_{image_hash}"

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        results = self.store.batch_get_into(keys, ptrs, sizes)
        return [res > 0 for res in results]

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        exists = self.store.batch_is_exist(keys)

        put_keys, put_ptrs, put_sizes, indices = [], [], [], []
        success_map = [True] * len(hashes)

        for i, status in enumerate(exists):
            if status != 1:
                put_keys.append(keys[i])
                put_ptrs.append(ptrs[i])
                put_sizes.append(sizes[i])
                indices.append(i)

        if put_keys:
            results = self.store.batch_put_from(put_keys, put_ptrs, put_sizes)
            for i, res in enumerate(results):
                success_map[indices[i]] = res == 0
        return success_map

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        results = self.store.batch_is_exist(keys)
        return [res == 1 for res in results]
