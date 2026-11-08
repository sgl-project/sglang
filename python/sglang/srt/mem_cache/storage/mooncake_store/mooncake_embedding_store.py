import logging
from typing import Any, List

from sglang.srt.mem_cache.embedding_store import EmbeddingStore
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_TENANT_ID,
    MooncakeBaseStore,
)

logger = logging.getLogger(__name__)


class MooncakeEmbeddingStore(MooncakeBaseStore, EmbeddingStore):
    def __init__(
        self,
        storage_config: Any = None,
    ):
        super().__init__()

        MooncakeDistributedStore = self._import_mooncake_store()
        self.store = MooncakeDistributedStore()
        self.config = self._load_config(storage_config)
        setup_kwargs = {}
        if self.config.tenant_id != DEFAULT_TENANT_ID:
            setup_kwargs["tenant_id"] = self.config.tenant_id
        try:
            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                16 * 1024 * 1024,  # Internal local buffer size
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
                **setup_kwargs,
            )
        except TypeError as e:
            if "tenant_id" in setup_kwargs and "tenant_id" in str(e):
                raise RuntimeError(
                    "The installed Mooncake version does not support tenant_id "
                    "in MooncakeDistributedStore.setup(). Please upgrade "
                    "Mooncake to use non-default Mooncake tenants with SGLang."
                ) from e
            raise
        if ret_code != 0:
            raise RuntimeError(f"Failed to setup Mooncake Embedding Store: {ret_code}")

        logger.info("Mooncake Embedding Store initialized successfully.")

    def get_key(self, mm_hash: str) -> str:
        return f"emb_{mm_hash}"

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return self._batch_get(hashes, ptrs, sizes, self.store.batch_get_into)

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        return self._batch_put(hashes, ptrs, sizes, self.store.batch_put_from)

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        keys = [self.get_key(h) for h in hashes]
        return [res == 1 for res in self._batch_exist(keys)]

    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        return self._batch_get(
            hashes, ptrs, sizes, self.store.batch_get_into_multi_buffers
        )

    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        return self._batch_put(
            hashes, ptrs, sizes, self.store.batch_put_from_multi_buffers
        )

    def _batch_get(self, hashes, ptrs, sizes, get_into) -> List[bool]:
        if not len(hashes) == len(ptrs) == len(sizes):
            raise ValueError("Mooncake key, pointer and size counts differ")
        if not all(map(self._valid_object_buffers, ptrs, sizes)):
            return [False] * len(hashes)
        keys = [self.get_key(h) for h in hashes]
        results = get_into(keys, ptrs, sizes)
        checked = self._checked_io_results(results, sizes, is_set=False)
        return [res > 0 for res in checked]

    def _batch_put(self, hashes, ptrs, sizes, put_from) -> List[bool]:
        if not len(hashes) == len(ptrs) == len(sizes):
            raise ValueError("Mooncake key, pointer and size counts differ")
        if not all(map(self._valid_object_buffers, ptrs, sizes)):
            return [False] * len(hashes)
        keys = [self.get_key(h) for h in hashes]
        exists = self._batch_exist(keys)
        if any(status not in (0, 1) for status in exists):
            return [False] * len(hashes)

        indices = [i for i, status in enumerate(exists) if status != 1]
        success_map = [True] * len(hashes)
        if indices:
            put_sizes = [sizes[i] for i in indices]
            results = put_from(
                [keys[i] for i in indices],
                [ptrs[i] for i in indices],
                put_sizes,
            )
            checked = self._checked_io_results(results, put_sizes, is_set=True)
            for i, res in zip(indices, checked):
                success_map[i] = res == 0
        return success_map
