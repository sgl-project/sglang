import logging
import os

import torch
import torch.distributed
from aibrix_kvcache.common.absl_logging import log_every_n_seconds
from aibrix_kvcache_storage import AibrixKVCacheStorage

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "63886"


class AIBrixKVCacheStorageTest:
    def test_with_page_size(self):
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="test",
        )
        for page_size in range(1, 3):
            logger.info(f"page_size: {page_size}")
            batch_size = 2
            head_num = 1
            layer_num = 64
            head_dim = 128
            kv_cache = MHATokenToKVPool(
                1024,
                page_size,
                torch.float16,
                head_num,
                head_dim,
                layer_num,
                "cpu",
                False,
                0,
                layer_num,
            )
            mem_pool = MHATokenToKVPoolHost(kv_cache, 2, 0, page_size, "layer_first")
            query_length = batch_size * 2
            partial = batch_size
            self.aibrix_kvcache = AibrixKVCacheStorage(config, mem_pool)
            target_shape = (2, layer_num, page_size, head_num, head_dim)
            rand_tensor = [
                torch.rand(target_shape, dtype=torch.float16)
                for _ in range(query_length)
            ]
            keys = ["hash" + str(i) for i in range(query_length)]
            partial_keys = keys[batch_size:query_length]
            assert self.aibrix_kvcache.batch_exists(keys) == 0
            assert self.aibrix_kvcache.batch_set(keys, rand_tensor)
            get_tensor = [
                torch.rand(target_shape, dtype=torch.float16).flatten()
                for _ in range(query_length)
            ]
            self.aibrix_kvcache.batch_get(keys, get_tensor)
            for i in range(query_length):
                assert torch.equal(get_tensor[i], rand_tensor[i].flatten())
            ret = self.aibrix_kvcache.batch_exists(keys)
            assert self.aibrix_kvcache.batch_exists(keys) == query_length
            assert self.aibrix_kvcache.batch_exists(partial_keys) == partial
            partial_get_tensor = [
                torch.rand(target_shape, dtype=torch.float16).flatten()
                for _ in range(partial)
            ]
            self.aibrix_kvcache.batch_get(partial_keys, partial_get_tensor)
            for i in range(partial):
                assert torch.equal(
                    partial_get_tensor[i], rand_tensor[i + partial].flatten()
                )
            log_every_n_seconds(
                logger,
                logging.INFO,
                self.aibrix_kvcache.kv_cache_manager.metrics.summary(),
                1,
            )


if __name__ == "__main__":
    setup()
    test = AIBrixKVCacheStorageTest()
    test.test_with_page_size()
