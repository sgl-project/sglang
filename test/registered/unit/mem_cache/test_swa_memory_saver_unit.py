import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class _FakeTokenToKVPool:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).created.append(self)

    def get_kv_size_bytes(self):
        return 0, 0

    def get_contiguous_buf_infos(self):
        return [], [], []


class TestSWAMemorySaverPropagation(CustomTestCase):
    def setUp(self):
        _FakeTokenToKVPool.created.clear()

    def test_enable_memory_saver_is_propagated(self):
        adapter = object()
        with patch(
            "sglang.srt.mem_cache.swa_memory_pool.TorchMemorySaverAdapter.create",
            return_value=adapter,
        ) as create_adapter:
            pool = SWAKVPool(
                size=16,
                size_swa=8,
                page_size=1,
                dtype=torch.float16,
                head_num=8,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0, 3],
                enable_kvcache_transpose=False,
                device="cpu",
                token_to_kv_pool_class=_FakeTokenToKVPool,
                enable_memory_saver=True,
            )

        create_adapter.assert_called_once_with(enable=True)
        self.assertIs(pool.memory_saver_adapter, adapter)
        self.assertEqual(len(_FakeTokenToKVPool.created), 2)
        self.assertTrue(_FakeTokenToKVPool.created[0].kwargs["enable_memory_saver"])
        self.assertTrue(_FakeTokenToKVPool.created[1].kwargs["enable_memory_saver"])

    def test_enable_memory_saver_defaults_to_false(self):
        adapter = object()
        with patch(
            "sglang.srt.mem_cache.swa_memory_pool.TorchMemorySaverAdapter.create",
            return_value=adapter,
        ) as create_adapter:
            pool = SWAKVPool(
                size=16,
                size_swa=8,
                page_size=1,
                dtype=torch.float16,
                head_num=8,
                head_dim=128,
                swa_attention_layer_ids=[1, 2],
                full_attention_layer_ids=[0, 3],
                enable_kvcache_transpose=False,
                device="cpu",
                token_to_kv_pool_class=_FakeTokenToKVPool,
            )

        create_adapter.assert_called_once_with(enable=False)
        self.assertIs(pool.memory_saver_adapter, adapter)
        self.assertEqual(len(_FakeTokenToKVPool.created), 2)
        self.assertFalse(_FakeTokenToKVPool.created[0].kwargs["enable_memory_saver"])
        self.assertFalse(_FakeTokenToKVPool.created[1].kwargs["enable_memory_saver"])


if __name__ == "__main__":
    unittest.main()