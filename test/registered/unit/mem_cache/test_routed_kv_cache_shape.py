"""Shape discovery on a unified pool whose pipeline stage starts on a recurrent
layer: the stage's first layer has no token-KV route, and its first KV layer has
another global id (an MLA/KDA hybrid with `--dcp-size > 1` asks this before the
forward). The routed cache answers with the owner view's buffer shape."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.test_utils import CustomTestCase


class TestRoutedKVCacheShape(CustomTestCase):
    def test_a_stage_starting_on_a_recurrent_layer_reports_the_owner_shape(self):
        mla = MLATokenToKVPool(
            size=32,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        # Layers 0-2 of the stage are recurrent; full attention is layer 3.
        cache = RoutedKVCache(
            arena=types.SimpleNamespace(device="cpu"),
            owner="full",
            layer_maps=[("full", mla, {3: 0})],
            page_size=1,
            start_layer=0,
            end_layer=None,
            size=32,
            head_num=1,
            head_dim=6,
        )
        self.assertEqual(cache.get_kv_buffer_shape(), mla.get_kv_buffer_shape())
        with self.assertRaises(KeyError):
            cache.get_kv_buffer(cache.start_layer)


if __name__ == "__main__":
    unittest.main()
