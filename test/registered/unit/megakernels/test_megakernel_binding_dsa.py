# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the megakernel binding's DSA (sparse MLA) view: it is the running
DSA backend and its pool, other backends have none, and a replaced decode buffer is
caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.megakernels import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


class DsaPool:
    index_page_size = 64

    def __init__(self):
        self.keys = [torch.zeros(4, 1, 8) for _ in range(2)]
        self.index_k = [torch.zeros(4, 12) for _ in range(2)]

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_k[layer_id]

    def set_mla_kv_buffer(self, layer, loc, k, k_rope):
        return layer, loc, k, k_rope


def dsa_backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata="metadata",
        dsa_decode_impl="trtllm",
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        kv_cache_dim=656,
        physical_page_size=64,
        use_mha=False,
        kv_cache_dtype=torch.float8_e4m3fn,
        use_fused_topk=True,
        dsa_topk_backend="topk backend",
        dcp_size=1,
        dsa_index_kpool=1,
        hisparse_coordinator=None,
        workspace_buffer=torch.zeros(16, dtype=torch.uint8),
        _multi_ctas_kv_counter_buffer=torch.zeros(8, dtype=torch.uint8),
        get_indexer_metadata=lambda layer_id, fb: ("indexer", layer_id, fb),
    )


class TestMegakernelBindingDsa(CustomTestCase):
    def test_the_dsa_view_is_the_running_dsa_backend(self):
        backend = dsa_backend(DsaPool())
        with patch(BACKEND, return_value=backend):
            dsa = mb.MegakernelBinding(model=torch.nn.Module()).forward_view().dsa
        same = {
            "kv_lora_rank": "kv_lora_rank",
            "qk_nope_head_dim": "qk_nope_head_dim",
            "qk_rope_head_dim": "qk_rope_head_dim",
            "kv_cache_dim": "kv_cache_dim",
            "physical_page_size": "physical_page_size",
            "decode_impl": "dsa_decode_impl",
            "use_mha": "use_mha",
            "kv_cache_dtype": "kv_cache_dtype",
            "use_fused_topk": "use_fused_topk",
            "topk_backend": "dsa_topk_backend",
            "dcp_size": "dcp_size",
            "index_kpool": "dsa_index_kpool",
            "workspace_buffer": "workspace_buffer",
            "multi_ctas_kv_counter_buffer": "_multi_ctas_kv_counter_buffer",
            "indexer_metadata": "get_indexer_metadata",
        }
        for member, attr in same.items():
            self.assertIs(getattr(dsa, member), getattr(backend, attr), member)
        self.assertFalse(dsa.hisparse)
        backend.use_mha = True  # read live: the backend sets it per forward
        self.assertTrue(dsa.use_mha)
        self.assertEqual(dsa.indexer_metadata(3, "fb"), ("indexer", 3, "fb"))

    def test_a_backend_without_dsa_has_no_dsa_view(self):
        backend = SimpleNamespace(token_to_kv_pool=DsaPool(), forward_metadata=None)
        with patch(BACKEND, return_value=backend):
            binding = mb.MegakernelBinding(model=torch.nn.Module())
            self.assertIsNone(binding.forward_view().dsa)
            binding.assert_live()

    def test_the_pool_view_offers_the_mla_and_index_k_buffers(self):
        pool = DsaPool()
        view = mb.KvPoolView(pool)
        self.assertIs(view.get_key_buffer(1), pool.keys[1])
        self.assertIs(view.get_index_k_with_scale_buffer(0), pool.index_k[0])
        self.assertEqual(view.index_page_size, 64)
        self.assertEqual(view.set_mla_kv_buffer(1, 2, 3, 4), (1, 2, 3, 4))
        # a pool without index-K pages
        self.assertIsNone(mb.KvPoolView(SimpleNamespace()).index_page_size)

    def test_a_replaced_dsa_buffer_fails_assert_live(self):
        for attr, name in (
            ("workspace_buffer", "dsa_workspace"),
            ("_multi_ctas_kv_counter_buffer", "dsa_kv_counter"),
        ):
            with self.subTest(buffer=attr):
                backend = dsa_backend(DsaPool())
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                with patch(BACKEND, return_value=backend):
                    binding.forward_view()
                    binding.assert_live()
                    setattr(backend, attr, torch.zeros(32, dtype=torch.uint8))
                    with self.assertRaisesRegex(RuntimeError, name):
                        binding.assert_live()


if __name__ == "__main__":
    unittest.main()
