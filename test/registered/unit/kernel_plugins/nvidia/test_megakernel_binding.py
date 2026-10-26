"""CPU checks of the megakernel binding: the view is the running forward's attention
backend and KV pool, and a source replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def set_swa_key_buffer_radix_fused_norm_rope(self, **kw):
        return kw


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata="core"),
        trtllm_attn=False,
        forward=lambda *a, **k: "attention",
        store_cache=lambda **k: None,
        get_swa_out_cache_loc=lambda fb: "swa_loc",
        forward_low_ratio_sources=lambda **k: "sources",
        forward_core_compressor=lambda **k: "compressed",
    )


class TestMegakernelBinding(CustomTestCase):
    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = Pool()
        with patch(BACKEND, return_value=backend(pool)):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])

    def test_a_replaced_source_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        with patch(BACKEND, return_value=attn_backend):
            binding.forward_view()
            self.assertIs(binding.engram_history(), model.engram_hasher.history)
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch(BACKEND, return_value=backend(Pool())):
            binding.forward_view()
        with patch(BACKEND, return_value=backend(Pool())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()


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
