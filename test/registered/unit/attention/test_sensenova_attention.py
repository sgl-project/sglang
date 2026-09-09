import unittest

import torch
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    prepare_flash_kv_cache,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    NeoUnifyAttentionMask,
    Qwen3Attention,
    create_block_causal_mask,
    create_neo_attention_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestSenseNovaAttention(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.config = NEOLLMConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
        )
        self.config._attn_implementation = "eager"
        self.layer = Qwen3Attention(self.config, 0).eval()

    @torch.no_grad()
    def test_prefill_matches_original(self):
        ids = torch.tensor([0, 1, 1, 1, 2, 3, 3, 4])
        indexes = torch.stack((ids, torch.zeros_like(ids), torch.zeros_like(ids)))
        hidden = torch.randn(1, 8, 64)
        mask = create_neo_attention_mask(ids, "torch")
        self.assertIsInstance(mask, NeoUnifyAttentionMask)
        legacy_cache = DynamicCache(config=self.config)
        new_cache = DynamicCache(config=self.config)
        legacy, _ = self.layer.forward_und(
            hidden, indexes, create_block_causal_mask(ids), legacy_cache
        )
        actual, _ = self.layer.forward_und(hidden, indexes, mask, new_cache)
        torch.testing.assert_close(actual, legacy, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            new_cache.layers[0].keys, legacy_cache.layers[0].keys, rtol=0, atol=0
        )

    @torch.no_grad()
    def test_denoise_reuses_prefix_and_overwrites_suffix(self):
        prefix = torch.randn(1, 3, 64)
        ids = torch.arange(3)
        indexes = torch.stack((ids, torch.zeros_like(ids), torch.zeros_like(ids)))
        cache = DynamicCache(config=self.config)
        self.layer.forward_und(prefix, indexes, create_block_causal_mask(ids), cache)
        prepare_flash_kv_cache(cache, current_len=5, batch_size=1)
        layer_cache = cache.layers[0]
        original_k = layer_cache.flash_k_cache[:, :3].clone()
        original_v = layer_cache.flash_v_cache[:, :3].clone()
        pointer = layer_cache.flash_k_cache.data_ptr()
        gen_ids = torch.full((5,), 3)
        gen_indexes = torch.stack((gen_ids, torch.arange(5), torch.zeros_like(gen_ids)))
        for _ in range(3):
            hidden = torch.randn(1, 5, 64)
            self.config.neo_denoise_backend = "legacy"
            expected, _ = self.layer.forward_gen(
                hidden, gen_indexes, None, cache, update_cache=False
            )
            self.config.neo_denoise_backend = "torch"
            actual, _ = self.layer.forward_gen(
                hidden, gen_indexes, None, cache, update_cache=False
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertEqual(cache.get_seq_length(), 3)
            self.assertEqual(layer_cache.flash_k_cache.data_ptr(), pointer)
            torch.testing.assert_close(
                layer_cache.flash_k_cache[:, :3], original_k, rtol=0, atol=0
            )
            torch.testing.assert_close(
                layer_cache.flash_v_cache[:, :3], original_v, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
