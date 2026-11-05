import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.cp.bcg import interleave_bcg


class TestGlm52CPBCG(unittest.TestCase):
    def test_glm52_supported_configuration(self):
        config = SimpleNamespace(
            dp_size=1,
            ep_size=1,
            tp_size=4,
            device="cuda",
            attn_cp_size=4,
            dsa_prefill_backend="trtllm",
            kv_cache_dtype="fp8_e4m3",
            dcp_size=1,
            enable_two_batch_overlap=False,
            enable_lora=False,
            lora_paths=None,
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                architectures=["GlmMoeDsaForCausalLM"],
            )
        )
        with (
            patch.object(interleave_bcg, "resolving_view", return_value=config),
            patch.object(interleave_bcg, "resolved_view", return_value=config),
            patch.object(interleave_bcg, "model_config_of", return_value=model),
        ):
            supports = interleave_bcg.InterleaveCPBCGInput.supports
            self.assertTrue(supports(config))
            for cp_size in (2, 4, 8):
                with self.subTest(cp_size=cp_size):
                    config.tp_size = config.attn_cp_size = cp_size
                    self.assertTrue(supports(config))
            config.tp_size = config.attn_cp_size = 4
            for field, value in (
                ("dp_size", 2),
                ("ep_size", 2),
                ("dcp_size", 2),
                ("device", "cpu"),
                ("dsa_prefill_backend", "flashmla_sparse"),
                ("kv_cache_dtype", "bf16"),
                ("enable_two_batch_overlap", True),
                ("enable_lora", True),
                ("lora_paths", ["adapter"]),
            ):
                with self.subTest(field=field, value=value):
                    old = getattr(config, field)
                    setattr(config, field, value)
                    self.assertFalse(supports(config))
                    setattr(config, field, old)
            # Preserve the pre-existing GLM-5.3 adapter gate.
            model.hf_config.architectures = ["Glm5NextForConditionalGeneration"]
            config.tp_size = config.attn_cp_size = 8
            self.assertTrue(supports(config))

    def test_index_cache_uses_live_global_rows(self):
        self._check_index_store(key_rows=6, graph_tokens=8, expected_rows=6)

    def test_eager_index_cache_preserves_all_rows(self):
        self._check_index_store(key_rows=8, graph_tokens=None, expected_rows=8)

    def _check_index_store(self, key_rows, graph_tokens, expected_rows):
        batch = SimpleNamespace(
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            extend_seq_lens_cpu=[2, 4],
            attn_cp_metadata=SimpleNamespace(graph_num_tokens=graph_tokens),
        )
        key = torch.zeros(key_rows, 128, dtype=torch.bfloat16)
        buffer = object()
        pool = SimpleNamespace(index_page_size=64)
        with (
            patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
            patch.object(dsa_indexer, "_is_cuda", True),
            patch.object(dsa_indexer, "_is_fp8_fnuz", False),
            patch.object(dsa_indexer, "can_use_dsa_fused_store", return_value=True),
            patch.object(
                dsa_indexer,
                "_index_k_write_targets",
                side_effect=lambda pool, layer, loc: [(buffer, loc, None)],
            ),
            patch.object(dsa_indexer, "fused_store_index_k_cache") as store,
        ):
            dsa_indexer.Indexer._store_index_k_cache(
                SimpleNamespace(),
                batch,
                0,
                key,
            )
        actual_key, actual_buffer, locations, page_size = store.call_args.args
        self.assertIs(actual_key, key)
        self.assertIs(actual_buffer, buffer)
        self.assertEqual(page_size, 64)
        self.assertEqual(locations.shape[0], expected_rows)
        torch.testing.assert_close(locations, batch.out_cache_loc[:expected_rows])
        self.assertEqual(batch.out_cache_loc.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
