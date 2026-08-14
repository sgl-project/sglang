import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    get_dsa_fp8_packed_cache_dim,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.model_executor.pool_configurator import (
    _get_npu_dsa_indexer_layer_count,
    _get_npu_dsa_indexer_size_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_pool(**overrides):
    args = {
        "size": 4,
        "page_size": 2,
        "dtype": torch.bfloat16,
        "kv_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "index_head_dim": 16,
        "layer_num": 6,
        "device": "cpu",
        "enable_memory_saver": False,
        "start_layer": 10,
        "end_layer": 16,
        "indexer_layer_ids": (10, 12, 15),
    }
    args.update(overrides)
    return NPUMLATokenToKVPool(**args)


class TestNPUMLAIndexerMemoryPool(CustomTestCase):
    def test_compact_layout_maps_absolute_layers(self):
        pool = _make_pool()

        self.assertEqual(pool.indexer_layer_ids, (10, 12, 15))
        self.assertEqual(pool.num_indexer_layers, 3)
        self.assertEqual(pool.get_index_k_buffer(12).shape, (3, 2, 1, 16))
        with self.assertRaisesRegex(ValueError, "not a physical Indexer layer"):
            pool.get_index_k_buffer(11)

        expected_layer_ids = list(range(10, 16)) * 2 + [10, 12, 15]
        self.assertEqual(pool.get_kv_layer_ids(), expected_layer_ids)
        self.assertEqual(
            [len(pool._get_cpu_offload_layer_buffers(i)) for i in range(6)],
            [3, 2, 3, 2, 2, 3],
        )

    def test_legacy_none_allocates_every_local_layer(self):
        pool = _make_pool(indexer_layer_ids=None)
        self.assertEqual(pool.indexer_layer_ids, tuple(range(10, 16)))
        self.assertEqual(pool.num_indexer_layers, 6)

    def test_empty_mapping_allocates_no_indexer_slots(self):
        pool = _make_pool(indexer_layer_ids=())
        self.assertEqual(pool.index_k_buffer.shape[0], 0)
        self.assertEqual(len(pool.get_contiguous_buf_infos()[0]), 12)

    def test_fp8_scale_layout_and_byte_accounting(self):
        packed_cache_dim = get_dsa_fp8_packed_cache_dim(
            kv_lora_rank=512, qk_rope_head_dim=64
        )
        pool = _make_pool(
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            enable_npu_quant_lightning_indexer=True,
            kv_cache_dim=packed_cache_dim,
        )

        self.assertTrue(pool.dsa_kv_cache_store_fp8)
        self.assertEqual(pool.k_buffer.shape[-1], packed_cache_dim)
        self.assertEqual(pool.v_buffer.shape[-1], 0)
        self.assertEqual(pool.index_k_scale_buffer.dtype, torch.float32)
        self.assertEqual(pool.index_k_scale_buffer.shape, (3, 3, 2, 1))
        self.assertEqual(len(pool.get_contiguous_buf_infos()[0]), 18)
        self.assertEqual(pool.get_state_layer_ids(), [10, 12, 15] * 2)

        slots = 6
        expected_bytes = 6 * slots * packed_cache_dim + 3 * slots * (128 + 4)
        self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)

    def test_rejects_invalid_layer_mappings(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _make_pool(indexer_layer_ids=(10, 10))
        with self.assertRaisesRegex(ValueError, "in increasing"):
            _make_pool(indexer_layer_ids=(12, 10))
        with self.assertRaisesRegex(ValueError, "local stage range"):
            _make_pool(indexer_layer_ids=(9, 10))


class TestNPUIndexerPoolSizing(CustomTestCase):
    def test_indexer_bytes_match_storage_dtype_and_scale(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_topk=2048,
                index_head_dim=128,
            )
        )
        self.assertEqual(
            _get_npu_dsa_indexer_size_per_token(
                model_config, torch.bfloat16, has_scale_cache=False
            ),
            256,
        )
        self.assertEqual(
            _get_npu_dsa_indexer_size_per_token(
                model_config, torch.float8_e4m3fn, has_scale_cache=True
            ),
            132,
        )

    def test_compact_and_transfer_compatible_layer_counts(self):
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        kvc = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_mode="null", enable_hierarchical_cache=False
            ),
            is_draft_worker=False,
            model_config=SimpleNamespace(
                hf_config=hf_config, num_nextn_predict_layers=None
            ),
            layer_info=SimpleNamespace(start_layer=0, end_layer=78),
        )
        self.assertEqual(_get_npu_dsa_indexer_layer_count(kvc, 78), 21)

        kvc.server_args.disaggregation_mode = "decode"
        self.assertEqual(_get_npu_dsa_indexer_layer_count(kvc, 78), 78)
        kvc.server_args.disaggregation_mode = "null"
        kvc.server_args.enable_hierarchical_cache = True
        self.assertEqual(_get_npu_dsa_indexer_layer_count(kvc, 78), 78)


if __name__ == "__main__":
    unittest.main()
