"""On A3 the index-K budget and the pool must elide the same layers.

[Test Category] Correctness
[Test Target] mem_cache/kv_cache_configurator.py  (_build_ascend_mla_kv_pool)
              model_executor/pool_configurator.py (_compute_dsa_indexer_cell_size)

If only one side elides, the pool overruns its budget or wastes the elided rows.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor import pool_configurator
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

INDEX_HEAD_DIM = 128
# GLM-5.2-like: only some layers own an Indexer, the rest reuse top-k.
INDEXER_TYPES = [
    "full",
    "shared",
    "shared",
    "full",
    "shared",
    "shared",
    "full",
    "shared",
]
NUM_LAYERS = len(INDEXER_TYPES)
INDEXER_LAYERS = (0, 3, 6)


def _kvc():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_topk=2048,
                index_head_dim=INDEX_HEAD_DIM,
                indexer_types=INDEXER_TYPES,
            ),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=INDEX_HEAD_DIM,
        ),
        layer_info=types.SimpleNamespace(
            start_layer=0, end_layer=NUM_LAYERS, num_effective_layers=NUM_LAYERS
        ),
        server_args=types.SimpleNamespace(enable_hisparse=False),
        mambaish_config=None,
        is_draft_worker=False,
        kv_cache_dtype=torch.bfloat16,
        device="cpu",
    )


class TestNpuIndexKElision(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(page_size=128)
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.hardware_backend.npu.utils.is_npu_arch35", return_value=False)
    @patch.object(pool_configurator, "_is_npu", True)
    def test_a3_budget_and_pool_count_the_same_indexer_layers(self, _):
        kvc = _kvc()
        with patch(
            "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMLATokenToKVPool"
        ) as pool_cls:
            KVCacheConfigurator._build_ascend_mla_kv_pool(
                kvc, max_total_num_tokens=1024, is_dsa_model=True
            )
        self.assertEqual(pool_cls.call_args.kwargs["indexer_layer_ids"], INDEXER_LAYERS)

        cell_size = (
            pool_configurator.DefaultPoolConfigurator._compute_dsa_indexer_cell_size(
                None, kvc=kvc, num_layers=NUM_LAYERS
            )
        )
        bf16_bytes = 2
        self.assertEqual(cell_size, INDEX_HEAD_DIM * len(INDEXER_LAYERS) * bf16_bytes)


if __name__ == "__main__":
    unittest.main()
