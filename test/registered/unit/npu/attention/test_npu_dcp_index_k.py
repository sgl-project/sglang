import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import pool_configurator
from sglang.srt.hardware_backend.npu.attention import dsa_index_k
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuDcpIndexK(unittest.TestCase):
    def test_kimi_k31_dcp_requires_sharded_index_k(self):
        parallel = SimpleNamespace(attn_dcp_size=2, attn_cp_size=1)
        disagg = SimpleNamespace(
            disaggregation_mode="null", enable_pd_role_switch=False
        )
        kvc = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["KimiK31ForConditionalGeneration"], index_topk=2048
                )
            ),
            is_draft_worker=False,
            use_mla_backend=True,
            mambaish_config=object(),
            kv_cache_dtype=torch.bfloat16,
            modelslim_cache_layout=None,
        )
        scored_indexer = Mock(return_value=None)
        with patch.multiple(
            pool_configurator,
            get_parallel=lambda: parallel,
            get_memory=lambda: SimpleNamespace(
                enable_hierarchical_cache=False, enable_hisparse=False
            ),
            get_disagg=lambda: disagg,
            get_scored_indexer=scored_indexer,
        ):
            with self.assertRaisesRegex(RuntimeError, "scored lightning_indexer_v2"):
                pool_configurator.resolve_kimi_k31_index_k_sharding(kvc)
            scored_indexer.return_value = object()
            self.assertTrue(pool_configurator.resolve_kimi_k31_index_k_sharding(kvc))
            disagg.disaggregation_mode = "decode"
            with self.assertRaisesRegex(ValueError, "requires sharded Index-K"):
                pool_configurator.resolve_kimi_k31_index_k_sharding(kvc)
            parallel.attn_dcp_size = 1
            self.assertFalse(pool_configurator.resolve_kimi_k31_index_k_sharding(kvc))

    def test_prefill_page_plan_preserves_request_boundaries(self):
        pages, block_table = dsa_index_k.plan_index_k_prefill_gather(
            torch.tensor([[4, 3], [8, 99], [99, 99]], dtype=torch.int32),
            torch.tensor([9, 4, 0]),
            page_size=4,
            dcp_size=2,
        )
        self.assertEqual(pages.tolist(), [4, 3, 8])
        self.assertEqual(block_table.tolist(), [[0, 1, 2, 0], [4, 0, 0, 0], [0] * 4])

    def test_verify_preserves_fp32_scores_and_each_query_frontier(self):
        # Rank 1 owns positions 4..7 and sees 0, 1 and 2 keys for these rows.
        seq_lens = torch.tensor([0, 1, 2], dtype=torch.int32)
        local_indices = torch.tensor([[[0, -1]], [[0, 1]], [[1, 0]]], dtype=torch.int32)
        local_scores = torch.tensor([[[2.0, 0.0]], [[1.002, 2.0]], [[1.003, 1.002]]])
        peer_scores = torch.tensor([[1.001, 0.0]] * 3)
        peer_indices = torch.tensor([[0, 1]] * 3, dtype=torch.int32)
        peer_candidates = torch.stack((peer_scores.view(torch.int32), peer_indices), 1)
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            all_gather=lambda local, dim: torch.cat((peer_candidates, local), dim=dim),
        )
        scored_indexer = Mock(return_value=(local_indices, local_scores))
        with patch.object(
            dsa_index_k, "get_scored_indexer", return_value=scored_indexer
        ):
            result = dsa_index_k.score_sharded_index_k(
                torch.zeros((3, 1, 128), dtype=torch.bfloat16),
                torch.zeros((3, 4, 1, 128), dtype=torch.bfloat16),
                torch.zeros((3, 1), dtype=torch.bfloat16),
                2,
                seq_lens,
                torch.zeros((3, 1), dtype=torch.int32),
                group,
            )
        self.assertEqual(result.tolist(), [[0, 1], [4, 0], [5, 4]])
        kwargs = scored_indexer.call_args.kwargs
        self.assertIs(kwargs["seqused_k"], seq_lens)
        self.assertEqual(kwargs["cu_seqlens_q"].tolist(), [0, 1, 2, 3])
        self.assertEqual(kwargs["return_value"], 1)


if __name__ == "__main__":
    unittest.main()
