import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"


class TestDSV4NonPagedIndexer(unittest.TestCase):
    def _is_eligible(self, **overrides):
        backend = SimpleNamespace(hisparse_coordinator=None)
        c4_indexer = SimpleNamespace(use_fp4_indexer=overrides.get("fp4", False))
        forward_batch = SimpleNamespace(
            forward_mode=overrides.get("mode", ForwardMode.EXTEND),
            _original_forward_mode=None,
            tbo_parent_token_range=overrides.get("tbo"),
            batch_size=overrides.get("batch_size", 1),
        )
        metadata = SimpleNamespace(
            use_prefill_cuda_graph=overrides.get("prefill_graph", False)
        )
        with (
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(
                overrides.get("enabled", True)
            ),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch(f"{_INDEXER}.is_hip", return_value=False),
            patch(f"{_INDEXER}.get_attention_cp_size", return_value=1),
            patch(
                f"{_INDEXER}.is_in_tc_piecewise_cuda_graph",
                return_value=overrides.get("piecewise_graph", False),
            ),
            patch(f"{_INDEXER}.is_in_breakable_cuda_graph", return_value=False),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            return C4IndexerBackendMixin._can_use_nonpaged_indexer(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=forward_batch,
                indexer_metadata=metadata,
            )

    def test_eligibility_is_fail_closed(self):
        self.assertIs(envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.default, False)
        self.assertTrue(self._is_eligible())
        for case in (
            {"enabled": False},
            {"mode": ForwardMode.DECODE},
            {"batch_size": 2},
            {"tbo": (1, 2)},
            {"prefill_graph": True},
            {"piecewise_graph": True},
            {"fp4": True},
        ):
            with self.subTest(case=case):
                self.assertFalse(self._is_eligible(**case))

    def test_single_request_plan_contract(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        query_rows = 4
        batch = SimpleNamespace(
            seq_lens=torch.tensor([262], dtype=torch.int32),
            seq_lens_cpu=[262],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        page_table = torch.tensor([[3, 1]], dtype=torch.int32).repeat(query_rows, 1)
        c4_seq_lens = torch.tensor([62, 63, 64, 65], dtype=torch.int32)

        def build_plan():
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        plan = build_plan()
        self.assertEqual(
            (plan.seq_len_sum, plan.max_seqlen_k, plan.query_rows),
            (65, 128, query_rows),
        )
        torch.testing.assert_close(plan.page_table, page_table[:1])
        torch.testing.assert_close(plan.ke, c4_seq_lens)

        metadata.nonpaged_plan = None
        batch.extend_seq_lens_cpu = [2, 2]
        self.assertIsNone(build_plan())


if __name__ == "__main__":
    unittest.main()
