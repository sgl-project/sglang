"""Contracts for KPool writes using metadata prepared before or after padding."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaIndexerKPoolPadding(CustomTestCase):
    def _run_write(self, physical, logical, plan_rows, width=6):
        indexer = object.__new__(dsa_indexer_kpool.IndexerKPool)
        indexer.index_kpool_compress_ape = torch.empty(4, 128)
        indexer.scale_fmt = None
        key = torch.randn(physical, 128)
        score = torch.randn_like(key)
        indexer._get_q_k_bf16 = MagicMock(return_value=(None, key, score))
        indexer._compute_gate_score_if_missing = MagicMock(return_value=score)
        pool = MagicMock()
        pool.get_compress_tail_buffers.return_value = (None, None)
        plan = SimpleNamespace(
            num_draft_tokens=width,
            req=torch.arange(plan_rows),
            write_start=torch.full((plan_rows,), 23, dtype=torch.int32),
            tail_logical_start=torch.full((plan_rows,), 20, dtype=torch.int32),
            write_loc=torch.arange(plan_rows * 2).view(plan_rows, 2),
            effective_n_per_batch=None,
        )
        metadata = SimpleNamespace(attn_metadata=SimpleNamespace(kpool_write_plan=plan))
        out_cache_loc = torch.arange(physical) + 100
        if logical is not None:
            out_cache_loc[logical:] = 0
        forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=logical,
            out_cache_loc=out_cache_loc,
        )
        with (
            patch.object(dsa_indexer_kpool, "is_cuda", return_value=True),
            patch.object(dsa_indexer_kpool, "get_token_to_kv_pool", return_value=pool),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index."
                "kpool_write_tail_and_maybe_compress"
            ) as write,
        ):
            indexer._forward_cuda_target_verify(
                x=torch.randn(physical, 64),
                q_lora=torch.randn(physical, 32),
                positions=torch.arange(physical),
                forward_batch=forward_batch,
                layer_id=7,
                act_quant=MagicMock(),
                metadata=metadata,
                enable_dual_stream=False,
                return_indices=False,
            )
        kwargs = write.call_args.kwargs
        # Request metadata belongs to the planner, including its dummy groups.
        for argument, field in (
            ("req_pool_indices", "req"),
            ("write_start", "write_start"),
            ("tail_logical_start", "tail_logical_start"),
            ("write_loc", "write_loc"),
            ("effective_n_per_batch", "effective_n_per_batch"),
        ):
            self.assertIs(kwargs[argument], getattr(plan, field))
        planned = plan_rows * width
        torch.testing.assert_close(kwargs["key"], key[:planned])
        torch.testing.assert_close(kwargs["score"], score[:planned])
        torch.testing.assert_close(kwargs["out_cache_loc"], out_cache_loc[:planned])
        return kwargs

    def test_target_verify_preserves_complete_padding_groups(self):
        # TP8 / width6 aligns 228 tokens to 240, with two dummy requests.
        kwargs = self._run_write(physical=240, logical=228, plan_rows=40)
        self.assertEqual(kwargs["key"].shape[0], 240)
        self.assertEqual(kwargs["out_cache_loc"][228:].count_nonzero().item(), 0)

    def test_draft_extend_uses_pre_padding_plan(self):
        # Draft extend retains the plan built for three real requests.
        kwargs = self._run_write(physical=24, logical=18, plan_rows=3)
        self.assertEqual(kwargs["key"].shape[0], 18)

    def test_draft_extend_accepts_partial_physical_padding_group(self):
        self._run_write(physical=8, logical=6, plan_rows=1)

    def test_unpadded_write(self):
        self._run_write(physical=12, logical=12, plan_rows=2)

    def test_graph_bucket_does_not_depend_on_host_logical_length(self):
        self._run_write(physical=24, logical=None, plan_rows=4)

    def test_empty_plan(self):
        self._run_write(physical=8, logical=0, plan_rows=0)

    def test_plan_cannot_exceed_input(self):
        with self.assertRaisesRegex(AssertionError, "more token rows than its input"):
            self._run_write(physical=8, logical=6, plan_rows=2)


if __name__ == "__main__":
    unittest.main()
