import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components import dspark_planner
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalVerifyTierNumTokens(CustomTestCase):
    def test_no_budget_returns_sentinel(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=None,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            -1,
        )

    def test_budget_adds_to_anchor_floor(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=10,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            18,
        )

    # Clamp/floor variants (verify-all clamp, min_verify_len floor, min=0) are
    # covered by the TestBusyIdleGraphKeyIdentity sweep bounds.


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestDpTierGatherAdmission(CustomTestCase):
    def test_pd_decode_can_enable_compact_dp_tier_gather(self):
        spec = SimpleNamespace(
            speculative_dspark_align_verify_tokens_to_graph_tier=False,
            speculative_dspark_confidence_sts_path=None,
            speculative_dspark_sps_table_path=None,
            speculative_skip_dp_mlp_sync=False,
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_cp_size=1, pp_size=1)
        schedule = SimpleNamespace(disable_overlap_schedule=False)
        budget_planner = SimpleNamespace(lag_steps=2)

        with (
            patch.object(dspark_planner, "get_spec", return_value=spec),
            patch.object(dspark_planner, "get_parallel", return_value=parallel),
            patch.object(dspark_planner, "get_schedule", return_value=schedule),
            patch.object(
                dspark_planner,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
                create=True,
            ),
            patch.object(
                dspark_planner,
                "read_ragged_verify_mode",
                return_value=RaggedVerifyMode.COMPACT,
            ),
            patch.object(dspark_planner, "is_dp_attention_enabled", return_value=True),
            patch.object(dspark_planner, "require_mlp_tp_gather", return_value=True),
            patch.object(dspark_planner, "build_sps_cost_table", return_value=object()),
            patch.object(
                dspark_planner, "is_uninitialized_sps_table", return_value=False
            ),
            patch.object(
                dspark_planner,
                "HostConfidenceBudgetPlanner",
                return_value=budget_planner,
            ),
            patch.object(
                dspark_planner.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER,
                "get",
                return_value=False,
            ),
        ):
            planner = DSparkVerifyPlanner(
                draft_model=SimpleNamespace(confidence_head=object()),
                gamma=5,
                model_runner=SimpleNamespace(),
                device="cpu",
                tp_rank=1,
                verify_num_draft_tokens=6,
                tp_sync=SimpleNamespace(),
            )

        self.assertTrue(planner._dp_tier_gather_enabled)


class TestDraftDpSyncMetadata(CustomTestCase):
    def test_preserves_unscaled_request_counts_for_cuda_graph_admission(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")

        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_decode_cuda_graph=True,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer._fill_dp_moe_sync_metadata(forward_batch, batch)

        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            [1, 3, 0, 2],
        )
        self.assertEqual(forward_batch.global_num_tokens_cpu, [6, 18, 0, 12])
        # Metadata fill sets only the invariant GLOBAL count; the LOCAL
        # num_token_non_padded is derived later when the draft forward localizes.
        self.assertEqual(forward_batch.global_num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.global_num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.global_num_token_non_padded_cpu, 6)
        self.assertTrue(forward_batch.can_run_decode_cuda_graph)


class TestBusyIdleGraphKeyIdentity(CustomTestCase):
    def test_busy_and_idle_floors_agree_on_random_topologies(self):
        rng = random.Random(20260703)
        for _ in range(2000):
            verify_num_draft_tokens = rng.randint(2, 8)
            min_verify_len = rng.randint(0, verify_num_draft_tokens - 1)
            effective_min = max(min_verify_len, 1)
            num_ranks = rng.randint(1, 8)
            contributions = []
            num_reqs_per_rank = []
            for _ in range(num_ranks):
                if rng.random() < 0.3:
                    num_reqs_per_rank.append(0)
                    contributions.append(0)
                    continue
                bs = rng.randint(1, 512)
                budget = rng.randint(0, bs * verify_num_draft_tokens)
                num_reqs_per_rank.append(bs)
                contributions.append(
                    local_verify_tier_num_tokens(
                        bs=bs,
                        verify_token_budget=budget,
                        verify_num_draft_tokens=verify_num_draft_tokens,
                        min_verify_len=min_verify_len,
                    )
                )
            tier_num_tokens = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            )
            global_num_reqs = max(num_reqs_per_rank)
            if tier_num_tokens is None:
                self.assertEqual(global_num_reqs, 0)
                continue

            self.assertGreaterEqual(tier_num_tokens, global_num_reqs * effective_min)
            self.assertLessEqual(
                tier_num_tokens, global_num_reqs * verify_num_draft_tokens
            )

            busy_floor = min(tier_num_tokens, global_num_reqs * verify_num_draft_tokens)
            self.assertEqual(busy_floor, tier_num_tokens)

            idle_lens_total = global_num_reqs
            idle_bucket_input = max(idle_lens_total, tier_num_tokens)
            self.assertEqual(idle_bucket_input, tier_num_tokens)


if __name__ == "__main__":
    unittest.main()
