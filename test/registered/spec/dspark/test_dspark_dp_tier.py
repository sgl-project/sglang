import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestDraftDpSyncMetadata(CustomTestCase):
    def test_gpu_only_draft_keeps_cpu_lengths_absent(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
        from sglang.test.test_utils import published_topology

        # The proposer resolves its logits layout from the published topology.
        self.enterContext(published_topology())
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        def forward(batch):
            self.assertIsNone(batch.seq_lens_cpu)
            self.assertIsNone(batch.seq_lens_sum)
            torch.testing.assert_close(batch.seq_lens, torch.tensor([10]))
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=hidden), can_run_graph=True
            )

        runner = SimpleNamespace(
            device="cpu",
            attn_backend=object.__new__(TRTLLMHAAttnBackend),
            decode_cuda_graph_runner=None,
            kv_index_translator=None,
            forward=forward,
        )
        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(sample_from_anchor=False),
            draft_model_runner=runner,
            gamma=2,
            mask_token_id=1,
            draft_block_spec_info=None,
            tp_sync=None,
            random_seed=0,
            greedy_draft=False,
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10]),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            req_pool_indices=torch.tensor([0]),
            can_run_decode_cuda_graph=True,
            is_extend_in_batch=False,
            dp_spec_prefill_coordination_applied=False,
        )
        window = SimpleNamespace(
            positions_2d=torch.tensor([[10, 11, 12]]),
            verify_cache_loc_2d=torch.tensor([[10, 11, 12]]),
            kv_loc_plan=SimpleNamespace(bind=Mock()),
        )
        for reserved_cpu in (None, torch.tensor([100])):
            result = proposer._run_forward(
                batch=batch,
                draft_input=SimpleNamespace(
                    bonus_tokens=torch.tensor([2]),
                    nxt_kv_lens_cpu=reserved_cpu,
                    nxt_kv_lens_sum=100,
                ),
                verify_window=window,
                bs=1,
                device="cpu",
                embed_module=torch.nn.Embedding(16, 2),
            )
            torch.testing.assert_close(result.raw_hidden, hidden[1:])

    def test_preserves_unscaled_request_counts_for_cuda_graph_admission(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")
        proposer._num_token_non_padded = torch.empty((1,), dtype=torch.int32)

        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_decode_cuda_graph=True,
            is_extend_in_batch=False,
            dp_spec_prefill_coordination_applied=False,
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
        # The LOCAL count reuses the proposer's persistent buffer; the GLOBAL
        # count is the invariant every DP rank agrees on.
        self.assertIs(
            forward_batch.num_token_non_padded, proposer._num_token_non_padded
        )
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 6)
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
