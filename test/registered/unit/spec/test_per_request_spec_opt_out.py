"""Unit tests for per-request speculative decoding opt-out behavior."""

from types import SimpleNamespace
from unittest.mock import patch
import torch

from sglang.srt.managers.schedule_batch import ForwardMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPerRequestSpecOptOut(CustomTestCase):
    def test_eagle_sample_clamps_opted_out_requests_to_single_token(self):
        """In a mixed batch, opted-out requests must always have accept_lens clamped to 1."""
        bs = 3
        draft_token_num = 4
        vocab_size = 64

        # Request 0: regular spec (can accept draft tokens)
        # Request 1: opt-out spec (must only accept 1 token, i.e. target bonus)
        # Request 2: regular spec (can accept draft tokens)
        spec_opt_out_mask = torch.tensor([False, True, False], dtype=torch.bool)

        sampling_info = SimpleNamespace(
            temperatures=torch.ones(bs, 1),
            top_ks=torch.ones(bs, dtype=torch.int32),
            top_ps=torch.ones(bs),
            min_ps=torch.zeros(bs),
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            sampling_seed=None,
            has_watermark_candidates=False,
            has_spec_opt_out=True,
            is_all_spec_opt_out=False,
            spec_opt_out_mask=spec_opt_out_mask,
        )

        batch = SimpleNamespace(
            device=torch.device("cpu"),
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int64),
            sampling_info=sampling_info,
            forward_mode=ForwardMode.TARGET_VERIFY,
            has_grammar=False,
            hisparse_coordinator=None,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        )

        candidates = torch.full((bs, draft_token_num), 42, dtype=torch.int64)
        verify_input = SimpleNamespace(
            draft_token=candidates,
            draft_token_num=draft_token_num,
            max_tree_depth=draft_token_num,
            tree_topk=1,
            retrieve_index=torch.arange(bs).unsqueeze(1).repeat(1, draft_token_num),
            retrieve_next_token=torch.full((bs, draft_token_num), -1, dtype=torch.long),
            retrieve_next_sibling=torch.full((bs, draft_token_num), -1, dtype=torch.long),
            custom_mask=None,
            positions=None,
            draft_probs=None,
        )

        # Build dummy target logits such that all candidates would match (argmax == 42)
        next_token_logits = torch.zeros(bs, draft_token_num, vocab_size)
        next_token_logits[:, :, 42] = 100.0  # argmax is 42
        logits_output = SimpleNamespace(next_token_logits=next_token_logits)

        spec_config = SimpleNamespace(speculative_use_rejection_sampling=False)
        with torch.no_grad(), patch("sglang.srt.speculative.eagle_utils.get_spec", return_value=spec_config):
            predict, accept_lens, accept_index = eagle_sample(
                verify_input=verify_input,
                batch=batch,
                logits_output=logits_output,
            )

        # Without opt-out, greedy matching on identical candidates would accept draft tokens.
        # But Request 1 has disable_speculative_decoding=True, so accept_lens[1] MUST be 1.
        self.assertEqual(accept_lens[1].item(), 1)
        # Non-opted out requests accept more than 1 token
        self.assertGreaterEqual(accept_lens[0].item(), 1)
        self.assertGreaterEqual(accept_lens[2].item(), 1)
