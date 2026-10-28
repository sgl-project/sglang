# SPDX-License-Identifier: Apache-2.0
"""Clover penalty history, row alignment and request validation contracts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.clover_config import validate_clover_request
from sglang.srt.speculative.clover_penalty import (
    CloverPenaltyState,
    apply_clover_penalties,
)
from sglang.srt.speculative.clover_utils import clover_correct_drafts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request(history=(), **params):
    return SimpleNamespace(
        sampling_params=SamplingParams(**params),
        output_ids=list(history),
        origin_input_ids=[5, 5, 5],
        eos_token_ids={7},
        tokenizer=None,
        return_logprob=False,
        return_sampling_mask=False,
        custom_logit_processor=None,
        input_embeds=None,
        positional_embed_overrides=None,
        return_hidden_states=False,
    )


class TestCloverPenalties(CustomTestCase):
    def test_default_bias_resolution_partial_override_and_explicit_disable(self):
        class Tokenizer:
            unk_token_id = None

            def encode(self, text, add_special_tokens=False):
                return [6]

            def __len__(self):
                return 8

        req = request([1] * 4096, custom_params={"unrelated": True})
        req.tokenizer = Tokenizer()
        self.assertIsNone(validate_clover_request(req))
        custom = req.sampling_params.custom_params
        self.assertEqual(custom["dllm_length_penalty"], 10)
        self.assertEqual(custom["dllm_think_end_bias"], 10)
        self.assertEqual(custom["dllm_think_end_start"], 4096)
        self.assertEqual(custom["dllm_think_end_token_id"], 6)
        self.assertTrue(custom["unrelated"])
        logits = torch.zeros(1, 4, 8)
        apply_clover_penalties(logits, [req], torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(logits[0, 0, 6].item(), 10 / 4096)
        self.assertEqual(logits[0, :, 7].count_nonzero().item(), 0)
        disabled = request(
            [1], custom_params={"dllm_length_penalty": 0, "dllm_think_end_bias": 0}
        )
        self.assertIsNone(validate_clover_request(disabled))
        scores = torch.randn(1, 4, 8)
        before = scores.clone()
        apply_clover_penalties(scores, [disabled], torch.ones(1, 4))
        torch.testing.assert_close(scores, before)

    def test_cached_history_across_preview_verify_reorder_and_retraction(self):
        state = CloverPenaltyState()
        a = request([1, 1], frequency_penalty=0.5, presence_penalty=1)
        b = request([2], frequency_penalty=1)

        def check(reqs, visible):
            actual = torch.zeros(len(reqs), 2, 8)
            apply_clover_penalties(actual, reqs, torch.tensor(visible), state=state)
            expected = torch.zeros_like(actual)
            for row, req in enumerate(reqs):
                sp = req.sampling_params
                for token in range(8):
                    count = req.output_ids.count(token)
                    seen = count > 0
                    expected[row, :, token] = (
                        -count * sp.frequency_penalty - seen * sp.presence_penalty
                    )
            torch.testing.assert_close(actual, expected)

        check([a, b], [[1, 0], [2, 0]])
        check([a, b], [[1, 3], [2, 4]])  # Uncommitted visible drafts.
        a.output_ids.extend([4, 4])
        check([b, a], [[2, 0], [4, 0]])  # Reorder and incremental counts.
        del a.output_ids[1:]
        check([a], [[1, 0]])  # In-place truncation, dropped request.
        a.output_ids = [6]  # Same-length replacement must invalidate the cache.
        check([a], [[6, 0]])
        check([b, a], [[2, 5], [6, 0]])
        c = request([7], frequency_penalty=0.25)
        check([c], [[7, 0]])  # Reused workspace, new request.

    def test_hard_minimum_survives_zero_acceptance_threshold(self):
        req = request([1], min_new_tokens=3)
        candidates = torch.tensor([[1, 7, 2, 3]])
        logits = torch.zeros(1, 4, 8)
        apply_clover_penalties(logits, [req], candidates)
        self.assertEqual(clover_correct_drafts(logits, candidates, 0, 0.3).item(), 0)

    def test_committed_counts_only_and_additive_before_scaling(self):
        req = request(
            [1, 1, 2], frequency_penalty=0.5, presence_penalty=1, repetition_penalty=2
        )
        logits = torch.tensor([[[4.0, 4.0, -2.0, 4.0, 4.0, 4.0, 4.0, 4.0]]]).repeat(
            1, 4, 1
        )
        apply_clover_penalties(logits, [req], torch.tensor([[2, 3, 3, 0]]))
        expected = torch.tensor([4.0, 1.0, -7.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        torch.testing.assert_close(logits[0], expected.expand(4, -1))
        self.assertEqual(req.output_ids, [1, 1, 2])

    def test_length_ramp_alignment_stop_union_and_minimum(self):
        req = request(
            [1, 2, 3],
            min_new_tokens=5,
            stop_token_ids=[6],
            custom_params={
                "dllm_length_penalty": 8.0,
                "dllm_length_penalty_start": 3,
                "dllm_length_penalty_window": 4,
            },
        )
        preview = torch.zeros(1, 4, 8)
        verify = torch.zeros_like(preview)
        visible = torch.tensor([[3, 0, 0, 0]])
        apply_clover_penalties(preview, [req], visible, preview=True)
        apply_clover_penalties(verify, [req], visible)
        torch.testing.assert_close(preview[:, 1:], verify[:, :-1])
        torch.testing.assert_close(
            verify[0, :, 7], torch.tensor([-float("inf"), -float("inf"), 6.0, 8.0])
        )
        torch.testing.assert_close(verify[0, :, 6], verify[0, :, 7])
        self.assertFalse(torch.isnan(verify).any())
        free = request(
            [1, 2, 3],
            ignore_eos=True,
            stop_token_ids=[6],
            custom_params=req.sampling_params.custom_params,
        )
        unbiased = torch.zeros_like(verify)
        apply_clover_penalties(unbiased, [free], visible)
        self.assertFalse(unbiased[0, :, 6:].any())

    def test_think_bias_first_position_and_committed_close(self):
        req = request(
            [1, 2],
            custom_params={
                "dllm_think_end_bias": 4.0,
                "dllm_think_end_start": 2,
                "dllm_think_end_window": 4,
                "dllm_think_end_token_id": 6,
            },
        )
        for preview, visible, expected in (
            (True, [2, 0, 0, 0], [0.0, 1.0, 0.0, 0.0]),
            (True, [2, 6, 0, 0], [0.0, 0.0, 0.0, 0.0]),
            (False, [2, 6, 3, 4], [1.0, 0.0, 0.0, 0.0]),
        ):
            logits = torch.zeros(1, 4, 8)
            apply_clover_penalties(
                logits, [req], torch.tensor([visible]), preview=preview
            )
            torch.testing.assert_close(logits[0, :, 6], torch.tensor(expected))
        req.output_ids.append(6)
        logits = torch.zeros(1, 4, 8)
        apply_clover_penalties(logits, [req], torch.tensor([[6, 1, 2, 3]]))
        self.assertEqual(logits.count_nonzero().item(), 0)

    def test_invalid_custom_controls_rejected(self):
        for key, value in (
            ("dllm_length_penalty", float("nan")),
            ("dllm_think_end_bias", -1),
            ("dllm_length_penalty_start", 1.5),
            ("dllm_length_penalty_window", 0),
            ("dllm_think_end_token_id", -1),
            ("dllm_think_end_bias", 1),  # No tokenizer or explicit token.
        ):
            with self.subTest(key=key, value=value):
                self.assertIsNotNone(
                    validate_clover_request(
                        request(custom_params={"dllm_think_end_bias": 0, key: value})
                    )
                )


if __name__ == "__main__":
    unittest.main()
