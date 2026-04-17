"""Unit tests for ReasoningEosRedirectLogitProcessor.

No server / no model loading. All tensors are CPU-only and tiny.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.custom_logit_processor import (
    CustomLogitProcessor,
    ReasoningEosRedirectLogitProcessor,
)
from sglang.test.test_utils import CustomTestCase


def _make_req(origin_input_ids=None, output_ids=None, temperature=1.0):
    sampling_params = SimpleNamespace(temperature=temperature)
    return SimpleNamespace(
        origin_input_ids=list(origin_input_ids or []),
        output_ids=list(output_ids or []),
        sampling_params=sampling_params,
    )


def _make_logits(values, vocab=10, fill=-10.0):
    """Build a (1, vocab) logits tensor with specific positions set."""
    logits = torch.full((1, vocab), fill, dtype=torch.float32)
    for tok, v in values.items():
        logits[0, tok] = v
    return logits


class TestReasoningEosRedirectSerialization(CustomTestCase):
    def test_to_str_round_trip(self):
        s = ReasoningEosRedirectLogitProcessor.to_str()
        proc = CustomLogitProcessor.from_str(s)
        self.assertIsInstance(proc, ReasoningEosRedirectLogitProcessor)


class TestReasoningEosRedirectTriggers(CustomTestCase):
    def setUp(self):
        self.processor = ReasoningEosRedirectLogitProcessor()
        self.base_params = {
            "think_end_token_id": 7,
            "think_start_token_id": 8,
            "redirect_eos_token_ids": [5, 6],
            "prob_threshold": 0.5,
        }

    def _params(self, req, **overrides):
        merged = dict(self.base_params)
        merged.update(overrides)
        merged["__req__"] = req
        return [merged]

    def test_redirect_triggers_when_eos_prob_above_threshold(self):
        # EOS tokens dominate (logit 5.0 vs others -10/0), so softmax mass
        # on {5,6} is ~1.0. Should trigger.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0, 1: 0.5})
        req = _make_req(origin_input_ids=[8, 1, 2, 3])
        out = self.processor(logits.clone(), self._params(req))
        self.assertEqual(out[0, 5].item(), float("-inf"))
        self.assertEqual(out[0, 6].item(), float("-inf"))
        self.assertEqual(int(torch.argmax(out[0]).item()), 7)

    def test_skip_when_eos_prob_below_threshold(self):
        # Token 1 dominates; EOS mass is negligible.
        logits = _make_logits({5: 0.0, 6: 0.0, 7: 0.0, 1: 5.0})
        req = _make_req(origin_input_ids=[8])
        original = logits.clone()
        out = self.processor(logits.clone(), self._params(req))
        self.assertTrue(torch.equal(out, original))

    def test_skip_when_already_left_reasoning(self):
        # think_end token already in output_ids → skip even if EOS dominates.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[8], output_ids=[7, 1, 2])
        original = logits.clone()
        out = self.processor(logits.clone(), self._params(req))
        self.assertTrue(torch.equal(out, original))

    def test_skip_when_not_yet_in_reasoning(self):
        # No think_start_id present anywhere, force_reasoning=False.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[1, 2, 3], output_ids=[])
        original = logits.clone()
        out = self.processor(logits.clone(), self._params(req))
        self.assertTrue(torch.equal(out, original))

    def test_force_reasoning_triggers_without_think_start(self):
        # Model with no think_start in chat template; force_reasoning=True.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[1, 2, 3], output_ids=[])
        out = self.processor(
            logits.clone(),
            self._params(req, think_start_token_id=None, force_reasoning=True),
        )
        self.assertEqual(int(torch.argmax(out[0]).item()), 7)

    def test_temperature_scales_eos_probability(self):
        # Without temperature scaling (T=1):
        #   logits {5:1.0, 7:0.0, 1:0.5, others:-10}
        #   softmax mass on {5} is ~0.4 (below threshold 0.5).
        # With T=0.1, logits effectively become {5:10.0, 7:0.0, 1:5.0, ...}
        #   so EOS mass ~ 1.0 (well above 0.5). Should trigger.
        logits = _make_logits({5: 1.0, 7: 0.0, 1: 0.5})
        # First, verify the no-trigger case at T=1:
        req_t1 = _make_req(origin_input_ids=[8], temperature=1.0)
        out_t1 = self.processor(
            logits.clone(),
            [
                {
                    "__req__": req_t1,
                    "think_end_token_id": 7,
                    "think_start_token_id": 8,
                    "redirect_eos_token_ids": [5],
                    "prob_threshold": 0.5,
                }
            ],
        )
        self.assertNotEqual(out_t1[0, 5].item(), float("-inf"))

        # Now with T=0.1, should trigger.
        req_t01 = _make_req(origin_input_ids=[8], temperature=0.1)
        out_t01 = self.processor(
            logits.clone(),
            [
                {
                    "__req__": req_t01,
                    "think_end_token_id": 7,
                    "think_start_token_id": 8,
                    "redirect_eos_token_ids": [5],
                    "prob_threshold": 0.5,
                }
            ],
        )
        self.assertEqual(out_t01[0, 5].item(), float("-inf"))
        self.assertEqual(int(torch.argmax(out_t01[0]).item()), 7)

    def test_think_start_in_origin_ids_counts_as_in_reasoning(self):
        # Some chat templates prefill <think> in the prompt. Detection should
        # use both origin_input_ids and output_ids.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[100, 8])  # think_start in prompt
        out = self.processor(logits.clone(), self._params(req))
        self.assertEqual(int(torch.argmax(out[0]).item()), 7)

    def test_empty_eos_set_does_nothing(self):
        logits = _make_logits({5: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[8])
        original = logits.clone()
        out = self.processor(
            logits.clone(),
            [
                {
                    "__req__": req,
                    "think_end_token_id": 7,
                    "think_start_token_id": 8,
                    "redirect_eos_token_ids": [],
                    "prob_threshold": 0.5,
                }
            ],
        )
        self.assertTrue(torch.equal(out, original))

    def test_missing_required_params_skip_safely(self):
        logits = _make_logits({5: 5.0, 7: 0.0})
        req = _make_req(origin_input_ids=[8])
        original = logits.clone()
        # Missing think_end_token_id
        out = self.processor(
            logits.clone(),
            [
                {
                    "__req__": req,
                    "think_start_token_id": 8,
                    "redirect_eos_token_ids": [5],
                    "prob_threshold": 0.5,
                }
            ],
        )
        self.assertTrue(torch.equal(out, original))

    def test_only_changes_eos_and_think_end_indices(self):
        # All other token logits must be untouched.
        logits = _make_logits({5: 5.0, 6: 5.0, 7: 0.0, 1: 0.5, 2: 0.3, 3: 0.1})
        req = _make_req(origin_input_ids=[8])
        out = self.processor(logits.clone(), self._params(req))
        # Untouched indices keep original value
        for tok in (1, 2, 3, 4, 8, 9):
            self.assertAlmostEqual(out[0, tok].item(), logits[0, tok].item(), places=5)
        # EOS is masked
        self.assertEqual(out[0, 5].item(), float("-inf"))
        self.assertEqual(out[0, 6].item(), float("-inf"))
        # think_end is strictly the largest
        self.assertEqual(int(torch.argmax(out[0]).item()), 7)

    def test_batch_independent_processing(self):
        # Two requests in one batch: only the first one should be redirected.
        logits = torch.full((2, 10), -10.0, dtype=torch.float32)
        # row 0: EOS dominates, in reasoning → trigger
        logits[0, 5] = 5.0
        logits[0, 6] = 5.0
        logits[0, 7] = 0.0
        # row 1: EOS dominates, but already past reasoning (output_ids has think_end)
        logits[1, 5] = 5.0
        logits[1, 6] = 5.0
        logits[1, 7] = 0.0

        req0 = _make_req(origin_input_ids=[8])
        req1 = _make_req(origin_input_ids=[8], output_ids=[7])
        params = [
            {**self.base_params, "__req__": req0},
            {**self.base_params, "__req__": req1},
        ]
        out = self.processor(logits.clone(), params)
        # row 0 redirected
        self.assertEqual(out[0, 5].item(), float("-inf"))
        self.assertEqual(int(torch.argmax(out[0]).item()), 7)
        # row 1 unchanged
        self.assertEqual(out[1, 5].item(), 5.0)
        self.assertEqual(out[1, 7].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
