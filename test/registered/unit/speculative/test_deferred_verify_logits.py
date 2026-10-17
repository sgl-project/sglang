"""Deferred graph logits must follow the current request, including fallback."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.deferred_verify_logits import (
    can_reduce_verify_logits,
    finalize_verify_logits,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request():
    return SimpleNamespace(
        return_logprob=False,
        has_grammar=False,
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            has_watermark_candidates=False,
            penalizer_orchestrator=None,
        ),
    )


class TestDeferredVerifyLogits(CustomTestCase):
    def setUp(self):
        super().setUp()
        for setting in (
            envs.SGLANG_SANITIZE_NAN_LOGITS,
            envs.SGLANG_ENABLE_NAN_LOGITS_CHECK,
            envs.SGLANG_ENABLE_ASYNC_ASSERT,
        ):
            context = setting.override(False)
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def test_request_changes_reject_reduction(self):
        # Reusing one captured graph must not reuse a prior greedy policy.
        batch = request()
        self.assertTrue(can_reduce_verify_logits(batch))
        for owner, field, value in (
            (batch, "return_logprob", True),
            (batch, "has_grammar", True),
            (batch.sampling_info, "is_all_greedy", False),
            (batch.sampling_info, "has_custom_logit_processor", True),
            (batch.sampling_info, "has_watermark_candidates", True),
            (batch.sampling_info, "acc_additive_penalties", torch.zeros(1)),
            (batch.sampling_info, "acc_scaling_penalties", torch.ones(1)),
            (batch.sampling_info, "logit_bias", torch.zeros(1)),
            (batch.sampling_info, "grammar_mask", object()),
            (batch.sampling_info, "sampling_mask_batch_indices", torch.zeros(1)),
            (
                batch.sampling_info,
                "sampling_support_logprobs_capture_indices",
                torch.zeros(1),
            ),
            (
                batch.sampling_info,
                "penalizer_orchestrator",
                SimpleNamespace(is_required=True),
            ),
        ):
            with self.subTest(field=field):
                previous = getattr(owner, field, None)
                setattr(owner, field, value)
                self.assertFalse(can_reduce_verify_logits(batch))
                setattr(owner, field, previous)
                self.assertTrue(can_reduce_verify_logits(batch))
        self.assertFalse(can_reduce_verify_logits(batch, grammar_mask=object()))

    def test_diagnostic_modes_materialize_full_logits(self):
        for setting in (
            envs.SGLANG_SANITIZE_NAN_LOGITS,
            envs.SGLANG_ENABLE_NAN_LOGITS_CHECK,
            envs.SGLANG_ENABLE_ASYNC_ASSERT,
        ):
            with self.subTest(setting=str(setting)), setting.override(True):
                self.assertFalse(can_reduce_verify_logits(request()))

    def test_logprob_fallback_restores_scores_without_mutating_graph_storage(self):
        # Subsequent logprob/penalty consumers need all FP32 scores; retaining
        # only winning IDs or softcapping borrowed BF16 storage breaks replay.
        batch = request()
        batch.return_logprob = True
        raw = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.bfloat16)
        output = SimpleNamespace(
            next_token_logits=None,
            deferred_verify_logits=raw,
            deferred_verify_softcap=None,
        )
        self.assertIsNone(finalize_verify_logits(output, batch))
        self.assertIsNone(output.deferred_verify_logits)
        self.assertEqual(output.next_token_logits.dtype, torch.float32)
        torch.testing.assert_close(output.next_token_logits, raw.float())
        output.next_token_logits.add_(5)
        torch.testing.assert_close(
            raw, torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
