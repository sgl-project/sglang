import unittest
from types import SimpleNamespace

import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class BooleanMaskBackend:
    @staticmethod
    def apply_vocab_mask(*, logits, vocab_mask):
        logits.masked_fill_(vocab_mask, float("-inf"))


class TestDFlashVerifyGrammarMask(CustomTestCase):
    def make_sampling_info(self):
        return SamplingBatchInfo(
            temperatures=torch.ones(2, 1),
            top_ps=torch.ones(2),
            top_ks=torch.ones(2, dtype=torch.int32),
            min_ps=torch.zeros(2),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=4,
            device="cpu",
        )

    def make_stale_mask(self, rows=2):
        # A previous extend batch can leave every token masked out, or even have
        # a different number of requests from the current verify batch.
        return GrammarMask(BooleanMaskBackend(), torch.ones(rows, 4, dtype=torch.bool))

    def adjust(self, logits, sampling_info):
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=2,
        )

    def test_stale_mask_does_not_poison_unconstrained_verify(self):
        for dtype in (torch.float32, torch.bfloat16):
            for stale_rows in (2, 3):
                with self.subTest(dtype=dtype, stale_rows=stale_rows):
                    sampling_info = self.make_sampling_info()
                    sampling_info.grammar_mask = self.make_stale_mask(stale_rows)
                    logits = torch.tensor(
                        [[0, 1, 4, 2], [1, 0, 5, 2], [0, 1, 2, 4], [1, 0, 2, 5]],
                        dtype=dtype,
                    )
                    expected = logits.clone()

                    self.adjust(logits, sampling_info)

                    torch.testing.assert_close(logits, expected)
                    self.assertEqual(logits.argmax(dim=-1).tolist(), [2, 2, 3, 3])
                    self.assertIsNone(sampling_info.grammar_mask)

    def test_current_verify_mask_is_applied_per_position(self):
        sampling_info = self.make_sampling_info()
        sampling_info.grammar_mask = self.make_stale_mask()
        logits = torch.tensor([[0.0, 1.0, 4.0, 2.0]] * 4)
        # Each verify-tree position has its own grammar state. Do not broadcast
        # the previous per-request mask over these positions.
        current_mask = GrammarMask(
            BooleanMaskBackend(),
            torch.tensor(
                [
                    [True, False, True, True],
                    [True, True, False, True],
                    [True, True, True, False],
                    [False, True, True, True],
                ]
            ),
        )

        self.adjust(logits, sampling_info)
        current_mask.apply(logits)

        self.assertEqual(logits.argmax(dim=-1).tolist(), [1, 2, 3, 0])
        torch.testing.assert_close(
            logits[torch.arange(4), torch.tensor([1, 2, 3, 0])],
            torch.tensor([1.0, 4.0, 2.0, 0.0]),
        )
        self.assertEqual(torch.isneginf(logits).sum().item(), 12)

    def test_live_penalties_and_logit_bias_survive_stale_mask(self):
        for stale_mask in (None, self.make_stale_mask()):
            with self.subTest(has_stale_mask=stale_mask is not None):
                sampling_info = self.make_sampling_info()
                sampling_info.grammar_mask = stale_mask
                # Only the external penalizer is replaced; apply_logits_bias is
                # the real SamplingBatchInfo implementation.
                penalty = torch.tensor([[-1.0, 0, -2, 0], [0, -1, 0, -2]])
                sampling_info.penalizer_orchestrator = SimpleNamespace(
                    is_required=True, apply=lambda logits: logits.add_(penalty)
                )
                sampling_info.logit_bias = torch.tensor([[0.0, 3, 0, 0], [0, 0, 3, 0]])
                logits = torch.ones(4, 4)

                self.adjust(logits, sampling_info)

                torch.testing.assert_close(
                    logits,
                    torch.tensor(
                        [[0.0, 4, -1, 1], [0, 4, -1, 1], [1, 0, 4, -1], [1, 0, 4, -1]]
                    ),
                )
                self.assertIsNone(sampling_info.grammar_mask)

    def test_logit_bias_without_live_penalizer_survives_stale_mask(self):
        sampling_info = self.make_sampling_info()
        sampling_info.grammar_mask = self.make_stale_mask()
        sampling_info.logit_bias = torch.tensor([[0.0, 2, 0, 0], [0, 0, 3, 0]])
        logits = torch.ones(4, 4)

        self.adjust(logits, sampling_info)

        torch.testing.assert_close(
            logits,
            torch.tensor([[1.0, 3, 1, 1], [1, 3, 1, 1], [1, 1, 4, 1], [1, 1, 4, 1]]),
        )
        self.assertIsNone(sampling_info.grammar_mask)

    def test_precomputed_penalties_survive_stale_mask(self):
        sampling_info = self.make_sampling_info()
        sampling_info.grammar_mask = self.make_stale_mask()
        sampling_info.acc_linear_penalties = torch.tensor(
            [[-1.0, 0, -2, 0], [0, -1, 0, -2]]
        )
        sampling_info.logit_bias = torch.tensor([[0.0, 3, 0, 0], [0, 0, 3, 0]])
        logits = torch.ones(4, 4, dtype=torch.bfloat16)

        self.adjust(logits, sampling_info)

        torch.testing.assert_close(
            logits,
            torch.tensor(
                [[0, 4, -1, 1], [0, 4, -1, 1], [1, 0, 4, -1], [1, 0, 4, -1]],
                dtype=torch.bfloat16,
            ),
        )
        self.assertIsNone(sampling_info.grammar_mask)


if __name__ == "__main__":
    unittest.main()
