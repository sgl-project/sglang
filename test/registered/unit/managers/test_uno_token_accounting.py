"""CPU regressions for UNO aggregate token accounting."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    MetricsSink,
)
from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
from sglang.srt.managers.scheduler_components.step_counters import StepCounters
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestUnoTokenAccounting(CustomTestCase):
    def setUp(self):
        self.batch_size = 2
        # Two rows accepting 4 and 3 tokens, 2 of each not drafted: 3 drafts.
        self.result = GenerationBatchResult(
            accept_lens=torch.tensor([4, 3]),
            num_non_draft_tokens_per_req=2,
        )

    def test_generated_token_count_includes_both_non_draft_tokens(self):
        self.assertEqual(self.result.num_correct_drafts_per_req(), [2, 1])
        self.assertEqual(self.result.get_num_generated_tokens(self.batch_size), 7)
        one_bonus = GenerationBatchResult(accept_lens=torch.tensor([3, 2]))
        self.assertEqual(one_bonus.get_num_generated_tokens(self.batch_size), 5)
        # Without speculation each row generates its one token.
        self.assertEqual(
            GenerationBatchResult().get_num_generated_tokens(self.batch_size), 2
        )

    def test_spec_metrics_keep_generated_and_draft_counts_separate(self):
        reporter = MetricsSink.__new__(MetricsSink)
        reporter.spec_num_accept_tokens = 0
        reporter.spec_num_correct_drafts = 0
        reporter.spec_num_forward_ct = 0
        reporter.spec_num_block_accept_tokens = 0
        reporter.spec_num_cap_tokens = 0

        reporter.update_spec_metrics(
            self.batch_size,
            sum(self.result.num_correct_drafts_per_req()),
            num_accept_tokens=self.result.get_num_generated_tokens(self.batch_size),
        )

        self.assertEqual(reporter.spec_num_accept_tokens, 7)
        self.assertEqual(reporter.spec_num_correct_drafts, 3)
        self.assertEqual(reporter.spec_num_forward_ct, 2)

    def test_decode_moment_receives_full_generated_token_count(self):
        step_counters = StepCounters()
        step_counters._prev_step = (1, 10.0, False)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: False,
                is_decode=lambda: True,
                is_target_verify=lambda: False,
            ),
            reqs=[SimpleNamespace(rid="req-0"), SimpleNamespace(rid="req-1")],
            forward_iter=2,
        )

        step_counters.record(
            PendingStep(
                step=batch,
                result=self.result,
                launch_ts=10.001,
                after_idle_gap=False,
            )
        )

        self.assertEqual(step_counters.decode_moment_totals[5], 7)


if __name__ == "__main__":
    unittest.main()
