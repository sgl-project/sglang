import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import (
    build_replayssm_staggered_force_flush_mask,
)
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FORBIDDEN_TOKENS = ("self.running_batch", "self.last_batch", "self.cur_batch")

DECISION_METHODS = (
    Scheduler.get_next_batch_to_run,
    Scheduler.get_new_batch_prefill,
    Scheduler._get_new_batch_prefill_raw,
    Scheduler._abort_on_running_timeout,
    Scheduler.is_disable_overlap_for_batch,
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run,
    SchedulerDisaggregationPrefillMixin.process_prefill_chunk,
    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch,
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run,
)


class TestDecisionMethodsHaveNoHiddenBatchChannel(unittest.TestCase):
    def test_decision_methods_take_batches_as_params_not_self(self):
        """The batch decision tree must receive running/last batch as params, never via self.*."""
        for method in DECISION_METHODS:
            source = inspect.getsource(inspect.unwrap(method))
            self.assertIn(
                f"def {method.__name__}",
                source,
                msg=f"failed to read the real source of {method.__qualname__}",
            )
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    source,
                    msg=(
                        f"{method.__qualname__} references {token}; pass the batch "
                        "explicitly and return it via NextBatchPlan instead."
                    ),
                )


class TestMtpPhaseBoundaryOverlap(unittest.TestCase):
    @staticmethod
    def _batch(*, is_extend: bool, is_speculative: bool = True):
        return SimpleNamespace(
            is_extend_in_batch=is_extend,
            forward_mode=SimpleNamespace(
                is_extend=lambda: is_extend,
                is_decode=lambda: not is_extend,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: not is_speculative),
            grammar_needs_sync=lambda: False,
        )

    def _scheduler(self, *, require_mlp_sync: bool):
        scheduler = object.__new__(Scheduler)
        scheduler.require_mlp_sync = require_mlp_sync
        scheduler.result_queue = [object()]
        return scheduler

    @patch(
        "sglang.srt.managers.scheduler.envs."
        "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get",
        return_value=False,
    )
    def test_mtp_phase_crossing_keeps_overlap(self, _disable_consecutive_prefill):
        extend = self._batch(is_extend=True)
        decode = self._batch(is_extend=False)

        for require_mlp_sync in (False, True):
            scheduler = self._scheduler(require_mlp_sync=require_mlp_sync)
            self.assertFalse(scheduler.is_disable_overlap_for_batch(decode, extend))
            self.assertFalse(scheduler.is_disable_overlap_for_batch(extend, decode))


class TestReplaySSMPhasePolicy(unittest.TestCase):
    def test_staggered_policy_balances_each_decode_iteration(self):
        batch_size = 64
        window = 16
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64)
        initial_seq_lens = torch.full((batch_size,), 100, dtype=torch.int64)
        masks = torch.stack(
            [
                build_replayssm_staggered_force_flush_mask(
                    initial_seq_lens + step, req_pool_indices, window
                )
                for step in range(window)
            ]
        )
        self.assertTrue(torch.equal(masks.sum(dim=1), torch.full((window,), 4)))
        self.assertTrue(torch.equal(masks.sum(dim=0), torch.ones(batch_size)))

    def test_staggered_policy_handles_noncontiguous_slots(self):
        seq_lens = torch.tensor([7, 19, 33, 65], dtype=torch.int64)
        req_pool_indices = torch.tensor([11, 3, 29, 40], dtype=torch.int64)
        mask = build_replayssm_staggered_force_flush_mask(
            seq_lens, req_pool_indices, window=8
        )
        expected = torch.remainder(seq_lens + req_pool_indices, 8) == 0
        self.assertTrue(torch.equal(mask, expected))

    def test_staggered_policy_validates_inputs(self):
        with self.assertRaisesRegex(ValueError, "window must be >= 1"):
            build_replayssm_staggered_force_flush_mask(
                torch.zeros(1), torch.zeros(1), window=0
            )
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            build_replayssm_staggered_force_flush_mask(
                torch.zeros(2), torch.zeros(1), window=8
            )


if __name__ == "__main__":
    unittest.main()
