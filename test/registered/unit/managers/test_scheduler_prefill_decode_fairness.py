"""Regression tests for decode starvation under sustained prefill load (#32549)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(*, empty: bool, prefill_only: bool = False, decoding_reqs=None):
    batch = MagicMock()
    batch.is_empty.return_value = empty
    batch.is_prefill_only = prefill_only
    batch.batch_is_full = False
    batch.reqs = [] if empty else [object()]
    batch.decoding_reqs = decoding_reqs
    return batch


def _scheduler(*, max_consecutive_prefills: int, speculative: bool) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._abort_on_waiting_timeout = MagicMock()
    scheduler._abort_on_running_timeout = MagicMock()
    scheduler.dllm_config = None
    scheduler.dllm_manager = None
    scheduler.enable_hisparse = False
    scheduler.enable_fpm = False
    scheduler.require_mlp_sync = speculative
    scheduler.spec_algorithm = MagicMock()
    scheduler.spec_algorithm.is_none.return_value = not speculative
    scheduler.server_args = SimpleNamespace(speculative_skip_dp_mlp_sync=False)
    scheduler.dp_attn_adapter = MagicMock()
    scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
        lambda batch, **_: batch
    )
    scheduler.ngram_embedding_manager = MagicMock()
    scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
        lambda batch, **_: batch
    )
    scheduler.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    scheduler.tree_cache = MagicMock()
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.is_mixed_chunk = False
    scheduler.max_consecutive_prefill_batches = max_consecutive_prefills
    scheduler.consecutive_prefill_batches = 0
    return scheduler


class TestPrefillDecodeFairness(CustomTestCase):
    def _assert_bounded_prefill_streak(self, *, speculative: bool):
        scheduler = _scheduler(max_consecutive_prefills=2, speculative=speculative)
        running_batch = _batch(empty=False)
        prefill_batch = _batch(empty=False)
        scheduler.get_new_batch_prefill = MagicMock(
            side_effect=lambda running: NextBatchPlan(
                batch_to_run=prefill_batch, running_batch=running
            )
        )

        selected = []
        for _ in range(6):
            plan = Scheduler.get_next_batch_to_run(
                scheduler, running_batch=running_batch, last_batch=None
            )
            selected.append(
                "decode" if plan.batch_to_run is running_batch else "prefill"
            )

        self.assertEqual(
            selected,
            ["prefill", "prefill", "decode", "prefill", "prefill", "decode"],
        )
        self.assertEqual(scheduler.get_new_batch_prefill.call_count, 4)

    def test_bounds_prefill_streak(self):
        self._assert_bounded_prefill_streak(speculative=False)

    def test_bounds_prefill_streak_with_speculative_decode(self):
        self._assert_bounded_prefill_streak(speculative=True)

    def test_zero_disables_fairness_limit(self):
        scheduler = _scheduler(max_consecutive_prefills=0, speculative=True)
        running_batch = _batch(empty=False)
        prefill_batch = _batch(empty=False)
        scheduler.get_new_batch_prefill = MagicMock(
            side_effect=lambda running: NextBatchPlan(
                batch_to_run=prefill_batch, running_batch=running
            )
        )

        selected = [
            Scheduler.get_next_batch_to_run(
                scheduler, running_batch=running_batch, last_batch=None
            ).batch_to_run
            for _ in range(4)
        ]

        self.assertTrue(all(batch is prefill_batch for batch in selected))
        self.assertEqual(scheduler.consecutive_prefill_batches, 0)

    def test_active_chunked_request_continues_prefill(self):
        scheduler = _scheduler(max_consecutive_prefills=1, speculative=True)
        scheduler.consecutive_prefill_batches = 1
        scheduler.chunked_req = MagicMock()
        scheduler.chunked_req.extend_range.end = 0
        scheduler.chunked_req.prefix_indices = []
        running_batch = _batch(empty=False)
        prefill_batch = _batch(empty=False)
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=NextBatchPlan(
                batch_to_run=prefill_batch, running_batch=running_batch
            )
        )

        plan = Scheduler.get_next_batch_to_run(
            scheduler, running_batch=running_batch, last_batch=None
        )

        self.assertIs(plan.batch_to_run, prefill_batch)

        scheduler.chunked_req = None
        plan = Scheduler.get_next_batch_to_run(
            scheduler, running_batch=running_batch, last_batch=None
        )
        self.assertIs(plan.batch_to_run, running_batch)

    def test_mixed_prefill_that_advances_decode_resets_streak(self):
        scheduler = _scheduler(max_consecutive_prefills=1, speculative=False)
        scheduler.is_mixed_chunk = True
        running_batch = _batch(empty=False)
        mixed_batch = _batch(empty=False, decoding_reqs=[object()])
        scheduler.get_new_batch_prefill = MagicMock(
            return_value=NextBatchPlan(
                batch_to_run=mixed_batch, running_batch=running_batch
            )
        )

        plan = Scheduler.get_next_batch_to_run(
            scheduler, running_batch=running_batch, last_batch=None
        )

        self.assertIs(plan.batch_to_run, mixed_batch)
        self.assertEqual(scheduler.consecutive_prefill_batches, 0)


if __name__ == "__main__":
    unittest.main()
