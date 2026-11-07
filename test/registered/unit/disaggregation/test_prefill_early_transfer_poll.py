"""Early polling retires old KV transfers before the next admission pass."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEarlyPrefillPoll(CustomTestCase):
    def run_admission(self, enabled, pp_size):
        events = []
        running = SimpleNamespace(batch_is_full=True)

        def admit(batch):
            events.append("admit")
            return SimpleNamespace(running_batch=batch, batch_to_run=None)

        scheduler = SimpleNamespace(
            process_disagg_prefill_transfer_queue=lambda: events.append("poll"),
            process_pending_chunked_abort=lambda: None,
            _process_hicache_events=lambda **kwargs: None,
            resolve_waiting_queue_bootstrap=lambda: None,
            process_prefill_chunk=lambda **kwargs: None,
            get_new_batch_prefill=admit,
            dp_attn_adapter=SimpleNamespace(
                maybe_prepare_mlp_sync_batch=lambda batch: batch
            ),
        )
        method = inspect.unwrap(
            SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run
        )
        with (
            patch.object(
                envs.SGLANG_DISAGG_PREFILL_POLL_TRANSFERS_BEFORE_FORWARD,
                "get",
                return_value=enabled,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_parallel",
                return_value=SimpleNamespace(pp_size=pp_size),
            ),
        ):
            result = method(scheduler, running, None)
        self.assertIs(result.running_batch, running)
        return events

    def test_poll_precedes_admission(self):
        self.assertEqual(self.run_admission(True, 1), ["poll", "admit"])

    def test_disabled_policy_uses_existing_late_poll(self):
        self.assertEqual(self.run_admission(False, 1), ["admit"])

    def test_pipeline_parallel_defers_to_existing_consensus(self):
        self.assertEqual(self.run_admission(True, 2), ["admit"])

    def retirement_before_next_work(self, loop_name, enabled, pp_size, idle):
        """An older transfer completes while the next batch is being prepared."""

        class StopLoop(Exception):
            pass

        transfer = SimpleNamespace(complete=False, retired=False)
        observed = []
        batch = None if idle else SimpleNamespace()
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.gracefully_exit = False
        scheduler.scheduler_stage_metrics = None
        scheduler.increment_scheduler_iterations = lambda: None
        scheduler.ingest_requests = lambda **kwargs: False
        scheduler._engine_paused = False
        scheduler._deferred_input_requests = []
        scheduler.waiting_queue = SimpleNamespace(join=lambda reqs: None)
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=lambda: []
        )
        scheduler.running_batch = SimpleNamespace(batch_is_full=False)
        scheduler.last_batch = None
        scheduler.chunked_req = None
        scheduler.enable_staging = False
        scheduler.process_pending_chunked_abort = lambda: None
        scheduler._process_hicache_events = lambda **kwargs: None
        scheduler.resolve_waiting_queue_bootstrap = lambda: None
        scheduler.process_prefill_chunk = lambda **kwargs: None
        scheduler.get_new_batch_prefill = lambda running: SimpleNamespace(
            running_batch=running, batch_to_run=batch
        )
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda value: value
        )
        scheduler._is_continuous_input_polling_enabled = lambda: False
        scheduler._is_transfer_wait_polling_enabled = lambda: False

        def prepare(value, **kwargs):
            transfer.complete = True
            return value

        def poll():
            if transfer.complete:
                transfer.retired = True

        def next_work(*args):
            observed.append(transfer.retired)
            raise StopLoop

        scheduler.ngram_embedding_manager = SimpleNamespace(prepare_for_forward=prepare)
        scheduler.run_batch = next_work
        scheduler.on_idle = next_work
        with (
            patch.object(scheduler, "process_disagg_prefill_transfer_queue", poll),
            envs.SGLANG_DISAGG_PREFILL_POLL_TRANSFERS_BEFORE_FORWARD.override(enabled),
            patch(
                "sglang.srt.disaggregation.prefill.get_parallel",
                return_value=SimpleNamespace(pp_size=pp_size),
            ),
            patch("sglang.srt.disaggregation.prefill.set_schedule_time_batch"),
            self.assertRaises(StopLoop),
        ):
            getattr(scheduler, loop_name)()
        self.assertEqual(len(observed), 1)
        return observed[0]

    def test_transfer_completed_during_preparation_retires_before_forward_or_idle(self):
        for loop in ("normal", "overlap"):
            for idle in (False, True):
                with self.subTest(loop=loop, idle=idle):
                    self.assertTrue(
                        self.retirement_before_next_work(
                            f"event_loop_{loop}_disagg_prefill", True, 1, idle
                        )
                    )

    def test_post_preparation_poll_respects_disabled_and_pipeline_guards(self):
        for loop in ("normal", "overlap"):
            for enabled, pp_size in ((False, 1), (True, 2)):
                with self.subTest(loop=loop, enabled=enabled, pp_size=pp_size):
                    self.assertFalse(
                        self.retirement_before_next_work(
                            f"event_loop_{loop}_disagg_prefill", enabled, pp_size, False
                        )
                    )


if __name__ == "__main__":
    unittest.main()
