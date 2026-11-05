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


if __name__ == "__main__":
    unittest.main()
