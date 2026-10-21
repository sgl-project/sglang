"""Under PP, prefill KV sends must wait for the forward that writes the pages."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, enable_overlap=False):
        self.enable_overlap = enable_overlap
        self.enable_staging = False
        self.forward_stream = object()
        self.device_module = SimpleNamespace(Event=MagicMock)
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        self.chunked_req = None
        self.sent_wait_events = []

    def checkpoint_disagg_prefill(self, req):
        pass

    def check_bootstrap(self, req):
        return True

    def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
        # The sender consumes the event when the chunk is queued.
        sender = req.disagg_kv_sender
        self.sent_wait_events.append(sender._early_send_wait_event)
        sender._early_send_wait_event = None


class _Req:
    # Hashable: process_prefill_chunk keeps requests in a set.
    def __init__(self):
        self.disagg_kv_sender = SimpleNamespace(_early_send_wait_event=None)
        self.pending_bootstrap = False
        self.prefix_len = 8
        self.host_hit_length = 0
        self.start_send_idx = 0


class TestPrefillPPKVSendWait(CustomTestCase):
    def _send_chunked_req(self, pp_size):
        scheduler = _Scheduler()
        scheduler.chunked_req = _Req()
        running_batch = SimpleNamespace(batch_is_full=True)
        with published_topology(pp_size=pp_size):
            scheduler.process_prefill_chunk(
                last_batch=None, running_batch=running_batch
            )
        return scheduler

    def test_pp_chunk_send_waits_for_forward_stream(self):
        scheduler = self._send_chunked_req(pp_size=2)
        (event,) = scheduler.sent_wait_events
        self.assertIsNotNone(event)
        event.record.assert_called_once_with(scheduler.forward_stream)

    def test_non_pp_chunk_send_does_not_wait(self):
        scheduler = self._send_chunked_req(pp_size=1)
        self.assertEqual(scheduler.sent_wait_events, [None])

    def test_cached_prefix_early_send_wait(self):
        for pp_size, enable_overlap, expect_wait in (
            (1, False, False),
            (1, True, True),
            (2, False, True),
        ):
            with self.subTest(pp_size=pp_size, enable_overlap=enable_overlap):
                scheduler = _Scheduler(enable_overlap=enable_overlap)
                with (
                    envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True),
                    published_topology(pp_size=pp_size),
                ):
                    scheduler.maybe_send_cached_prefix_chunk(_Req())
                (event,) = scheduler.sent_wait_events
                if expect_wait:
                    event.record.assert_called_once_with(scheduler.forward_stream)
                else:
                    self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
