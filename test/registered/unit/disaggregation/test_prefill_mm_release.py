import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid: str, *, session=None):
    return SimpleNamespace(
        rid=rid,
        disagg_kv_sender=MagicMock(),
        pending_bootstrap=False,
        finished_reason=None,
        cache_request_handle=None,
        time_stats=MagicMock(),
        bootstrap_host="decode",
        bootstrap_room=0,
        kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
        return_logprob=False,
        return_routed_experts=False,
        multimodal_inputs=MagicMock(),
        session=session,
    )


def _scheduler(reqs):
    scheduler = MagicMock()
    scheduler.disagg_prefill_inflight_queue = list(reqs)
    scheduler._release_mm_inputs = types.MethodType(
        Scheduler._release_mm_inputs, scheduler
    )
    return scheduler


class TestPrefillMmRelease(CustomTestCase):
    def _drain(self, scheduler, polls):
        with (
            patch.object(
                prefill, "poll_and_all_reduce_attn_cp_tp_group", return_value=polls
            ),
            patch.object(prefill, "release_kv_cache"),
            patch.object(prefill, "maybe_release_metadata_buffer"),
        ):
            return SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )

    def test_transferred_request_releases_features(self):
        done, transferring = _req("done"), _req("transferring")
        mm_inputs = done.multimodal_inputs
        scheduler = _scheduler([done, transferring])

        self._drain(scheduler, [KVPoll.Success, KVPoll.Transferring])

        mm_inputs.release_features.assert_called_once_with()
        self.assertIsNone(done.multimodal_inputs)
        self.assertIsNotNone(transferring.multimodal_inputs)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [transferring])

    def test_failed_transfer_releases_features(self):
        failed = _req("failed")
        mm_inputs = failed.multimodal_inputs
        scheduler = _scheduler([failed])

        self._drain(scheduler, [KVPoll.Failed])

        mm_inputs.release_features.assert_called_once_with()
        self.assertIsNone(failed.multimodal_inputs)

    def test_bootstrap_failure_releases_features_after_streaming(self):
        req = _req("bootstrap-failed")
        mm_inputs = req.multimodal_inputs
        scheduler = _scheduler([])
        streamed_mm_inputs = []
        scheduler.output_streamer.stream_output.side_effect = lambda reqs, *_: (
            streamed_mm_inputs.extend(r.multimodal_inputs for r in reqs)
        )

        with (
            get_parallel().override(tp_rank=0),
            patch.object(prefill, "release_kv_cache"),
            patch.object(prefill, "maybe_release_metadata_buffer"),
        ):
            SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertEqual(streamed_mm_inputs, [mm_inputs])
        mm_inputs.release_features.assert_called_once_with()
        self.assertIsNone(req.multimodal_inputs)

    def test_session_request_keeps_features(self):
        req = _req("session", session=object())
        scheduler = _scheduler([req])

        self._drain(scheduler, [KVPoll.Success])

        req.multimodal_inputs.release_features.assert_not_called()


if __name__ == "__main__":
    unittest.main()
