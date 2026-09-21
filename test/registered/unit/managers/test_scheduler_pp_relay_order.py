import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeEvent:
    def record(self, stream):
        pass


class _FakeStream:
    def wait_stream(self, stream):
        pass


class TestSchedulerPPRelayOrder(unittest.TestCase):
    def _make_scheduler(self, is_last_rank):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        scheduler.ps = SimpleNamespace(pp_rank=0)
        scheduler.copy_stream_ctx = nullcontext()
        scheduler.copy_stream = _FakeStream()
        scheduler.schedule_stream = _FakeStream()
        scheduler.device_module = SimpleNamespace(
            Event=_FakeEvent,
            current_stream=lambda: _FakeStream(),
        )
        return scheduler

    def _run_relay(self, is_last_rank):
        scheduler = self._make_scheduler(is_last_rank)
        events = []
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
        )
        scheduler._pp_send_output_to_next_stage = lambda *args: (
            events.append("send") or ["output-work"]
        )
        scheduler._pp_recv_dict_from_prev_stage = lambda: (
            events.append("recv") or {"next_token_ids": object()}
        )
        scheduler._pp_prep_batch_result = lambda *args: events.append("prep")
        scheduler._pp_send_dict_to_next_stage = lambda *args, **kwargs: (
            events.append("relay") or ["relay-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: events.append(("commit", work))

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin._pp_can_skip_output_comm",
            return_value=False,
        ):
            _, _, _, send_work = scheduler._pp_send_recv_and_preprocess_output_tensors(
                next_first_rank_mb_id=0,
                next_mb_id=1,
                mbs=[target, target],
                mb_metadata=[None, None],
                last_rank_comm_queue=[],
                pp_outputs=None,
                relay_output_immediately=True,
            )

        return events, send_work

    def test_last_rank_injects_output_before_commit(self):
        events, send_work = self._run_relay(is_last_rank=True)

        self.assertEqual(
            events[:4], ["send", "recv", "prep", ("commit", ["output-work"])]
        )
        self.assertEqual(send_work, [])

    def test_non_last_rank_forwards_received_output_before_commit(self):
        events, send_work = self._run_relay(is_last_rank=False)

        self.assertEqual(
            events[:4], ["recv", "prep", "relay", ("commit", ["relay-work"])]
        )
        self.assertEqual(send_work, [])

    def test_proxy_exchange_is_committed_before_next_ring(self):
        scheduler = self._make_scheduler(is_last_rank=False)
        events = []
        scheduler._pp_send_dict_to_next_stage = lambda *args, **kwargs: (
            events.append(("send", kwargs["msg_type"])) or ["proxy-work"]
        )
        scheduler._pp_commit_comm_work = lambda work: events.append(("commit", work))

        scheduler._pp_send_and_commit_proxy({"hidden_states": object()})

        self.assertEqual(
            events,
            [("send", "proxy"), ("commit", ["proxy-work"])],
        )


if __name__ == "__main__":
    unittest.main()
