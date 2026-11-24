"""Skipping the PP output send must keep the last rank's wait on its forward.

With SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM, the last PP rank does not return
the placeholder output of a pure middle chunked-prefill batch. The output send
was also where the scheduler stream waited for that batch's forward (through
its ready_event). Without that wait, the skip result's d2h_event completes at
once, and the next chunk's KV transfer can read pages the forward is still
writing.
"""

import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_last_rank_scheduler(stream):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(
        pp_group=SimpleNamespace(
            is_last_rank=True, send_tensor_dict=Mock(return_value=["work"])
        ),
        device_module=SimpleNamespace(current_stream=lambda: stream),
        pp_comm_stream_ctx=nullcontext(),
        attn_tp_group=None,
    )
    return scheduler


def _make_batch(last_chunk, forward_mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        forward_mode=forward_mode,
        reqs=[object()],
        contains_last_prefill_chunk=last_chunk,
        return_logprob=False,
    )


class TestPPSkipOutputComm(CustomTestCase):
    def test_last_rank_waits_on_forward_whether_or_not_output_is_sent(self):
        cases = [
            # (skip env, last chunk, expect send)
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ]
        for skip_env, last_chunk, expect_send in cases:
            with self.subTest(skip_env=skip_env, last_chunk=last_chunk):
                stream = Mock()
                scheduler = _make_last_rank_scheduler(stream)
                q_event = object()
                queue = deque([(q_event, PPProxyTensors({}))])
                with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(skip_env):
                    work = scheduler._pp_send_output_to_next_stage(
                        0, [_make_batch(last_chunk)], queue, None
                    )
                stream.wait_event.assert_called_once_with(q_event)
                self.assertEqual(
                    scheduler.pp_group.send_tensor_dict.called, expect_send
                )
                self.assertEqual(work, ["work"] if expect_send else [])
                self.assertFalse(queue)

    def test_prebuilt_batch_neither_sends_nor_waits(self):
        stream = Mock()
        scheduler = _make_last_rank_scheduler(stream)
        queue = deque([(object(), PPProxyTensors({}))])
        batch = _make_batch(last_chunk=True, forward_mode=ForwardMode.PREBUILT)
        work = scheduler._pp_send_output_to_next_stage(0, [batch], queue, None)
        stream.wait_event.assert_not_called()
        scheduler.pp_group.send_tensor_dict.assert_not_called()
        self.assertEqual(work, [])
        self.assertFalse(queue)

    def test_npu_skip_marker_send_keeps_forward_wait(self):
        for last_chunk in (False, True):
            with self.subTest(last_chunk=last_chunk):
                stream = Mock()
                scheduler = _make_last_rank_scheduler(stream)
                scheduler._pp_send_dict_to_next_stage = Mock(return_value=["work"])
                q_event = object()
                queue = deque([(q_event, PPProxyTensors({"x": 1}))])
                with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True):
                    *_, work = scheduler._pp2_only_send_recv_output_tensors_npu(
                        0, 1, [_make_batch(last_chunk), None], [None, None], queue, None
                    )
                stream.wait_event.assert_called_once_with(q_event)
                send_dict = scheduler._pp_send_dict_to_next_stage.call_args.args[0]
                self.assertEqual(send_dict.get("__skip__", False), not last_chunk)
                self.assertEqual(work, ["work"])


if __name__ == "__main__":
    unittest.main()
