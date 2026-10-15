import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, lora_id=None):
        self.lora_id = lora_id


def _record(reqs, mode=ForwardMode.DECODE, chunked_req=None):
    step = SimpleNamespace(reqs=reqs, forward_mode=mode, chunked_req=chunked_req)
    return PendingStep(
        step=step,
        result=None,
        launch_ts=0.0,
        after_idle_gap=False,
    )


class TestPendingSteps(unittest.TestCase):
    def test_counts_follow_push_and_commit(self):
        a, b = _Req(lora_id="x"), _Req()
        chunked = _Req(lora_id="y")
        pending = PendingSteps()
        # A prefill step whose middle chunk commits no output token.
        pending.push(_record([a, chunked], ForwardMode.EXTEND, chunked_req=chunked))
        pending.push(_record([a, b]))

        self.assertEqual(pending.num_inflight_outputs(a), 2)
        self.assertEqual(pending.num_inflight_outputs(b), 1)
        self.assertEqual(pending.num_inflight_outputs(chunked), 0)
        self.assertEqual(pending.lora_ids(), {"x", "y", None})

        pending.popleft()
        self.assertEqual(pending.num_inflight_outputs(a), 1)
        self.assertEqual(pending.lora_ids(), {"x", None})

        pending.popleft()
        self.assertEqual(pending.num_inflight_outputs(a), 0)
        self.assertEqual(pending.lora_ids(), set())
        self.assertIsNone(pending.newest())


if __name__ == "__main__":
    unittest.main()
