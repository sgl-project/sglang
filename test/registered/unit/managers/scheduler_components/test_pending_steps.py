import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import (
    ReqKvInfo,
    ScheduleBatch,
    new_tokens_required_next_decode,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, lora_id=None, committed=0, allocated=0):
        self.lora_id = lora_id
        self.kv = ReqKvInfo(kv_committed_len=committed, kv_allocated_len=allocated)
        self.is_retracted = False
        self.done = False
        self.beam_group = None

    def finished(self):
        return self.done


class _Step(SimpleNamespace):
    prefill_reqs = ScheduleBatch.prefill_reqs
    decode_reqs = ScheduleBatch.decode_reqs


def _record(
    reqs,
    mode=ForwardMode.DECODE,
    chunked_req=None,
    num_decode_rows=0,
    spec=SpeculativeAlgorithm.NONE,
):
    step = _Step(
        reqs=reqs,
        forward_mode=mode,
        chunked_req=chunked_req,
        num_decode_rows=num_decode_rows,
        spec_algorithm=spec,
    )
    return PendingStep(
        step=step,
        result=None,
        slot=ScheduleSlot(),
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


class TestInflightKv(unittest.TestCase):
    def test_counts_kv_the_commits_add(self):
        a, b, c = _Req(), _Req(), _Req()
        pending = PendingSteps()
        # A mixed step: a prefills, b decodes one token.
        pending.push(_record([a, b], ForwardMode.MIXED, num_decode_rows=1))
        pending.push(_record([a, b]))
        # A speculative verify commits its accepted run when it resolves.
        pending.push(_record([c], spec=SpeculativeAlgorithm.EAGLE))

        self.assertEqual(pending.num_inflight_kv(a), 1)
        self.assertEqual(pending.num_inflight_kv(b), 2)
        self.assertEqual(pending.num_inflight_kv(c), 0)

        # Rows commit_step_counters skips add nothing.
        b.done = True
        a.is_retracted = True
        self.assertEqual(pending.num_inflight_kv(a), 0)
        self.assertEqual(pending.num_inflight_kv(b), 0)

        b.done = False
        pending.popleft()
        self.assertEqual(pending.num_inflight_kv(b), 1)

    def test_next_decode_pages_count_uncommitted_steps(self):
        """The next decode needs a page where the KV of the launched steps
        ends on a page boundary: committed plus the pending decode tokens."""
        page_size = 4
        # One decode step in flight for each, and allocated lengths past it
        # (on which reading them answers 0).
        at_boundary = _Req(committed=7, allocated=9)
        mid_page = _Req(committed=5, allocated=7)
        pending = PendingSteps()
        pending.push(_record([at_boundary, mid_page]))

        def num_tokens():
            return new_tokens_required_next_decode(
                [at_boundary, mid_page], page_size, SpeculativeAlgorithm.NONE, pending
            )

        # KV ends at 8 and 6.
        self.assertEqual(num_tokens(), page_size)
        # The step commits; nothing moves.
        pending.popleft()
        at_boundary.kv.kv_committed_len += 1
        mid_page.kv.kv_committed_len += 1
        self.assertEqual(num_tokens(), page_size)
        # Two more in flight: KV ends at 10 and 8.
        pending.push(_record([at_boundary, mid_page]))
        pending.push(_record([at_boundary, mid_page]))
        self.assertEqual(num_tokens(), page_size)
        self.assertEqual(pending.num_inflight_kv(mid_page), 2)
        # Without those steps, nothing ends on a boundary at 9 and 7.
        pending.popleft()
        self.assertEqual(num_tokens(), 0)


if __name__ == "__main__":
    unittest.main()
