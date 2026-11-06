import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.decode_preemption import (
    PREEMPT_COOLDOWN_S,
    DecodePreemptor,
    plan_decode_preemption,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import request_retirer
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
MAX_NEW = 8192  # above the clip, so every reserve is CLIP * ratio


def _req(rid, *, input_len, output_len=0, device_prefix=0, queued_at=None):
    req = Req(
        rid,
        "",
        array("q", [1] * input_len),
        SamplingParams(max_new_tokens=MAX_NEW),
    )
    req.output_ids.extend([2] * output_len)
    req.prefix_len = device_prefix
    req.uniboost_arrival_s = queued_at
    return req


def _running(rid, *, input_len, output_len):
    return _req(rid, input_len=input_len, output_len=output_len)


def _plan(waiting, running, have, **kw):
    args = dict(
        now_s=100.0,
        waiting_queue=waiting,
        running_reqs=running,
        have_tokens=have,
        new_token_ratio=0.5,
        page_size=PAGE,
        wait_s=30.0,
        quantum_tokens=512,
        max_per_req=2,
    )
    args.update(kw)
    return plan_decode_preemption(**args)


class TestPlanDecodePreemption(CustomTestCase):
    def test_trigger_requires_a_memory_deficit_on_the_adder_budget(self):
        # The deficit must use PrefillAdder's budget: available + evictable
        # minus every running request's clipped decode reserve. A waiter that
        # fits is starving for another reason, so releasing decodes would
        # only add stalls.
        waiter = _req("w", input_len=100_000, device_prefix=40_000, queued_at=50.0)
        need = 100_000 - 40_000 + CLIP_MAX_NEW_TOKENS + PAGE
        running = [_running("r", input_len=90_000, output_len=4_000)]
        reserve = min(MAX_NEW - 4_000, CLIP_MAX_NEW_TOKENS) * 0.5
        self.assertIsNone(_plan([waiter], running, have=int(need + reserve)))
        plan = _plan([waiter], running, have=int(need + reserve) - 1)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.target_rid, "w")
        self.assertEqual(plan.deficit_tokens, 1)

    def test_ignore_eos_decodes_reserve_their_full_remainder(self):
        # PrefillAdder reserves ratio 1.0 for ignore_eos requests; a planner
        # using the decayed ratio would see a deficit admission never sees.
        waiter = _req("w", input_len=10_000, queued_at=0.0)
        need = 10_000 + CLIP_MAX_NEW_TOKENS + PAGE
        r = _running("r", input_len=20_000, output_len=1_000)
        r.sampling_params.ignore_eos = True
        reserve = min(MAX_NEW - 1_000, CLIP_MAX_NEW_TOKENS)
        self.assertIsNone(_plan([waiter], [r], have=need + reserve))
        self.assertIsNotNone(_plan([waiter], [r], have=need + reserve - 1))

    def test_victims_are_the_minimal_longest_attained_prefix(self):
        # Heavy-tailed outputs: the most decoded request has the longest
        # expected remaining hold, so release in that order and stop as soon
        # as the waiter fits -- not every eligible decode.
        waiter = _req("w", input_len=150_000, queued_at=0.0)
        running = [
            _running("young", input_len=100_000, output_len=100),  # < quantum
            _running("mid", input_len=60_000, output_len=3_000),
            _running("short", input_len=60_000, output_len=800),
            _running("long", input_len=60_000, output_len=5_000),
        ]
        plan = _plan([waiter], running, have=0)
        self.assertEqual(
            [running[i].rid for i in plan.victim_indices], ["long", "mid", "short"]
        )
        # Room for most of the need already: one release suffices.
        plan = _plan([waiter], running, have=120_000)
        self.assertEqual([running[i].rid for i in plan.victim_indices], ["long"])

    def test_no_plan_when_eligible_releases_cannot_cover_the_deficit(self):
        waiter = _req("w", input_len=250_000, queued_at=0.0)
        running = [_running("r", input_len=20_000, output_len=1_000)]
        self.assertIsNone(_plan([waiter], running, have=0))

    def test_quantum_doubles_per_release_and_the_cap_ends_releases(self):
        # Progress guarantee: after k releases a request must decode
        # quantum * 2**k tokens since its latest admission before the next,
        # and max_per_req releases end it.
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        r = _running("r", input_len=200_000, output_len=3_000)
        r.num_preemptions = 1
        r.preempt_resume_output_len = 3_000 - 1_023  # 1023 < 512 << 1
        self.assertIsNone(_plan([waiter], [r], have=0))
        r.preempt_resume_output_len = 3_000 - 1_024
        self.assertIsNotNone(_plan([waiter], [r], have=0))
        r.num_preemptions = 2  # == max_per_req
        r.preempt_resume_output_len = 0
        self.assertIsNone(_plan([waiter], [r], have=0))

    def test_only_first_token_waiters_trigger(self):
        # A released decode re-queues with output already streamed; letting it
        # trigger releases of its own would cascade stalls between decodes.
        released = _req("rel", input_len=100_000, output_len=2_000, queued_at=0.0)
        running = [_running("r", input_len=200_000, output_len=6_000)]
        self.assertIsNone(_plan([released], running, have=0))
        fresh = _req("w", input_len=100_000, queued_at=0.0)
        plan = _plan([released, fresh], running, have=0)
        self.assertEqual(plan.target_rid, "w")
        # Not yet starving: no trigger.
        recent = _req("w2", input_len=100_000, queued_at=80.0)
        self.assertIsNone(_plan([recent], running, have=0))


class TestDecodePreemptor(CustomTestCase):
    def test_reserved_waiter_heads_the_queue_until_admitted(self):
        # The released memory must reach the waiter: it heads the sorted queue
        # (the misfit break then stops later arrivals from taking the memory)
        # until admitted; admission ends the reservation and restarts the
        # released decode's quantum.
        pre = DecodePreemptor(wait_s=30.0, quantum_tokens=512, max_per_req=2, dp_rank=0)
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        victim = _running("v", input_len=200_000, output_len=6_000)
        plan = pre.maybe_plan(
            now_s=100.0,
            waiting_queue=[waiter],
            running_reqs=[victim],
            have_tokens=0,
            new_token_ratio=0.5,
            page_size=PAGE,
        )
        pre.on_preempted(plan, [victim], now_s=100.0)
        self.assertEqual(victim.num_preemptions, 1)

        # The scheduler's waiting queue is an IndexedReqList (no list
        # mutators); the promoted order is applied with replace().
        a, b = _req("a", input_len=10), _req("b", input_len=10)
        queue = IndexedReqList(ListKind.WAITING)
        queue.join([a, b, waiter, victim])
        queue.replace(pre.promote_target(queue))
        self.assertEqual([r.rid for r in queue], ["w", "a", "b", "v"])

        # Within the cooldown no second plan, even with a starving waiter.
        self.assertIsNone(
            pre.maybe_plan(
                now_s=100.0 + PREEMPT_COOLDOWN_S / 2,
                waiting_queue=queue,
                running_reqs=[_running("x", input_len=200_000, output_len=9_000)],
                have_tokens=0,
                new_token_ratio=0.5,
                page_size=PAGE,
            )
        )

        pre.on_admitted([waiter, victim])
        self.assertEqual(victim.preempt_resume_output_len, 6_000)
        self.assertEqual([r.rid for r in pre.promote_target([a, b])], ["a", "b"])

    def test_off_and_unsupported_trees(self):
        common = dict(quantum_tokens=512, max_per_req=2, dp_rank=0)
        ok = dict(tree_supports_mamba=False, tree_supports_swa=False)
        # Off must construct nothing (the scheduler path stays bit-identical).
        self.assertIsNone(DecodePreemptor.maybe_create(wait_s=0.0, **common, **ok))
        self.assertIsNotNone(DecodePreemptor.maybe_create(wait_s=30.0, **common, **ok))
        for bad in (
            dict(ok, tree_supports_swa=True),
            dict(ok, tree_supports_mamba=True),
        ):
            with self.assertRaises(ValueError):
                DecodePreemptor.maybe_create(wait_s=30.0, **common, **bad)


class TestPreemptDecodeRelease(CustomTestCase):
    def test_release_checkpoints_into_the_tree_and_keeps_the_output(self):
        # The point of preempting instead of retracting: the KV is checkpointed
        # into the radix tree, so the request resumes from its own cache; OOM
        # retraction drops the uncached tail.
        keep = _running("keep", input_len=10, output_len=5)
        victim = _running("victim", input_len=10, output_len=7)
        tree_cache = object()
        retirer = RequestRetirer(
            tree_cache=tree_cache,
            beam_coordinator=Mock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=Mock(),
        )
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [keep, victim]
        batch.filter_batch = Mock()

        with patch.object(request_retirer, "release_kv_cache") as rel:
            preempted = batch.preempt_decode((1,), retirer)
        rel.assert_called_once_with(victim, tree_cache, checkpoint=True)
        self.assertEqual(preempted, [victim])
        batch.filter_batch.assert_called_once_with(keep_indices=[0])
        self.assertTrue(victim.is_retracted)
        self.assertEqual(victim.retraction_count, 1)
        self.assertEqual(len(victim.output_ids), 7)


class TestSchedulerPreemptGlue(CustomTestCase):
    def test_released_decodes_leave_running_and_rejoin_the_waiting_queue(self):
        # A released request is no longer in any batch; if it does not
        # re-enter the waiting queue it never finishes, and if it stays in the
        # running set the scheduler counts it twice.
        sched = Scheduler.__new__(Scheduler)
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        victim = _running("v", input_len=200_000, output_len=6_000)
        sched.waiting_queue = [waiter]
        sched.decode_preemptor = DecodePreemptor(
            wait_s=30.0, quantum_tokens=512, max_per_req=2, dp_rank=0
        )
        sched.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 0)
        sched.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
        sched.new_token_ratio_tracker = SimpleNamespace(current=0.5)
        sched.logical_page_size = PAGE
        sched.decode_offload_manager = None
        sched.running_reqs = Mock()
        sched.request_retirer = Mock()
        requeued = []
        sched._add_request_to_queue = lambda req, is_retracted=False: requeued.append(
            (req, is_retracted)
        )
        batch = SimpleNamespace(
            reqs=[victim],
            preempt_decode=Mock(side_effect=lambda idx, retirer: [victim]),
        )
        with patch("time.perf_counter", return_value=100.0):
            sched._preempt_for_starving_waiters(batch)
        batch.preempt_decode.assert_called_once_with((0,), sched.request_retirer)
        sched.running_reqs.leave.assert_called_once_with([victim])
        self.assertEqual(requeued, [(victim, True)])


if __name__ == "__main__":
    unittest.main()
