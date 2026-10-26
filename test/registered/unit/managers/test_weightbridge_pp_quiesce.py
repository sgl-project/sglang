"""Unit tests for emptying a PP pipeline before a WeightBridge weight update.

The quiesce must finish every micro-batch already launched (each processed exactly once, as an
idle event_loop_pp pass would), then requeue every unfinished request and leave no micro-batch
state behind, so the loop resumes from an empty pipeline under the new weights.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PP_SIZE = 2
LOOP_SIZE = 3  # pp_size + one async batch


def _req(name: str, finished: bool = False) -> SimpleNamespace:
    return SimpleNamespace(rid=name, finished=lambda: finished)


def _batch(*reqs) -> SimpleNamespace:
    return SimpleNamespace(reqs=list(reqs), is_empty=lambda: not reqs)


def _scheduler(*, mbs, running_mbs, last_mbs, chunked_req=None) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.pp_loop_size = LOOP_SIZE
    sched.mbs = list(mbs)
    sched.running_mbs = list(running_mbs)
    sched.last_mbs = list(last_mbs)
    sched.mb_metadata = [object()] * LOOP_SIZE
    sched.chunked_req = chunked_req
    sched.pp_outputs = "stale"
    sched.req_to_token_pool = sched.token_to_kv_pool_allocator = sched.tree_cache = None
    sched.hisparse_coordinator = None
    sched.waiting_queue = []
    sched.processed = []
    sched.exchanges = []

    sched._pp_commit_proxy_send_work = lambda: None

    def exchange(next_first_rank_mb_id, next_mb_id):
        sched.exchanges.append((next_first_rank_mb_id, next_mb_id))
        return (
            f"outputs-{next_mb_id}",
            f"result-{next_mb_id}",
            SimpleNamespace(synchronize=lambda: None),
        )

    sched._pp_commit_send_output_work_and_preprocess_output_tensors = exchange
    sched._pp_process_batch_result = lambda batch, result: sched.processed.append(
        (batch, result)
    )
    sched._add_request_to_queue = sched.waiting_queue.append
    return sched


class TestWeightBridgePPQuiesce(CustomTestCase):
    def setUp(self):
        self.retracted = []
        patches = [
            patch.object(
                scheduler_pp_mixin,
                "get_parallel",
                lambda: SimpleNamespace(pp_size=PP_SIZE),
            ),
            patch.object(
                scheduler_pp_mixin,
                "retract_all",
                lambda *, reqs, **_: self.retracted.extend(reqs),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_finishes_each_launched_micro_batch_once_in_loop_order(self):
        # At the start of slot 1, slot 1's batch was processed one pass ago; slots 2 and 0 are pending.
        b0, b1, b2 = _batch(_req("a")), _batch(_req("b")), _batch(_req("c"))
        sched = _scheduler(
            mbs=[b0, b1, b2],
            running_mbs=[_batch(), _batch(), _batch()],
            last_mbs=[None, None, None],
        )

        sched.pp_retract_all_for_weight_update(mb_id=1)

        self.assertEqual(sched.processed, [(b2, "result-2"), (b0, "result-0")])
        # Same (next_first_rank_mb_id, next_mb_id) sequence as event_loop_pp's passes 1, 2, 0.
        self.assertEqual(sched.exchanges, [(0, 2), (1, 0), (2, 1)])
        self.assertEqual(sched.pp_outputs, "outputs-1")

    def test_requeues_unfinished_requests_and_resets_micro_batch_state(self):
        running, done, prefilled, chunked = (
            _req("run"),
            _req("done", True),
            _req("pre"),
            _req("chunk"),
        )
        sched = _scheduler(
            mbs=[None, None, None],
            running_mbs=[_batch(running, done), _batch(), _batch(running)],
            last_mbs=[None, _batch(prefilled), None],
            chunked_req=chunked,
        )

        sched.pp_retract_all_for_weight_update(mb_id=0)

        self.assertEqual({r.rid for r in sched.waiting_queue}, {"run", "pre", "chunk"})
        self.assertEqual(
            len(sched.waiting_queue), 3
        )  # a request in two slots is requeued once
        self.assertEqual(self.retracted, sched.waiting_queue)
        self.assertIsNone(sched.chunked_req)
        self.assertEqual(sched.mbs, [None] * LOOP_SIZE)
        self.assertEqual(sched.last_mbs, [None] * LOOP_SIZE)
        self.assertEqual(sched.mb_metadata, [None] * LOOP_SIZE)
        self.assertTrue(all(b.is_empty() for b in sched.running_mbs))
        self.assertIs(sched.running_batch, sched.running_mbs[0])
        self.assertIsNone(sched.last_batch)


if __name__ == "__main__":
    unittest.main()
