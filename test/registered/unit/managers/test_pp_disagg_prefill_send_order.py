"""The PP PD prefill loop forwards requests and transfer outcomes before launch.

Release ids, requests, capacity digests and transferred ids travel to the next
stage on one FIFO channel, so the next stage must receive them in the order
they are sent.
"""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ITERATIONS = 8


class _ChannelDry(Exception):
    pass


def _stage(is_first_rank, is_last_rank, depth):
    s = Scheduler.__new__(Scheduler)
    loop_size = 2 + depth

    def init_pp_loop_state():
        s.pp_loop_size = loop_size
        s.mbs = [None] * loop_size
        s.last_mbs = [None] * loop_size
        s.running_mbs = [SimpleNamespace(batch_is_full=False) for _ in range(loop_size)]
        s.pp_outputs = None
        s.mb_metadata = [None] * loop_size
        s.last_rank_comm_queue = deque()
        s.send_req_work = []

    s.init_pp_loop_state = init_pp_loop_state
    s.pp_group = SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank)
    s.increment_scheduler_iterations = Mock()
    s._pp_mark_capacity_digest = Mock()
    s._pp_commit_comm_work = Mock()
    s._pp_commit_proxy_send_work = Mock()
    s.process_bootstrapped_queue = Mock()
    s.process_pending_chunked_abort = Mock()
    s.process_prefill_chunk = Mock()
    s._process_hicache_events = Mock()
    s.pause_state = SimpleNamespace(may_launch=lambda: True)
    s.get_new_batch_prefill = lambda running_batch: SimpleNamespace(
        batch_to_run=Mock(), running_batch=running_batch
    )
    s.dp_attn_adapter = SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda b: b)
    s.enable_staging = False
    s._pp_recv_proxy_tensors = Mock(return_value=None)
    s._pp_commit_send_output_work_and_preprocess_output_tensors = Mock(
        return_value=(None, None, Mock())
    )
    s._pp_process_batch_result = Mock()
    s.process_disagg_prefill_transfer_queue = Mock()
    s._pp_send_proxy_to_next_stage = Mock()
    s._pp_answer_release_requests = Mock()
    s._pp_apply_war_barrier = Mock()
    s.invariant_checker = Mock()
    s.disagg_prefill_transfer_queue = []
    s.on_idle = Mock()
    return s


class TestPPDisaggPrefillSendOrder(CustomTestCase):
    def _run_sender(self, depth):
        """First (non-last) stage: record what it sends to the next stage."""
        s = _stage(is_first_rank=True, is_last_rank=False, depth=depth)
        channel, trace, labels = [], [], {}
        iterations = iter(range(ITERATIONS))

        def ingest():
            next(iterations)
            trace.append("ingest")
            return []

        def transferred_ids():
            rids = {}
            labels[id(rids)] = "transferred"
            return rids

        def recv_release():
            # From the last stage, on another channel.
            rids = {}
            labels[id(rids)] = "release"
            return rids

        def send_pyobj(data, async_send=False):
            channel.append(labels[id(data)])
            trace.append(labels[id(data)])
            return []

        def send(label):
            channel.append(label)
            trace.append(label)
            return []

        def launch(*args, **kwargs):
            trace.append("launch")
            return Mock(), Mock()

        s._pp_ingest_requests = ingest
        s._pp_check_capacity_digest = Mock()
        s._pp_pd_get_prefill_transferred_ids = transferred_ids
        s._pp_recv_pyobj_from_prev_stage = recv_release
        s._pp_send_pyobj_to_next_stage = send_pyobj
        s._pp_send_reqs_to_next_stage = lambda reqs: send("reqs")
        s._pp_send_capacity_digest = lambda: send("digest")
        s._pp_launch_batch = launch
        with get_parallel().override(pp_size=2, pp_rank=0, pp_async_batch_depth=depth):
            with self.assertRaises(StopIteration):
                Scheduler.event_loop_pp_disagg_prefill(s)
        return channel, trace

    def _run_receiver(self, channel, depth):
        """Last stage: every receive must get the kind it expects."""
        s = _stage(is_first_rank=False, is_last_rank=True, depth=depth)
        pending = deque(channel)
        ingested = 0

        def recv(kind):
            if not pending:
                raise _ChannelDry
            self.assertEqual(pending.popleft(), kind)
            return {}

        def ingest():
            nonlocal ingested
            recv("reqs")
            ingested += 1
            return []

        s._pp_ingest_requests = ingest
        s._pp_check_capacity_digest = lambda: recv("digest")
        s._pp_pd_get_prefill_transferred_ids = lambda: recv("transferred")
        s._pp_recv_pyobj_from_prev_stage = lambda: recv("release")
        # The last stage sends release ids on to the first stage only.
        s._pp_send_pyobj_to_next_stage = Mock(return_value=[])
        s._pp_launch_batch = Mock(return_value=(Mock(), Mock()))
        with get_parallel().override(pp_size=2, pp_rank=1, pp_async_batch_depth=depth):
            with self.assertRaises(_ChannelDry):
                Scheduler.event_loop_pp_disagg_prefill(s)
        return ingested

    def test_next_stage_receives_in_send_order(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                channel, _ = self._run_sender(depth)
                self.assertIn("release", channel)
                self.assertEqual(self._run_receiver(channel, depth), ITERATIONS)

    def test_requests_and_transfers_go_out_before_launch(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                _, trace = self._run_sender(depth)
                iterations = []
                for event in trace:
                    if event == "ingest":
                        iterations.append([])
                    else:
                        iterations[-1].append(event)
                for events in iterations:
                    launch = events.index("launch")
                    for kind in ("reqs", "digest", "transferred"):
                        self.assertLess(events.index(kind), launch)


if __name__ == "__main__":
    unittest.main()
