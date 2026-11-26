"""PD decode seats in arrival order (SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER).

Stock seats landed requests in landing order; with the switch the waiting queue is
kept in decode-side arrival order, the same on every rank.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from test_decode_retraction_transfer_progress import SMALL_POOL, _DecodeNode, _Receiver

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
)
from sglang.srt.disaggregation.decode_seat_age import SeatAgeOrder, init_seat_age_order
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SWITCH = envs.SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER


def _reqs(*rids):
    return [
        SimpleNamespace(rid=rid, owner_list=None, decode_arrival_seq=None)
        for rid in rids
    ]


def _rids(queue):
    return [req.rid for req in queue]


class TestSeatAgeDecision(CustomTestCase):
    def test_declines_with_pipeline_parallelism_and_priority_scheduling(self):
        with SWITCH.override(False):
            self.assertIsNone(init_seat_age_order(pp_size=1, priority_scheduling=False))
        with SWITCH.override(True):
            self.assertIsNone(init_seat_age_order(pp_size=2, priority_scheduling=False))
            self.assertIsNone(init_seat_age_order(pp_size=1, priority_scheduling=True))
            self.assertIsInstance(
                init_seat_age_order(pp_size=1, priority_scheduling=False),
                SeatAgeOrder,
            )


class TestSeatAgeQueueInit(CustomTestCase):
    """The real queue constructors: the transfer queue decides once, and the
    preallocation queue numbers requests with the order the joins sort by."""

    def setUp(self):
        enter_override(
            self, get_context().override_server_args(disaggregation_mode="decode")
        )

    def _queues(self, *, priority_scheduling):
        parallel = SimpleNamespace(
            attn_tp_group=SimpleNamespace(cpu_group=None),
            tp_rank=0,
            pp_rank=0,
            pp_size=1,
            attn_tp_rank=0,
        )
        scheduler = MagicMock(
            enable_overlap=False,
            enable_hisparse=False,
            enable_decode_hicache=False,
            enable_priority_scheduling=priority_scheduling,
        )
        scheduler.tp_worker.is_hybrid_swa = False
        with (
            patch.object(decode_module, "get_parallel", return_value=parallel),
            patch.object(
                DecodePreallocQueue, "_init_kv_manager", return_value=MagicMock()
            ),
        ):
            transfer_queue = DecodeTransferQueue(
                req_to_metadata_buffer_idx_allocator=MagicMock(),
                metadata_buffers=MagicMock(),
                scheduler=scheduler,
                tree_cache=MagicMock(),
            )
            prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=MagicMock(),
                draft_token_to_kv_pool=None,
                req_to_metadata_buffer_idx_allocator=MagicMock(),
                metadata_buffers=MagicMock(),
                scheduler=scheduler,
                transfer_queue=transfer_queue,
                tree_cache=MagicMock(),
                gpu_id=0,
                bootstrap_port=0,
                max_total_num_tokens=0,
                num_reserved_decode_tokens=0,
                transfer_backend=MagicMock(),
                draft_kv_ring=None,
            )
        return transfer_queue, prealloc_queue

    def test_the_preallocation_queue_shares_the_transfer_queues_order(self):
        for switch, priority_scheduling, engaged in (
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ):
            with (
                SWITCH.override(switch),
                self.subTest(switch=switch, priority_scheduling=priority_scheduling),
            ):
                transfer_queue, prealloc_queue = self._queues(
                    priority_scheduling=priority_scheduling
                )
                self.assertIs(
                    isinstance(transfer_queue.seat_age_order, SeatAgeOrder), engaged
                )
                self.assertIs(
                    prealloc_queue.seat_age_order, transfer_queue.seat_age_order
                )


class TestSeatAgeOrder(CustomTestCase):
    def test_late_landings_take_their_arrival_place(self):
        order = SeatAgeOrder()
        a, b, c, d, e = _reqs("a", "b", "c", "d", "e")
        for req in (a, b, c, d, e):
            order.stamp(req)
        waiting = IndexedReqList(ListKind.WAITING)
        order.join(waiting, [c, e])
        order.join(waiting, [d, a])
        self.assertEqual(_rids(waiting), ["a", "c", "d", "e"])
        order.join(waiting, [b])
        self.assertEqual(_rids(waiting), ["a", "b", "c", "d", "e"])

    def test_a_retracted_request_keeps_its_arrival_number(self):
        order = SeatAgeOrder()
        a, b, c = _reqs("a", "b", "c")
        order.stamp(a)
        order.stamp(b)
        waiting = IndexedReqList(ListKind.WAITING)
        retracted = IndexedReqList(ListKind.PD_RETRACTED)
        order.join(waiting, [a, b])
        move([a], retracted)
        self.assertEqual(order.stamp(a), 0)
        order.stamp(c)
        order.join(waiting, [c])
        order.join(waiting, [a])
        self.assertEqual(_rids(waiting), ["a", "b", "c"])

    def test_unnumbered_requests_number_in_queue_then_join_order(self):
        """A request the preallocation queue never numbered is numbered at its join
        from the queue's own order, never a clock, so every rank numbers alike."""
        order = SeatAgeOrder()
        x, y, z, w = _reqs("x", "y", "z", "w")
        order.stamp(w)
        waiting = IndexedReqList(ListKind.WAITING)
        move([x], waiting)
        order.join(waiting, [z, y, w])
        self.assertEqual(
            [(req.rid, req.decode_arrival_seq) for req in waiting],
            [("w", 0), ("x", 1), ("z", 2), ("y", 3)],
        )


class _HandshakeReceiver(_Receiver):
    """The harness's transport after the prefill handshake."""

    conclude_state = None

    def init(self, prefill_dp_rank):
        self.decode_req.waiting_for_input = True


class TestSeatAgeAdmission(CustomTestCase):
    """The real decode queues, arrivals through DecodePreallocQueue.add."""

    def setUp(self):
        world_size = patch("torch.distributed.get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_mode="decode", disable_radix_cache=True, page_size=1
            ),
        )
        get_context().override(
            "test", disaggregation_decode_retraction_backup="cpu_tensor"
        )

    def _arrive(self, node, rid, *, prompt_len=100, max_new_tokens, done_at):
        """Through DecodePreallocQueue.add; only the receiver's construction and the
        prefill DP rank lookup are stubbed."""
        prealloc = node.scheduler.disagg_decode_prealloc_queue
        req = node.request(rid, prompt_len, max_new_tokens)

        def enqueue(req, is_rebootstrap=False):
            receiver = _HandshakeReceiver(node, req.bootstrap_room, done_at)
            receiver.decode_req = DecodeRequest(req=req, kv_receiver=receiver)
            move([receiver.decode_req], prealloc.queue)
            return receiver.decode_req

        with (
            patch.object(prealloc, "_create_receiver_and_enqueue", enqueue),
            patch.object(prealloc, "_resolve_prefill_dp_rank", return_value=0),
        ):
            prealloc.add(req)
        return req

    def _node(self, order, running):
        node = _DecodeNode(SMALL_POOL, rows=running)
        node.scheduler.disagg_decode_transfer_queue.seat_age_order = order
        node.scheduler.disagg_decode_prealloc_queue.seat_age_order = order
        return node

    def _seatings(self, node, reqs):
        """The requests in the order they take a decode seat, each time they do."""
        seated, running = [], set()
        while not all(req.finished() for req in reqs) and node.step_count < 4000:
            node.step()
            batch = node.scheduler.running_batch.reqs
            seated += [req.rid for req in batch if req.rid not in running]
            running = {req.rid for req in batch}
        return seated

    def _seat_order(self, order):
        """One running seat: a, b and c arrive in that order behind a running
        request and land in the order b, c, a."""
        node = self._node(order, running=1)
        reqs = [self._arrive(node, "running", max_new_tokens=40, done_at=0)]
        for rid, done_at in (("a", 30), ("b", 5), ("c", 10)):
            reqs.append(self._arrive(node, rid, max_new_tokens=4, done_at=done_at))
        return self._seatings(node, reqs)

    def test_a_freed_seat_goes_to_the_earliest_arrival(self):
        self.assertEqual(self._seat_order(SeatAgeOrder()), ["running", "a", "b", "c"])

    def test_stock_seats_in_landing_order(self):
        self.assertEqual(self._seat_order(None), ["running", "b", "c", "a"])

    def _resume_order(self, order):
        """Two running seats, taken by first and a. The KV pool fills and a, which
        arrived before b and c, is retracted; b takes the freed seat and c lands
        and waits. When first finishes, a resumes and joins the waiting queue."""
        node = self._node(order, running=2)
        reqs = [
            self._arrive(
                node, rid, prompt_len=prompt_len, max_new_tokens=new, done_at=at
            )
            for rid, prompt_len, new, at in (
                ("first", 6000, 1300, 0),
                ("a", 2000, 1500, 10),
                ("b", 3000, 1500, 50),
                ("c", 3000, 400, 100),
            )
        ]
        seated = self._seatings(node, reqs)
        self.assertEqual([rids for _, rids, _ in node.retractions], [["a"]])
        return seated

    def test_a_resumed_request_keeps_its_place(self):
        self.assertEqual(
            self._resume_order(SeatAgeOrder()), ["first", "a", "b", "a", "c"]
        )

    def test_stock_resumes_behind_later_landings(self):
        self.assertEqual(self._resume_order(None), ["first", "a", "b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
