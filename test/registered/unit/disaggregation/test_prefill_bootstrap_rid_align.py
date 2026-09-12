"""Unit tests for PrefillBootstrapQueue.pop_bootstrapped TP/CP rid alignment.

When attn TP or CP world size > 1, ranks may have divergent bootstrap queues.
pop_bootstrapped must intersect rids across the group before polling so
poll_and_all_reduce_attn_cp_tp_group does not hang / crash on length mismatch.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid: str):
    return SimpleNamespace(
        rid=rid,
        disagg_kv_sender=MagicMock(name=f"sender-{rid}"),
        prefill_attempt_count=0,
        is_retracted=False,
        time_stats=SimpleNamespace(set_wait_queue_entry_time=MagicMock()),
    )


def _make_queue(rids, *, tp_world_size=1, cp_world_size=1):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.pp_size = 1
    queue.queue = [_make_req(rid) for rid in rids]
    tp_group = object() if tp_world_size > 1 else None
    cp_group = object() if cp_world_size > 1 else None
    queue.scheduler = SimpleNamespace(
        attn_tp_cpu_group=tp_group,
        attn_cp_cpu_group=cp_group,
        handle_bootstrap_failure=MagicMock(),
    )
    queue.finalize_bootstrap = MagicMock(return_value=True)
    queue.ensure_metadata_buffer = MagicMock(return_value=True)
    queue._tp_world_size = tp_world_size
    queue._cp_world_size = cp_world_size
    return queue


def _world_size_side_effect(queue):
    def _get_world_size(group):
        if group is queue.scheduler.attn_tp_cpu_group:
            return queue._tp_world_size
        if group is queue.scheduler.attn_cp_cpu_group:
            return queue._cp_world_size
        return 1

    return _get_world_size


class TestPrefillBootstrapRidAlign(CustomTestCase):
    def test_empty_queue_without_rid_align_returns_early(self):
        queue = _make_queue([], tp_world_size=1, cp_world_size=1)

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as poll,
        ):
            self.assertEqual(queue.pop_bootstrapped(), [])
            self.assertEqual(queue.pop_bootstrapped(return_failed_reqs=True), ([], []))
            poll.assert_not_called()

    def test_empty_queue_rid_align_all_ranks_idle(self):
        queue = _make_queue([], tp_world_size=2)

        def all_reduce(tensor, op=None, group=None):
            # Peer ranks are also empty → MAX stays 0.
            return None

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_reduce",
                side_effect=all_reduce,
            ) as reduce,
            patch("sglang.srt.disaggregation.prefill.dist.all_gather_object") as gather,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as poll,
        ):
            self.assertEqual(queue.pop_bootstrapped(), [])
            reduce.assert_called()
            gather.assert_not_called()
            poll.assert_not_called()

    def test_empty_local_queue_skips_poll_when_peers_have_reqs(self):
        """Local rank empty but peer has reqs: after intersection, local polls nothing."""
        queue = _make_queue([], tp_world_size=2)

        def all_reduce(tensor, op=None, group=None):
            tensor[0] = 1  # peer has requests

        def all_gather_object(gathered, obj, group=None):
            # Local empty; peer has rid-a. Intersection is empty.
            gathered[0] = list(obj)
            gathered[1] = ["rid-a"]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_reduce",
                side_effect=all_reduce,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as poll,
        ):
            self.assertEqual(queue.pop_bootstrapped(), [])
            poll.assert_not_called()
            self.assertEqual(queue.queue, [])

    def test_rid_align_polls_only_common_intersection(self):
        """Local [a,b], peer [a,c] → only poll a; b stays with poll=None."""
        queue = _make_queue(["rid-a", "rid-b"], tp_world_size=2)
        req_a, req_b = queue.queue

        def all_gather_object(gathered, obj, group=None):
            gathered[0] = list(obj)
            gathered[1] = ["rid-a", "rid-c"]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.WaitingForInput],
            ) as poll,
            patch(
                "sglang.srt.disaggregation.prefill.should_force_retry",
                return_value=False,
            ),
        ):
            bootstrapped = queue.pop_bootstrapped()

        poll.assert_called_once()
        senders = poll.call_args.args[0]
        self.assertEqual(len(senders), 1)
        self.assertIs(senders[0], req_a.disagg_kv_sender)
        self.assertEqual(bootstrapped, [req_a])
        self.assertEqual(queue.queue, [req_b])
        queue.finalize_bootstrap.assert_called_once_with(req_a)

    def test_rid_align_cp_group_also_intersects(self):
        queue = _make_queue(["rid-a", "rid-b"], cp_world_size=2)
        req_a, req_b = queue.queue

        def all_gather_object(gathered, obj, group=None):
            gathered[0] = list(obj)
            gathered[1] = ["rid-b"]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.WaitingForInput],
            ) as poll,
            patch(
                "sglang.srt.disaggregation.prefill.should_force_retry",
                return_value=False,
            ),
        ):
            bootstrapped = queue.pop_bootstrapped()

        senders = poll.call_args.args[0]
        self.assertEqual(len(senders), 1)
        self.assertIs(senders[0], req_b.disagg_kv_sender)
        self.assertEqual(bootstrapped, [req_b])
        self.assertEqual(queue.queue, [req_a])

    def test_rid_align_failed_common_req(self):
        queue = _make_queue(["rid-fail"], tp_world_size=2)
        req = queue.queue[0]

        def all_gather_object(gathered, obj, group=None):
            gathered[0] = list(obj)
            gathered[1] = ["rid-fail"]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Failed],
            ),
        ):
            bootstrapped, failed = queue.pop_bootstrapped(return_failed_reqs=True)

        self.assertEqual(bootstrapped, [])
        self.assertEqual(failed, [req])
        self.assertEqual(queue.queue, [])
        queue.scheduler.handle_bootstrap_failure.assert_called_once_with(req)

    def test_rid_align_maps_polls_by_sorted_common_order(self):
        """Multiple common rids: local queue order != sorted intersection order.

        In pop_bootstrapped, aligned_reqs is built from the *sorted* intersection,
        while polls are mapped back onto the local queue via rid->poll dict. This
        pins the zip between aligned_reqs (sorted) and the poll return so a poll
        lands on the right request even when the local queue is unsorted.
        """
        queue = _make_queue(["rid-c", "rid-a", "rid-b"], tp_world_size=2)
        req_c, req_a, req_b = queue.queue

        def all_gather_object(gathered, obj, group=None):
            gathered[0] = list(obj)
            gathered[1] = ["rid-a", "rid-b"]  # sorted peer view

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Failed, KVPoll.WaitingForInput],
            ) as poll,
            patch(
                "sglang.srt.disaggregation.prefill.should_force_retry",
                return_value=False,
            ),
        ):
            bootstrapped, failed = queue.pop_bootstrapped(return_failed_reqs=True)

        # aligned_reqs is sorted by rid: [rid-a, rid-b], so poll[0] is rid-a's
        # (Failed) and poll[1] is rid-b's (WaitingForInput), regardless of the
        # unsorted local-queue insertion order [rid-c, rid-a, rid-b].
        senders = poll.call_args.args[0]
        self.assertIs(senders[0], req_a.disagg_kv_sender)
        self.assertIs(senders[1], req_b.disagg_kv_sender)
        self.assertEqual(bootstrapped, [req_b])
        self.assertEqual(failed, [req_a])
        self.assertEqual(queue.queue, [req_c])
        queue.scheduler.handle_bootstrap_failure.assert_called_once_with(req_a)

    def test_rid_align_tp_and_cp_intersect_in_sequence(self):
        """tp == cp (both sizes equal, here 2): the two all_gathers intersect in order.

        pop_bootstrapped runs all_gather once per active group in deterministic
        tp-then-cp order, so the mock keys off the call index instead of group
        identity. The first pass narrows to peers that have rid-a; the second
        narrows further to rid-a only.
        """
        queue = _make_queue(["rid-a", "rid-b"], tp_world_size=2, cp_world_size=2)
        req_a, req_b = queue.queue

        phase = 0

        def all_gather_object(gathered, obj, group=None):
            nonlocal phase
            gathered[0] = list(obj)
            gathered[1] = ["rid-a", "rid-c"] if phase == 0 else ["rid-a"]
            phase += 1

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.WaitingForInput],
            ) as poll,
            patch(
                "sglang.srt.disaggregation.prefill.should_force_retry",
                return_value=False,
            ),
        ):
            bootstrapped = queue.pop_bootstrapped()

        # Both groups gathered (tp pass then cp pass); the sequential intersection
        # kills rid-b on every rank, so only rid-a is polled.
        self.assertEqual(phase, 2)
        senders = poll.call_args.args[0]
        self.assertEqual(len(senders), 1)
        self.assertIs(senders[0], req_a.disagg_kv_sender)
        self.assertEqual(bootstrapped, [req_a])
        self.assertEqual(queue.queue, [req_b])
        queue.finalize_bootstrap.assert_called_once_with(req_a)

    def test_no_common_rids_leaves_local_queue_untouched(self):
        queue = _make_queue(["rid-local-only"], tp_world_size=2)

        def all_gather_object(gathered, obj, group=None):
            gathered[0] = list(obj)
            gathered[1] = ["rid-peer-only"]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.dist.get_world_size",
                side_effect=_world_size_side_effect(queue),
            ),
            patch("sglang.srt.disaggregation.prefill.dist.all_reduce"),
            patch(
                "sglang.srt.disaggregation.prefill.dist.all_gather_object",
                side_effect=all_gather_object,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as poll,
        ):
            self.assertEqual(queue.pop_bootstrapped(), [])

        poll.assert_not_called()
        self.assertEqual([r.rid for r in queue.queue], ["rid-local-only"])


if __name__ == "__main__":
    unittest.main()
