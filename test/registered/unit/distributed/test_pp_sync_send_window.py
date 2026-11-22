"""Unit tests for srt/distributed/pp_sync_window."""

import os
import tempfile
import time
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.pp_sync_window import PPReceiveAhead, PPSyncSendWindow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _round(window: PPSyncSendWindow, sends: int) -> list[_FakeWork]:
    """One scheduler round: open it, then issue this round's sends."""
    window.begin_round()
    works = [_FakeWork() for _ in range(sends)]
    window.add(works)
    return works


class TestPPSyncSendWindow(CustomTestCase):
    """Sends inside the window are never waited on, so a rank is not tied to the
    next rank's progress; sends older than the window are, so the backlog stays
    bounded (the leak #28916 fixed with the single-round drain)."""

    def test_rounds_within_window_are_not_waited(self):
        window = PPSyncSendWindow(rounds=3)
        rounds = [_round(window, sends=2) for _ in range(3)]
        self.assertFalse(any(w.waited for works in rounds for w in works))
        self.assertEqual(len(window), 3)

        _round(window, sends=1)
        self.assertTrue(all(w.waited for w in rounds[0]))
        self.assertFalse(any(w.waited for works in rounds[1:] for w in works))
        self.assertEqual(len(window), 3)

    def test_window_one_is_the_single_round_drain(self):
        window = PPSyncSendWindow(rounds=1)
        first = _round(window, sends=2)
        self.assertFalse(any(w.waited for w in first))

        second = _round(window, sends=1)
        self.assertTrue(all(w.waited for w in first))
        self.assertFalse(any(w.waited for w in second))
        self.assertEqual(len(window), 1)

    def test_non_positive_depth_still_keeps_the_open_round(self):
        """A window of zero rounds would pop the round it just opened and leave
        nowhere to record the sends."""
        window = PPSyncSendWindow(rounds=0)
        first = _round(window, sends=1)
        self.assertFalse(first[0].waited)
        _round(window, sends=1)
        self.assertTrue(first[0].waited)


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.001)


class TestPPReceiveAhead(CustomTestCase):
    """The thread posts the channel's receives ahead of the scheduler, in order
    and at most ``depth`` ahead; a failed receive fails every later pop."""

    def test_pops_in_receive_order_and_receives_at_most_depth_ahead(self):
        calls = []

        def recv_one():
            calls.append(len(calls))
            return calls[-1]

        receiver = PPReceiveAhead(recv_one=recv_one, depth=2, name="t-recv")
        _wait_until(lambda: receiver._queue.qsize() == 2)
        # Both permits are taken: a third receive waits for a pop.
        self.assertEqual(len(calls), 2)
        self.assertEqual(receiver.pop(), 0)
        _wait_until(lambda: len(calls) == 3)
        self.assertEqual([receiver.pop() for _ in range(4)], [1, 2, 3, 4])
        receiver.close()

    def test_a_failed_receive_fails_this_and_every_later_pop(self):
        messages = iter([0])

        def recv_one():
            return next(messages)

        receiver = PPReceiveAhead(recv_one=recv_one, depth=4, name="t-recv-fail")
        self.assertEqual(receiver.pop(), 0)
        for _ in range(2):
            with self.assertRaises(RuntimeError) as ctx:
                receiver.pop()
            self.assertIsInstance(ctx.exception.__cause__, StopIteration)

    def test_close_stops_posting_receives_and_fails_later_pops(self):
        calls = []

        def recv_one():
            calls.append(len(calls))
            return calls[-1]

        receiver = PPReceiveAhead(recv_one=recv_one, depth=1, name="t-recv-close")
        self.assertEqual(receiver.pop(), 0)
        # The pop's permit lets the thread receive one more, then it waits.
        _wait_until(lambda: receiver._queue.qsize() == 1)
        receiver.close()
        receiver._thread.join(timeout=10)
        self.assertFalse(receiver._thread.is_alive())
        self.assertEqual(len(calls), 2)
        with self.assertRaises(RuntimeError):
            receiver.pop()


def _receive_ahead_worker(rank, store_path, rounds):
    store = dist.FileStore(store_path, 2)
    dist.init_process_group(
        "gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    group = object.__new__(GroupCoordinator)
    group.__dict__.update(
        rank=rank,
        rank_in_group=rank,
        ranks=[0, 1],
        world_size=2,
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
    )

    def message(i):
        return {"lens": torch.arange(i, i + 4), "odd": torch.full((3,), i), "seq": i}

    if rank == 0:
        works = []
        for i in range(rounds + 1):
            works += group.send_tensor_dict(message(i), dst=1, async_send=True)
        assert [group.recv_object(src=1) for _ in range(rounds)] == list(range(rounds))
        for work in works:
            work.wait()
    else:
        receiver = PPReceiveAhead(
            recv_one=lambda: group.recv_tensor_dict_host_part(src=0),
            depth=2,
            name="t-recv-gloo",
        )
        works = []
        for i in range(rounds):
            # The scheduler thread sends on the group the receive thread reads.
            works += group.send_object(i, dst=0, async_send=True)
            got = group.recv_tensor_dict(src=0, host_part=receiver.pop())
            want = message(i)
            assert got.keys() == want.keys() and got["seq"] == i
            assert torch.equal(got["lens"], want["lens"])
            assert torch.equal(got["odd"], want["odd"])
        # The receive posted after the last pop takes the sender's next message.
        receiver.close()
        receiver._thread.join(timeout=30)
        assert not receiver._thread.is_alive()
        for work in works:
            work.wait()
    dist.barrier()
    dist.destroy_process_group()


class TestPPReceiveAheadOnGloo(CustomTestCase):
    def test_receives_ahead_while_the_scheduler_thread_sends_on_the_group(self):
        """Rank 1's thread receives tensor dicts on a gloo group while its main
        thread sends on the same group; messages arrive whole and in order."""
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(
                _receive_ahead_worker,
                args=(os.path.join(tmp, "store"), 20),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
