"""Unit tests for the PP relay (rank 0's decisions forwarded down the pipeline)."""

import socket
import time
import unittest

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.distributed.pp_relay import PPRelay
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _round(relay: PPRelay, *, sends: int) -> list[_FakeWork]:
    # one scheduler round: bound the outstanding sends, then issue this round's
    relay.open_round()
    works = [_FakeWork() for _ in range(sends)]
    relay.work_rounds[-1].extend(works)
    return works


class TestPPRelayWindow(CustomTestCase):
    """Sends inside the window are never waited on, so a rank is not tied to the next
    rank's progress; sends older than the window are, so the list stays bounded."""

    def test_rounds_within_window_are_not_waited(self):
        relay = PPRelay(group=None, pp_rank=0, pp_size=2, tag=1, window=3)
        rounds = [_round(relay, sends=2) for _ in range(3)]
        self.assertFalse(any(w.waited for works in rounds for w in works))

        _round(relay, sends=1)
        self.assertTrue(all(w.waited for w in rounds[0]))
        self.assertFalse(any(w.waited for works in rounds[1:] for w in works))
        self.assertEqual(len(relay.work_rounds), 3)

    def test_window_one_is_the_single_round_drain(self):
        relay = PPRelay(group=None, pp_rank=0, pp_size=2, tag=1, window=1)
        first = _round(relay, sends=2)
        self.assertFalse(any(w.waited for w in first))

        _round(relay, sends=1)
        self.assertTrue(all(w.waited for w in first))
        self.assertEqual(len(relay.work_rounds), 1)

    def test_inactive_relay_passes_data_through(self):
        relay = PPRelay(group=None, pp_rank=0, pp_size=1, tag=1, window=4)
        data = torch.tensor([3, 4])
        self.assertIs(relay.relay_tensor(data), data)
        self.assertEqual(relay.relay_object(({1}, {2})), ({1}, {2}))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gloo_pp_rank(rank: int, port: int, result: dict) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}"
    )
    group = torch.distributed.group.WORLD
    elapsed = {}
    for phase, window, rounds in (("drain", 1, 2), ("window", 4, 4)):
        relay = PPRelay(group=group, pp_rank=rank, pp_size=2, tag=7, window=window)
        if rank == 1:
            time.sleep(1.0)
        start = time.perf_counter()
        received = []
        for i in range(rounds):
            relay.open_round()
            data = torch.tensor([i, 10 * i], dtype=torch.int64)
            received.append(relay.relay_tensor(data).tolist())
            received.append(relay.relay_object(({i}, {"round": i})))
        elapsed[phase] = time.perf_counter() - start
        for works in relay.work_rounds:
            for work in works:
                work.wait()
        if rank == 1:
            expected = []
            for i in range(rounds):
                expected += [[i, 10 * i], ({i}, {"round": i})]
            assert received == expected, received
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        result.update(elapsed)


class TestPPRelayGloo(CustomTestCase):
    """With real gloo sends, rank 0 must issue a window's worth of rounds while rank 1
    has posted no recv; a single-round window blocks rank 0 on rank 1 instead. Rank 1
    checks that tensors and objects arrive in order."""

    def test_leader_runs_ahead_by_window(self):
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Manager() as manager:
            result = manager.dict()
            port = _free_port()
            procs = [
                ctx.Process(target=_gloo_pp_rank, args=(rank, port, result))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=60)
            self.assertTrue(all(proc.exitcode == 0 for proc in procs), procs)
            elapsed = dict(result)
        self.assertGreater(elapsed["drain"], 0.5, elapsed)
        self.assertLess(elapsed["window"], 0.5, elapsed)


if __name__ == "__main__":
    unittest.main()
