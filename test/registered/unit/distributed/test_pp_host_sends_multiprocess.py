"""PP host sends and the torch profiler stop, on real gloo ranks (CPU).

A gloo send started under the torch profiler ends its RecordFunction in wait();
waited after profiler.stop(), that write lands in freed host memory. The stop
must wait every send of this rank first, without blocking a pipeline whose
output relay closes a ring.
"""

import os
import tempfile
import unittest
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.utils import pp_sync_polls
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow
from sglang.srt.observability.profile_utils import _ProfilerTorch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

REQ_TAG, OUT_TAG = 11, 12


def _init(rank, world, tmp):
    # A gloo send waited twice raises once this timeout passes.
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(os.path.join(tmp, "store"), world),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    group = object.__new__(GroupCoordinator)
    group.__dict__.update(
        rank=rank,
        rank_in_group=rank,
        ranks=list(range(world)),
        world_size=world,
        cpu_group=dist.group.WORLD,
    )
    own = [dist.new_group([r], backend="gloo") for r in range(world)][rank]
    profiler = _ProfilerTorch(
        with_stack=False,
        record_shapes=False,
        activities=["CPU"],
        output_dir=tmp,
        output_prefix=f"rank{rank}",
        output_suffix="",
        profile_id="t",
        cpu_group=own,
        first_rank_in_node=True,
    )
    return group, profiler


def _stop(profiler, sends):
    profiler.stop()
    # Every send issued under the profiler has ended before stop() returned.
    open_sends = [w for w in sends if not w.work.is_completed()]
    assert not open_sends, f"{len(open_sends)} sends still open at stop()"


def _poll_relay_worker(rank, tmp):
    """Rank 0 profiles main's PD poll relay with 63 rounds of sends un-waited, and
    stops the profiler halfway; the relay then keeps going on both ranks."""
    with published_topology(role="scheduler", ranks={"world_rank": rank}, pp_size=2):
        group, profiler = _init(rank, 2, tmp)
        window = PPSyncSendWindow(rounds=64)
        sends = []
        if rank == 0:
            profiler.start()
        for i in range(160):
            polls = [i % 5] * 16 if rank == 0 else None
            assert pp_sync_polls(polls, group, rank, 2, window) == [i % 5] * 16
            sends.extend(window._outstanding[-1])
            if rank == 0 and i == 80:
                _stop(profiler, sends)
        dist.barrier()
        dist.destroy_process_group()


# Iteration at which each stage stops its profiler, and whether the stop is decided
# at the iteration's end or at the next iteration's ingest (a manual /stop_profile).
RING_SCENARIOS = (((3, 5, 4), False), ((8, 2, 5), False), ((3, 5, 4), True))


def _ring_worker(rank, tmp):
    """Three stages in lockstep: a request relay sent at the end of each iteration
    (waited at the next), main's poll relay, and the last stage's output relay back
    to stage 0 (received pp_size - 1 iterations later). Each stage stops its
    profiler at its own iteration."""
    pp = 3
    with published_topology(role="scheduler", ranks={"world_rank": rank}, pp_size=pp):
        group, profiler = _init(rank, pp, tmp)
        for stop_at, at_ingest in RING_SCENARIOS:
            _ring_scenario(group, profiler, rank, pp, stop_at[rank], at_ingest)
            dist.barrier()
        dist.destroy_process_group()


def _ring_scenario(group, profiler, rank, pp, stop_at, at_ingest):
    window = PPSyncSendWindow(rounds=pp)
    req_work, out_work, sends = [], [], []
    profiler.start()
    iterations = 10
    for t in range(iterations):
        if rank > 0:
            assert group.recv_object(src=rank - 1, tag=REQ_TAG) == t
        if at_ingest and t == stop_at + 1:
            _stop(profiler, sends)
        for w in req_work:
            w.wait()
        polls = [t] * 8 if rank == 0 else None
        assert pp_sync_polls(polls, group, rank, pp, window) == [t] * 8
        sends.extend(window._outstanding[-1])
        if rank == pp - 1:
            for w in out_work:
                w.wait()
            out_work = group.send_object(
                ("out", t), dst=0, async_send=True, tag=OUT_TAG
            )
            sends.extend(out_work)
        if rank == 0 and t >= pp - 1:
            assert group.recv_object(src=pp - 1, tag=OUT_TAG) == ("out", t - (pp - 1))
        if rank < pp - 1:
            req_work = group.send_object(t, dst=rank + 1, async_send=True, tag=REQ_TAG)
            sends.extend(req_work)
        if not at_ingest and t == stop_at:
            _stop(profiler, sends)
    if rank == 0:
        for t in range(iterations - (pp - 1), iterations):
            assert group.recv_object(src=pp - 1, tag=OUT_TAG) == ("out", t)
    for w in req_work + out_work:
        w.wait()
    for _ in range(pp):
        window.begin_round()


class TestProfilerStopOnGlooRanks(CustomTestCase):
    def test_stop_ends_every_send_of_the_poll_relay_window(self):
        """Main's poll relay keeps pp_size + depth rounds of gloo sends un-waited.
        The stop must end every send started under the profiler, and the window's
        own later wait of a drained send must return, not wait for it again."""
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_poll_relay_worker, args=(tmp,), nprocs=2, join=True)

    def test_stages_stopping_at_different_iterations_do_not_deadlock(self):
        """The stop waits only sends of iterations up to the current one, which the
        loop itself waits by the next iteration's commits, so it closes no cycle
        through the output ring; the same holds for a stop decided at ingest."""
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_ring_worker, args=(tmp,), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
