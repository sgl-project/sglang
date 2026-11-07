"""A band's page-usage ledger decides moves and reuse from logical state only:
two ranks fed the same work, whose signals complete at different times, make
the same decisions."""

import random
import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer, PageUsage
from sglang.srt.mem_cache.kv.alloc.signals import StreamTransfer, TransferRelease
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Signal:
    """A signal whose completion the test controls per rank."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired


class _Rank:
    def __init__(self):
        self.usage = PageUsage()
        self.forwards = []  # recorded forward events, in order
        self.transfers = []  # (signal, release or None)


def _decisions(rank: _Rank):
    """Everything a move, reuse or placement decision reads, named by
    recording order instead of by the rank's own objects."""
    usage = rank.usage
    usage.prune()
    forward_index = {id(e): i for i, e in enumerate(rank.forwards)}
    release_index = {
        id(release): i
        for i, (_, release) in enumerate(rank.transfers)
        if release is not None
    }
    return (
        tuple(usage.forward_retired(e) for e in rank.forwards),
        tuple(
            usage.transfer_ended(release)
            for _, release in rank.transfers
            if release is not None
        ),
        usage.writes_unknown(),
        tuple(forward_index[id(use.signal)] for use in usage.forwards()),
        # Transfers that may stop a walk: those with a logical release.
        tuple(
            release_index[id(use.release)]
            for use in usage.transfers()
            if use.writes and use.release is not None
        ),
        frozenset(usage.pending_pages),
        # Fenced bytes stop an opportunistic walk.
        tuple(usage.fence_meets(b, b + 1) for b in range(_FENCE_BYTES)),
    )


_FENCE_BYTES = 32


class TestPageUsageAcrossRanks(CustomTestCase):
    def test_decisions_do_not_depend_on_signal_timing(self):
        timing_differed = False
        for seed in range(300):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                ranks = [_Rank(), _Rank()]
                next_page = 0
                for _ in range(40):
                    op = rng.choice(
                        [
                            "forward",
                            "retire",
                            "transfer",
                            "release",
                            "vacate",
                            "drain",
                            "fence",
                            "write",
                        ]
                    )
                    if op == "forward":
                        unknown = rng.random() < 0.2
                        ids = torch.tensor([rng.randrange(64)])
                        for rank in ranks:
                            event = _Signal()
                            rank.forwards.append(event)
                            rank.usage.latest_forward = event
                            rank.usage.record_forward(
                                event, ids, writes_unknown=unknown
                            )
                    elif op == "retire" and ranks[0].forwards:
                        i = rng.randrange(len(ranks[0].forwards))
                        for rank in ranks:
                            rank.usage.retire_forward(rank.forwards[i])
                    elif op == "transfer":
                        logical = rng.random() < 0.5
                        writes = rng.random() < 0.5
                        ids = torch.tensor([rng.randrange(64)])
                        for rank in ranks:
                            signal = _Signal()
                            release = TransferRelease() if logical else None
                            rank.transfers.append((signal, release))
                            rank.usage.record_transfer(
                                signal,
                                ids,
                                writes=writes,
                                release=release,
                                unit_tokens=None,
                            )
                    elif op == "release" and ranks[0].transfers:
                        i = rng.randrange(len(ranks[0].transfers))
                        for rank in ranks:
                            release = rank.transfers[i][1]
                            if release is not None:
                                release.released = True
                    elif op == "vacate":
                        pages = [next_page, next_page + 1]
                        next_page += 2
                        for rank in ranks:
                            readers = tuple(
                                release
                                for _, release in rank.transfers[-2:]
                                if release is not None
                            )
                            key = (rank.usage.latest_forward, *readers)
                            rank.usage.vacate(
                                key,
                                (),
                                page_bytes=1,
                                pages=pages,
                                batch=torch.tensor(pages),
                            )
                    elif op == "drain":
                        drained = [
                            sorted(
                                p
                                for use in rank.usage.take_ready_vacated(urgent=False)
                                for p in use.pages
                            )
                            for rank in ranks
                        ]
                        self.assertEqual(drained[0], drained[1])
                    elif op == "fence":
                        # Bytes a move took out of the band while readers ran.
                        first = rng.randrange(_FENCE_BYTES - 4)
                        runs = [(first, first + rng.randrange(1, 4))]
                        for rank in ranks:
                            readers = tuple(
                                (signal, release) if release is not None else signal
                                for signal, release in rank.transfers[-1:]
                            )
                            key = tuple(
                                k
                                for k in (rank.usage.latest_forward, *readers)
                                if k is not None
                            )
                            if key:
                                rank.usage.vacate(
                                    key if len(key) > 1 else key[0], runs, page_bytes=1
                                )
                    elif op == "write":
                        # A copy into some bytes waits the fences they meet;
                        # how many it waits may differ, as it skips fired ones.
                        lo = rng.randrange(_FENCE_BYTES - 2)
                        for rank in ranks:
                            rank.usage.take_fences_meeting(lo, lo + 2)
                    # Signals complete independently on each rank.
                    for rank in ranks:
                        for signal in [*rank.forwards, *(s for s, _ in rank.transfers)]:
                            if not signal.fired and rng.random() < 0.3:
                                signal.fired = True
                    if len(ranks[0].usage.transfers()) != len(
                        ranks[1].usage.transfers()
                    ):
                        timing_differed = True
                    self.assertEqual(_decisions(ranks[0]), _decisions(ranks[1]))
        # The ranks did see different timing: transfers without a logical
        # release were dropped at different steps.
        self.assertTrue(timing_differed)


class TestFencesSplitByReaders(CustomTestCase):
    def test_a_wait_for_part_of_a_fence_keeps_the_rest(self):
        """A move vacates bytes 0-5 while a forward runs. On one rank a host
        copy is still seen reading byte 5 (its event has not fired there), so
        that byte's fence carries the copy too and is a fence apart; on the
        other it is part of one fence. A schedule-stream copy into bytes 0-1
        waits the forward on both ranks; bytes 2-5 stay fenced on the forward
        on both, so a take that prefers bytes no running forward reads chooses
        alike on every rank."""
        guarded = []
        for split in (False, True):
            usage = PageUsage()
            forward = _Signal()
            usage.register_forward(forward)
            usage.latest_forward = forward
            if split:
                usage.vacate(forward, [(0, 5)], page_bytes=1)
                copy = StreamTransfer(_Signal())
                usage.vacate((forward, copy), [(5, 6)], page_bytes=1)
            else:
                usage.vacate(forward, [(0, 6)], page_bytes=1)
            usage.take_clearance(0, 2, (Consumer.SCHEDULE_COPY,))
            guarded.append(
                [usage.fences_need(b, b + 1, (Consumer.ZERO,)) for b in range(6)]
            )
        self.assertEqual(guarded[0], guarded[1])
        self.assertEqual(guarded[0], [False, False, True, True, True, True])


if __name__ == "__main__":
    unittest.main()
