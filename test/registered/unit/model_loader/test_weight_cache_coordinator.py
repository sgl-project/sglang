"""CPU tests for the vote of snapshot loading (weight_cache/coordinator.py)."""

import unittest

import msgspec

from sglang.srt.weight_cache import coordinator
from sglang.srt.weight_cache.coordinator import Ballot, VoteError, vote
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Group:
    """The parts of a GroupCoordinator a vote uses. ``others`` is what the peers
    pass to their own vote, in rank order; ``arrives`` replaces what the gather
    brings from them."""

    def __init__(self, *, rank, others, cpu_group=object(), arrives=None, ranks=None):
        self.rank_in_group = rank
        self.world_size = len(others) + 1
        self.cpu_group = cpu_group
        if ranks is not None:
            self.ranks = ranks
        self._arrives = (
            [coordinator._cast(other) for other in others]
            if arrives is None
            else arrives
        )
        self.calls = []

    def barrier(self):
        self.calls.append("barrier")

    def all_gather_object(self, obj):
        self.calls.append("gather")
        gathered = list(self._arrives)
        gathered.insert(self.rank_in_group, obj)
        return gathered


def _pair(name, value):
    """One field of a msgpack map."""
    return msgspec.msgpack.encode(name) + msgspec.msgpack.encode(value)


def _vote_on_every_rank(ballots):
    """What each rank gets, or the report it raises, when the ranks cast ``ballots``."""
    outcomes = []
    for rank, mine in enumerate(ballots):
        others = ballots[:rank] + ballots[rank + 1 :]
        try:
            outcomes.append(vote(group=_Group(rank=rank, others=others), ballot=mine))
        except VoteError as e:
            outcomes.append(str(e))
    return outcomes


class TestVote(CustomTestCase):
    def _one_report(self, ballots):
        reports = _vote_on_every_rank(ballots)
        self.assertIsInstance(reports[0], str)
        self.assertEqual(reports, [reports[0]] * len(ballots))
        return reports[0]

    def test_one_rank_votes_without_a_collective(self):
        ballot = Ballot(agree=b"a", payload=b"p")
        alone = _Group(rank=0, others=[])
        for group in (None, alone):
            self.assertEqual(vote(group=group, ballot=ballot), [ballot])
            for error in ("boom", ""):
                with self.assertRaisesRegex(VoteError, f"rank 0: {error}"):
                    vote(group=group, ballot=Ballot(error=error))
        self.assertEqual(alone.calls, [])

    def test_every_rank_gets_the_ballots_in_rank_order(self):
        for ranks in (2, 3):
            ballots = [
                Ballot(agree=b"same", payload=bytes([rank]) * (48 if rank == 0 else 32))
                for rank in range(ranks)
            ]
            self.assertEqual(_vote_on_every_rank(ballots), [ballots] * ranks)

    def test_barrier_comes_before_the_gather(self):
        group = _Group(rank=1, others=[Ballot()])
        vote(group=group, ballot=Ballot())
        self.assertEqual(group.calls, ["barrier", "gather"])

    def test_an_error_on_any_rank_fails_every_rank_with_one_report(self):
        report = self._one_report(
            [Ballot(agree=b"x"), Ballot(error="no snapshot"), Ballot(agree=b"x")]
        )
        self.assertTrue(report.startswith("3 rank(s) voted and cannot go on"), report)
        self.assertIn("rank 1: no snapshot", report)
        self.assertNotIn("disagree", report)
        self.assertIn(
            "rank 0: leader", self._one_report([Ballot(error="leader"), Ballot()])
        )
        report = self._one_report([Ballot(error="e0"), Ballot(error="e1")])
        self.assertIn("rank 0: e0; rank 1: e1", report)

    def test_different_agree_values_fail_every_rank_with_one_report(self):
        same_start = b"\x92\xd9\x40" + b"f" * 64
        report = self._one_report(
            [Ballot(agree=same_start + b"hit"), Ballot(agree=same_start + b"miss")]
        )
        self.assertIn("rank 0 hold 70 bytes with digest ", report)
        self.assertIn("rank 1 hold 71 bytes with digest ", report)
        report = self._one_report(
            [Ballot(agree=b"\x01"), Ballot(agree=b"\x02"), Ballot(agree=b"\x01")]
        )
        first, second = report.split("ranks [0, 2] hold ")[1].split("; rank 1 hold ")
        self.assertNotEqual(first, second)

    def test_a_failed_rank_is_not_compared(self):
        # Its agree value was never made: only the ranks that can go on are compared.
        report = self._one_report(
            [Ballot(agree=b"x"), Ballot(error="e1"), Ballot(error="e2")]
        )
        self.assertIn("rank 1: e1; rank 2: e2", report)
        self.assertNotIn("disagree", report)

    def test_errors_and_disagreement_are_both_reported_on_every_rank(self):
        report = self._one_report(
            [Ballot(agree=b"\x01"), Ballot(agree=b""), Ballot(error="e")]
        )
        self.assertIn("rank 2: e", report)
        self.assertIn("rank 1 hold 0 bytes", report)

    def test_a_ballot_that_cannot_be_sent_is_cast_as_that_ranks_error(self):
        class Renamed(Ballot, frozen=True, kw_only=True, rename={"error": "reason"}):
            pass

        class Text(str):
            def encode(self, *args):
                raise RuntimeError("no")

        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no")

        class Raises(bytes):
            def __len__(self):
                raise Unprintable()

        too_big = Ballot(payload=bytes(coordinator._MAX_BALLOT_BYTES))
        cases = {
            "Ballot.agree and Ballot.payload must be bytes": Ballot(payload=object()),
            "Ballot.error is not a str": Ballot(error=4),
            "TypeError: Ballot.error is not a str": Ballot(error=Text("veto")),
            "must be bytes": Ballot(agree=Raises(b"x")),
            "the limit is": too_big,
            "it is not a Ballot": "not a ballot",
            # A subclass can rename its fields on the wire: its veto would be lost.
            "TypeError: it is not a Ballot": Renamed(error="checkpoint corrupt"),
        }
        for expected, bad in cases.items():
            report = self._one_report([Ballot(), bad])
            self.assertIn("rank 1: this rank's ballot cannot be cast", report)
            self.assertIn(expected, report)
            with self.assertRaisesRegex(VoteError, "rank 0: this rank's ballot"):
                vote(group=None, ballot=bad)
        # Text that UTF-8 cannot carry is escaped, not dropped.
        report = self._one_report([Ballot(error="bad \ud800")])
        self.assertIn("rank 0: bad \\ud800", report)

    def test_a_gathered_ballot_it_cannot_read_is_that_ranks_error(self):
        good = coordinator._cast(Ballot())
        wire = msgspec.msgpack.decode(good)
        unreadable_ballots = (
            b"\xc1",
            None,
            msgspec.msgpack.encode({}),  # a veto under other names would read as a yes
            msgspec.msgpack.encode(dict(wire, agree=3)),
            msgspec.msgpack.encode(dict(wire, unexpected="fail")),
            msgspec.msgpack.encode({k: v for k, v in wire.items() if k != "error"}),
            msgspec.msgpack.encode(dict(wire, payload=bytes(1 << 20))),
            bytearray(good),
            # A field sent twice: the veto first, then no error.
            b"\x84" + _pair("error", "checkpoint corrupt") + good[1:],
            b"\x84" + good[1:] + _pair("agree", b"x"),
            b"\x84" + good[1:] + _pair("payload", b"x"),
            # The same ballot in another encoding of its map.
            b"\xde\x00\x03" + good[1:],
        )
        for unreadable in unreadable_ballots:
            group = _Group(rank=0, others=[None, None], arrives=[good, unreadable])
            with self.assertRaisesRegex(VoteError, "rank 2: its ballot cannot be read"):
                vote(group=group, ballot=Ballot())

    def test_a_failed_gather_is_not_a_vote(self):
        class Broken(_Group):
            def all_gather_object(self, obj):
                raise RuntimeError("connection closed by peer")

        with self.assertRaisesRegex(RuntimeError, "connection closed by peer"):
            vote(group=Broken(rank=0, others=[Ballot()]), ballot=Ballot())

    def test_a_group_without_its_process_group_does_not_vote(self):
        destroyed = _Group(rank=0, others=[Ballot()], cpu_group=None)
        with self.assertRaisesRegex(VoteError, "no CPU process group"):
            vote(group=destroyed, ballot=Ballot())
        self.assertEqual(destroyed.calls, [])

    def test_a_group_whose_ranks_are_not_ascending_does_not_vote(self):
        # Its process group returns the ballots in another order than its ranks.
        permuted = _Group(rank=0, others=[Ballot()], ranks=[1, 0])
        with self.assertRaisesRegex(VoteError, r"not in ascending order: \[1, 0\]"):
            vote(group=permuted, ballot=Ballot())
        self.assertEqual(permuted.calls, [])
        ascending = _Group(rank=1, others=[Ballot()], ranks=[2, 5])
        self.assertEqual(len(vote(group=ascending, ballot=Ballot())), 2)

    def test_the_report_names_every_failing_rank_within_its_bound(self):
        report = self._one_report([Ballot(error="x" * 10**6), Ballot()])
        self.assertLess(len(report), coordinator._MAX_ERROR_CHARS + 100)
        report = self._one_report([Ballot(error="y" * 4000)] * 8)
        self.assertIn("ranks [0, 1, 2, 3, 4, 5, 6, 7]: yyy", report)
        self.assertLess(len(report), 4100)
        ballots = [Ballot(error=f"{rank} " + "z" * 4000) for rank in range(8)]
        report = self._one_report(ballots)
        for rank in range(8):
            self.assertIn(f"rank {rank}: {rank} zzz", report)
        self.assertLess(len(report), coordinator._MAX_REPORT_CHARS + 400)


if __name__ == "__main__":
    unittest.main()
