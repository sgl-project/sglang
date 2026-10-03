"""Unit tests for how wide a rank's readmission is — no server, no Mooncake.

A rank has to rejoin exactly the groups its peers readmit it to. If a survivor calls
``recover_ranks`` on a group the returning rank never called ``join_group`` on, or the
other way round, Mooncake sizes that group's next collective from a bitmap the two
sides disagree about. ``mlp_sync`` runs over ``tp_group``, so a plain fault recovery
hits it immediately.

Shrink announces a departure in every live group, so a bare fault has to be recovered
just as widely. A scale regrow must stay narrow: its joiner is inactive rather than
faulted, and ``joinGroup`` takes only an isolated or inactive rank. These tests pin
both breadths and, most importantly, that the two sides of each agree.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

WORLD_SIZE = 4
FAULTED_RANK = 2


class FakePg:
    """Stands in for a ProcessGroup. Identity is all the code under test uses."""

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"FakePg({self.label})"


class FakeGroup:
    """The parts of GroupCoordinator the recover and join paths read."""

    def __init__(self, name, ranks):
        self.unique_name = name
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.device_group = FakePg(f"{name}.device")
        self.cpu_group = FakePg(f"{name}.cpu")
        self.use_message_queue_broadcaster = False
        self.mq_broadcaster = None


class RecordingMooncake:
    """Records which PGs each side touched, which is the whole point here."""

    def __init__(self):
        self.recovered = []
        self.joined = []
        self.peer_state_ok = True

    def recover_ranks(self, pg, ranks):
        self.recovered.append((pg.label, tuple(ranks)))

    def join_group(self, pg):
        self.joined.append(pg.label)

    def get_peer_state(self, pg, ranks):
        return [self.peer_state_ok] * len(ranks)

    def deactivate_ranks(self, pg, ranks):
        pass

    def as_module(self):
        module = types.ModuleType("mooncake.pg")
        module.recover_ranks = self.recover_ranks
        module.join_group = self.join_group
        module.get_peer_state = self.get_peer_state
        module.deactivate_ranks = self.deactivate_ranks
        return module


class _GroupFixture(CustomTestCase):
    """A world group plus the sub-groups a shrink would have deactivated."""

    def setUp(self):
        import sglang.srt.elastic_ep.elastic_ep as eep

        self.eep = eep
        ranks = list(range(WORLD_SIZE))
        self.world = FakeGroup("world:0", ranks)
        self.tp = FakeGroup("tp:0", ranks)
        self.moe_ep = FakeGroup("moe_ep:0", ranks)
        self.attn = FakeGroup("attn_tp:0", ranks)
        # Nothing to join in a group of one, and no peer holds the other side.
        self.solo = FakeGroup("solo:0", [FAULTED_RANK])
        self.groups = [self.world, self.tp, self.moe_ep, self.attn, self.solo]

        self.mooncake = RecordingMooncake()
        self.default_pg = FakePg("default_world")

        patches = [
            mock.patch.dict(
                sys.modules,
                {
                    "mooncake": types.ModuleType("mooncake"),
                    "mooncake.pg": self.mooncake.as_module(),
                },
            ),
            mock.patch.object(
                self.eep, "_iter_live_parallel_groups", lambda: iter(self.groups)
            ),
            mock.patch.object(
                self.eep.torch.distributed, "group", mock.Mock(WORLD=self.default_pg)
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        # _WORLD is in the same registry the iterator walks, so both paths have to
        # recognise and skip it; its own PGs are handled by the WORLD-scope step.
        world_patch = mock.patch.object(self.eep.parallel_state, "_WORLD", self.world)
        world_patch.start()
        self.addCleanup(world_patch.stop)

    def sub_group_labels(self):
        """Every non-WORLD PG a shrink would have deactivated for the faulted rank."""
        labels = set()
        for group in (self.tp, self.moe_ep, self.attn):
            labels.add(group.device_group.label)
            labels.add(group.cpu_group.label)
        return labels


class TestBareFaultRecoveryIsSymmetric(_GroupFixture):
    def test_survivor_readmits_every_sub_group(self):
        self.assertTrue(self.eep._recover_parallel_groups([FAULTED_RANK]))
        recovered = {label for label, _ in self.mooncake.recovered}
        self.assertEqual(
            recovered,
            self.sub_group_labels(),
            "a group left out here mis-sizes its next collective",
        )

    def test_returning_rank_joins_exactly_what_survivors_readmit(self):
        self.eep._join_world_group(include_subgroups=True, include_parallel_groups=True)
        joined = set(self.mooncake.joined)
        # The WORLD-scope PGs are on both sides of the handshake too, but they are the
        # part that already worked; the sub-groups are what this test exists for.
        self.assertEqual(
            joined & self.sub_group_labels(),
            self.sub_group_labels(),
            "the returning rank must join every group a survivor readmits it to",
        )

    def test_the_two_sides_agree(self):
        self.eep._recover_parallel_groups([FAULTED_RANK])
        readmitted = {label for label, _ in self.mooncake.recovered}
        self.mooncake.recovered.clear()
        self.eep._join_world_group(include_subgroups=True, include_parallel_groups=True)
        joined = set(self.mooncake.joined)
        self.assertEqual(
            readmitted,
            joined & self.sub_group_labels(),
            "readmit and rejoin must cover the same groups",
        )

    def test_world_group_is_not_taken_twice(self):
        """_WORLD rides in the same registry; the WORLD-scope step already has it."""
        self.eep._recover_parallel_groups([FAULTED_RANK])
        for label, _ in self.mooncake.recovered:
            self.assertNotIn("world:0", label)

        self.eep._join_world_group(include_subgroups=True, include_parallel_groups=True)
        self.assertEqual(
            self.mooncake.joined.count("world:0.cpu"),
            1,
            "once from the WORLD-scope step; twice trips joinGroup's state check",
        )

    def test_a_solo_group_is_skipped_on_both_sides(self):
        """No peer holds the other half of a one-rank group."""
        self.eep._recover_parallel_groups([FAULTED_RANK])
        self.eep._join_world_group(include_subgroups=True, include_parallel_groups=True)
        touched = {label for label, _ in self.mooncake.recovered} | set(
            self.mooncake.joined
        )
        self.assertNotIn(self.solo.device_group.label, touched)
        self.assertNotIn(self.solo.cpu_group.label, touched)

    def test_peers_that_never_arrive_leave_it_for_a_later_tick(self):
        """Bounded, so a joiner that never shows up does not wedge the event loop."""
        self.mooncake.peer_state_ok = False
        with mock.patch.object(self.eep, "_PEER_STATE_POLL_INTERVAL_SEC", 0.0):
            ok = self.eep._wait_for_peer_state(
                self.tp.cpu_group, [FAULTED_RANK], budget_s=0.01
            )
        self.assertFalse(ok)


class TestScaleRegrowStaysNarrow(_GroupFixture):
    def test_a_scale_joiner_does_not_touch_the_sub_groups(self):
        """Its slot is inactive, not faulted, and joinGroup rejects it there."""
        self.eep._join_world_group(include_subgroups=True)
        self.assertEqual(
            set(self.mooncake.joined) & self.sub_group_labels(),
            set(),
            "a scale regrow joiner must stay out of tp / moe_ep / attn",
        )

    def test_an_append_joiner_takes_world_only(self):
        self.eep._join_world_group()
        self.assertEqual(self.mooncake.joined, [self.default_pg.label])


if __name__ == "__main__":
    unittest.main()
