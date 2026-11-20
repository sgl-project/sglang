"""A rank must rejoin exactly the groups its peers readmit it to, or Mooncake mis-sizes
that group's next collective. No server, no Mooncake."""

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
    def __init__(self, label):
        self.label = label


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


class RecordingMooncake(types.ModuleType):
    """Stands in for ``mooncake.pg`` and records which PGs each side touched."""

    def __init__(self):
        super().__init__("mooncake.pg")
        self.recovered, self.joined, self.peer_state_ok = [], [], True

    def recover_ranks(self, pg, ranks):
        self.recovered.append((pg.label, tuple(ranks)))

    def join_group(self, pg):
        self.joined.append(pg.label)

    def get_peer_state(self, pg, ranks):
        return [self.peer_state_ok] * len(ranks)

    def deactivate_ranks(self, pg, ranks):
        pass


class _GroupFixture(CustomTestCase):
    """A world group plus the sub-groups a shrink would have deactivated."""

    def setUp(self):
        import sglang.srt.elastic_ep.elastic_ep as eep

        self.eep = eep
        self.world, self.tp, self.moe_ep, self.attn = (
            FakeGroup(name, range(WORLD_SIZE))
            for name in ("world:0", "tp:0", "moe_ep:0", "attn_tp:0")
        )
        # Nothing to join in a group of one, and no peer holds the other side.
        self.solo = FakeGroup("solo:0", [FAULTED_RANK])
        self.groups = [self.world, self.tp, self.moe_ep, self.attn, self.solo]
        self.mooncake = RecordingMooncake()
        self.default_pg = FakePg("default_world")
        modules = {
            "mooncake": types.ModuleType("mooncake"),
            "mooncake.pg": self.mooncake,
        }
        for patch in (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                eep, "_iter_live_parallel_groups", lambda: iter(self.groups)
            ),
            mock.patch.object(
                eep.torch.distributed, "group", mock.Mock(WORLD=self.default_pg)
            ),
            # _WORLD is in the registry the iterator walks, so both paths must skip it;
            # its own PGs are handled by the WORLD-scope step.
            mock.patch.object(eep.parallel_state, "_WORLD", self.world),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def sub_group_labels(self):
        """Every non-WORLD PG a shrink would have deactivated for the faulted rank."""
        return {
            pg.label
            for group in (self.tp, self.moe_ep, self.attn)
            for pg in (group.device_group, group.cpu_group)
        }

    def join_all(self):
        self.eep._join_world_group(include_subgroups=True, include_parallel_groups=True)


class TestBareFaultRecoveryIsSymmetric(_GroupFixture):
    def test_the_two_sides_agree_on_every_sub_group(self):
        self.assertTrue(self.eep._recover_parallel_groups([FAULTED_RANK]))
        readmitted = {label for label, _ in self.mooncake.recovered}
        self.assertEqual(
            readmitted,
            self.sub_group_labels(),
            "a group left out here mis-sizes its next collective",
        )
        self.mooncake.recovered.clear()
        self.join_all()
        # WORLD-scope PGs are on both sides too; the sub-groups are what this covers.
        self.assertEqual(
            set(self.mooncake.joined) & self.sub_group_labels(),
            readmitted,
            "readmit and rejoin must cover the same groups",
        )

    def test_world_group_is_not_taken_twice(self):
        """_WORLD rides in the same registry; the WORLD-scope step already has it."""
        self.eep._recover_parallel_groups([FAULTED_RANK])
        for label, _ in self.mooncake.recovered:
            self.assertNotIn("world:0", label)
        self.join_all()
        self.assertEqual(
            self.mooncake.joined.count("world:0.cpu"),
            1,
            "once from the WORLD-scope step; twice trips joinGroup's state check",
        )

    def test_a_solo_group_is_skipped_on_both_sides(self):
        """No peer holds the other half of a one-rank group."""
        self.eep._recover_parallel_groups([FAULTED_RANK])
        self.join_all()
        touched = {label for label, _ in self.mooncake.recovered}
        touched.update(self.mooncake.joined)
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
