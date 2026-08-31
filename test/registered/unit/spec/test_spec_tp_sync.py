import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTpGroup:
    def __init__(self, *, world_size: int, rank_in_group: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.cpu_group = object()
        self.broadcasts = []

    def broadcast(self, values: torch.Tensor, *, src: int):
        self.broadcasts.append((values, src))


class TestParseSpecTpSync(CustomTestCase):
    def test_default_enables_all_decision_sites(self):
        self.assertEqual(envs.SGLANG_SPEC_TP_SYNC.default, "all")
        self.assertEqual(
            parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.default),
            frozenset(SpecTpSyncSite),
        )

    def test_all_and_off_presets(self):
        self.assertEqual(parse_spec_tp_sync("all"), frozenset(SpecTpSyncSite))
        self.assertEqual(parse_spec_tp_sync("off"), frozenset())
        self.assertEqual(parse_spec_tp_sync("none"), frozenset())

    def test_init_and_rng_presets(self):
        init = parse_spec_tp_sync("init")
        self.assertEqual(
            init,
            frozenset(
                {
                    SpecTpSyncSite.DSPARK_MEM,
                    SpecTpSyncSite.DFLASH_MEM,
                }
            ),
        )

        rng = parse_spec_tp_sync("rng")
        self.assertTrue(init <= rng)
        self.assertIn(SpecTpSyncSite.DSPARK_DRAFT_SAMPLE, rng)
        self.assertIn(SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE, rng)
        self.assertIn(SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE, rng)
        self.assertNotIn(SpecTpSyncSite.DSPARK_PLAN, rng)

    def test_composes_names_numbers_and_negation(self):
        sites = parse_spec_tp_sync(" all, -dspark_plan, -6, dspark-plan ")
        self.assertIn(SpecTpSyncSite.DSPARK_PLAN, sites)
        self.assertNotIn(SpecTpSyncSite.DSPARK_GRAPH_GREEDY, sites)
        self.assertEqual(len(sites), len(SpecTpSyncSite) - 1)

    def test_unknown_site_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown token"):
            parse_spec_tp_sync("not-a-sync-site")


class TestSpecTpSync(CustomTestCase):
    def test_multi_rank_broadcasts_enabled_site_from_rank_zero(self):
        group = _FakeTpGroup(world_size=2, rank_in_group=1)
        values = torch.tensor([1, 2, 3])
        with envs.SGLANG_SPEC_TP_SYNC.override("all"):
            sync = SpecTpSync(group)
            returned = sync.sync(SpecTpSyncSite.DSPARK_PLAN, values)

        self.assertIs(returned, values)
        self.assertEqual(len(group.broadcasts), 1)
        broadcast_values, src = group.broadcasts[0]
        self.assertIs(broadcast_values, values)
        self.assertEqual(src, 0)

    def test_disabled_site_does_not_broadcast(self):
        group = _FakeTpGroup(world_size=2)
        with envs.SGLANG_SPEC_TP_SYNC.override("init"):
            sync = SpecTpSync(group)
            sync.sync(SpecTpSyncSite.DSPARK_PLAN, torch.tensor([1]))

        self.assertEqual(group.broadcasts, [])

    def test_single_rank_is_always_a_noop(self):
        group = _FakeTpGroup(world_size=1)
        with envs.SGLANG_SPEC_TP_SYNC.override("all"):
            sync = SpecTpSync(group)
            sync.sync(SpecTpSyncSite.DSPARK_PLAN, torch.tensor([1]))

        self.assertFalse(sync.enabled(SpecTpSyncSite.DSPARK_PLAN))
        self.assertEqual(group.broadcasts, [])

    def test_available_memory_uses_distributed_min_when_enabled(self):
        group = _FakeTpGroup(world_size=2)
        with envs.SGLANG_SPEC_TP_SYNC.override("all"):
            sync = SpecTpSync(group)
            with patch(
                "sglang.srt.speculative.spec_tp_sync.get_available_gpu_memory",
                return_value=7.5,
            ) as get_memory:
                result = sync.available_memory_gb(
                    SpecTpSyncSite.DSPARK_MEM,
                    torch.device("cpu"),
                    3,
                    group=group,
                )

        self.assertEqual(result, 7.5)
        get_memory.assert_called_once_with(
            torch.device("cpu"),
            3,
            distributed=True,
            cpu_group=group.cpu_group,
        )

    def test_available_memory_stays_local_when_disabled(self):
        group = _FakeTpGroup(world_size=2)
        with envs.SGLANG_SPEC_TP_SYNC.override("off"):
            sync = SpecTpSync(group)
            with patch(
                "sglang.srt.speculative.spec_tp_sync.get_available_gpu_memory",
                return_value=8.0,
            ) as get_memory:
                result = sync.available_memory_gb(
                    SpecTpSyncSite.DSPARK_MEM,
                    torch.device("cpu"),
                    0,
                    group=group,
                )

        self.assertEqual(result, 8.0)
        get_memory.assert_called_once_with(
            torch.device("cpu"),
            0,
            distributed=False,
            cpu_group=None,
        )


if __name__ == "__main__":
    unittest.main()
