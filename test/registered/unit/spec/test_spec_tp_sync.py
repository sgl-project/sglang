import datetime
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DRAFT = SpecTpSyncSite.DSPARK_DRAFT


def _check_cpu_budget_broadcast(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=3,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        # Attention TP's leader can be global rank 1, not global rank 0.
        groups = [
            (ranks, dist.new_group(ranks, backend="gloo")) for ranks in ([0], [1, 2])
        ]
        ranks, group = next(item for item in groups if rank in item[0])
        sync = SpecTpSync.__new__(SpecTpSync)
        sync._tp_group = SimpleNamespace(ranks=ranks, cpu_group=group)
        sync._sites = frozenset({SpecTpSyncSite.DSPARK_PLAN})
        for leader_budget in (-1, 0, 7):
            value = torch.tensor(
                [leader_budget if rank == ranks[0] else 99], dtype=torch.int64
            )
            sync.sync_cpu(SpecTpSyncSite.DSPARK_PLAN, value)
            assert value.item() == leader_budget, (rank, value)
        value = torch.tensor([rank], dtype=torch.int64)
        sync.sync_cpu(SpecTpSyncSite.DSPARK_TARGET, value)
        assert value.item() == rank
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(), "requires distributed Gloo"
)
class TestCpuBudgetBroadcast(CustomTestCase):
    def test_attention_tp_subgroup_uses_its_own_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _check_cpu_budget_broadcast,
                args=(f"{directory}/rendezvous",),
                nprocs=3,
                join=True,
            )


class TestParseSpecTpSync(CustomTestCase):
    def test_retired_draft_handles_select_the_merged_site(self):
        for token in ("dspark-graph-greedy", "dspark-draft-sample", "2", "6"):
            with self.subTest(token=token):
                self.assertEqual(parse_spec_tp_sync(token), frozenset({DRAFT}))

    def test_negating_a_retired_handle_drops_the_merged_site(self):
        sites = parse_spec_tp_sync("all,-6")
        self.assertNotIn(DRAFT, sites)
        self.assertEqual(sites, frozenset(SpecTpSyncSite) - {DRAFT})

    def test_draft_noise_is_not_an_rng_site(self):
        # Ranks with the same logits draw the same draft noise.
        self.assertNotIn(DRAFT, parse_spec_tp_sync("rng"))
        self.assertIn(DRAFT, parse_spec_tp_sync("dspark-draft"))


if __name__ == "__main__":
    unittest.main()
