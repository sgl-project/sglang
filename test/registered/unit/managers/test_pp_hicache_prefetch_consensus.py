import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPHiCachePrefetchConsensus(unittest.TestCase):
    def _make_scheduler(self, *, pp_rank: int, readiness: dict[str, bool]):
        reqs = [SimpleNamespace(rid=rid, finished_reason=None) for rid in readiness]
        scheduler = SchedulerPPMixin()
        scheduler.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=8)
        scheduler.pp_group = SimpleNamespace(is_first_rank=pp_rank == 0)
        scheduler.enable_hicache_storage = True
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=reqs)
        scheduler.tree_cache = SimpleNamespace(
            check_prefetch_progress=MagicMock(side_effect=lambda rid: readiness[rid])
        )
        return scheduler

    def test_local_filter_checks_all_candidates_and_preserves_order(self):
        scheduler = self._make_scheduler(
            pp_rank=3,
            readiness={"req-a": True, "req-b": False, "req-c": True},
        )

        ready_rids = scheduler._pp_filter_hicache_ready_rids(
            ["req-c", "req-b", "req-a"]
        )

        self.assertEqual(ready_rids, ["req-c", "req-a"])
        self.assertEqual(
            scheduler.tree_cache.check_prefetch_progress.call_count,
            3,
        )

    def test_first_stage_withholds_locally_unready_request(self):
        scheduler = self._make_scheduler(
            pp_rank=0,
            readiness={"req-ready": True, "req-wait": False},
        )
        scheduler.get_rids = MagicMock(return_value=(["req-ready", "req-wait"], []))

        good_rids, bad_rids = scheduler._pp_pd_get_bootstrapped_ids()

        self.assertEqual(good_rids, ["req-ready"])
        self.assertEqual(bad_rids, [])

    def test_later_stage_intersects_previous_and_local_readiness(self):
        scheduler = self._make_scheduler(
            pp_rank=5,
            readiness={
                "req-a": True,
                "req-b": False,
                "req-c": True,
                "req-local-only": True,
            },
        )
        scheduler._pp_recv_pyobj_from_prev_stage = MagicMock(
            return_value=[["req-c", "req-b", "req-a"], ["bad-prev"]]
        )
        scheduler.get_rids = MagicMock(
            return_value=(
                ["req-a", "req-b", "req-c", "req-local-only"],
                ["bad-local", "bad-prev"],
            )
        )

        good_rids, bad_rids = scheduler._pp_pd_get_bootstrapped_ids()

        self.assertEqual(good_rids, ["req-c", "req-a"])
        self.assertEqual(bad_rids, ["bad-prev", "bad-local"])

    def test_non_pp_path_keeps_existing_scheduler_behavior(self):
        scheduler = self._make_scheduler(
            pp_rank=0,
            readiness={"req": False},
        )
        scheduler.ps.pp_size = 1

        ready_rids = scheduler._pp_filter_hicache_ready_rids(["req"])

        self.assertEqual(ready_rids, ["req"])
        scheduler.tree_cache.check_prefetch_progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
