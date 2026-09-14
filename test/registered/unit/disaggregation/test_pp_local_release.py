import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def request(rid):
    return SimpleNamespace(rid=rid)


class TestPPLocalRelease(unittest.TestCase):
    def test_duplicate_rid_waits_for_every_entry(self):
        first, retry = request("x"), request("x")
        scheduler = SimpleNamespace(disagg_prefill_inflight_queue=[first, retry])
        polls = {id(first): KVPoll.Success, id(retry): KVPoll.Transferring}
        self.assertEqual(
            SchedulerPPMixin._pp_pd_local_terminal_rids(scheduler, polls), []
        )

    def test_held_rids_remain_in_release_ring(self):
        current = request("current")
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[current],
            pp_release_held={"held": [request("held")]},
            pp_group=SimpleNamespace(is_first_rank=True),
            _pp_pd_local_terminal_rids=lambda _polls: ["current"],
        )
        self.assertEqual(
            SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(
                scheduler, {id(current): KVPoll.Success}
            ),
            ["current", "held"],
        )

    def test_later_stage_keeps_the_ring_intersection_order(self):
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[],
            pp_release_held={},
            pp_group=SimpleNamespace(is_first_rank=False),
            _pp_pd_local_terminal_rids=lambda _polls: ["a", "b"],
            _pp_recv_pyobj_from_prev_stage=lambda: ["b"],
        )
        self.assertEqual(
            SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(scheduler, {}),
            ["b"],
        )

    @patch("sglang.srt.managers.scheduler_pp_mixin._PP_LOCAL_RELEASE", True)
    def test_local_conclusion_needs_no_separate_admission_flag(self):
        req = request("x")
        scheduler = SimpleNamespace(
            pp_release_held={},
            _pp_pd_local_terminal_rids=lambda _polls: ["x"],
            _pp_pd_aligned_polls=lambda _polls: [KVPoll.Success],
            process_disagg_prefill_inflight_queue=lambda *args, **kwargs: [req],
        )
        SchedulerPPMixin._pp_pd_release_local_transfers(
            scheduler, {id(req): KVPoll.Success}
        )
        self.assertEqual(scheduler.pp_release_held, {"x": [req]})

    def test_ring_confirmation_releases_only_named_requests(self):
        first, second = request("a"), request("b")
        released = []
        scheduler = SimpleNamespace(
            pp_release_held={"a": [first], "b": [second]},
            release_prefill_transfer_pools=released.append,
        )
        SchedulerPPMixin._pp_pd_confirm_release(scheduler, ["a"])
        self.assertEqual(released, [first])
        self.assertEqual(scheduler.pp_release_held, {"b": [second]})


if __name__ == "__main__":
    unittest.main()
