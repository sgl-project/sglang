import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.lpm_opportunity import competition_opportunity
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def req(i, shared=True, size=8192):
    r = FakeReq(rid=str(i), num_matched_prefix_tokens=0)
    r.origin_input_ids = array(
        "q", ([7 + i % 8 if shared else i + 100] * 4096 + [i + 100] * (size - 4096))
    )
    r.output_ids = array("q")
    r.extra_key = None
    r.cache_salt = None
    return r


class TestOpportunity(unittest.TestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.cache = RadixCache.create_simulated()
        self.cache.token_to_kv_pool_allocator = SimpleNamespace(size=32768)
        self.p = SchedulePolicy("lpm", self.cache, False, False, False)

    def test_low_density_high_risk_opens_long_queue_batch(self):
        rs = [req(i, False, size=4096) for i in range(129)]
        rs[0].origin_input_ids = array("q", [777] * 6144)
        rs[1].origin_input_ids = array("q", [888] * 32768)
        rs[2].origin_input_ids = array("q", [777] * 6144)
        self.cache.token_to_kv_pool_allocator.size = 4096
        # One repeated 4096-token checkpoint: below the former 20% density gate.
        self.assertLess(4096 / sum(len(r.origin_input_ids) for r in rs[:64]), 0.2)
        with (
            patch.object(self.p, "_compute_prefix_matches", return_value=set()),
            patch.object(self.p, "_touch_waiting_prefixes"),
        ):
            self.p.calc_priority(rs)
        self.assertTrue(self.p._bounded_active)
        self.assertEqual(len(self.p._bounded_members), 64)
        self.assertEqual(self.p._lpm_gate_competition["risk_tokens"], 6144)
        self.assertEqual(self.p._lpm_gate_competition["ratio"], 1.5)

    def test_risk_threshold_is_inclusive(self):
        rs = [req(i, False, size=4096) for i in range(3)]
        rs[0].origin_input_ids = rs[2].origin_input_ids = array("q", [777] * 6144)
        rs[1].origin_input_ids = array("q", [888] * 32768)
        self.assertTrue(competition_opportunity(rs, 4096)["allowed"])
        self.assertFalse(competition_opportunity(rs, 4097)["allowed"])

    def test_dense_prefixes_without_competition_do_not_open(self):
        rs = [req(i) for i in range(129)]
        self.cache.token_to_kv_pool_allocator.size = 1000000
        self.assertFalse(self.p._should_start_bounded_batch(rs))
        self.assertEqual(self.p._lpm_gate_competition["risk_tokens"], 0)

    def test_namespaces_not_combined(self):
        rs = [req(i) for i in range(64)]
        for i, r in enumerate(rs):
            r.cache_salt = str(i)
        self.assertEqual(competition_opportunity(rs, 32768)["risk_tokens"], 0)
        for i, r in enumerate(rs):
            r.cache_salt = None
            r.extra_key = str(i)
        self.assertEqual(competition_opportunity(rs, 32768)["risk_tokens"], 0)

    def test_short_queue_never_probes_risk(self):
        rs = [req(i) for i in range(128)]
        with (
            patch.object(self.p, "_should_start_bounded_batch") as gate,
            patch.object(self.p, "_compute_prefix_matches", return_value=set()),
            patch.object(self.p, "_touch_waiting_prefixes"),
        ):
            self.p.calc_priority(rs)
        gate.assert_not_called()
        self.assertFalse(self.p._bounded_active)

    def test_no_eviction_history_required(self):
        rs = [req(i) for i in range(129)]
        self.assertTrue(self.p._should_start_bounded_batch(rs))
        self.p._lpm_gate_next_check = 0
        self.assertTrue(self.p._should_start_bounded_batch(rs))

    def test_failed_gate_preserves_fcfs_and_no_members(self):
        rs = [req(i, False) for i in range(129)]
        with patch.object(self.p, "_touch_waiting_prefixes"):
            result = self.p.calc_priority(rs)
        self.assertEqual(result, rs)
        self.assertFalse(self.p._bounded_members)

    def test_only_first64_used_and_retry_throttled(self):
        rs = [req(i, False) for i in range(64)] + [req(i) for i in range(64, 150)]
        self.assertFalse(self.p._should_start_bounded_batch(rs))
        self.assertFalse(
            self.p._should_start_bounded_batch([req(i) for i in range(150)])
        )
        self.p._lpm_gate_next_check = 0
        self.assertTrue(
            self.p._should_start_bounded_batch([req(i) for i in range(150)])
        )

    def test_committed_batch_finishes_without_rechecking_gate(self):
        rs = [req(i) for i in range(129)]
        with (
            patch.object(self.p, "_compute_prefix_matches", return_value=set()),
            patch.object(self.p, "_touch_waiting_prefixes"),
        ):
            self.p.calc_priority(rs)
            self.assertEqual(len(self.p._bounded_members), 64)
            self.p.retire_waiting_members(rs[:60])
            with patch.object(
                self.p,
                "_should_start_bounded_batch",
                side_effect=AssertionError("must not recheck"),
            ):
                self.p.calc_priority(rs[60:])
            self.assertEqual(len(self.p._bounded_members), 4)

    def test_unsupported_hierarchical_cache_fails_closed(self):
        self.p.enable_hierarchical_cache = True
        self.assertFalse(
            self.p._should_start_bounded_batch([req(i) for i in range(129)])
        )


if __name__ == "__main__":
    unittest.main()
