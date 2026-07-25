import unittest

from sglang.srt.mem_cache.concurrency_limit import (
    ConcurrencyLimit,
    format_concurrency_report,
    heuristic_limit,
    kv_capacity_limit,
    resolve_concurrency_limit,
    state_pool_limit,
    user_request_limit,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _limit(source, value, remedy="raise it"):
    return ConcurrencyLimit(
        source=source, value=value, detail=f"{source} detail", remedy=remedy
    )


class TestConcurrencyLimit(CustomTestCase):
    def test_binds_to_the_smallest_limit(self):
        binding = resolve_concurrency_limit(
            [_limit("kv_capacity", 400), _limit("mamba_state_pool", 26)]
        )
        self.assertEqual(binding.value, 26)
        self.assertEqual(binding.source, "mamba_state_pool")

    def test_ties_report_the_first_listed(self):
        binding = resolve_concurrency_limit(
            [_limit("max_running_requests", 74), _limit("kv_capacity", 74)]
        )
        self.assertEqual(binding.source, "max_running_requests")

    def test_downgrade_names_the_binding_limit_and_remedy(self):
        limits = [
            ConcurrencyLimit("max_running_requests", 74, "--max-running-requests=74"),
            _limit("kv_capacity", 400),
            _limit("mamba_state_pool", 26, remedy="--max-mamba-cache-size 370"),
        ]
        binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(
            binding, limits, requested=limits[0]
        )
        self.assertTrue(is_downgrade)
        self.assertIn("reduced from the requested 74 to 26", message)
        self.assertIn("bound by mamba_state_pool", message)
        self.assertIn("--max-mamba-cache-size 370", message)
        self.assertIn("kv_capacity=400", message)

    def test_request_honored_is_not_a_downgrade(self):
        limits = [
            ConcurrencyLimit("max_running_requests", 32, "--max-running-requests=32"),
            _limit("kv_capacity", 400),
        ]
        binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(
            binding, limits, requested=limits[0]
        )
        self.assertFalse(is_downgrade)
        self.assertIn("bound by max_running_requests", message)
        # The user's own request carries no remedy.
        self.assertNotIn("To raise it", message)

    def test_sole_limit_omits_the_other_limits_clause(self):
        limits = [_limit("kv_capacity", 400)]
        _, message = format_concurrency_report(
            resolve_concurrency_limit(limits), limits
        )
        self.assertNotIn("other limits", message)

    def test_no_request_still_reports_binding_and_remedy(self):
        limits = [_limit("kv_capacity", 400), _limit("heuristic_estimate", 2048)]
        binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(binding, limits)
        self.assertFalse(is_downgrade)
        self.assertEqual(binding.value, 400)
        self.assertIn("bound by kv_capacity", message)
        self.assertIn("To raise it", message)


class TestLimitConstructors(CustomTestCase):
    def test_user_request_is_per_dp_worker(self):
        limit = user_request_limit(128, attn_dp_size=4)
        self.assertEqual(limit.value, 32)
        self.assertIn("--max-running-requests=128 / 4 dp workers", limit.detail)

    def test_user_request_detail_omits_the_dp_split_when_single(self):
        self.assertEqual(
            user_request_limit(128, attn_dp_size=1).detail,
            "--max-running-requests=128",
        )

    def test_downgrade_keeps_the_global_request_visible(self):
        requested = user_request_limit(296, attn_dp_size=4)
        limits = [requested, _limit("mamba_state_pool", 26)]
        _, message = format_concurrency_report(
            resolve_concurrency_limit(limits), limits, requested
        )
        self.assertIn("reduced from the requested 74", message)
        self.assertIn("--max-running-requests=296 / 4 dp workers", message)

    def test_kv_capacity_halves_the_token_pool(self):
        self.assertEqual(kv_capacity_limit(798208).value, 399104)

    def test_heuristic_is_clamped(self):
        # 2M / 1M * 512 = 1024 -> floor; 10M / 1K * 512 = 5M -> ceiling.
        self.assertEqual(heuristic_limit(2 * 1024**2, 1024**2).value, 2048)
        self.assertEqual(heuristic_limit(10**7, 1024).value, 4096)

    def test_state_pool_remedy_sizes_for_the_target(self):
        limit = state_pool_limit(131, slots_per_request=5, attn_dp_size=1, target=74)
        self.assertEqual(limit.value, 26)
        self.assertIn("--max-mamba-cache-size 370", limit.remedy)

    def test_state_pool_remedy_scales_back_to_the_global_flag(self):
        # The pool size read from server_args is per shard, but
        # --max-mamba-cache-size is global and gets divided by attn_dp_size.
        limit = state_pool_limit(131, slots_per_request=5, attn_dp_size=4, target=74)
        self.assertEqual(limit.value, 26)
        self.assertIn("--max-mamba-cache-size 1480", limit.remedy)
        self.assertIn("per shard", limit.detail)

    def test_state_pool_without_a_target_suggests_no_size(self):
        # Any size we could invent is either +1 request or far past what fits.
        limit = state_pool_limit(131, slots_per_request=5, attn_dp_size=1)
        self.assertNotIn("--max-mamba-cache-size", limit.remedy)
        self.assertIn("--max-running-requests", limit.remedy)

    def test_state_pool_target_zero_is_not_treated_as_unset(self):
        limit = state_pool_limit(131, slots_per_request=5, attn_dp_size=1, target=0)
        self.assertIn("--max-mamba-cache-size 0", limit.remedy)


if __name__ == "__main__":
    unittest.main()
