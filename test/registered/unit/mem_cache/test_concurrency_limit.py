import unittest

from sglang.srt.mem_cache.concurrency_limit import (
    ConcurrencyLimit,
    format_concurrency_report,
    resolve_concurrency_limit,
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
        limits = [_limit("kv_capacity", 400), _limit("mamba_state_pool", 26)]
        resolved, binding = resolve_concurrency_limit(limits)
        self.assertEqual(resolved, 26)
        self.assertEqual(binding.source, "mamba_state_pool")

    def test_ties_report_the_first_listed(self):
        limits = [_limit("max_running_requests", 74), _limit("kv_capacity", 74)]
        _, binding = resolve_concurrency_limit(limits)
        self.assertEqual(binding.source, "max_running_requests")

    def test_downgrade_names_the_binding_limit_and_remedy(self):
        limits = [
            ConcurrencyLimit("max_running_requests", 74, "--max-running-requests=74"),
            _limit("kv_capacity", 400),
            _limit("mamba_state_pool", 26, remedy="set --max-mamba-cache-size 370"),
        ]
        resolved, binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(
            resolved, binding, limits, requested=74
        )
        self.assertTrue(is_downgrade)
        self.assertIn("reduced from the requested 74 to 26", message)
        self.assertIn("mamba_state_pool", message)
        self.assertIn("set --max-mamba-cache-size 370", message)
        # The non-binding limits stay visible for context.
        self.assertIn("kv_capacity=400", message)

    def test_request_honored_is_not_a_downgrade(self):
        limits = [
            ConcurrencyLimit("max_running_requests", 32, "--max-running-requests=32"),
            _limit("kv_capacity", 400),
        ]
        resolved, binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(
            resolved, binding, limits, requested=32
        )
        self.assertFalse(is_downgrade)
        self.assertIn("max_running_requests=32", message)

    def test_no_request_still_reports_the_binding_limit(self):
        limits = [_limit("kv_capacity", 400), _limit("estimate", 2048)]
        resolved, binding = resolve_concurrency_limit(limits)
        is_downgrade, message = format_concurrency_report(resolved, binding, limits)
        self.assertFalse(is_downgrade)
        self.assertEqual(resolved, 400)
        self.assertIn("bound by kv_capacity", message)


if __name__ == "__main__":
    unittest.main()
