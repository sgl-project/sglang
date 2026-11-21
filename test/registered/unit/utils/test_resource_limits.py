import resource
import unittest
from unittest.mock import patch

from sglang.benchmark import utils as benchmark_utils
from sglang.srt.utils import process
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResourceLimits(CustomTestCase):
    def _apply(self, helper, limits, target):
        def set_limit(kind, value):
            soft, hard = value
            if hard != resource.RLIM_INFINITY and soft > hard:
                raise ValueError("soft limit exceeds hard limit")
            limits[kind] = value

        with (
            patch.object(resource, "getrlimit", side_effect=limits.__getitem__),
            patch.object(resource, "setrlimit", side_effect=set_limit),
        ):
            helper(target)

    def test_unlimited_limits_remain_unlimited(self):
        # Linux uses -1; macOS uses a positive sentinel. Exercise both without
        # changing this process's actual resource limits.
        for infinity in (-1, 2**63 - 1):
            with patch.object(resource, "RLIM_INFINITY", infinity):
                for helper in (process.set_ulimit, benchmark_utils.set_ulimit):
                    with self.subTest(helper=helper.__module__, infinity=infinity):
                        limits = {
                            resource.RLIMIT_NOFILE: (infinity, infinity),
                            resource.RLIMIT_STACK: (infinity, infinity),
                        }
                        original = limits.copy()
                        self._apply(helper, limits, 128)
                        self.assertEqual(limits, original)

    def test_finite_limits_are_only_raised(self):
        for helper in (process.set_ulimit, benchmark_utils.set_ulimit):
            for soft in (64, 256):
                with self.subTest(helper=helper.__module__, soft=soft):
                    limits = {
                        resource.RLIMIT_NOFILE: (soft, 1024),
                        resource.RLIMIT_STACK: (soft * 1024, 1024 * 1024),
                    }
                    self._apply(helper, limits, 128)
                    self.assertEqual(
                        limits[resource.RLIMIT_NOFILE], (max(soft, 128), 1024)
                    )
                    expected_stack = (
                        max(soft, 128) if helper is process.set_ulimit else soft
                    )
                    self.assertEqual(
                        limits[resource.RLIMIT_STACK],
                        (expected_stack * 1024, 1024 * 1024),
                    )

    def test_finite_hard_cap_still_prevents_raising(self):
        for helper in (process.set_ulimit, benchmark_utils.set_ulimit):
            with self.subTest(helper=helper.__module__):
                limits = {
                    resource.RLIMIT_NOFILE: (32, 64),
                    resource.RLIMIT_STACK: (32 * 1024, 64 * 1024),
                }
                original = limits.copy()
                self._apply(helper, limits, 128)
                self.assertEqual(limits, original)


if __name__ == "__main__":
    unittest.main()
