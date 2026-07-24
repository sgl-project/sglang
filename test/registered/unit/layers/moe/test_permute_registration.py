import unittest

from sglang.srt.layers.moe.moe_runner.base import PermuteMethodPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _noop_permute(*args, **kwargs):
    return None


class TestPermuteRegistrationValidation(CustomTestCase):
    """An unknown runner-backend name must be rejected at register_{pre,post}_permute,
    not left to fail later at get_{pre,post}_permute() lookup time."""

    def test_pre_permute_rejects_unknown_runner_backend(self):
        """register_pre_permute validates its runner slot (arg 2)."""
        with self.assertRaises(ValueError):
            PermuteMethodPool.register_pre_permute(
                "standard", "not_a_runner", _noop_permute
            )

    def test_post_permute_rejects_a2a_name_in_runner_slot(self):
        """'mori' is a valid MoeA2ABackend but not a MoeRunnerBackend; register_post_permute
        validates its runner slot (arg 1 -- the order flips vs register_pre_permute)."""
        with self.assertRaises(ValueError):
            PermuteMethodPool.register_post_permute("mori", "standard", _noop_permute)

    def test_rejected_registration_does_not_mutate_pool(self):
        """A rejected name leaves the registry untouched (validation precedes insert)."""
        before = dict(PermuteMethodPool._pre_permute_methods)
        with self.assertRaises(ValueError):
            PermuteMethodPool.register_pre_permute(
                "standard", "not_a_runner", _noop_permute
            )
        self.assertEqual(PermuteMethodPool._pre_permute_methods, before)


if __name__ == "__main__":
    unittest.main()
