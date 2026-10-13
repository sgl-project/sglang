import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPriorityPolicyValidation(CustomTestCase):
    def test_cli_does_not_advertise_dead_priority_policy(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        action = next(
            a for a in parser._actions if "--schedule-policy" in a.option_strings
        )
        self.assertNotIn("priority", action.choices)

    def test_python_api_rejects_dead_priority_policy_with_recovery_hint(self):
        for enabled in (False, True):
            with self.subTest(enable_priority_scheduling=enabled):
                args = ServerArgs(
                    model_path="dummy",
                    schedule_policy="priority",
                    enable_priority_scheduling=enabled,
                )
                with self.assertRaisesRegex(ValueError, "--enable-priority-scheduling"):
                    args.resolve_once()

    def test_supported_priority_policies_remain_accepted(self):
        for policy in ("fcfs", "lof"):
            with self.subTest(policy=policy):
                args = ServerArgs(
                    model_path="dummy",
                    schedule_policy=policy,
                    enable_priority_scheduling=True,
                )
                args.resolve_once()


if __name__ == "__main__":
    unittest.main()
