"""Unit tests for migrated ServerArgs CLI metadata."""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import human_readable_int
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestServerArgsMigratedCliMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)
        cls.actions_by_option = {
            option: action
            for action in cls.parser._actions
            for option in action.option_strings
        }

    def test_argparse_shape_is_preserved_for_representative_migrated_options(self):
        self.assertEqual(self.actions_by_option["--dtype"].default, ServerArgs.dtype)
        self.assertEqual(
            self.actions_by_option["--dtype"].choices,
            ["auto", "half", "float16", "bfloat16", "float", "float32"],
        )
        self.assertIs(self.actions_by_option["--dtype"].type, str)
        self.assertIs(
            self.actions_by_option["--max-total-tokens"].type, human_readable_int
        )
        self.assertIs(
            self.actions_by_option["--max-prefill-tokens"].type, human_readable_int
        )
        self.assertIs(
            self.actions_by_option["--prefill-delayer-forward-passes-buckets"].type,
            float,
        )
        self.assertEqual(
            self.actions_by_option["--prefill-delayer-forward-passes-buckets"].nargs,
            "+",
        )
        self.assertEqual(
            self.actions_by_option["--schedule-policy"].choices,
            ["lpm", "random", "fcfs", "dfs-weight", "lof", "priority", "routing-key"],
        )
        self.assertEqual(
            self.actions_by_option["--load-balance-method"].choices,
            [
                "auto",
                "round_robin",
                "follow_bootstrap_room",
                "total_requests",
                "total_tokens",
            ],
        )

    def test_data_parallel_aliases_keep_old_usage(self):
        for option in ("--data-parallel-size", "--dp-size"):
            with self.subTest(option=option):
                args = self.parser.parse_args(["--model", "dummy", option, "3"])
                self.assertEqual(args.dp_size, 3)
                self.assertEqual(ServerArgs.from_cli_args(args).dp_size, 3)

    def test_migrated_and_manual_options_parse_together(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--dtype",
                "bfloat16",
                "--max-total-tokens",
                "1024",
                "--prefill-delayer-forward-passes-buckets",
                "1.5",
                "2.5",
                "--data-parallel-size",
                "2",
                "--load-balance-method",
                "total_tokens",
                "--tp-size",
                "4",
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertEqual(server_args.dtype, "bfloat16")
        self.assertEqual(server_args.max_total_tokens, 1024)
        self.assertEqual(server_args.prefill_delayer_forward_passes_buckets, [1.5, 2.5])
        self.assertEqual(server_args.dp_size, 2)
        self.assertEqual(server_args.load_balance_method, "total_tokens")
        self.assertEqual(server_args.tp_size, 4)


if __name__ == "__main__":
    unittest.main()
