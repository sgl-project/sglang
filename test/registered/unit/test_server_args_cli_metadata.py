"""Unit tests for migrated ServerArgs CLI metadata."""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestServerArgsMigratedCliMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_data_parallel_aliases_keep_old_usage(self):
        for option in ("--data-parallel-size", "--dp-size"):
            with self.subTest(option=option):
                args = self.parser.parse_args(["--model", "dummy", option, "3"])
                self.assertEqual(args.dp_size, 3)
                self.assertEqual(ServerArgs.from_cli_args(args).dp_size, 3)

    def test_request_chat_template_requires_explicit_opt_in(self):
        for flags, expected in (([], False), (["--trust-request-chat-template"], True)):
            with self.subTest(flags=flags):
                args = self.parser.parse_args(["--model", "dummy", *flags])
                self.assertIs(
                    ServerArgs.from_cli_args(args).trust_request_chat_template, expected
                )

    def test_prefill_max_context_accepts_human_readable_values(self):
        for option in (
            "--cuda-graph-prefill-max-context",
            "--context-bucket",
        ):
            with self.subTest(option=option):
                args = self.parser.parse_args(["--model", "dummy", option, "200k"])

                self.assertEqual(
                    ServerArgs.from_cli_args(args).cuda_graph_prefill_max_context,
                    200_000,
                )

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
