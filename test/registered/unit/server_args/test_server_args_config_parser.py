"""Unit tests for server argument config action classification."""

import argparse
import unittest

from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAliasStoreAction,
    DeprecatedStoreTrueAction,
)
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestConfigArgumentMerger(unittest.TestCase):
    def test_canonical_store_action_wins_shared_destination(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mamba-radix-cache-strategy")
        parser.add_argument(
            "--mamba-scheduler-strategy",
            dest="mamba_radix_cache_strategy",
            action=DeprecatedAliasStoreAction,
            new_flag="--mamba-radix-cache-strategy",
        )

        args = ConfigArgumentMerger(parser)._convert_config_to_args(
            {"mamba-radix-cache-strategy": "no_buffer"}
        )

        self.assertEqual(args, ["--mamba-radix-cache-strategy", "no_buffer"])

    def test_canonical_store_true_action_wins_shared_destination(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--incremental-streaming-output", action="store_true")
        parser.add_argument(
            "--stream-output",
            dest="incremental_streaming_output",
            action=DeprecatedStoreTrueAction,
            new_flag="--incremental-streaming-output",
        )

        args = ConfigArgumentMerger(parser)._convert_config_to_args(
            {"incremental-streaming-output": True}
        )

        self.assertEqual(args, ["--incremental-streaming-output"])

    def test_action_without_supported_destination_remains_unsupported(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--stream-output",
            dest="incremental_streaming_output",
            action=DeprecatedStoreTrueAction,
            new_flag="--incremental-streaming-output",
        )

        merger = ConfigArgumentMerger(parser)
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported config option 'incremental_streaming_output'.*"
            "DeprecatedStoreTrueAction",
        ):
            merger._convert_config_to_args({"incremental-streaming-output": True})


if __name__ == "__main__":
    unittest.main()
