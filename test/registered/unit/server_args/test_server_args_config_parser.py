"""Unit tests for YAML configuration argument merging."""

import argparse
import os
import tempfile
import unittest

from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAliasStoreAction,
    DeprecatedStoreTrueAction,
)
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestConfigArgumentMerger(CustomTestCase):
    def _write_config(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            config_file.write(content)
        self.addCleanup(os.unlink, config_file.name)
        return config_file.name

    def test_canonical_store_action_is_not_hidden_by_deprecated_alias(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--cuda-graph-max-bs-decode", type=int)
        parser.add_argument(
            "--cuda-graph-max-bs",
            action=DeprecatedAliasStoreAction,
            dest="cuda_graph_max_bs_decode",
            new_flag="--cuda-graph-max-bs-decode",
            type=int,
        )
        config_path = self._write_config("cuda-graph-max-bs-decode: 8\n")

        merged_args = ConfigArgumentMerger(parser).merge_config_with_args(
            ["--config", config_path]
        )
        parsed_args = parser.parse_args(merged_args)

        self.assertEqual(parsed_args.cuda_graph_max_bs_decode, 8)

    def test_canonical_store_true_is_not_hidden_by_deprecated_alias(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--enable-feature", action="store_true")
        parser.add_argument(
            "--old-enable-feature",
            action=DeprecatedStoreTrueAction,
            dest="enable_feature",
            new_flag="--enable-feature",
        )
        config_path = self._write_config("enable-feature: true\n")

        merged_args = ConfigArgumentMerger(parser).merge_config_with_args(
            ["--config", config_path]
        )
        parsed_args = parser.parse_args(merged_args)

        self.assertTrue(parsed_args.enable_feature)

    def test_action_without_supported_canonical_option_is_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--item", action="append")
        config_path = self._write_config("item: value\n")

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported config option 'item' with action '_AppendAction'",
        ):
            ConfigArgumentMerger(parser).merge_config_with_args(
                ["--config", config_path]
            )


if __name__ == "__main__":
    unittest.main()
