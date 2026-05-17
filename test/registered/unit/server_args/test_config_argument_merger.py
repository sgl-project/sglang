"""Unit tests for srt/server_args_config_parser"""

import argparse
import os
import tempfile
import unittest

from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


def _make_parser():
    """Create a simple argparse parser for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="default-model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--config", type=str)
    parser.add_argument("--served-model-name", nargs="+")
    return parser


class TestExtractConfigFilePath(CustomTestCase):
    def setUp(self):
        self.merger = ConfigArgumentMerger()

    def test_no_config_flag(self):
        result = self.merger._extract_config_file_path(["--port", "9000"])
        self.assertIsNone(result)

    def test_single_config_flag(self):
        result = self.merger._extract_config_file_path(
            ["--port", "9000", "--config", "my.yaml"]
        )
        self.assertEqual(result, "my.yaml")

    def test_multiple_config_flags_raises(self):
        with self.assertRaises(ValueError, msg="Multiple config files"):
            self.merger._extract_config_file_path(
                ["--config", "a.yaml", "--config", "b.yaml"]
            )

    def test_config_flag_at_end_without_value_raises(self):
        with self.assertRaises(ValueError, msg="No config file specified"):
            self.merger._extract_config_file_path(["--port", "9000", "--config"])


class TestValidateYamlFile(CustomTestCase):
    def setUp(self):
        self.merger = ConfigArgumentMerger()

    def test_non_yaml_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError, msg="YAML format"):
                self.merger._validate_yaml_file(path)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError, msg="not found"):
            self.merger._validate_yaml_file("/nonexistent/file.yaml")

    def test_valid_yaml_file(self):
        path = _write_yaml("key: value\n")
        try:
            self.merger._validate_yaml_file(path)  # should not raise
        finally:
            os.unlink(path)

    def test_yml_extension_accepted(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        f.write("key: value\n")
        f.close()
        try:
            self.merger._validate_yaml_file(f.name)  # should not raise
        finally:
            os.unlink(f.name)


class TestConvertConfigToArgs(CustomTestCase):
    def test_scalar_values(self):
        merger = ConfigArgumentMerger()
        result = merger._convert_config_to_args({"port": 9000, "model-path": "gpt2"})
        self.assertIn("--port", result)
        self.assertIn("9000", result)
        self.assertIn("--model-path", result)
        self.assertIn("gpt2", result)

    def test_boolean_store_true_true(self):
        parser = _make_parser()
        merger = ConfigArgumentMerger(parser=parser)
        result = merger._convert_config_to_args({"trust_remote_code": True})
        self.assertIn("--trust_remote_code", result)

    def test_boolean_store_true_false(self):
        parser = _make_parser()
        merger = ConfigArgumentMerger(parser=parser)
        result = merger._convert_config_to_args({"trust_remote_code": False})
        self.assertNotIn("--trust_remote_code", result)

    def test_boolean_non_store_true(self):
        """Boolean values for non-store_true args should emit --key true/false."""
        merger = ConfigArgumentMerger()
        result = merger._convert_config_to_args({"some_flag": True})
        self.assertEqual(result, ["--some_flag", "true"])
        result = merger._convert_config_to_args({"some_flag": False})
        self.assertEqual(result, ["--some_flag", "false"])

    def test_list_values(self):
        merger = ConfigArgumentMerger()
        result = merger._convert_config_to_args(
            {"served-model-name": ["model-a", "model-b"]}
        )
        self.assertEqual(result, ["--served-model-name", "model-a", "model-b"])

    def test_empty_list_skipped(self):
        merger = ConfigArgumentMerger()
        result = merger._convert_config_to_args({"served-model-name": []})
        self.assertEqual(result, [])

    def test_dash_underscore_normalization(self):
        """Keys with dashes are normalized to underscores for store_true lookup."""
        parser = _make_parser()
        merger = ConfigArgumentMerger(parser=parser)
        result = merger._convert_config_to_args({"trust-remote-code": True})
        self.assertIn("--trust-remote-code", result)


class TestMergeConfigWithArgs(CustomTestCase):
    def test_no_config_returns_original(self):
        merger = ConfigArgumentMerger()
        args = ["--port", "9000"]
        self.assertEqual(merger.merge_config_with_args(args), args)

    def test_config_values_merged(self):
        yaml_path = _write_yaml("port: 7000\n")
        try:
            merger = ConfigArgumentMerger()
            result = merger.merge_config_with_args(
                ["--model-path", "gpt2", "--config", yaml_path]
            )
            self.assertIn("--port", result)
            self.assertIn("7000", result)
            self.assertIn("--model-path", result)
            self.assertNotIn("--config", result)
            self.assertNotIn(yaml_path, result)
        finally:
            os.unlink(yaml_path)

    def test_cli_overrides_config(self):
        """CLI args come after config args, so argparse gives them precedence."""
        yaml_path = _write_yaml("port: 7000\n")
        try:
            parser = _make_parser()
            merger = ConfigArgumentMerger(parser=parser)
            merged = merger.merge_config_with_args(
                ["--port", "9000", "--config", yaml_path]
            )
            # CLI --port 9000 should appear after config --port 7000
            # so argparse will use 9000
            ns = parser.parse_args(merged)
            self.assertEqual(ns.port, 9000)
        finally:
            os.unlink(yaml_path)

    def test_config_fills_defaults(self):
        """Config values fill in for unspecified CLI args."""
        yaml_path = _write_yaml("port: 7000\ntp_size: 4\n")
        try:
            parser = _make_parser()
            merger = ConfigArgumentMerger(parser=parser)
            merged = merger.merge_config_with_args(["--config", yaml_path])
            ns = parser.parse_args(merged)
            self.assertEqual(ns.port, 7000)
            self.assertEqual(ns.tp_size, 4)
        finally:
            os.unlink(yaml_path)

    def test_empty_config_file(self):
        yaml_path = _write_yaml("")
        try:
            merger = ConfigArgumentMerger()
            result = merger.merge_config_with_args(
                ["--port", "9000", "--config", yaml_path]
            )
            self.assertIn("--port", result)
            self.assertIn("9000", result)
        finally:
            os.unlink(yaml_path)

    def test_non_dict_config_raises(self):
        yaml_path = _write_yaml("- item1\n- item2\n")
        try:
            merger = ConfigArgumentMerger()
            with self.assertRaises(ValueError, msg="dictionary"):
                merger.merge_config_with_args(["--config", yaml_path])
        finally:
            os.unlink(yaml_path)

    def test_config_with_boolean_and_list(self):
        yaml_path = _write_yaml(
            "trust_remote_code: true\nserved_model_name:\n  - a\n  - b\n"
        )
        try:
            parser = _make_parser()
            merger = ConfigArgumentMerger(parser=parser)
            merged = merger.merge_config_with_args(["--config", yaml_path])
            ns = parser.parse_args(merged)
            self.assertTrue(ns.trust_remote_code)
            self.assertEqual(ns.served_model_name, ["a", "b"])
        finally:
            os.unlink(yaml_path)


class TestUnsupportedActions(CustomTestCase):
    def test_unsupported_action_raises(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--count", action="count")
        parser.add_argument("--config", type=str)
        merger = ConfigArgumentMerger(parser=parser)
        with self.assertRaises(ValueError, msg="Unsupported"):
            merger._convert_config_to_args({"count": 3})


class TestLegacyInterface(CustomTestCase):
    def test_boolean_actions_list(self):
        merger = ConfigArgumentMerger(boolean_actions=["verbose", "debug"])
        result = merger._convert_config_to_args({"verbose": True, "debug": False})
        self.assertIn("--verbose", result)
        self.assertNotIn("--debug", result)


if __name__ == "__main__":
    unittest.main()
