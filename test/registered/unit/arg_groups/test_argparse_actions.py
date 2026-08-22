"""Unit tests for srt/arg_groups/argparse_actions.py — no server, no model loading.

Argparse actions used by the server CLI: LoRA path parsing (plain paths or
JSON entries) and the family of deprecated-flag shims.
"""

import argparse
import json
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.argparse_actions import (
    DeprecatedAction,
    DeprecatedAliasStoreAction,
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
    LoRAPathAction,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLoRAPathAction(CustomTestCase):
    """LoRAPathAction accepts plain path strings and JSON entries."""

    def _make_parser(self, nargs="+"):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lora-paths", action=LoRAPathAction, nargs=nargs)
        return parser

    def test_plain_paths_are_stripped(self):
        ns = self._make_parser().parse_args(["--lora-paths", " /models/a ", "  /models/b"])
        self.assertEqual(ns.lora_paths, ["/models/a", "/models/b"])

    def test_json_entry_is_parsed(self):
        entry = json.dumps({"lora_path": "/models/a", "lora_name": "lora-a"})
        ns = self._make_parser().parse_args(["--lora-paths", entry])
        self.assertEqual(ns.lora_paths, [{"lora_path": "/models/a", "lora_name": "lora-a"}])

    def test_mixed_plain_and_json_entries(self):
        entry = json.dumps({"lora_path": "/models/b", "lora_name": "lora-b"})
        ns = self._make_parser().parse_args(["--lora-paths", "/models/a", entry])
        self.assertEqual(len(ns.lora_paths), 2)
        self.assertEqual(ns.lora_paths[0], "/models/a")
        self.assertEqual(ns.lora_paths[1], {"lora_path": "/models/b", "lora_name": "lora-b"})

    def test_json_missing_required_keys_raises_assertion(self):
        bad = json.dumps({"path": "/models/a"})
        parser = self._make_parser()
        with self.assertRaises(AssertionError):
            parser.parse_args(["--lora-paths", bad])

    def test_malformed_json_raises_json_error(self):
        # Only strings that both start AND end with braces go through json.loads;
        # a brace-open string without the closing brace is treated as a plain path.
        parser = self._make_parser()
        with self.assertRaises(json.JSONDecodeError):
            parser.parse_args(["--lora-paths", "{bad json}"])

    def test_brace_open_but_not_closed_is_plain_path(self):
        ns = self._make_parser().parse_args(["--lora-paths", "{not a json entry"])
        self.assertEqual(ns.lora_paths, ["{not a json entry"])

    def test_values_not_a_list_raises_assertion(self):
        # Direct __call__ invocation with a scalar value hits the type guard.
        action = LoRAPathAction(option_strings=["--lora-paths"], dest="lora_paths")
        with self.assertRaises(AssertionError):
            action(None, argparse.Namespace(), "/models/a", "--lora-paths")

    def test_absent_option_defaults_to_none(self):
        # argparse pre-fills the dest with the action default (None) when the
        # option is not given; callers treat it truthily (same as an empty list).
        parser = argparse.ArgumentParser()
        parser.add_argument("--lora-paths", action=LoRAPathAction, nargs="*")
        ns = parser.parse_args([])
        self.assertIsNone(ns.lora_paths)

    def test_option_without_values_stores_empty_list(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lora-paths", action=LoRAPathAction, nargs="*")
        ns = parser.parse_args(["--lora-paths"])
        self.assertEqual(ns.lora_paths, [])


class TestDeprecatedAction(CustomTestCase):
    def test_error_message_aborts_parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--old-flag", action=DeprecatedAction, error_message="removed")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--old-flag"])

    def test_without_error_message_warns_and_continues(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--old-flag", action=DeprecatedAction)
        with patch(
            "sglang.srt.arg_groups.argparse_actions.print_deprecated_warning"
        ) as mock_warn:
            parser.parse_args(["--old-flag"])
        mock_warn.assert_called_once()
        self.assertIn("--old-flag", mock_warn.call_args[0][0])
        self.assertIn("deprecated", mock_warn.call_args[0][0].lower())


class TestDeprecatedStoreTrueAction(CustomTestCase):
    def test_stores_true_and_mentions_replacement(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--old-flag", action=DeprecatedStoreTrueAction, new_flag="--new-flag"
        )
        with patch(
            "sglang.srt.arg_groups.argparse_actions.print_deprecated_warning"
        ) as mock_warn:
            ns = parser.parse_args(["--old-flag"])
        self.assertTrue(ns.old_flag)
        mock_warn.assert_called_once()
        self.assertIn("Use '--new-flag' instead.", mock_warn.call_args[0][0])

    def test_defaults_false_when_absent(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--old-flag", action=DeprecatedStoreTrueAction)
        ns = parser.parse_args([])
        self.assertFalse(ns.old_flag)


class TestDeprecatedStoreConstAction(CustomTestCase):
    def test_stores_const_value(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--old-flag",
            action=DeprecatedStoreConstAction,
            const_value="disabled",
            new_flag="--new-flag",
        )
        with patch(
            "sglang.srt.arg_groups.argparse_actions.print_deprecated_warning"
        ) as mock_warn:
            ns = parser.parse_args(["--old-flag"])
        self.assertEqual(ns.old_flag, "disabled")
        mock_warn.assert_called_once()
        self.assertIn("'--old-flag'", mock_warn.call_args[0][0])
        self.assertIn("--new-flag", mock_warn.call_args[0][0])

    def test_defaults_none_when_absent(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--old-flag", action=DeprecatedStoreConstAction)
        ns = parser.parse_args([])
        self.assertIsNone(ns.old_flag)


class TestDeprecatedAliasStoreAction(CustomTestCase):
    def test_stores_value_and_mentions_replacement(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--old-alias", action=DeprecatedAliasStoreAction, new_flag="--new-flag"
        )
        with patch(
            "sglang.srt.arg_groups.argparse_actions.print_deprecated_warning"
        ) as mock_warn:
            ns = parser.parse_args(["--old-alias", "value"])
        self.assertEqual(ns.old_alias, "value")
        mock_warn.assert_called_once()
        self.assertIn("Use '--new-flag' instead.", mock_warn.call_args[0][0])

    def test_default_survives_when_absent(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--old-alias", dest="config", action=DeprecatedAliasStoreAction, default="orig"
        )
        ns = parser.parse_args([])
        self.assertEqual(ns.config, "orig")


if __name__ == "__main__":
    unittest.main()