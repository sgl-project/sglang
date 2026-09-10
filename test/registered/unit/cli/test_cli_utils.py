# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

from sglang.cli.utils import try_get_model_path
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTryGetModelPathFromConfig(unittest.TestCase):
    def _detect_from_config(self, content, extra_argv=()):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            config_file = f.name

        try:
            return try_get_model_path(["--config", config_file, *extra_argv])
        finally:
            os.unlink(config_file)

    def test_model_path_key_from_config(self):
        self.assertEqual(self._detect_from_config("model-path: X\n"), "X")

    def test_model_path_underscore_key_from_config(self):
        self.assertEqual(self._detect_from_config("model_path: X\n"), "X")

    def test_model_alias_key_from_config(self):
        self.assertEqual(self._detect_from_config("model: X\n"), "X")

    def test_cli_flag_takes_precedence_over_config(self):
        self.assertEqual(
            self._detect_from_config(
                "model-path: CFG_VAL\n", extra_argv=["--model-path", "CLI_VAL"]
            ),
            "CLI_VAL",
        )

    def test_nonexistent_config_file_returns_none(self):
        self.assertIsNone(try_get_model_path(["--config", "/no/such/file.yaml"]))

    def test_config_flag_without_path_returns_none(self):
        self.assertIsNone(try_get_model_path(["--config"]))

    def test_no_config_and_no_model_flag_returns_none(self):
        self.assertIsNone(try_get_model_path(["--some-flag", "value"]))

    def test_config_without_model_key_returns_none(self):
        self.assertIsNone(self._detect_from_config("host: 127.0.0.1\n"))

    def test_last_config_model_key_in_file_order_wins(self):
        self.assertEqual(
            self._detect_from_config("model-path: First\nmodel: Second\n"),
            "Second",
        )

    def test_model_path_flag_still_detected(self):
        self.assertEqual(try_get_model_path(["--model-path", "X"]), "X")

    def test_model_equals_form_still_detected(self):
        self.assertEqual(try_get_model_path(["--model=X"]), "X")

    def test_non_string_config_key_does_not_raise(self):
        self.assertEqual(
            self._detect_from_config("123: junk\nmodel-path: Qwen/Qwen2-7B\n"),
            "Qwen/Qwen2-7B",
        )

    def test_only_non_string_key_returns_none(self):
        self.assertIsNone(self._detect_from_config("123: junk\n"))

    def test_empty_config_file_returns_none(self):
        self.assertIsNone(self._detect_from_config(""))

    def test_non_dict_config_root_returns_none(self):
        self.assertIsNone(self._detect_from_config("- a\n- b\n"))

    def test_falsy_or_non_string_model_value_returns_none(self):
        self.assertIsNone(self._detect_from_config('model-path: ""\n'))
        self.assertIsNone(self._detect_from_config("model-path: 123\n"))
        self.assertIsNone(self._detect_from_config("model-path: [a, b]\n"))

    def test_model_space_form_still_detected(self):
        self.assertEqual(try_get_model_path(["--model", "X"]), "X")


if __name__ == "__main__":
    unittest.main()
