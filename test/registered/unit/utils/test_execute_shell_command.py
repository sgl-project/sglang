"""CPU-only subprocess argument regressions; no server or model is launched."""

import json
import os
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import execute_shell_command

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExecuteShellCommand(CustomTestCase):
    def _run(self, arguments: str, prefix: str = ""):
        script = "import json, os, sys; print(json.dumps([sys.argv[1:], os.getenv('SGLANG_QUOTED_TEST')]))"
        command = f"{prefix}{shlex.quote(sys.executable)} -c {shlex.quote(script)} {arguments}"
        popen = subprocess.Popen
        with patch(
            "sglang.utils.subprocess.Popen",
            side_effect=lambda *args, **kwargs: popen(
                *args, stdout=subprocess.PIPE, **kwargs
            ),
        ):
            process = execute_shell_command(command)
        try:
            output, _ = process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        self.assertEqual(process.returncode, 0, output)
        return json.loads(output)

    def test_preserves_quoted_model_path_and_json_argument(self):
        args, _ = self._run(
            "--model-path '/models/my model' --overrides '{\"name\": \"two words\"}'"
        )
        self.assertEqual(
            args,
            [
                "--model-path",
                "/models/my model",
                "--overrides",
                '{"name": "two words"}',
            ],
        )

    def test_preserves_quoted_environment_value_without_changing_parent(self):
        before = os.environ.get("SGLANG_QUOTED_TEST")
        args, value = self._run(
            "--port 30000", prefix="SGLANG_QUOTED_TEST='two words' "
        )
        self.assertEqual(args, ["--port", "30000"])
        self.assertEqual(value, "two words")
        self.assertEqual(os.environ.get("SGLANG_QUOTED_TEST"), before)

    def test_preserves_escaped_spaces_empty_arguments_and_backslashes(self):
        args, _ = self._run(r"one\ two '' 'C:\models\example'")
        self.assertEqual(args, ["one two", "", r"C:\models\example"])

    def test_still_supports_multiline_commands(self):
        continuation = " " + "\\" + "\n"
        args, value = self._run(
            "--port" + continuation + "30000",
            prefix="SGLANG_QUOTED_TEST=plain" + continuation,
        )
        self.assertEqual(args, ["--port", "30000"])
        self.assertEqual(value, "plain")

    def test_preserves_double_quoted_arguments_without_shell_expansion(self):
        args, _ = self._run(r'"a \"quoted\" value" "$(echo should-not-run)"')
        self.assertEqual(args, ['a "quoted" value', "$(echo should-not-run)"])

    def test_rejects_plain_environment_assignments_without_an_executable(self):
        with self.assertRaisesRegex(ValueError, "no executable"):
            execute_shell_command("SGLANG_QUOTED_TEST=plain")

    def test_rejects_environment_assignments_without_an_executable(self):
        with self.assertRaisesRegex(ValueError, "no executable"):
            execute_shell_command("SGLANG_QUOTED_TEST='two words'")


if __name__ == "__main__":
    unittest.main()
