"""Unit tests for sglang.srt.utils.log_utils."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A worker whose only sglang import is log_utils, like a spawned detokenizer.
_WORKER = """
import logging
from types import SimpleNamespace

from sglang.srt.utils.log_utils import configure_logger

configure_logger(SimpleNamespace(log_level="info"))
print(logging.getLevelName(logging.getLogger("custom").level))
"""


class TestConfigureLoggerFromConfigPath(CustomTestCase):
    def test_config_path_applies_without_other_logging_config_importers(self):
        """SGLANG_LOGGING_CONFIG_PATH must apply in a process where nothing
        else has imported ``logging.config``."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {"custom": {"level": "ERROR"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp, "logging.json")
            config_path.write_text(json.dumps(config))
            result = subprocess.run(
                [sys.executable, "-c", _WORKER],
                env={**os.environ, "SGLANG_LOGGING_CONFIG_PATH": str(config_path)},
                capture_output=True,
                text=True,
                timeout=300,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "ERROR")


if __name__ == "__main__":
    unittest.main()
