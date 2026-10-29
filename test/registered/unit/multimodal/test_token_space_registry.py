"""Startup registration of token-space strategies.

import managers.multimodal_processor -> no token-space model module imported
register_token_space_models()
  one module fails to import -> warned and skipped, the others still registered
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.multimodal.token_space_mm_process import models
from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_STRATEGY_MAPPING,
    register_token_space_models,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODELS = "sglang.srt.multimodal.token_space_mm_process.models"


class TestStrategyRegistry(unittest.TestCase):
    def test_importing_the_manager_leaves_strategy_modules_unloaded(self):
        """Every server imports the manager; only a multimodal one loads strategies."""
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, sglang.srt.managers.multimodal_processor; "
                f"print(sorted(m for m in sys.modules if m.startswith({_MODELS!r})))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(loaded, "[]")

    def test_a_module_that_fails_to_import_is_skipped(self):
        """One broken strategy module must not stop every server from starting."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "ok.py").write_text(
                "from sglang.srt.multimodal.token_space_mm_process.process_strategy "
                "import TokenSpaceProcessStrategy\n"
                "class OkStrategy(TokenSpaceProcessStrategy):\n"
                "    models = ('OkForConditionalGeneration',)\n"
            )
            (package / "broken.py").write_text("import a_module_that_does_not_exist\n")
            self.addCleanup(_forget_modules)
            with (
                patch.object(models, "__path__", [directory]),
                patch.dict(TOKEN_SPACE_STRATEGY_MAPPING, clear=True),
                self.assertLogs(
                    "sglang.srt.multimodal.token_space_mm_process.registry", "WARNING"
                ) as logs,
            ):
                register_token_space_models()
                registered = list(TOKEN_SPACE_STRATEGY_MAPPING)
        self.assertEqual(registered, ["OkForConditionalGeneration"])
        self.assertIn(f"{_MODELS}.broken", logs.output[0])


def _forget_modules():
    for name in (f"{_MODELS}.ok", f"{_MODELS}.broken"):
        sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
