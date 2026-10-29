"""Startup registration of token-space strategies.

import managers.multimodal_processor -> no token-space model module imported
import_processors(models package)
  one module fails to import -> warned and skipped, the others still registered
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers.multimodal_processor import import_processors
from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_STRATEGY_MAPPING,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PACKAGE = "broken_token_space_models"
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
            package = Path(directory) / _PACKAGE
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "ok.py").write_text(
                "from sglang.srt.multimodal.token_space_mm_process.process_strategy "
                "import TokenSpaceProcessStrategy\n"
                "class OkStrategy(TokenSpaceProcessStrategy):\n"
                "    models = ('OkForConditionalGeneration',)\n"
            )
            (package / "broken.py").write_text("import a_module_that_does_not_exist\n")
            sys.path.insert(0, directory)
            self.addCleanup(sys.path.remove, directory)
            self.addCleanup(_forget_package)
            with (
                patch.dict(TOKEN_SPACE_STRATEGY_MAPPING, clear=True),
                self.assertLogs(
                    "sglang.srt.managers.multimodal_processor", "WARNING"
                ) as logs,
            ):
                import_processors(_PACKAGE)
                registered = list(TOKEN_SPACE_STRATEGY_MAPPING)
        self.assertEqual(registered, ["OkForConditionalGeneration"])
        self.assertIn(f"{_PACKAGE}.broken", logs.output[0])


def _forget_package():
    for name in [name for name in sys.modules if name.startswith(_PACKAGE)]:
        del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
