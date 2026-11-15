"""Startup registration of token-space strategies.

register_models_package(models package)
  one module fails to import -> warned and skipped, the others still registered
"""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.multimodal.token_space_mm_process.registry import (
    TOKEN_SPACE_STRATEGY_MAPPING,
    register_models_package,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PACKAGE = "broken_token_space_models"


class TestStrategyRegistry(unittest.TestCase):
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
                    "sglang.srt.multimodal.token_space_mm_process.registry", "WARNING"
                ) as logs,
            ):
                register_models_package(importlib.import_module(_PACKAGE))
                registered = list(TOKEN_SPACE_STRATEGY_MAPPING)
        self.assertEqual(registered, ["OkForConditionalGeneration"])
        self.assertIn(f"{_PACKAGE}.broken", logs.output[0])


def _forget_package():
    for name in [name for name in sys.modules if name.startswith(_PACKAGE)]:
        del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
