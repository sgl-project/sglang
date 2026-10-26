"""Draft-model ``set_embed_and_head`` / ``set_embed`` stay device-neutral.

The EAGLE-family draft worker calls them at init on every device, so a bare
``torch.cuda.*()`` call there breaks speculative decoding off CUDA.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODELS_ROOT = Path(next(iter(sglang.__path__))) / "srt" / "models"


def _unguarded_cuda_calls_in_set_embed(path: Path):
    """``torch.cuda.*()`` statements directly in a ``set_embed*`` body.

    Calls nested under an ``if`` (``torch.cuda.is_available()``, ``_is_xpu``) are
    treated as guarded.
    """
    tree = ast.parse(path.read_text())
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("set_embed")):
            continue
        for stmt in fn.body:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            callee = ast.unparse(stmt.value.func)
            if callee.startswith("torch.cuda."):
                yield f"{path.relative_to(_MODELS_ROOT)}:{stmt.lineno}: {fn.name} calls {callee}()"


class TestSetEmbedDeviceNeutral(CustomTestCase):
    def test_no_unguarded_cuda_calls(self):
        """A bare ``torch.cuda.synchronize()`` here crashed NEXTN/MTP draft init on
        XPU ("Torch not compiled with CUDA enabled")."""
        offenders = [
            hit
            for path in sorted(_MODELS_ROOT.rglob("*.py"))
            for hit in _unguarded_cuda_calls_in_set_embed(path)
        ]
        self.assertEqual(
            offenders,
            [],
            "use current_platform.synchronize() / empty_cache() instead",
        )


if __name__ == "__main__":
    unittest.main()
