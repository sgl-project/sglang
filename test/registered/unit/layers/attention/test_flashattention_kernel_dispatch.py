import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BACKEND_SOURCE = (
    Path(sglang.__file__).parent / "srt/layers/attention/flashattention_backend.py"
)
_ENTRY_POINTS = ("flash_attn_with_kvcache", "flash_attn_varlen_func")


class TestFlashAttentionKernelDispatch(CustomTestCase):
    def test_kernel_calls_use_selected_entry_points(self):
        # A call through any name other than self.flash_attn_* bypasses the build
        # __init__ selected, e.g. generic FA4 on SM12x, or FA3 when `ver` is omitted.
        tree = ast.parse(_BACKEND_SOURCE.read_text())
        bypasses = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _ENTRY_POINTS:
                bypasses.append(f"line {node.lineno}: {func.id}(...)")
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _ENTRY_POINTS
                and not (isinstance(func.value, ast.Name) and func.value.id == "self")
            ):
                bypasses.append(f"line {node.lineno}: {ast.unparse(func)}(...)")
        self.assertEqual(bypasses, [])


if __name__ == "__main__":
    unittest.main()
