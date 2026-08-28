"""CPU-only contracts for Triton PDL-aware launcher calls."""

import ast
import importlib.util
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Load the CI marker directly so this source-only test does not import the full
# sglang package (or any GPU-only dependencies) during collection.
_CI_REGISTER_PATH = _REPO_ROOT / "python/sglang/test/ci/ci_register.py"
_CI_REGISTER_SPEC = importlib.util.spec_from_file_location(
    "sglang_ci_register", _CI_REGISTER_PATH
)
assert _CI_REGISTER_SPEC is not None
assert _CI_REGISTER_SPEC.loader is not None
_ci_register = importlib.util.module_from_spec(_CI_REGISTER_SPEC)
_CI_REGISTER_SPEC.loader.exec_module(_ci_register)
register_cpu_ci = _ci_register.register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_PDL_LAUNCHES = (
    (
        "python/sglang/kernels/ops/elementwise/elementwise.py",
        "fused_gate_sigmoid_mul_add",
        "_fused_gate_sigmoid_mul_add_kernel",
    ),
    (
        "python/sglang/kernels/ops/moe/pack_topk_ids.py",
        "triton",
        "_pack_topk_ids_triton_kernel",
    ),
    (
        "python/sglang/kernels/ops/moe/inkling_moe.py",
        "silu_and_mul_triton",
        "_silu_and_mul_triton_kernel",
    ),
    (
        "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        "static_quant_fp8",
        "_static_quant_fp8",
    ),
)


def _find_function(tree: ast.AST, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"function {function_name!r} was not found")


def _find_kernel_launch(function: ast.FunctionDef, kernel_name: str) -> ast.Call:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == kernel_name:
            return node
    raise AssertionError(
        f"launch of {kernel_name!r} was not found in {function.name!r}"
    )


class TestPdlLaunchKwargs(unittest.TestCase):
    def test_pdl_is_explicitly_passed_to_each_pdl_aware_launch(self):
        for source_path, function_name, kernel_name in _PDL_LAUNCHES:
            with self.subTest(source_path=source_path):
                source = (_REPO_ROOT / source_path).read_text()
                function = _find_function(ast.parse(source), function_name)
                launch = _find_kernel_launch(function, kernel_name)
                use_pdl = next(
                    (
                        keyword.value.id
                        for keyword in launch.keywords
                        if keyword.arg == "USE_PDL"
                        and isinstance(keyword.value, ast.Name)
                    ),
                    None,
                )
                self.assertEqual(
                    use_pdl,
                    "use_pdl",
                    f"{source_path}:{function_name} must pass USE_PDL=use_pdl",
                )


if __name__ == "__main__":
    unittest.main()
