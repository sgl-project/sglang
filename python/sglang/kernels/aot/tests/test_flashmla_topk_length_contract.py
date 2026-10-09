"""
Regression contract tests for FlashMLA topk_length propagation.

Ensures the full chain — dsa_backend._compute_flashmla_metadata → get_mla_metadata
(sgl_kernel + PPU wrapper) — consistently accepts and forwards topk_length, so the
scheduler metadata and kernel page-walk use the same effective topk per query.

A "half-migrated" state (kernel passes topk_length but scheduler does not) is the
exact bug this contract detects.

Reference: B1 commit 6dd5c0cc6e on v0.5.18_rel.
"""

import ast
import inspect
import os
import sys
import textwrap
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate repo root from this file (tests/ → aot/ → kernels/ → sglang/ → python/ → root)
# ---------------------------------------------------------------------------
_REPO_ROOT = (
    Path(__file__).resolve().parents[5]
)  # python/sglang/kernels/aot/tests → root


def _parse_file(rel_path: str) -> ast.Module:
    """Parse a Python file by repo-relative path and return its AST."""
    full = _REPO_ROOT / rel_path
    assert full.exists(), f"File not found: {full}"
    return ast.parse(full.read_text(), filename=str(full))


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Find a top-level or class-level function definition by name."""
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"Function '{name}' not found in AST")


def _get_param_names(func_def: ast.FunctionDef) -> list[str]:
    """Return the list of parameter names (excluding *args/**kwargs)."""
    args = func_def.args
    names = [a.arg for a in args.posonlyargs + args.args]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return names


def _find_call_with_keyword(tree: ast.Module, func_name: str, keyword: str) -> bool:
    """Check if any call to `func_name` in the tree passes `keyword` as a kwarg."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Match both `get_mla_metadata(...)` and `module.get_mla_metadata(...)`
            callee = node.func
            callee_name = None
            if isinstance(callee, ast.Name):
                callee_name = callee.id
            elif isinstance(callee, ast.Attribute):
                callee_name = callee.attr
            if callee_name == func_name:
                for kw in node.keywords:
                    if kw.arg == keyword:
                        return True
    return False


class TestTopkLengthSignatureContract(unittest.TestCase):
    """Verify all three files in the topk_length chain have the parameter."""

    def test_sgl_kernel_flash_mla_has_topk_length_param(self):
        tree = _parse_file("python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py")
        func = _find_function_def(tree, "get_mla_metadata")
        params = _get_param_names(func)
        self.assertIn(
            "topk_length",
            params,
            "sgl_kernel.flash_mla.get_mla_metadata must accept topk_length",
        )

    def test_ppu_flash_mla_has_topk_length_param(self):
        tree = _parse_file(
            "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py"
        )
        func = _find_function_def(tree, "get_mla_metadata")
        params = _get_param_names(func)
        self.assertIn(
            "topk_length",
            params,
            "PPU flash_mla.get_mla_metadata must accept topk_length",
        )

    def test_ppu_flash_mla_forwards_topk_length(self):
        """PPU wrapper must forward topk_length to _flashmla.get_mla_metadata."""
        tree = _parse_file(
            "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py"
        )
        self.assertTrue(
            _find_call_with_keyword(tree, "get_mla_metadata", "topk_length"),
            "PPU flash_mla.get_mla_metadata must pass topk_length= to _flashmla",
        )


class TestTopkLengthPropagationContract(unittest.TestCase):
    """Verify dsa_backend._compute_flashmla_metadata passes topk_length to scheduler."""

    def test_compute_flashmla_metadata_passes_topk_length(self):
        tree = _parse_file("python/sglang/srt/layers/attention/dsa_backend.py")
        func = _find_function_def(tree, "_compute_flashmla_metadata")
        # Walk the function body for a call to get_mla_metadata with topk_length=
        found = False
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                callee = node.func
                callee_name = None
                if isinstance(callee, ast.Name):
                    callee_name = callee.id
                elif isinstance(callee, ast.Attribute):
                    callee_name = callee.attr
                if callee_name == "get_mla_metadata":
                    for kw in node.keywords:
                        if kw.arg == "topk_length":
                            found = True
                            break
        self.assertTrue(
            found,
            "_compute_flashmla_metadata must pass topk_length= to get_mla_metadata. "
            "This is the half-migration detection: if kernel uses topk_length but "
            "scheduler does not, odd-bs CUDA Graph page-walk faults will occur.",
        )


class TestTopkLengthNoneDefault(unittest.TestCase):
    """Verify non-PPU/non-kpool callers are unaffected (topk_length defaults to None)."""

    def test_sgl_kernel_topk_length_has_none_default(self):
        tree = _parse_file("python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py")
        func = _find_function_def(tree, "get_mla_metadata")
        # Find the default for topk_length
        args = func.args
        all_args = args.posonlyargs + args.args + args.kwonlyargs
        defaults_for_args = [None] * (len(args.args) - len(args.defaults)) + list(
            args.defaults
        )
        # Also check kwonly defaults
        all_defaults = defaults_for_args + list(args.kw_defaults)
        all_params = args.args + args.kwonlyargs

        for param, default in zip(all_params, all_defaults):
            if param.arg == "topk_length":
                self.assertIsNotNone(default, "topk_length must have a default value")
                self.assertIsInstance(
                    default,
                    ast.Constant,
                    "topk_length default must be a constant (None)",
                )
                self.assertIsNone(
                    default.value,
                    "topk_length must default to None for non-PPU compatibility",
                )
                return
        self.fail("topk_length parameter not found in get_mla_metadata signature")

    def test_ppu_topk_length_has_none_default(self):
        tree = _parse_file(
            "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py"
        )
        func = _find_function_def(tree, "get_mla_metadata")
        args = func.args
        all_params = args.args + args.kwonlyargs
        defaults_for_args = [None] * (len(args.args) - len(args.defaults)) + list(
            args.defaults
        )
        all_defaults = defaults_for_args + list(args.kw_defaults)

        for param, default in zip(all_params, all_defaults):
            if param.arg == "topk_length":
                self.assertIsNotNone(default, "topk_length must have a default")
                self.assertIsInstance(default, ast.Constant)
                self.assertIsNone(
                    default.value,
                    "PPU topk_length must default to None",
                )
                return
        self.fail("topk_length parameter not found in PPU get_mla_metadata signature")


class TestFlashMLABackendUnaffected(unittest.TestCase):
    """Verify flashmla_backend.py calls do NOT pass topk_length (dense-only paths)."""

    def test_flashmla_backend_no_topk_length(self):
        tree = _parse_file("python/sglang/srt/layers/attention/flashmla_backend.py")
        has_topk_length = _find_call_with_keyword(
            tree, "get_mla_metadata", "topk_length"
        )
        self.assertFalse(
            has_topk_length,
            "flashmla_backend.py should NOT pass topk_length (dense attention path)",
        )


if __name__ == "__main__":
    unittest.main()
