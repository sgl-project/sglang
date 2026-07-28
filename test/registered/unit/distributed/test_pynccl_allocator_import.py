"""Regression test for https://github.com/sgl-project/sglang/issues/28999.

``pynccl_allocator`` is imported eagerly across the codebase (model_runner,
linear, dp_attention, fp8, ... and the Ascend NPU graph runners), so importing
a torch>=2.8-only symbol at module scope aborts startup on backends pinned to
older torch. On Ascend, ``torch_npu`` pins torch 2.7, whose
``torch.cuda.memory`` exposes neither ``_cuda_beginAllocateCurrentThreadToPool``
nor ``_cuda_endAllocateToPool`` -- and a ``from ... import (a, b, c)`` statement
is all-or-nothing, so one missing name takes the whole import down.

Those symbols are re-exported from ``torch._C``, so the call sites reach them
lazily via ``torch._C.<name>``. This test guards the module-scope import list
statically: it detects a regression on any torch version, without a GPU or NPU,
and without importing the module under test.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# test/registered/unit/distributed/<this file> -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = (
    REPO_ROOT / "python/sglang/srt/distributed/device_communicators/pynccl_allocator.py"
)


def _import_time_nodes(tree: ast.Module):
    """Yield nodes that execute when the module is imported.

    Descends into module-level ``try`` / ``if`` blocks -- a guarded import still
    runs at import time -- but not into functions or classes, where imports are
    deferred until the symmetric-memory path actually calls them.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


class TestPyncclAllocatorImportGuard(CustomTestCase):
    def test_no_import_time_private_cuda_memory_symbols(self):
        self.assertTrue(
            SOURCE_PATH.is_file(),
            f"cannot locate pynccl_allocator.py at {SOURCE_PATH}; "
            "update REPO_ROOT if the tree layout changed",
        )
        tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))

        offenders = [
            alias.name
            for node in _import_time_nodes(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "torch.cuda.memory"
            for alias in node.names
            if alias.name.startswith("_cuda_")
        ]

        self.assertEqual(
            offenders,
            [],
            "pynccl_allocator must not import private torch.cuda.memory symbols "
            f"at module scope (found {offenders}); these are absent on torch<2.8 "
            "and break startup on Ascend NPU. Reach them via torch._C.<name> at "
            "the call site instead.",
        )


if __name__ == "__main__":
    unittest.main()
