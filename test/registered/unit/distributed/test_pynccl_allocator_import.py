"""Regression test for https://github.com/sgl-project/sglang/issues/28999.

``pynccl_allocator`` must not import private ``torch.cuda.memory`` symbols at
module scope: they are absent before torch 2.8 and abort startup on Ascend NPU.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# test/registered/unit/distributed/<this file> -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = (
    REPO_ROOT / "python/sglang/srt/distributed/device_communicators/pynccl_allocator.py"
)


def _import_time_nodes(tree: ast.Module):
    """Yield nodes that run at import time, including ``try`` / ``if`` bodies."""
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
