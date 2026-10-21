"""Real provider methods test admission, geometry, and finalization with GPU boundaries mocked."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
PROVIDER = ROOT / "python/sglang/srt/lora/moe/base_gemm_provider"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


class TestMaterializedFinalize(CustomTestCase):
    """Execute the host dispatch with real CPU buffers and mocked GPU launches."""

    def test_materialized_finalize_preserves_strided_weights(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        finalize = types.ModuleType("sglang.kernels.ops.lora.moe.finalize")
        finalize.invoke_small_finalize = mock.Mock()
        reorder = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        reorder.post_reorder_deepgemm = mock.Mock()
        # Production hands over strided top-k weight views; post_reorder_deepgemm
        # addresses them contiguously, so even a small batch must not take the
        # gather path and must receive a contiguous copy.
        tokens = 64
        weights = torch.arange(tokens * 6).view(tokens, 6)[:, ::2]
        ids = torch.zeros((tokens, 3), dtype=torch.int64)
        output = torch.empty((tokens, 5))
        delta = torch.empty((tokens, 3, 5))
        with mock.patch.dict(
            sys.modules, {finalize.__name__: finalize, reorder.__name__: reorder}
        ):
            MoeBaseProvider().finalize(
                SimpleNamespace(pair_to_row=torch.arange(tokens * 3)),
                torch.empty((tokens * 3, 5)),
                ids,
                weights,
                2.5,
                output,
                lora_delta=delta,
            )
        finalize.invoke_small_finalize.assert_not_called()
        call = reorder.post_reorder_deepgemm.call_args
        actual = call.args[4]
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, weights)
        self.assertIs(call.kwargs["lora_delta"], delta)


if __name__ == "__main__":
    unittest.main()
