"""Keep compact DeepGEMM's SwiGLU clamp within its existing workspace."""

import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from sglang.srt.layers.moe.moe_runner.deep_gemm import _apply_swiglu_limit
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _NoNewTensorStorage(TorchDispatchMode):
    def __init__(self, workspace):
        super().__init__()
        self.storage_ptr = workspace.untyped_storage().data_ptr()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                if tensor.untyped_storage().data_ptr() != self.storage_ptr:
                    raise AssertionError(f"SwiGLU clamp allocated new storage: {func}")
        return result


class TestDeepGemmSwigluLimit(CustomTestCase):
    def test_clamp_reuses_workspace_and_preserves_values(self):
        # A compact GLM prefill can already be near the memory limit. Copies
        # of the gate/up halves and their concatenation must not be live here.
        values = [-float("inf"), -11, -10, -1, 0, 9, 10, 11, float("inf"), float("nan")]
        expected_gate = [-float("inf"), -11, -10, -1, 0, 9, 10, 10, 10, float("nan")]
        expected_up = [-10, -10, -10, -1, 0, 9, 10, 10, 10, float("nan")]
        for rows in (0, 1, 4):
            with self.subTest(rows=rows):
                workspace = torch.tensor(
                    [values + values], dtype=torch.bfloat16
                ).repeat(rows, 1)
                expected = torch.tensor(
                    [expected_gate + expected_up], dtype=torch.bfloat16
                ).repeat(rows, 1)
                with _NoNewTensorStorage(workspace):
                    result = _apply_swiglu_limit(workspace, swiglu_limit=10)
                self.assertIs(result, workspace)
                torch.testing.assert_close(
                    result, expected, rtol=0, atol=0, equal_nan=True
                )


if __name__ == "__main__":
    unittest.main()
