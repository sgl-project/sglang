import unittest
from contextlib import nullcontext
from unittest.mock import call, patch

import torch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper import entrypoint as deep_gemm_entrypoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPpuBf16DeepgemmWarmup(unittest.TestCase):
    def test_bf16_executor_registration(self):
        with patch.object(compile_utils.torch, "empty") as empty:
            executor = compile_utils._BaseWarmupExecutor.create(
                compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16BF16,
                max_m=4,
                n=8,
                k=16,
                num_groups=1,
            )

        self.assertIsInstance(executor, compile_utils._BF16WarmupExecutor)
        self.assertEqual(
            empty.call_args_list,
            [
                call((4, 16), device="cuda", dtype=torch.bfloat16),
                call((8, 16), device="cuda", dtype=torch.bfloat16),
                call((4, 8), device="cuda", dtype=torch.bfloat16),
            ],
        )

    def test_bf16_memory_requirement(self):
        required = compile_utils._BaseWarmupExecutor.get_memory_requirement(
            compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16BF16,
            max_m=4,
            n=8,
            k=16,
            num_groups=1,
        )
        expected = (4 * 16 * 2 + 8 * 16 * 2 + 4 * 8 * 2) / (1 << 30)
        self.assertEqual(required, expected)

    def test_entrypoint_uses_bf16_execution_hook(self):
        lhs = torch.empty(2, 4, dtype=torch.bfloat16)
        rhs = torch.empty(3, 4, dtype=torch.bfloat16)
        out = torch.empty(2, 3, dtype=torch.bfloat16)

        with (
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as hook,
            patch.object(deep_gemm_entrypoint, "deep_gemm") as deep_gemm,
        ):
            deep_gemm_entrypoint.gemm_nt_bf16bf16bf16(lhs, rhs, out)

        hook.assert_called_once_with(
            2,
            3,
            4,
            1,
            compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16BF16,
        )
        deep_gemm.bf16_gemm_nt.assert_called_once_with(lhs, rhs, out)


if __name__ == "__main__":
    unittest.main()
