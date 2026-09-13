"""Unit tests for compile-time dimensions in the DeepGEMM wrappers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.test.test_utils import CustomTestCase


class TestDeepGemmCompiledDims(CustomTestCase):
    def setUp(self):
        self.lhs = (
            torch.empty((1, 128), dtype=torch.float8_e4m3fn),
            torch.empty((1, 1), dtype=torch.float32),
        )
        self.rhs = (
            torch.empty((64, 128), dtype=torch.float8_e4m3fn),
            torch.empty((1, 1), dtype=torch.float32),
        )
        self.out = torch.empty((1, 64), dtype=torch.bfloat16)

    def _patch_deep_gemm(self):
        fake_deep_gemm = MagicMock()
        patches = (
            patch.object(entrypoint, "deep_gemm", fake_deep_gemm, create=True),
            patch.object(entrypoint, "_sanity_check_input"),
            patch.object(
                entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ),
        )
        return fake_deep_gemm, patches

    def test_fp8_gemm_compiles_n_and_k(self):
        fake_deep_gemm, patches = self._patch_deep_gemm()
        with patches[0], patches[1], patches[2]:
            entrypoint.gemm_nt_f8f8bf16(self.lhs, self.rhs, self.out)

        self.assertEqual(
            fake_deep_gemm.fp8_gemm_nt.call_args.kwargs["compiled_dims"], "nk"
        )

    def test_mxfp8_gemm_compiles_n_and_k(self):
        fake_deep_gemm, patches = self._patch_deep_gemm()
        with patches[0], patches[1], patches[2]:
            entrypoint.gemm_nt_mxfp8_f8f8bf16(self.lhs, self.rhs, self.out)

        self.assertEqual(
            fake_deep_gemm.fp8_fp4_gemm_nt.call_args.kwargs["compiled_dims"],
            "nk",
        )

    def test_bf16_gemm_compiles_n_and_k(self):
        lhs = torch.empty((1, 128), dtype=torch.bfloat16)
        rhs = torch.empty((64, 128), dtype=torch.bfloat16)
        fake_deep_gemm, patches = self._patch_deep_gemm()
        with patches[0], patches[1], patches[2]:
            entrypoint.gemm_nt_bf16bf16f32(lhs, rhs, self.out)

        self.assertEqual(
            fake_deep_gemm.bf16_gemm_nt.call_args.kwargs["compiled_dims"], "nk"
        )


if __name__ == "__main__":
    unittest.main()
