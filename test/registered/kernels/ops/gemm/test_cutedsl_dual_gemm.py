"""Correctness tests for the Blackwell CuTe DSL dual GEMM."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 2048
_INTERMEDIATE_SIZE = 2048


def _make_inputs(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x, x_scale = scaled_fp8_quant(
        torch.randn(
            (1, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=True,
    )
    gate_up_weight, gate_up_weight_scale = scaled_fp8_quant(
        torch.randn(
            (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=True,
    )
    return (
        x,
        gate_up_weight,
        x_scale.reshape(-1),
        gate_up_weight_scale.reshape(-1),
    )


def _reference(
    x,
    gate_up_weight,
    x_scale,
    gate_up_weight_scale,
    output_scale,
):
    gate_up = fp8_scaled_mm(
        x,
        gate_up_weight.T,
        x_scale,
        gate_up_weight_scale,
        torch.bfloat16,
    )
    activation = silu_and_mul(gate_up)
    if output_scale is None:
        return scaled_fp8_quant(activation, use_per_token_if_dynamic=True)
    return scaled_fp8_quant(activation, scale=output_scale)


def _make_float_inputs(dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(
            (1, _HIDDEN_SIZE),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.25,
        torch.randn(
            (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.25,
    )


def _float_reference(x, gate_up_weight):
    return silu_and_mul(torch.nn.functional.linear(x, gate_up_weight))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCuteDSLDualGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def _check(self, dynamic_quant, seed, tactic=-1):
        if tactic < 0:
            from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8
        else:
            from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
                _dual_gemm_swiglu_fp8_with_tactic,
            )

        x, gate_up_weight, x_scale, gate_up_weight_scale = _make_inputs(seed)
        output_scale = None
        if not dynamic_quant:
            _, calibrated_scale = _reference(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                None,
            )
            output_scale = calibrated_scale.reshape(1)

        args = (x, gate_up_weight, x_scale, gate_up_weight_scale, output_scale)
        actual, actual_scale = (
            dual_gemm_swiglu_fp8(*args)
            if tactic < 0
            else _dual_gemm_swiglu_fp8_with_tactic(*args, tactic)
        )
        expected, expected_scale = _reference(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            output_scale,
        )
        torch.testing.assert_close(
            actual_scale,
            expected_scale.reshape_as(actual_scale),
            rtol=2e-2,
            atol=1e-5,
        )
        # Compare dequantized activations.  The fused fast sigmoid and the
        # production activation kernel are allowed their normal BF16 rounding
        # difference before E4M3 quantization.
        actual_dequantized = actual.float() * actual_scale
        expected_dequantized = expected.float() * expected_scale
        torch.testing.assert_close(
            actual_dequantized,
            expected_dequantized,
            rtol=5e-2,
            atol=5e-2,
        )

    def _check_float(self, dtype, seed):
        from sglang.kernels.ops.gemm import dual_gemm_swiglu

        x, gate_up_weight = _make_float_inputs(dtype, seed)
        actual = dual_gemm_swiglu(x, gate_up_weight)
        expected = _float_reference(x, gate_up_weight)
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_bf16(self):
        self._check_float(torch.bfloat16, seed=20261006)

    def test_fp16(self):
        self._check_float(torch.float16, seed=20261007)

    def test_fp8_dynamic(self):
        self._check(dynamic_quant=True, seed=20261008)

    def test_fp8_static(self):
        self._check(dynamic_quant=False, seed=20261009)

    def test_fp8_dynamic_two_cta(self):
        self._check(dynamic_quant=True, seed=20261008, tactic=7)

    def test_fp8_static_two_cta(self):
        self._check(dynamic_quant=False, seed=20261009, tactic=6)

    def test_can_use_dual_gemm(self):
        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import can_use_dual_gemm

        self.assertTrue(can_use_dual_gemm(1, 4096, 14336))
        self.assertTrue(can_use_dual_gemm(1, 3584, 18944))
        self.assertFalse(can_use_dual_gemm(2, 4096, 14336))
        self.assertFalse(can_use_dual_gemm(1, 4095, 14336))
        self.assertFalse(can_use_dual_gemm(1, 4096, 14335))
        with patch(
            "sglang.kernels.ops.gemm.cutedsl_dual_gemm.is_sm100_supported",
            return_value=False,
        ):
            self.assertFalse(can_use_dual_gemm(1, 4096, 14336))


if __name__ == "__main__":
    unittest.main()
