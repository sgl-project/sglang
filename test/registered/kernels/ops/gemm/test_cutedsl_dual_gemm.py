"""Correctness tests for the Blackwell CuTe DSL dual GEMM."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 2048
_INTERMEDIATE_SIZE = 2048


def _make_inputs(
    num_tokens,
    seed,
    input_per_token=True,
    hidden_size=_HIDDEN_SIZE,
    intermediate_size=_INTERMEDIATE_SIZE,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x, x_scale = scaled_fp8_quant(
        torch.randn(
            (num_tokens, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=input_per_token,
    )
    gate_up_weight, gate_up_weight_scale = scaled_fp8_quant(
        torch.randn(
            (2 * intermediate_size, hidden_size),
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
    use_per_token_if_dynamic,
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
        return scaled_fp8_quant(
            activation,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )
    quantized = (
        (activation.float() / output_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return quantized, output_scale


def _make_float_inputs(
    num_tokens,
    dtype,
    seed,
    hidden_size=_HIDDEN_SIZE,
    intermediate_size=_INTERMEDIATE_SIZE,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(
            (num_tokens, hidden_size),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.25,
        torch.randn(
            (2 * intermediate_size, hidden_size),
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

    def _check(
        self,
        quant_mode,
        num_tokens,
        seed,
        tactic=-1,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
    ):
        if tactic < 0:
            from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8
        else:
            from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
                _dual_gemm_swiglu_fp8_with_tactic,
            )

        x, gate_up_weight, x_scale, gate_up_weight_scale = _make_inputs(
            num_tokens,
            seed,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        output_scale = None
        if not quant_mode.is_dynamic:
            _, calibrated_scale = _reference(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                None,
                quant_mode.is_per_token,
            )
            output_scale = calibrated_scale

        args = (x, gate_up_weight, x_scale, gate_up_weight_scale, output_scale)
        actual, actual_scale = (
            dual_gemm_swiglu_fp8(
                *args,
                quant_mode=quant_mode,
            )
            if tactic < 0
            else _dual_gemm_swiglu_fp8_with_tactic(
                *args,
                quant_mode,
                tactic,
            )
        )
        expected, expected_scale = _reference(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            output_scale,
            quant_mode.is_per_token,
        )
        expected_scale_shape = (
            (num_tokens, 1) if quant_mode.is_per_token and num_tokens > 1 else (1,)
        )
        self.assertEqual(actual_scale.shape, expected_scale_shape)
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
            rtol=1e-1,
            atol=1e-1,
        )

    def _check_float(
        self,
        num_tokens,
        dtype,
        seed,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
    ):
        from sglang.kernels.ops.gemm import dual_gemm_swiglu

        x, gate_up_weight = _make_float_inputs(
            num_tokens,
            dtype,
            seed,
            hidden_size,
            intermediate_size,
        )
        actual = dual_gemm_swiglu(x, gate_up_weight)
        expected = _float_reference(x, gate_up_weight)
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_bf16(self):
        self._check_float(16, torch.bfloat16, seed=20261006)

    def test_fp16(self):
        self._check_float(16, torch.float16, seed=20261007)

    def test_fp8_dynamic_per_tensor(self):
        for num_tokens in (4, 16):
            with self.subTest(num_tokens=num_tokens):
                self._check(
                    DualGemmQuantMode.DYNAMIC_PER_TENSOR,
                    num_tokens,
                    seed=20261008,
                )

    def test_fp8_dynamic_per_token(self):
        for num_tokens in (4, 8, 16):
            with self.subTest(num_tokens=num_tokens):
                self._check(
                    DualGemmQuantMode.DYNAMIC_PER_TOKEN,
                    num_tokens,
                    seed=20261009,
                )

    def test_fp8_static_per_tensor(self):
        for num_tokens in (4, 16):
            with self.subTest(num_tokens=num_tokens):
                self._check(
                    DualGemmQuantMode.STATIC_PER_TENSOR,
                    num_tokens,
                    seed=20261010,
                )

    def test_fp8_static_per_token(self):
        for num_tokens in (4, 16):
            with self.subTest(num_tokens=num_tokens):
                self._check(
                    DualGemmQuantMode.STATIC_PER_TOKEN,
                    num_tokens,
                    seed=20261011,
                )

    def test_fp8_qwen_static_per_tensor_bs12(self):
        """Qwen BS12 must not select an eight-token-column MMA tactic."""
        self._check(
            DualGemmQuantMode.STATIC_PER_TENSOR,
            12,
            seed=20261015,
            hidden_size=3584,
            intermediate_size=18944,
        )

    def test_bf16_qwen_bs12(self):
        """The unquantized Qwen path must also use a 16-column MMA tactic."""
        self._check_float(
            12,
            torch.bfloat16,
            seed=20261016,
            hidden_size=3584,
            intermediate_size=18944,
        )

    def test_tactic_token_capacity_validation(self):
        """Explicit eight-column tactics must reject batches above eight."""
        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
            _dual_gemm_swiglu_fp8_run,
            _dual_gemm_swiglu_run,
        )

        x, weight, x_scale, weight_scale = _make_inputs(16, seed=20261017)
        with self.assertRaisesRegex(ValueError, "has 8 token columns"):
            _dual_gemm_swiglu_fp8_run(
                x,
                weight,
                x_scale,
                weight_scale,
                torch.ones((1,), device="cuda", dtype=torch.float32),
                int(DualGemmQuantMode.STATIC_PER_TENSOR),
                tactic=5,
            )

        x, weight = _make_float_inputs(16, torch.bfloat16, seed=20261018)
        with self.assertRaisesRegex(ValueError, "has 8 token columns"):
            _dual_gemm_swiglu_run(x, weight, tactic=1)

    def test_fp8_dynamic_two_cta(self):
        self._check(DualGemmQuantMode.DYNAMIC_PER_TOKEN, 1, seed=20261012, tactic=7)

    def test_fp8_static_two_cta(self):
        self._check(DualGemmQuantMode.STATIC_PER_TENSOR, 1, seed=20261013, tactic=6)

    def test_can_use_dual_gemm(self):
        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import can_use_dual_gemm

        static_mode = DualGemmQuantMode.STATIC_PER_TENSOR
        dynamic_mode = DualGemmQuantMode.DYNAMIC_PER_TOKEN
        self.assertTrue(can_use_dual_gemm(1, 4096, 14336, dynamic_mode))
        self.assertTrue(can_use_dual_gemm(4, 4096, 14336, dynamic_mode))
        self.assertTrue(can_use_dual_gemm(8, 4096, 14336, dynamic_mode))
        self.assertTrue(can_use_dual_gemm(16, 4096, 14336, dynamic_mode))
        self.assertTrue(can_use_dual_gemm(1, 3584, 18944, static_mode))
        self.assertTrue(can_use_dual_gemm(1, 8192, 28672, static_mode))
        self.assertFalse(can_use_dual_gemm(1, 8192, 28672, dynamic_mode))
        self.assertFalse(can_use_dual_gemm(17, 4096, 14336, static_mode))
        self.assertFalse(can_use_dual_gemm(1, 4095, 14336, static_mode))
        self.assertFalse(can_use_dual_gemm(1, 4096, 14335, static_mode))

        with patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=type("Properties", (), {"multi_processor_count": 132})(),
        ):
            self.assertFalse(can_use_dual_gemm(1, 3584, 18944, dynamic_mode))
            self.assertFalse(can_use_dual_gemm(16, 3584, 18944, dynamic_mode))
            self.assertTrue(can_use_dual_gemm(1, 3584, 18944, static_mode))

        with patch(
            "sglang.kernels.ops.gemm.cutedsl_dual_gemm.is_sm100_supported",
            return_value=False,
        ):
            self.assertFalse(can_use_dual_gemm(1, 4096, 14336, static_mode))

    def test_tactic_selection(self):
        from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
            _pick_float16_tactic,
            _pick_fp8_tactic,
        )

        num_sms = 148
        self.assertEqual(
            _pick_fp8_tactic(
                1,
                4096,
                14336,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            4,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                1,
                4096,
                14336,
                num_sms,
                DualGemmQuantMode.STATIC_PER_TENSOR,
            ),
            6,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                4,
                4096,
                14336,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            5,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                16,
                4096,
                14336,
                num_sms,
                DualGemmQuantMode.STATIC_PER_TOKEN,
            ),
            6,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                16,
                4096,
                14336,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            7,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                12,
                3584,
                18944,
                num_sms,
                DualGemmQuantMode.STATIC_PER_TENSOR,
            ),
            6,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                16,
                3584,
                18944,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            7,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                8,
                4096,
                4096,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            2,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                16,
                4096,
                11008,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            3,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                4,
                3584,
                14336,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            ),
            1,
        )
        self.assertEqual(
            _pick_fp8_tactic(
                4,
                5120,
                14336,
                num_sms,
                DualGemmQuantMode.DYNAMIC_PER_TENSOR,
            ),
            5,
        )
        self.assertEqual(
            _pick_float16_tactic(16, 4096, 14336, num_sms, torch.bfloat16),
            2,
        )
        self.assertEqual(
            _pick_float16_tactic(16, 4096, 14336, num_sms, torch.float16),
            3,
        )
        self.assertEqual(
            _pick_float16_tactic(16, 3584, 18944, num_sms, torch.bfloat16),
            2,
        )
        self.assertEqual(
            _pick_float16_tactic(4, 4096, 18944, num_sms, torch.bfloat16),
            2,
        )
        self.assertEqual(
            _pick_float16_tactic(4, 4096, 4096, num_sms, torch.float16),
            2,
        )
        self.assertEqual(
            _pick_float16_tactic(4, 4096, 7168, num_sms, torch.bfloat16),
            1,
        )
        self.assertEqual(
            _pick_float16_tactic(4, 4096, 11008, num_sms, torch.float16),
            2,
        )
        self.assertEqual(
            _pick_float16_tactic(8, 8192, 28672, num_sms, torch.float16),
            2,
        )

    def test_fp8_quant_mode_validation(self):
        from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8

        x, gate_up_weight, x_scale, gate_up_weight_scale = _make_inputs(
            4,
            seed=20261014,
        )
        scalar_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

        with self.assertRaisesRegex(ValueError, "must be None"):
            dual_gemm_swiglu_fp8(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                scalar_scale,
                DualGemmQuantMode.DYNAMIC_PER_TENSOR,
            )
        with self.assertRaisesRegex(ValueError, "must be a scale tensor"):
            dual_gemm_swiglu_fp8(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                None,
                DualGemmQuantMode.STATIC_PER_TENSOR,
            )
        with self.assertRaisesRegex(ValueError, "requires 4 output scale"):
            dual_gemm_swiglu_fp8(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                scalar_scale,
                DualGemmQuantMode.STATIC_PER_TOKEN,
            )
        with self.assertRaisesRegex(ValueError, "not valid for the FP8"):
            dual_gemm_swiglu_fp8(
                x,
                gate_up_weight,
                x_scale,
                gate_up_weight_scale,
                None,
                DualGemmQuantMode.UNQUANT,
            )


if __name__ == "__main__":
    unittest.main()
