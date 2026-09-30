"""Integration tests for the reusable dual GEMM model layer."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 2048
_INTERMEDIATE_SIZE = 2048


def _initialize_mlp_weights(mlp, dtype):
    generator = torch.Generator(device="cuda").manual_seed(20261010)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(
            torch.randn(
                mlp.gate_up_proj.weight.shape,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.02
        )
        mlp.down_proj.weight.copy_(
            torch.randn(
                mlp.down_proj.weight.shape,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.02
        )
    return mlp, generator


def _make_llama_mlp(quant_config=None, dtype=torch.bfloat16):
    from sglang.srt.models.llama import LlamaMLP

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with patch.dict(os.environ, {"SGLANG_ENABLE_DUAL_GEMM": "1"}):
            mlp = LlamaMLP(
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
                "silu",
                quant_config=quant_config,
                reduce_results=False,
                tp_rank=0,
                tp_size=1,
            ).cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(mlp, dtype)


def _make_qwen2_mlp(dtype=torch.bfloat16):
    from sglang.srt.models.qwen2 import Qwen2MLP

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_DUAL_GEMM": "1"}),
            get_parallel().override(tp_rank=0, tp_size=1),
        ):
            mlp = Qwen2MLP(
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
                "silu",
            ).cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(mlp, dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDualGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def test_decode_token_gate(self):
        """The integration must fall back outside the kernel's token contract."""
        mlp, _ = _make_llama_mlp()
        self.assertFalse(
            mlp.dual_gemm.can_run(
                torch.empty((0, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
            )
        )
        self.assertTrue(
            mlp.dual_gemm.can_run(
                torch.empty((16, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
            )
        )
        self.assertFalse(
            mlp.dual_gemm.can_run(
                torch.empty((17, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
            )
        )

    def test_float16_integration(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                mlp, generator = _make_llama_mlp(dtype=dtype)
                x = torch.randn(
                    (16, _HIDDEN_SIZE),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                with torch.inference_mode(), get_parallel().override(tp_group=object()):
                    gate_up, _ = mlp.gate_up_proj(x)
                    expected, _ = mlp.down_proj(mlp.act_fn(gate_up))
                    self.assertEqual(
                        mlp.dual_gemm.mode,
                        DualGemmQuantMode.UNQUANT,
                    )
                    actual = mlp(x)

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_qwen2_integration(self):
        """Qwen2 must route eligible MLP inputs through the shared layer."""
        mlp, generator = _make_qwen2_mlp()
        x = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )
        with (
            torch.inference_mode(),
            get_parallel().override(tp_group=object()),
            patch("sglang.srt.models.qwen2.get_exec", return_value=execution),
        ):
            gate_up, _ = mlp.gate_up_proj(x)
            expected, _ = mlp.down_proj(mlp.act_fn(gate_up))
            self.assertEqual(mlp.dual_gemm.mode, DualGemmQuantMode.UNQUANT)
            with patch.object(
                mlp.gate_up_proj,
                "forward",
                side_effect=AssertionError("Qwen2 used the unfused gate/up path"),
            ):
                actual = mlp(x)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_fp8_handoff_skips_down_quantization(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        mlp, generator = _make_llama_mlp(Fp8Config())
        for projection in (mlp.gate_up_proj, mlp.down_proj):
            projection.quant_method.process_weights_after_loading(projection)

        x = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        with torch.inference_mode(), get_parallel().override(tp_group=object()):
            gate_up, _ = mlp.gate_up_proj(x)
            expected, _ = mlp.down_proj(mlp.act_fn(gate_up))
            self.assertEqual(
                mlp.dual_gemm.mode,
                DualGemmQuantMode.DYNAMIC_PER_TOKEN,
            )
            with patch(
                "sglang.srt.layers.quantization.fp8_utils.sglang_per_token_quant_fp8",
                side_effect=AssertionError("down projection requantized its input"),
            ):
                actual = mlp(x)

        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=5e-1)

    def test_static_fp8_handoff_skips_down_quantization(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        quant_config = CompressedTensorsConfig.from_config(
            {
                "format": "naive-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "tensor",
                            "symmetric": True,
                            "dynamic": False,
                        },
                        "input_activations": {
                            "num_bits": 8,
                            "type": "float",
                            "strategy": "tensor",
                            "symmetric": True,
                            "dynamic": False,
                        },
                    }
                },
            }
        )
        mlp, generator = _make_llama_mlp(quant_config, dtype=torch.float16)
        with torch.no_grad():
            mlp.gate_up_proj.weight_scale.fill_(0.02)
            mlp.gate_up_proj.input_scale.fill_(0.01)
            mlp.down_proj.weight_scale.fill_(0.02)
            mlp.down_proj.input_scale.fill_(0.01)
        for projection in (mlp.gate_up_proj, mlp.down_proj):
            projection.quant_method.process_weights_after_loading(projection)

        x = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        quantized_x, x_scale = scaled_fp8_quant(x, mlp.gate_up_proj.input_scale)
        prequantized_x = quantized_x, x_scale, x.dtype
        with torch.inference_mode(), get_parallel().override(tp_group=object()):
            gate_up, _ = mlp.gate_up_proj(prequantized_x)
            activation = mlp.act_fn(gate_up)
            quantized_activation, activation_scale = scaled_fp8_quant(
                activation, mlp.down_proj.input_scale
            )
            expected, _ = mlp.down_proj(
                (quantized_activation, activation_scale, activation.dtype)
            )
            self.assertEqual(
                mlp.dual_gemm.mode,
                DualGemmQuantMode.STATIC_PER_TENSOR,
            )
            with (
                patch(
                    "sglang.kernels.ops.quantization.fp8_kernel.scaled_fp8_quant",
                    side_effect=AssertionError("MLP requantized its input"),
                ),
                patch(
                    "sglang.srt.layers.quantization.fp8_utils.static_quant_fp8",
                    side_effect=AssertionError("down projection requantized its input"),
                ),
                patch(
                    "sglang.srt.layers.quantization.fp8_utils._apply_fallback_scaled_mm",
                    side_effect=AssertionError(
                        "down projection used the unfused scaling fallback"
                    ),
                ),
            ):
                actual = mlp(prequantized_x)

        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=5e-1)


if __name__ == "__main__":
    unittest.main()
