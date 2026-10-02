"""Integration tests for the reusable dual GEMM model layer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm.cutedsl_dual_gemm import (
    DualGemmActivationType,
    DualGemmQuantMode,
)
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")

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
        with get_parallel().override(tp_rank=0, tp_size=1):
            mlp = LlamaMLP(
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
                "silu",
                quant_config=quant_config,
                reduce_results=False,
            ).cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(mlp, dtype)


def _make_qwen2_mlp(dtype=torch.bfloat16):
    from sglang.srt.models.qwen2 import Qwen2MLP

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with get_parallel().override(tp_rank=0, tp_size=1):
            mlp = Qwen2MLP(
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
                "silu",
            ).cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(mlp, dtype)


def _make_gemma_mlp(
    model_name,
    dtype=torch.bfloat16,
    activation_sparsity=0.0,
    quant_config=None,
):
    if model_name == "gemma":
        from sglang.srt.models.gemma import GemmaMLP

        constructor = lambda: GemmaMLP(
            _HIDDEN_SIZE, _INTERMEDIATE_SIZE, quant_config=quant_config
        )
    elif model_name == "gemma2":
        from sglang.srt.models.gemma2 import Gemma2MLP

        constructor = lambda: Gemma2MLP(
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
            "gelu_pytorch_tanh",
            "gelu_pytorch_tanh",
            quant_config=quant_config,
        )
    elif model_name in ("gemma3", "gemma4"):
        if model_name == "gemma3":
            from sglang.srt.models.gemma3_causal import Gemma3MLP as GemmaMLPClass
        else:
            from sglang.srt.models.gemma4_causal import Gemma4MLP as GemmaMLPClass

        constructor = lambda: GemmaMLPClass(
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
            "gelu_pytorch_tanh",
            quant_config=quant_config,
        )
    elif model_name == "gemma3n":
        from sglang.srt.models.gemma3n_causal import Gemma3nTextMLP

        constructor = lambda: Gemma3nTextMLP(
            _HIDDEN_SIZE,
            _INTERMEDIATE_SIZE,
            "gelu_pytorch_tanh",
            activation_sparsity=activation_sparsity,
            quant_config=quant_config,
        )
    else:
        raise ValueError(f"Unknown Gemma model: {model_name}")

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with get_parallel().override(tp_rank=0, tp_size=1):
            mlp = constructor().cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(mlp, dtype)


def _make_gemma4_diffusion_self_conditioning(dtype=torch.bfloat16):
    from sglang.srt.models.gemma4_diffusion import DiffusionGemmaSelfConditioning

    config = SimpleNamespace(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        rms_norm_eps=1e-6,
    )
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with get_parallel().override(tp_rank=0, tp_size=1):
            layer = DiffusionGemmaSelfConditioning(config).cuda()
    finally:
        torch.set_default_dtype(original_dtype)
    return _initialize_mlp_weights(layer, dtype)


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
                torch.empty((0, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16),
                mlp.gate_up_proj,
            )
        )
        self.assertTrue(
            mlp.dual_gemm.can_run(
                torch.empty((16, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16),
                mlp.gate_up_proj,
            )
        )
        self.assertFalse(
            mlp.dual_gemm.can_run(
                torch.empty((17, _HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16),
                mlp.gate_up_proj,
            )
        )

    def test_gate_up_lora_wrapper_falls_back(self):
        """Replacing gate/up with a LoRA wrapper must bypass the fused kernel."""

        class GateUpWrapper(torch.nn.Module):
            def __init__(self, base_layer):
                super().__init__()
                self.base_layer = base_layer
                self.called = False

            def forward(self, x):
                self.called = True
                return self.base_layer(x)

        mlp, generator = _make_llama_mlp()
        x = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        with (
            torch.inference_mode(),
            get_parallel().override(tp_group=object()),
            patch(
                "sglang.kernels.ops.gemm.dual_gemm_swiglu",
                side_effect=AssertionError("LoRA wrapper used the fused gate/up path"),
            ),
        ):
            gate_up, _ = mlp.gate_up_proj(x)
            expected, _ = mlp.down_proj(mlp.act_fn(gate_up))

            wrapper = GateUpWrapper(mlp.gate_up_proj)
            mlp.gate_up_proj = wrapper
            self.assertFalse(mlp.dual_gemm.can_run(x, mlp.gate_up_proj))
            actual = mlp(x)

        self.assertTrue(wrapper.called)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

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

    def test_gemma_integrations(self):
        """Every dense Gemma MLP must select its model-accurate GELU variant."""
        for model_name in ("gemma", "gemma2", "gemma3", "gemma3n", "gemma4"):
            with self.subTest(model_name=model_name):
                mlp, generator = _make_gemma_mlp(model_name)
                x = torch.randn(
                    (16, _HIDDEN_SIZE),
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                expected_activation = (
                    DualGemmActivationType.GELU
                    if model_name == "gemma"
                    else DualGemmActivationType.GELU_TANH
                )
                with torch.inference_mode(), get_parallel().override(tp_group=object()):
                    gate_up, _ = mlp.gate_up_proj(x)
                    expected, _ = mlp.down_proj(mlp.act_fn(gate_up))
                    self.assertEqual(mlp.dual_gemm.activation_type, expected_activation)
                    with patch.object(
                        mlp.gate_up_proj,
                        "forward",
                        side_effect=AssertionError(
                            f"{model_name} used the unfused gate/up path"
                        ),
                    ):
                        actual = mlp(x)

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_gemma3n_activation_sparsity_falls_back(self):
        """Gemma3n sparsification must run before GELU and cannot be bypassed."""
        mlp, generator = _make_gemma_mlp("gemma3n", activation_sparsity=0.5)
        x = torch.randn(
            (4, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        with torch.inference_mode(), get_parallel().override(tp_group=object()):
            gate_up, _ = mlp.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            expected, _ = mlp.down_proj(
                mlp.act_fn(torch.cat((mlp._gaussian_topk(gate), up), dim=-1))
            )
            actual = mlp(x)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_gemma4_diffusion_self_conditioning(self):
        """Gemma4 diffusion's standalone gated MLP must use tanh GELU."""
        layer, generator = _make_gemma4_diffusion_self_conditioning()
        inputs_embeds = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        signal = torch.randn(
            (16, _HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        with torch.inference_mode(), get_parallel().override(tp_group=object()):
            normed_signal = layer.pre_norm(signal)
            gate_up, _ = layer.gate_up_proj(normed_signal)
            projected, _ = layer.down_proj(layer.act_fn(gate_up))
            expected = layer.post_norm(inputs_embeds + projected)
            with patch.object(
                layer.gate_up_proj,
                "forward",
                side_effect=AssertionError("Gemma4 diffusion used the unfused path"),
            ):
                actual = layer(inputs_embeds, signal)

        self.assertEqual(
            layer.dual_gemm.activation_type, DualGemmActivationType.GELU_TANH
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)

    def test_gemma_fp8_integration(self):
        """Gemma's FP8 handoff must retain tanh-GELU through down projection."""
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        mlp, generator = _make_gemma_mlp("gemma2", quant_config=Fp8Config())
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
            actual = mlp(x)

        self.assertEqual(
            mlp.dual_gemm.activation_type, DualGemmActivationType.GELU_TANH
        )
        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=5e-1)

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
