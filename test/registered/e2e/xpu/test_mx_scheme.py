"""Loader tests for the shared MX (MXFP4/MXFP8) scheme path on Intel XPU."""

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Mxfp4MoE,
    CompressedTensorsW8A8MXFp8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import (
    MXFP8_GROUP_SIZE,
    dequant_mxfp8_weight,
)
from sglang.srt.layers.quantization.mxfp4 import _dequant_mxfp4_to_bf16_torch
from sglang.srt.model_loader.weight_utils import resolve_expert_param_name
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=15, suite="extra-test-1-gpu-xpu")


def _auto_round_mx_config(bits: int = 4, extra_config=None) -> dict:
    return {
        "quant_method": "auto-round",
        "packing_format": "auto_round:llm_compressor",
        "data_type": "mx_fp",
        "bits": bits,
        "group_size": MXFP8_GROUP_SIZE,
        "sym": True,
        "extra_config": extra_config,
    }


def _linear():
    with patch.object(RowParallelLinear, "__init__", lambda self, *a, **k: None):
        return RowParallelLinear()


class FakeFusedMoe(FusedMoE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # FusedMoE.__init__ is bypassed here; compressed_tensors gates the
        # MXFP4 MoE short-circuit on this field, so it must be set by hand.
        self.serves_fused_mxfp4 = True


class TestAutoRoundMXConfig(unittest.TestCase):
    def test_llm_compressor_mx_config_delegates_to_compressed_tensors(self):
        config = AutoRoundConfig.from_config(_auto_round_mx_config())
        self.assertIsInstance(config.mx_delegate, CompressedTensorsConfig)
        self.assertEqual(config.packing_format, "auto_round:llm_compressor")
        self.assertEqual(config.group_size, MXFP8_GROUP_SIZE)

    def test_mxfp4_w4a16_export_is_accepted(self):
        """AutoRound MXFP4 exports carry act_bits=16; the XPU expert kernel is W4A16.
        Rejecting act_bits != bits made every such checkpoint unloadable."""
        raw = _auto_round_mx_config()
        raw["act_bits"] = 16
        config = AutoRoundConfig.from_config(raw)
        group = config.mx_delegate.target_scheme_map["Linear"]
        self.assertEqual(group["weights"].num_bits, 4)
        self.assertIsNone(group.get("input_activations"))

    def test_mxfp8_export_keeps_activation_quantization(self):
        raw = _auto_round_mx_config(bits=8)
        config = AutoRoundConfig.from_config(raw)
        group = config.mx_delegate.target_scheme_map["Linear"]
        self.assertEqual(group["weights"].num_bits, 8)
        self.assertEqual(group["input_activations"].num_bits, 8)

    def test_compressed_tensors_metadata_from_other_provider_rejected(self):
        with self.assertRaisesRegex(ValueError, "only from AutoRound"):
            AutoRoundConfig.from_config(
                {
                    "provider": "someone-else",
                    "format": "mxfp4-pack-quantized",
                    "config_groups": {"group_0": {}},
                }
            )

    def test_auto_round_sglang_format_rejected(self):
        raw = _auto_round_mx_config()
        raw["packing_format"] = "auto_round:sglang"
        with self.assertRaisesRegex(ValueError, "Unsupported packing_format"):
            AutoRoundConfig.from_config(raw)

    def test_both_entry_points_agree_on_mxfp4_experts(self):
        config = AutoRoundConfig.from_config(_auto_round_mx_config())
        prefix = "model.layers.0.mlp.experts"
        ar_layer, ct_layer = FakeFusedMoe(), FakeFusedMoe()
        ar_method = config.get_quant_method(ar_layer, prefix)
        ct_method = config.mx_delegate.get_quant_method(ct_layer, prefix)
        self.assertIsInstance(ar_method, CompressedTensorsFusedMoEMethod)
        self.assertIs(type(ar_method), type(ct_method))
        self.assertIsInstance(ar_layer.scheme, CompressedTensorsW4A16Mxfp4MoE)
        self.assertFalse(ar_layer.scheme.has_input_global_scale)

    def test_bf16_override_becomes_a_compressed_tensors_ignore(self):
        config = AutoRoundConfig.from_config(
            _auto_round_mx_config(extra_config={"lm_head": {"bits": 16}})
        )
        self.assertIn("lm_head", config.mx_delegate.ignore)

    def test_mixed_recipe_builds_a_separate_mxfp8_group(self):
        config = AutoRoundConfig.from_config(
            _auto_round_mx_config(
                extra_config={".*self_attn.*": {"bits": 8, "act_bits": 8}}
            )
        )
        scheme_map = config.mx_delegate.target_scheme_map
        self.assertEqual(scheme_map["Linear"]["weights"].num_bits, 4)
        self.assertEqual(scheme_map["re:.*self_attn.*"]["weights"].num_bits, 8)
        self.assertEqual(config.mx_delegate.quant_format, "mixed-precision")
        self.assertEqual(
            config.mx_delegate.get_scheme_dict(
                _linear(), "model.layers.0.self_attn.o_proj"
            )["weights"].num_bits,
            8,
        )


class TestMXFp8DenseRouting(unittest.TestCase):
    """MXFP8 W8A8 is floating-point W8A8 like plain fp8, separated only by the
    GROUP/32 strategy; routing it to the fp8 branch loads the e8m0 scales as
    fp8 per-tensor scales and produces silent garbage."""

    def _args(self, strategy, group_size, dynamic):
        return QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=strategy,
            group_size=group_size,
            dynamic=dynamic,
        )

    def _config(self):
        return CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="mxfp8-quantized",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

    def test_mxfp8_args_match_and_plain_fp8_does_not(self):
        config = self._config()
        mxfp8_w = self._args(QuantizationStrategy.GROUP.value, 32, False)
        mxfp8_a = self._args(QuantizationStrategy.GROUP.value, 32, True)
        self.assertTrue(config._is_mxfp8_w8a8(mxfp8_w, mxfp8_a, "mxfp8-quantized"))
        self.assertFalse(config._is_mxfp8_w8a8(mxfp8_w, mxfp8_a, "float-quantized"))

        fp8_w = self._args(QuantizationStrategy.TENSOR.value, None, False)
        fp8_a = self._args(QuantizationStrategy.TENSOR.value, None, True)
        self.assertFalse(config._is_mxfp8_w8a8(fp8_w, fp8_a, "mxfp8-quantized"))

    def test_group_size_other_than_32_is_not_mx(self):
        config = self._config()
        w = self._args(QuantizationStrategy.GROUP.value, 128, False)
        a = self._args(QuantizationStrategy.GROUP.value, 128, True)
        self.assertFalse(config._is_mxfp8_w8a8(w, a, "mxfp8-quantized"))

    def test_weight_only_mxfp8_is_not_w8a8(self):
        config = self._config()
        w = self._args(QuantizationStrategy.GROUP.value, 32, False)
        self.assertFalse(config._is_mxfp8_w8a8(w, None, "mxfp8-quantized"))


class TestMixedPrecisionMoEDetection(unittest.TestCase):
    """A mixed MXFP4-experts / MXFP8-dense export names the routed experts in
    explicit per-layer targets, so the generic `Linear` entry describes the
    MXFP8 group. Probing only `FusedMoE`/`Linear` missed the experts and sent
    them to the CUDA-only FP4 Marlin MoE scheme instead of the XPU kernel."""

    def _config(self):
        mxfp4 = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP.value,
            group_size=32,
        )
        mxfp8 = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP.value,
            group_size=32,
        )
        return CompressedTensorsConfig(
            target_scheme_map={
                "re:.*mlp\\.experts\\..*": {
                    "weights": mxfp4,
                    "input_activations": None,
                },
                "Linear": {"weights": mxfp8, "input_activations": mxfp8},
            },
            ignore=[],
            quant_format="mixed-precision",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

    def test_routed_experts_detected_despite_mxfp8_linear_entry(self):
        config = self._config()
        self.assertIsInstance(
            config.get_moe_scheme(FakeFusedMoe(), "model.layers.0.mlp.experts"),
            CompressedTensorsW4A16Mxfp4MoE,
        )

    def test_dense_only_mxfp8_model_is_not_an_mxfp4_moe(self):
        config = self._config()
        config.target_scheme_map.pop("re:.*mlp\\.experts\\..*")
        with self.assertRaisesRegex(RuntimeError, "Unsupported FusedMoe scheme"):
            config.get_moe_scheme(FakeFusedMoe(), "model.layers.0.mlp.experts")


@unittest.skipUnless(torch.xpu.is_available(), "requires an Intel XPU")
class TestMXFp4ExpertKernel(unittest.TestCase):
    def test_packed_weight_only_matches_bf16_reference(self):
        generator = torch.Generator().manual_seed(7)
        experts, hidden, intermediate = 3, 256, 256
        layer = torch.nn.Module()
        scheme = CompressedTensorsW4A16Mxfp4MoE(has_input_global_scale=False)
        scheme.create_moe_runner(layer, MoeRunnerConfig())
        states = torch.randn(4, hidden, generator=generator).to(torch.bfloat16)
        weights = torch.tensor([[0.7, 0.3]] * 4, dtype=torch.float32)
        ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, -1]], dtype=torch.int32)

        def make_weights(name, output_size, input_size):
            packed = torch.randint(
                256,
                (experts, output_size, input_size // 2),
                dtype=torch.uint8,
                generator=generator,
            )
            scales = torch.randint(
                121,
                125,
                (experts, output_size, input_size // 32),
                dtype=torch.uint8,
                generator=generator,
            )
            setattr(layer, f"{name}_weight_packed", packed.xpu())
            setattr(layer, f"{name}_weight_scale", scales.xpu())
            return _dequant_mxfp4_to_bf16_torch(packed, scales)

        gate_up = make_weights("w13", 2 * intermediate, hidden)
        down = make_weights("w2", hidden, intermediate)
        topk = StandardTopKOutput(weights.xpu(), ids.xpu(), None)
        output = scheme.apply_weights(
            layer, StandardDispatchOutput(states.xpu(), None, topk)
        ).hidden_states.cpu()

        reference = torch.zeros_like(states, dtype=torch.float32)
        for token, (token_ids, token_weights) in enumerate(zip(ids, weights)):
            for expert, weight in zip(token_ids.tolist(), token_weights.tolist()):
                if expert < 0:
                    continue
                gate, up = (gate_up[expert] @ states[token]).chunk(2)
                activation = (torch.nn.functional.silu(gate.float()) * up).bfloat16()
                reference[token] += weight * (down[expert] @ activation).float()
        relative_l2 = (output.float() - reference).norm() / reference.norm()
        self.assertLess(relative_l2.item(), 0.015)


class TestMXFp8Dequant(unittest.TestCase):
    """DeepSeek MLA absorption has no block-scaled bmm, so kv_b_proj must be
    dequantized up front. The e8m0 scale is a biased exponent that decodes to
    its power of two directly -- reading it as a per-channel fp8 scale raised a
    shape mismatch, and treating it as a reciprocal would silently invert it."""

    def test_group_scales_apply_per_32_columns_as_powers_of_two(self):
        k = 64
        weight = torch.ones(2, k, dtype=torch.float32).to(torch.float8_e4m3fn)
        # Biased exponents 127 and 129 -> multipliers 1.0 and 4.0.
        scale = torch.tensor([[127, 129], [125, 127]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

        out = dequant_mxfp8_weight(weight, scale, out_dtype=torch.float32)

        self.assertEqual(out.shape, (2, k))
        self.assertTrue(torch.all(out[0, :32] == 1.0))
        self.assertTrue(torch.all(out[0, 32:] == 4.0))
        self.assertTrue(torch.all(out[1, :32] == 0.25))
        self.assertTrue(torch.all(out[1, 32:] == 1.0))

    def test_fallback_uses_dequantized_weights_and_bias(self):
        scheme = object.__new__(CompressedTensorsW8A8MXFp8)
        scheme.use_native = False
        scheme.params_dtype = torch.bfloat16
        layer = torch.nn.Module()
        layer.weight = torch.ones(1, 64).to(torch.float8_e4m3fn)
        layer.weight_scale = torch.tensor([[127, 128]], dtype=torch.uint8)
        scheme.process_weights_after_loading(layer)
        output = scheme.apply_weights(
            layer,
            torch.ones((2, 64), dtype=torch.bfloat16),
            torch.ones(1, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(output, torch.full((2, 1), 97, dtype=torch.bfloat16))


class TestDeepSeekExpertParamNames(unittest.TestCase):
    """Mxfp4MoEMethod fuses the routed experts into `w13_weight`, but a
    compressed-tensors checkpoint names the tensor `weight_packed`. The loader
    renamed only on NPU, so elsewhere the packed weights matched no parameter
    and were dropped without an error -- the experts stayed zero and generation
    degenerated, while the scales (already named `weight_scale`) loaded fine."""

    def test_packed_expert_tensor_falls_back_to_the_fused_name(self):
        self.assertEqual(
            resolve_expert_param_name(
                "layers.0.mlp.experts.w13_weight_packed",
                {"layers.0.mlp.experts.w13_weight": None},
            ),
            "layers.0.mlp.experts.w13_weight",
        )

    def test_scheme_registering_the_packed_name_keeps_it(self):
        name = "layers.0.mlp.experts.w13_weight_packed"
        self.assertEqual(
            resolve_expert_param_name(
                name, {name: None, "layers.0.mlp.experts.w13_weight": None}
            ),
            name,
        )

    def test_unclaimed_tensor_resolves_to_none(self):
        self.assertIsNone(
            resolve_expert_param_name("layers.0.mlp.experts.w13_weight_packed", {})
        )


if __name__ == "__main__":
    unittest.main()
