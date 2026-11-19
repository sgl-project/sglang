import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
    XPUCompressedTensorsWNA16MoE,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_WNA16_MOE_SCHEMES = (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)
EXPERTS_LAYER = "model.layers.0.mlp.experts"
PER_LAYER_EXPERT_TARGETS = [
    f"{EXPERTS_LAYER}.0.gate_proj",
    f"{EXPERTS_LAYER}.0.up_proj",
    f"{EXPERTS_LAYER}.0.down_proj",
]


def _make_wna16_moe_config(targets, num_bits, **weight_overrides):
    weights = {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
    }
    weights.update(weight_overrides)
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": weights,
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*mlp.gate$"],
    }


def _pack_int32_along_k(codes):
    """Pack [E, K, N] 4-bit codes as compressed-tensors does: [E, K // 8, N] int32."""
    experts, k, n = codes.shape
    words = torch.zeros(experts, k // 8, n, dtype=torch.int64)
    for i in range(8):
        words |= codes[:, i::8, :].to(torch.int64) << (4 * i)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return torch.nn.Parameter(words.to(torch.int32), requires_grad=False)


def _signed_nibbles(packed):
    """Decode [E, N, K // 2] uint8, low nibble first, as two's-complement INT4."""
    low, high = (packed & 0xF).to(torch.int64), (packed >> 4).to(torch.int64)
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles)


class TestWNA16MoENoLinearGroup(CustomTestCase):
    def setUp(self):
        # The XPU branch keys on a module-level flag, so pin it off unless a test turns it on.
        patcher = mock.patch.object(compressed_tensors, "_is_xpu", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _assert_wna16_moe(self, config_dict, expected_bits):
        quant_config = CompressedTensorsConfig.from_config(config_dict)
        self.assertNotIn("Linear", quant_config.target_scheme_map)

        layer = torch.nn.Module()
        scheme = quant_config.get_moe_scheme(layer, layer_name=EXPERTS_LAYER)

        self.assertIsInstance(scheme, _WNA16_MOE_SCHEMES)
        self.assertEqual(scheme.num_bits, expected_bits)
        self.assertEqual(scheme.group_size, 128)

    def test_regex_expert_targets_int4(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_regex_expert_targets_int8(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        self._assert_wna16_moe(config, expected_bits=8)

    def test_per_layer_fqn_expert_targets_int4(self):
        config = _make_wna16_moe_config(PER_LAYER_EXPERT_TARGETS, num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_blackwell_int4_auto_uses_triton(self):
        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"],
                        num_bits=4,
                        group_size=group_size,
                    )
                )

                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    override_platform(is_sm100=True),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_blackwell_auto_rejects_unvalidated_triton_layouts(self):
        cases = {
            "asymmetric": {"symmetric": False},
            "channel": {"strategy": "channel", "group_size": None},
            "group64": {"group_size": 64},
            "actorder": {"actorder": "group"},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"], num_bits=4, **overrides
                    )
                )
                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    override_platform(is_sm100=True),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
                self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def _xpu_moe_scheme(self, backend, **weight_overrides):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(
                ["re:.*mlp.experts.*"], num_bits=4, **weight_overrides
            )
        )
        with (
            mock.patch.object(compressed_tensors, "_is_xpu", True),
            mock.patch.object(
                compressed_tensors, "get_moe_runner_backend", return_value=backend
            ),
        ):
            return quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

    def test_xpu_int4_auto_uses_the_native_kernel(self):
        for group_size in (32, 64, 128, 256):
            with self.subTest(group_size=group_size):
                scheme = self._xpu_moe_scheme(
                    MoeRunnerBackend.AUTO, group_size=group_size
                )

                self.assertIsInstance(scheme, XPUCompressedTensorsWNA16MoE)

    def test_xpu_explicit_triton_keeps_the_triton_scheme(self):
        scheme = self._xpu_moe_scheme(MoeRunnerBackend.TRITON)

        # The native scheme subclasses the Triton one, so isinstance can't tell them apart.
        self.assertIs(type(scheme), CompressedTensorsWNA16TritonMoE)

    def test_xpu_native_scheme_refuses_triton_quant_info(self):
        scheme = self._xpu_moe_scheme(MoeRunnerBackend.AUTO)

        with self.assertRaisesRegex(NotImplementedError, "--moe-runner-backend triton"):
            scheme.get_triton_quant_info(torch.nn.Module())

    def test_xpu_rejects_unvalidated_layouts(self):
        rejected_by_both = {
            "asymmetric": (4, {"symmetric": False}),
            "channel": (4, {"strategy": "channel", "group_size": None}),
            "group16": (4, {"group_size": 16}),
            "actorder": (4, {"actorder": "group"}),
            "int8": (8, {}),
        }
        # The native kernel takes these group sizes; the Triton scheme isn't validated on them.
        rejected_by_triton = {
            "group64": (4, {"group_size": 64}),
            "group256": (4, {"group_size": 256}),
        }
        for backend, cases in (
            (MoeRunnerBackend.AUTO, rejected_by_both),
            (MoeRunnerBackend.TRITON, {**rejected_by_both, **rejected_by_triton}),
        ):
            for name, (num_bits, overrides) in cases.items():
                with self.subTest(backend=backend, name=name):
                    quant_config = CompressedTensorsConfig.from_config(
                        _make_wna16_moe_config(
                            ["re:.*mlp.experts.*"], num_bits=num_bits, **overrides
                        )
                    )
                    with (
                        mock.patch.object(compressed_tensors, "_is_xpu", True),
                        mock.patch.object(
                            compressed_tensors,
                            "get_moe_runner_backend",
                            return_value=backend,
                        ),
                        self.assertRaisesRegex(ValueError, "WNA16 MoE on XPU"),
                    ):
                        quant_config.get_moe_scheme(
                            torch.nn.Module(), layer_name=EXPERTS_LAYER
                        )

    def test_xpu_native_scheme_stores_signed_nibbles(self):
        """compressed-tensors stores code + 8; the native kernel reads signed INT4."""
        experts, hidden, intermediate, group_size = 2, 64, 32, 32
        scheme = self._xpu_moe_scheme(MoeRunnerBackend.AUTO, group_size=group_size)
        w13_codes = torch.randint(0, 16, (experts, hidden, 2 * intermediate))
        w2_codes = torch.randint(0, 16, (experts, intermediate, hidden))
        layer = torch.nn.Module()
        for name, codes in (("w13", w13_codes), ("w2", w2_codes)):
            k, n = codes.shape[1], codes.shape[2]
            setattr(layer, f"{name}_weight_packed", _pack_int32_along_k(codes))
            setattr(
                layer,
                f"{name}_weight_scale",
                torch.nn.Parameter(
                    torch.ones(experts, k // group_size, n, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )

        scheme.process_weights_after_loading(layer)
        scheme.process_weights_after_loading(layer)  # must not flip back

        for name, codes in (("w13", w13_codes), ("w2", w2_codes)):
            with self.subTest(weight=name):
                packed = getattr(layer, f"{name}_weight_packed")
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertTrue(
                    torch.equal(_signed_nibbles(packed), codes.transpose(1, 2) - 8)
                )

    def test_explicit_triton_rejects_unvalidated_layout(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4, symmetric=False)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            self.assertRaisesRegex(ValueError, "only supports symmetric INT4"),
        ):
            quant_config.get_moe_scheme(torch.nn.Module(), layer_name=EXPERTS_LAYER)

    def test_blackwell_explicit_marlin_is_preserved(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            override_platform(is_sm100=True),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)

    def test_blackwell_int8_auto_keeps_marlin(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            override_platform(is_sm100=True),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
        self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_asymmetric_group_actorder_loads_full_w2_zero_points(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(
                ["re:.*mlp.experts.*"],
                num_bits=4,
                symmetric=False,
                actorder="group",
            )
        )
        with mock.patch.object(
            compressed_tensors,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.MARLIN,
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        layer = torch.nn.Module()
        layer.moe_tp_size = 2
        scheme.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.w2_weight_scale.shape, (2, 2, 256))
        self.assertEqual(layer.w2_weight_zero_point.shape, (2, 2, 32))
        self.assertTrue(layer.w2_weight_scale.load_full_w2)
        self.assertTrue(layer.w2_weight_zero_point.load_full_w2)


if __name__ == "__main__":
    unittest.main()
