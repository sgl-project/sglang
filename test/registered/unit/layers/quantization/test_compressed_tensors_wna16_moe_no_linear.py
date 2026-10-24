import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16_moe as wna16_moe,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_zero_points
from sglang.srt.layers.quantization.utils import pack_cols
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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


class TestWNA16MoENoLinearGroup(CustomTestCase):
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


class TestWNA16MoEZeroPoints(CustomTestCase):
    """CT checkpoints must keep zero-point values and TP slices through loading."""

    @staticmethod
    def _make_layer(num_bits, group_size=32, tp_size=1, tp_rank=0):
        # Only exercise CPU weight loading; no distributed runner is constructed.
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.moe_tp_size = tp_size
        layer.moe_tp_rank = tp_rank
        layer.scheme = None
        layer.use_flashinfer_trtllm_moe = False
        layer.use_triton_kernels = False
        layer.use_presharded_weights = False
        layer._has_fused_shared = False
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        weights = QuantizationArgs(
            num_bits=num_bits,
            type="int",
            symmetric=False,
            strategy="channel" if group_size == -1 else "group",
            group_size=None if group_size == -1 else group_size,
        )
        layer.quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(
                ["Linear"], num_bits, **weights.model_dump(exclude={"num_bits"})
            )
        )
        method = CompressedTensorsWNA16MoE(layer.quant_config, weights)
        layer.quant_method = method
        method.create_weights(layer, 2, 256, 128 // tp_size, torch.bfloat16)
        return layer, method

    @staticmethod
    def _checkpoint_zero_points(num_bits, output_size, num_groups, seed):
        generator = torch.Generator().manual_seed(seed)
        offset = 1 << (num_bits - 1)
        values = torch.randint(
            -offset, offset, (output_size, num_groups), generator=generator
        ).to(torch.int8)
        packed = pack_to_int32(values, num_bits, packed_dim=0)
        # Canonical unsigned [groups, output] values, before Marlin permutation.
        return packed, values.t().to(torch.int32) + offset

    def test_checkpoint_zero_points_tp_loading(self):
        """Packed output rows must transpose before gate/up/down TP slicing."""
        for num_bits in (4, 8):
            for tp_size in (1, 2):
                for tp_rank in range(tp_size):
                    with self.subTest(bits=num_bits, tp=tp_size, rank=tp_rank):
                        layer, _ = self._make_layer(
                            num_bits, tp_size=tp_size, tp_rank=tp_rank
                        )
                        for expert in range(2):
                            for shard in ("w1", "w3", "w2"):
                                down = shard == "w2"
                                output_size, groups = (256, 4) if down else (128, 8)
                                packed, values = self._checkpoint_zero_points(
                                    num_bits,
                                    output_size,
                                    groups,
                                    seed=expert * 3
                                    + {"w1": 0, "w3": 1, "w2": 2}[shard],
                                )
                                name = (
                                    "w2_weight_zero_point"
                                    if down
                                    else "w13_weight_zero_point"
                                )
                                param = getattr(layer, name)
                                layer._weight_loader_impl(
                                    param, packed, name, shard, expert
                                )
                                if down:
                                    values = values.chunk(tp_size, dim=0)[tp_rank]
                                    actual = param[expert]
                                else:
                                    values = values.chunk(tp_size, dim=1)[tp_rank]
                                    actual = param[expert].chunk(2, dim=1)[
                                        shard == "w3"
                                    ]
                                expected = pack_cols(
                                    values.contiguous(), num_bits, *values.shape
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )

    def test_fused_checkpoint_zero_points_loading(self):
        """Fused checkpoints transpose output/groups without transposing experts."""
        layer, _ = self._make_layer(4)
        for name, shard, groups in (
            ("w13_weight_zero_point", "w13", 8),
            ("w2_weight_zero_point", "w2", 4),
        ):
            checkpoints, expected = [], []
            for expert in range(2):
                packed, values = self._checkpoint_zero_points(
                    4, 256, groups, seed=expert
                )
                checkpoints.append(packed)
                expected.append(pack_cols(values.contiguous(), 4, *values.shape))
            param = getattr(layer, name)
            layer.weight_loader_fused(param, torch.stack(checkpoints), name, shard)
            torch.testing.assert_close(param, torch.stack(expected), rtol=0, atol=0)

    def test_zero_points_repack_matches_dense_ct(self):
        """MoE CT zero-points must not receive AWQ's inverse interleave."""
        for num_bits in (4, 8):
            for group_size in (32, 128, -1):
                with self.subTest(bits=num_bits, group_size=group_size):
                    layer, method = self._make_layer(num_bits, group_size=group_size)
                    expected = {}
                    for name in ("w13_weight_zero_point", "w2_weight_zero_point"):
                        param = getattr(layer, name)
                        _, groups, packed_output = param.shape
                        outputs = packed_output * (32 // num_bits)
                        reference = []
                        for expert in range(2):
                            packed, values = self._checkpoint_zero_points(
                                num_bits, outputs, groups, seed=expert
                            )
                            param.data[expert].copy_(packed.t())
                            # The dense CT path applies Marlin's layout directly
                            # to canonical values, with no AWQ inverse permutation.
                            reference.append(
                                marlin_zero_points(values, groups, outputs, num_bits)
                            )
                        expected[name] = torch.stack(reference)
                    # Mock only GPU-only weight/scale kernels and workspace allocation.
                    # The real scheme still converts and replaces both zero-point tensors.
                    with (
                        mock.patch.object(
                            wna16_moe,
                            "gptq_marlin_moe_repack",
                            side_effect=lambda x, *args: x.clone(),
                        ),
                        mock.patch.object(
                            wna16_moe,
                            "marlin_moe_permute_scales",
                            side_effect=lambda x, *args: x.clone(),
                        ),
                        mock.patch.object(
                            wna16_moe,
                            "marlin_make_workspace",
                            return_value=torch.empty(0, dtype=torch.int32),
                        ),
                    ):
                        method.process_weights_after_loading(layer)
                    for name, reference in expected.items():
                        torch.testing.assert_close(
                            getattr(layer, name), reference, rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
