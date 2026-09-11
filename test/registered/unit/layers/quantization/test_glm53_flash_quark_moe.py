"""Correctness contracts for GLM-5.3-Flash Quark MXFP4 MoE."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.quark.schemes import (
    quark_w4a4_mxfp4_moe as quark_moe,
)
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _CapturingRunner:
    def __init__(self):
        self.quant_info = None

    def run(self, dispatch_output, quant_info):
        self.quant_info = quant_info
        return dispatch_output


class TestGLM53FlashQuarkMoE(CustomTestCase):
    def test_block_fp8_forwards_separated_layout_and_clamp(self):
        method = object.__new__(Fp8MoEMethod)
        method.block_quant = True
        method.is_fp4_expert = False
        method.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
        layer = SimpleNamespace(
            w13_weight=torch.zeros((1, 4, 4), dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros((1, 4, 2), dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            w2_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            hidden_pad=0,
            intermediate_pad=0,
            _aiter_gate_up_interleaved=False,
            dispatcher=SimpleNamespace(expert_mask_gpu=torch.tensor([True, False])),
        )
        gate_mode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = gate_mode
        with (
            patch.dict(
                sys.modules,
                {"aiter.ops.flydsl.moe_common": fake_moe_common},
            ),
            patch.object(fp8_module, "_use_aiter", True),
        ):
            quant_info = method.maybe_get_hip_aiter_quant_info(layer)

        self.assertIsNotNone(quant_info)
        self.assertEqual(quant_info.quant_type, AiterQuantType.PER_128X128)
        self.assertEqual(quant_info.swiglu_limit, 10.0)
        self.assertEqual(quant_info.fused_moe_kwargs, {"gate_mode": "separated"})
        self.assertIs(quant_info.expert_mask, layer.dispatcher.expert_mask_gpu)

    def test_apply_forwards_clamp_separated_layout_and_padding(self):
        scheme = object.__new__(QuarkW4A4MXFp4MoE)
        scheme.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
        scheme.runner = _CapturingRunner()

        layer = SimpleNamespace(
            w13_weight=torch.zeros((1, 4, 2), dtype=torch.uint8),
            w2_weight=torch.zeros((1, 2, 2), dtype=torch.uint8),
            w13_weight_scale=torch.ones((1, 4, 1), dtype=torch.uint8),
            w2_weight_scale=torch.ones((1, 2, 1), dtype=torch.uint8),
            hidden_pad=0,
            intermediate_pad=128,
            dispatcher=SimpleNamespace(expert_mask_gpu=torch.tensor([True, False])),
        )
        layer.w13_weight.is_shuffled = True
        gate_mode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = gate_mode

        with (
            patch.dict(
                sys.modules,
                {"aiter.ops.flydsl.moe_common": fake_moe_common},
            ),
            patch.object(quark_moe, "_is_gfx95", True),
            patch.object(quark_moe, "_is_gfx1250", False),
        ):
            marker = object()
            result = scheme.apply_weights(layer, marker)

        self.assertIs(result, marker)
        quant_info = scheme.runner.quant_info
        self.assertEqual(quant_info.quant_type, AiterQuantType.PER_1X32)
        self.assertEqual(quant_info.swiglu_limit, 10.0)
        self.assertEqual(quant_info.hidden_pad, 0)
        self.assertEqual(quant_info.intermediate_pad, 128)
        self.assertEqual(quant_info.fused_moe_kwargs, {"gate_mode": "separated"})
        self.assertIs(quant_info.expert_mask, layer.dispatcher.expert_mask_gpu)
        self.assertTrue(quant_info.w13_weight.is_shuffled)
        self.assertTrue(quant_info.w2_weight.is_shuffled)

    def test_preshuffle_requires_owned_aiter_runner(self):
        scheme = object.__new__(QuarkW4A4MXFp4MoE)
        scheme._owns_moe_runner = False
        scheme.is_checkpoint_mxfp4_serialized = True
        scheme.dequantization_config = None
        with self.assertRaisesRegex(RuntimeError, "owned AITER runner"):
            scheme.process_weights_after_loading(SimpleNamespace())

        scheme._owns_moe_runner = True
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(
            torch.zeros((1, 4, 2), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight = nn.Parameter(
            torch.zeros((1, 2, 2), dtype=torch.uint8), requires_grad=False
        )
        layer.w13_weight_scale = nn.Parameter(
            torch.ones((1, 4, 1), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_scale = nn.Parameter(
            torch.ones((1, 2, 1), dtype=torch.uint8), requires_grad=False
        )
        layer.dispatcher = SimpleNamespace(set_quant_config=lambda _config: None)

        with (
            patch.object(quark_moe, "_is_gfx1250", False),
            patch.object(quark_moe, "_is_shuffle_moe_mxfp4", True),
            patch.object(
                quark_moe, "e8m0_shuffle", side_effect=lambda x: x, create=True
            ),
            patch.object(
                quark_moe,
                "shuffle_weight",
                side_effect=lambda x, _layout: x.clone(),
                create=True,
            ) as shuffle,
        ):
            scheme.process_weights_after_loading(layer)

        self.assertEqual(shuffle.call_count, 2)
        self.assertTrue(layer.w13_weight.is_shuffled)
        self.assertTrue(layer.w2_weight.is_shuffled)

    def test_flash_packed_shapes_for_tp_partitions(self):
        weight_config = {"qscheme": "per_group"}
        input_config = {"qscheme": "per_group", "is_dynamic": True}
        for tp_size in (1, 2, 4, 8):
            with self.subTest(tp_size=tp_size):
                scheme = QuarkW4A4MXFp4MoE(
                    weight_config,
                    input_config,
                    is_checkpoint_mxfp4_serialized=True,
                )
                layer = nn.Module()
                intermediate = 2048 // tp_size
                with patch.object(quark_moe, "_use_aiter", True):
                    scheme.create_weights(
                        layer,
                        num_experts=288,
                        hidden_size=4096,
                        intermediate_size_per_partition=intermediate,
                        params_dtype=torch.bfloat16,
                        weight_loader=lambda *_args: None,
                    )
                self.assertEqual(
                    tuple(layer.w13_weight.shape),
                    (288, 2 * intermediate, 2048),
                )
                self.assertEqual(
                    tuple(layer.w2_weight.shape),
                    (288, 4096, intermediate // 2),
                )
                self.assertEqual(
                    tuple(layer.w13_weight_scale.shape),
                    (288, 2 * intermediate, 128),
                )
                self.assertEqual(
                    tuple(layer.w2_weight_scale.shape),
                    (288, 4096, intermediate // 32),
                )
                self.assertEqual(layer.hidden_pad, 0)
                self.assertEqual(layer.intermediate_pad, 0)

    def test_unsupported_runner_fails_during_construction(self):
        scheme = object.__new__(QuarkW4A4MXFp4MoE)
        backend = SimpleNamespace(
            value="triton",
            is_auto=lambda: False,
            is_aiter=lambda: False,
        )
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=backend,
            ),
            self.assertRaisesRegex(NotImplementedError, "requires the AITER"),
        ):
            scheme.create_moe_runner(
                SimpleNamespace(), SimpleNamespace(swiglu_limit=10.0)
            )

    def test_explicit_gate_mode_is_not_overwritten_by_aiter(self):
        captured = {}

        def fused_moe(**kwargs):
            captured.update(kwargs)
            return kwargs["hidden_states"]

        fake_fused_moe = types.ModuleType("aiter.fused_moe")
        fake_fused_moe.fused_moe = fused_moe
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )
        config = SimpleNamespace(
            no_combine=False,
            activation="silu",
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        )
        core = AiterRunnerCore(config)
        hidden = torch.zeros((2, 4), dtype=torch.bfloat16)
        runner_input = AiterRunnerInput(
            hidden_states=hidden,
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            quant_type=AiterQuantType.PER_1X32,
        )
        quant_info = AiterMoeQuantInfo(
            w13_weight=torch.zeros((1, 4, 2), dtype=torch.uint8),
            w2_weight=torch.zeros((1, 2, 2), dtype=torch.uint8),
            quant_type=AiterQuantType.PER_1X32,
            swiglu_limit=10.0,
            fused_moe_kwargs={"gate_mode": "separated"},
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "aiter.fused_moe": fake_fused_moe,
                    "aiter.ops.flydsl.moe_common": fake_moe_common,
                },
            ),
            patch.object(aiter_runner, "_aiter_quant_type", return_value="per_1x32"),
            patch.object(aiter_runner, "_aiter_activation", return_value="swiglu"),
        ):
            output = core.run(runner_input, quant_info, {})

        self.assertIs(output.hidden_states, hidden)
        self.assertEqual(captured["gate_mode"], "separated")
        self.assertEqual(captured["swiglu_limit"], 10.0)
        self.assertEqual(captured["hidden_pad"], 0)
        self.assertEqual(captured["intermediate_pad"], 0)


if __name__ == "__main__":
    unittest.main()
