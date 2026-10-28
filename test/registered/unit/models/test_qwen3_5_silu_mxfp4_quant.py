"""CPU tests for the Qwen3.5 dense-MLP SiLU+mul+MXFP4 quant producer (ROCm/AITER):
when qwen3_5 enables it, what it asks AITER for, and which batches take it.
AITER is faked; the numerics are covered by
test/registered/kernels/ops/activation/test_silu_and_mul_mxfp4_quant.py."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
    _AsmShuffledMxfp4Input,
)
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FP4X2, _E8M0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu


def _fake_aiter(act_mul_and_mxfp4_quant):
    """sys.modules entries for an AITER with this act_mul_and_mxfp4_quant
    (None: the activation module does not import)."""
    names = ("aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.triton.fusions")
    modules = {name: types.ModuleType(name) for name in names}
    modules["aiter"].dtypes = SimpleNamespace(fp4x2=_FP4X2, fp8_e8m0=_E8M0)
    activation = None
    if act_mul_and_mxfp4_quant is not None:
        activation = types.ModuleType("aiter.ops.triton.fusions.activation")
        activation.act_mul_and_mxfp4_quant = act_mul_and_mxfp4_quant
    modules["aiter.ops.triton.fusions.activation"] = activation
    return mock.patch.dict(sys.modules, modules)


def _act_mul_and_mxfp4_quant(
    x,
    activation,
    scaling_mode="even",
    shuffle=False,
    scale_shuffle_padding=False,
    round_to_input_dtype=False,
):
    m, n = x.shape[0], x.shape[1] // 2
    return (
        torch.zeros(m, n // 2, dtype=torch.uint8),
        torch.zeros((m + 255) // 256 * 256, n // 32, dtype=torch.uint8),
    )


def _act_mul_and_mxfp4_quant_without_rounding(
    x, activation, scaling_mode="even", shuffle=False, scale_shuffle_padding=False
):
    return _act_mul_and_mxfp4_quant(x, activation)


def _mxfp4_linear(use_asm=True):
    linear = nn.Module()
    linear.scheme = object.__new__(QuarkW4A4MXFP4)
    linear.use_aiter_asm_fp4_gemm = use_asm
    return linear


class TestEnable(CustomTestCase):
    def _enable(self, act_mul=_act_mul_and_mxfp4_quant, is_hip=True, down=None):
        mlp = SimpleNamespace(
            gate_up_proj=_mxfp4_linear(),
            down_proj=down if down is not None else _mxfp4_linear(),
            _enable_silu_mxfp4_quant_fusion=False,
        )
        with mock.patch.object(qwen3_5, "_is_hip", is_hip), _fake_aiter(act_mul):
            qwen3_5._maybe_enable_silu_mxfp4_quant_fusion(mlp)
        return mlp

    def test_enabled_for_mxfp4_mlp(self):
        mlp = self._enable()
        self.assertTrue(mlp._enable_silu_mxfp4_quant_fusion)
        self.assertEqual((mlp._mxfp4_x_dtype, mlp._mxfp4_scale_dtype), (_FP4X2, _E8M0))

    def test_disabled_otherwise(self):
        cases = {
            "not ROCm": dict(is_hip=False),
            "non-MXFP4 down_proj": dict(down=nn.Linear(4, 4)),
            "no AITER activation module": dict(act_mul=None),
            # Its output would be valid MXFP4 but not the unfused path's.
            "AITER without round_to_input_dtype": dict(
                act_mul=_act_mul_and_mxfp4_quant_without_rounding
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                mlp = self._enable(**kwargs)
                self.assertFalse(mlp._enable_silu_mxfp4_quant_fusion)


class TestProducer(CustomTestCase):
    def test_requests_the_unfused_numerics_and_layout(self):
        calls = []

        def act_mul_and_mxfp4_quant(x, activation, **kwargs):
            calls.append((x, activation, kwargs))
            return _act_mul_and_mxfp4_quant(x, activation)

        mlp = SimpleNamespace(_mxfp4_x_dtype=_FP4X2, _mxfp4_scale_dtype=_E8M0)
        gate_up = torch.zeros(3, 2 * 1024, dtype=torch.bfloat16)
        with _fake_aiter(act_mul_and_mxfp4_quant):
            out = Qwen2MoeMLP._silu_mxfp4_quant_fused(mlp, gate_up)
        ((x, activation, kwargs),) = calls
        self.assertIs(x, gate_up)
        self.assertEqual(activation, "silu")
        self.assertEqual(
            kwargs,
            {"scaling_mode": "roundup", "shuffle": True, "round_to_input_dtype": True},
        )
        self.assertIsInstance(out, _AsmShuffledMxfp4Input)
        x_q, x_scales = out
        self.assertEqual((x_q.dtype, tuple(x_q.shape)), (_FP4X2, (3, 512)))
        self.assertEqual((x_scales.dtype, tuple(x_scales.shape)), (_E8M0, (256, 32)))


class TestForward(CustomTestCase):
    def _forward(self, gate_up, enabled=True, down_asm=True):
        """Qwen2MoeMLP.forward on fake projections; returns what down_proj got."""
        down_proj = mock.Mock(return_value=("out", None))
        down_proj.use_aiter_asm_fp4_gemm = down_asm
        mlp = SimpleNamespace(
            gate_up_proj=lambda x: (x, None),
            down_proj=down_proj,
            act_fn=lambda x: "silu_and_mul",
            _enable_silu_mxfp4_quant_fusion=enabled,
            _enable_silu_fp4_quant_fusion=False,
            _silu_mxfp4_quant_fused=lambda x: "mxfp4_pair",
        )
        self.assertEqual(Qwen2MoeMLP.forward(mlp, gate_up), "out")
        ((down_input,), _) = down_proj.call_args
        return down_input

    def test_fused_path_feeds_down_proj(self):
        self.assertEqual(self._forward(torch.zeros(3, 64)), "mxfp4_pair")

    def test_unfused_path_otherwise(self):
        cases = {
            "fusion off": (torch.zeros(3, 64), dict(enabled=False)),
            "down_proj not on the ASM GEMM": (torch.zeros(3, 64), dict(down_asm=False)),
            "empty batch": (torch.zeros(0, 64), {}),
            "3-D input": (torch.zeros(1, 3, 64), {}),
        }
        for name, (gate_up, kwargs) in cases.items():
            with self.subTest(name):
                self.assertEqual(self._forward(gate_up, **kwargs), "silu_and_mul")


if __name__ == "__main__":
    unittest.main()
