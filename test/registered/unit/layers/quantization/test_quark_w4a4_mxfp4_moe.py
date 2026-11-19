"""CPU regression tests for Quark MXFP4 MoE loading and runner plumbing."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_scheme() -> QuarkW4A4MXFp4MoE:
    # is_checkpoint_mxfp4_serialized=True skips the gfx95 hardware check.
    return QuarkW4A4MXFp4MoE(
        weight_config={"qscheme": "per_group"},
        input_config={"qscheme": "per_group", "is_dynamic": True},
    )


def _fake_layer():
    # dtype uint8 so we never hit the float4_e2m1fn_x2 view branch (also guarded
    # by delattr in the apply_weights tests below).
    return SimpleNamespace(
        w13_weight=torch.zeros(1, dtype=torch.uint8),
        w2_weight=torch.zeros(1, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(1, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(1, dtype=torch.uint8),
        dispatcher=SimpleNamespace(expert_mask_gpu=torch.zeros(1)),
    )


def _capturing_runner(store):
    class _Runner:
        def run(self, dispatch_output, quant_info):
            store["quant_info"] = quant_info
            return "ran"

    return _Runner()


def test_apply_weights_forwards_swiglu_limit_when_clamped(monkeypatch):
    monkeypatch.delattr(torch, "float4_e2m1fn_x2", raising=False)
    scheme = _make_scheme()
    monkeypatch.setattr(
        "sglang.srt.layers.moe.moe_runner.aiter.is_gfx95_supported",
        lambda: True,
    )
    scheme.moe_runner_config = MoeRunnerConfig(
        activation="silu",
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    store = {}
    scheme.runner = _capturing_runner(store)

    result = scheme.apply_weights(_fake_layer(), dispatch_output=object())

    assert result == "ran"
    assert store["quant_info"].swiglu_limit == 7.0


def test_apply_weights_defaults_swiglu_limit_zero_without_clamp(monkeypatch):
    monkeypatch.delattr(torch, "float4_e2m1fn_x2", raising=False)
    scheme = _make_scheme()
    scheme.moe_runner_config = MoeRunnerConfig(activation="silu")  # no clamp
    store = {}
    scheme.runner = _capturing_runner(store)

    scheme.apply_weights(_fake_layer(), dispatch_output=object())

    assert store["quant_info"].swiglu_limit == 0.0


class TestOnlineWeightLoader(CustomTestCase):
    def test_nonlocal_w13_after_quantization(self):
        self._check_no_copy("w13")

    def test_nonlocal_w2_after_quantization(self):
        self._check_no_copy("w2")

    def test_invalid_weight_name(self):
        self._check_no_copy("invalid")

    def _check_no_copy(self, name):
        packed = torch.nn.Parameter(
            torch.ones(1, 2, 2, dtype=torch.uint8), requires_grad=False
        )
        layer = SimpleNamespace(
            w13_weight=packed,
            w2_weight=packed,
            _w13_loaded_numel=8,
            _w2_loaded_numel=8,
            _load_device=torch.device("cpu"),
        )
        original_loader = MagicMock()
        loader = _make_scheme().get_online_weight_loader(layer, original_loader)
        args = (
            packed,
            torch.ones(2, 4),
            f"{name}_weight",
            "w2" if name == "w2" else "w1",
            1,
        )
        with patch(
            "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe.dynamic_mxfp4_quant"
        ) as quantize:
            if name == "invalid":
                with self.assertRaisesRegex(ValueError, "Expected w13 or w2"):
                    loader(*args)
            else:
                loader(*args)
            quantize.assert_not_called()
        original_loader.assert_called_once()
        self.assertIs(layer.w13_weight, packed)
        self.assertIs(layer.w2_weight, packed)
        self.assertEqual(layer._w13_loaded_numel, 8)
        self.assertEqual(layer._w2_loaded_numel, 8)
        torch.testing.assert_close(packed, torch.ones_like(packed))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
