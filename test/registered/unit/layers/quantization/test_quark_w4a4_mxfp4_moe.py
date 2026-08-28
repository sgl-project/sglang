"""Unit tests for the Quark W4A4 MXFP4 MoE clamped-SwiGLU plumbing.

CPU-only, no model loading and no aiter/ROCm dependency: they exercise the
pure-Python control flow in ``QuarkW4A4MXFp4MoE.create_moe_runner`` and
``apply_weights`` that decides whether the AITER MoE runner is driven on the
clamped-SwiGLU path (``activation="swiglu"`` + ``swiglu_limit``) or the plain
SiLU path. See the regression these guard against: clamped-SwiGLU MXFP4
checkpoints (e.g. MiniMax-M3) declare ``activation="silu"`` + a
``gemm1_clamp_limit``, and without this plumbing the experts silently run plain
SiLU and emit garbage.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.utils as moe_utils
import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe as quark_moe
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_scheme() -> QuarkW4A4MXFp4MoE:
    # is_checkpoint_mxfp4_serialized=True skips the gfx95 hardware check.
    return QuarkW4A4MXFp4MoE(
        weight_config={"qscheme": "per_group"},
        input_config={"qscheme": "per_group", "is_dynamic": True},
    )


def _force_aiter_runner(monkeypatch):
    """Make ``create_moe_runner`` take the AITER branch and capture its config."""
    captured = {}

    def fake_moe_runner(backend, config):
        captured["backend"] = backend
        captured["config"] = config
        return object()

    monkeypatch.setattr(quark_moe, "MoeRunner", fake_moe_runner)
    monkeypatch.setattr(
        moe_utils, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    return captured


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


def test_create_moe_runner_translates_activation_to_swiglu_when_clamped(monkeypatch):
    captured = _force_aiter_runner(monkeypatch)
    scheme = _make_scheme()

    cfg = MoeRunnerConfig(activation="silu", gemm1_clamp_limit=7.0)
    scheme.create_moe_runner(layer=None, moe_runner_config=cfg)

    assert captured["config"].activation == "swiglu"
    # The original config is left untouched (dataclasses.replace returns a copy).
    assert cfg.activation == "silu"


def test_create_moe_runner_keeps_silu_without_clamp(monkeypatch):
    captured = _force_aiter_runner(monkeypatch)
    scheme = _make_scheme()

    cfg = MoeRunnerConfig(activation="silu")  # no clamp
    scheme.create_moe_runner(layer=None, moe_runner_config=cfg)

    assert captured["config"].activation == "silu"
    assert captured["config"] is cfg


def test_apply_weights_forwards_swiglu_limit_when_clamped(monkeypatch):
    monkeypatch.delattr(torch, "float4_e2m1fn_x2", raising=False)
    scheme = _make_scheme()
    scheme.moe_runner_config = MoeRunnerConfig(
        activation="swiglu", gemm1_clamp_limit=7.0
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
