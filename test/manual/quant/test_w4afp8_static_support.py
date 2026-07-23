# SPDX-License-Identifier: Apache-2.0
"""Test W4AFP8 static quantization: config detection, MoE routing, and scale lifecycle.

Covers the full static W4AFP8 path end-to-end:
  test/manual/quant/test_w4afp8_static_support.py

Changes under test:
  python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
    → _is_wint4afp8(): removed input_quant.dynamic requirement
  python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py
    → CompressedTensorsW4AFP8MoE: static input scales support

Pure config-parsing and scheme lifecycle — no GPU required.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4AFP8MoE,
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)

_WNA16_MOE_SCHEMES = (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)
_NUM_EXPERTS, _HIDDEN, _INTER = 4, 512, 256


# ============================================================================
# config builders
# ============================================================================


def _make_w4afp8_config(dynamic: bool) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token" if dynamic else "tensor",
                    "dynamic": dynamic,
                },
            }
        },
    }


def _make_w4a16_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                },
                "input_activations": None,
            }
        },
    }


# ============================================================================
# 1. _is_wint4afp8 detection
# ============================================================================


class TestDetection:
    """Verify static configs are recognized by _is_wint4afp8()."""

    def test_static(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4afp8_config(dynamic=False))
        d = next(iter(cfg.target_scheme_map.values()))
        assert cfg._is_wint4afp8(d["weights"], d["input_activations"])

    def test_dynamic(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4afp8_config(dynamic=True))
        d = next(iter(cfg.target_scheme_map.values()))
        assert cfg._is_wint4afp8(d["weights"], d["input_activations"])

    def test_w4a16_not_misdetected(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4a16_config())
        d = next(iter(cfg.target_scheme_map.values()))
        assert not cfg._is_wint4afp8(d["weights"], d["input_activations"])


# ============================================================================
# 2. MoE scheme routing
# ============================================================================


class _FakeLinearModule(nn.Module):
    pass


class TestMoERouting:
    """Verify static/dynamic W4AFP8 config routes to CompressedTensorsW4AFP8MoE."""

    MOE_LAYER = "model.layers.0.mlp.experts"
    layer: _FakeLinearModule

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.layer = _FakeLinearModule()

    def test_static_routes_to_moe(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4afp8_config(dynamic=False))
        s = cfg.get_moe_scheme(self.layer, layer_name=self.MOE_LAYER)
        assert isinstance(s, CompressedTensorsW4AFP8MoE)

    def test_dynamic_routes_to_moe(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4afp8_config(dynamic=True))
        s = cfg.get_moe_scheme(self.layer, layer_name=self.MOE_LAYER)
        assert isinstance(s, CompressedTensorsW4AFP8MoE)

    def test_w4a16_not_hijacked(self):
        cfg = CompressedTensorsConfig.from_config(_make_w4a16_config())
        s = cfg.get_moe_scheme(self.layer, layer_name=self.MOE_LAYER)
        assert isinstance(s, _WNA16_MOE_SCHEMES)


# ============================================================================
# 3. static scale lifecycle (create_weights → process_weights_after_loading)
# ============================================================================


def _make_scheme(dynamic: bool) -> CompressedTensorsW4AFP8MoE:
    cfg = CompressedTensorsConfig.from_config(_make_w4afp8_config(dynamic=dynamic))
    d = next(iter(cfg.target_scheme_map.values()))
    return CompressedTensorsW4AFP8MoE(cfg, d["weights"], d["input_activations"])


def _create_dummy_layer_and_weights(scheme: CompressedTensorsW4AFP8MoE):
    """Helper: create a layer with mock weights and run create_weights."""
    layer = _FakeLinearModule()
    scheme.create_weights(
        layer,
        _NUM_EXPERTS,
        _HIDDEN,
        _INTER,
        params_dtype=torch.float32,
        input_dim=0,
        output_dim=0,
        weight_loader=MagicMock(),
    )
    # Fill weights with random data so process_weights_after_loading can run
    for pname in ("w13_weight_packed", "w2_weight_packed"):
        p = getattr(layer, pname)
        p.data = torch.randint(-(2**31), 2**31 - 1, p.shape, dtype=torch.int32)
        p.data = p.data.to(torch.int32)  # ensure int32 for bit ops
    for pname in ("w13_weight_scale", "w2_weight_scale"):
        p = getattr(layer, pname)
        p.data = torch.randn(p.shape, dtype=torch.float32)
    return layer


def _layer_has_state(layer: nn.Module, static: bool):
    """Shared assertions for scale parameter state."""
    has_a13 = hasattr(layer, "a13_scale") and layer.a13_scale is not None
    has_a2 = hasattr(layer, "a2_scale") and layer.a2_scale is not None
    if static:
        assert has_a13 and has_a2, "static mode: scales should be created"
    else:
        assert not has_a13 and not has_a2, "dynamic mode: scales should be None"


class TestStaticScaleLifecycle:
    """Verify static scales flow through create_weights → process_weights_after_loading."""

    # ----- static path -----

    def test_static_flag(self):
        s = _make_scheme(dynamic=False)
        assert s.static_input_scales is True

    def test_static_create_weights(self):
        s = _make_scheme(dynamic=False)
        layer = _FakeLinearModule()
        s.create_weights(
            layer,
            _NUM_EXPERTS,
            _HIDDEN,
            _INTER,
            params_dtype=torch.float32,
            input_dim=0,
            output_dim=0,
            weight_loader=MagicMock(),
        )
        _layer_has_state(layer, static=True)
        assert layer.a13_scale.shape == (_NUM_EXPERTS,)
        assert layer.a2_scale.shape == (_NUM_EXPERTS,)

    def test_static_process_weights(self):
        s = _make_scheme(dynamic=False)
        layer = _create_dummy_layer_and_weights(s)
        # process_weights_after_loading should collapse per-expert scales to scalar
        s.process_weights_after_loading(layer)
        assert (
            layer.a13_scale.numel() == 1
        ), f"expected scalar, got {layer.a13_scale.shape}"
        assert (
            layer.a2_scale.numel() == 1
        ), f"expected scalar, got {layer.a2_scale.shape}"
        assert layer.is_w4afp8_converted is True

    # ----- dynamic path (regression) -----

    def test_dynamic_flag(self):
        s = _make_scheme(dynamic=True)
        assert s.static_input_scales is False

    def test_dynamic_create_weights(self):
        s = _make_scheme(dynamic=True)
        layer = _FakeLinearModule()
        s.create_weights(
            layer,
            _NUM_EXPERTS,
            _HIDDEN,
            _INTER,
            params_dtype=torch.float32,
            input_dim=0,
            output_dim=0,
            weight_loader=MagicMock(),
        )
        _layer_has_state(layer, static=False)

    def test_dynamic_process_weights(self):
        s = _make_scheme(dynamic=True)
        layer = _create_dummy_layer_and_weights(s)
        s.process_weights_after_loading(layer)
        assert layer.a13_scale is None
        assert layer.a2_scale is None
        assert layer.is_w4afp8_converted is True
