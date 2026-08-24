"""Unit tests for ROCm MegaMoE (AITER) dispatch wiring."""

from __future__ import annotations

import pytest


def test_mega_moe_build_routes_to_aiter_on_rocm(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe._is_hip", True)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe._use_aiter_mega_moe",
        lambda: True,
    )

    called = {}

    def fake_build(experts):
        called["experts"] = experts
        experts._mega_moe_weights_built = True

    class Experts:
        _mega_moe_weights_built = False

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_aiter.build_mega_moe_aiter_weights",
        fake_build,
    )

    from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights

    experts = Experts()
    build_mega_moe_experts_weights(experts)
    assert called["experts"] is experts


def test_validate_mtpr_rejects_non_power_of_two():
    from sglang.srt.layers.moe.mega_moe_aiter import _validate_mtpr

    with pytest.raises(ValueError, match="power of two"):
        _validate_mtpr(1000)


def test_validate_mtpr_accepts_power_of_two():
    from sglang.srt.layers.moe.mega_moe_aiter import _validate_mtpr

    _validate_mtpr(1024)


def test_is_mega_moe_aiter_enabled_requires_megamoe_backend(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe_aiter._is_hip", True)
    monkeypatch.setattr("sglang.srt.layers.moe.mega_moe_aiter._use_aiter", True)

    class FakeBackend:
        @staticmethod
        def is_megamoe():
            return False

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe_aiter.get_moe_a2a_backend",
        lambda: FakeBackend(),
    )

    from sglang.srt.layers.moe.mega_moe_aiter import is_mega_moe_aiter_enabled

    assert is_mega_moe_aiter_enabled() is False
