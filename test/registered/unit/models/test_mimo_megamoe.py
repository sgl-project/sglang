import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import mimo_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_target_runtime(monkeypatch, *, speculative=False):
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "1")
    monkeypatch.setenv("AITER_MEGA_SMALL_M", "1")
    monkeypatch.setattr(
        mimo_v2,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_megamoe=lambda: True),
    )
    monkeypatch.setattr(mimo_v2, "get_platform", lambda: SimpleNamespace(is_hip=True))
    monkeypatch.setattr(mimo_v2, "is_gfx95_supported", lambda: True)
    monkeypatch.setattr(
        mimo_v2,
        "get_flags",
        lambda: SimpleNamespace(moe=SimpleNamespace(in_speculative_scope=speculative)),
    )


def test_target_mimo_megamoe_exact_shape_is_eligible(monkeypatch):
    _patch_target_runtime(monkeypatch)

    assert mimo_v2._is_target_mimo_megamoe_eligible(_config(), 8, 8)


@pytest.mark.parametrize(
    ("config", "tp_size", "ep_size"),
    [
        (_config(hidden_size=7168), 8, 8),
        (_config(moe_intermediate_size=3072), 8, 8),
        (_config(n_routed_experts=256), 8, 8),
        (_config(num_experts_per_tok=6), 8, 8),
        (_config(), 4, 8),
        (_config(), 8, 4),
    ],
)
def test_target_mimo_megamoe_rejects_other_shapes_and_topologies(
    monkeypatch, config, tp_size, ep_size
):
    _patch_target_runtime(monkeypatch)

    assert not mimo_v2._is_target_mimo_megamoe_eligible(config, tp_size, ep_size)


def test_target_mimo_megamoe_rejects_dflash_draft(monkeypatch):
    _patch_target_runtime(monkeypatch, speculative=True)

    assert not mimo_v2._is_target_mimo_megamoe_eligible(_config(), 8, 8)


def test_mimo_megamoe_caller_preserves_normal_fallback(monkeypatch):
    moe = mimo_v2.MiMoV2MoE.__new__(mimo_v2.MiMoV2MoE)
    nn.Module.__init__(moe)
    moe._use_target_mega_moe = True
    moe._enable_a2a_moe = False
    expected = torch.ones(2, 3)
    monkeypatch.setattr(moe, "forward_normal", lambda hidden_states: expected)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
        lambda layer, hidden_states: False,
    )

    assert moe.forward(torch.zeros(2, 3)) is expected


def test_mimo_megamoe_caller_keeps_prefill_on_fallback(monkeypatch):
    moe = mimo_v2.MiMoV2MoE.__new__(mimo_v2.MiMoV2MoE)
    nn.Module.__init__(moe)
    moe._use_target_mega_moe = True
    moe._enable_a2a_moe = False
    expected = torch.ones(2, 3)
    monkeypatch.setattr(moe, "forward_normal", lambda hidden_states: expected)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False)
    )

    assert moe.forward(torch.zeros(2, 3), forward_batch) is expected


def test_mimo_megamoe_caller_uses_shared_adapter(monkeypatch):
    moe = mimo_v2.MiMoV2MoE.__new__(mimo_v2.MiMoV2MoE)
    nn.Module.__init__(moe)
    moe._use_target_mega_moe = True
    moe.experts = SimpleNamespace(_mega_moe_weights_built=True)
    expected = torch.ones(2, 3)
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
        lambda layer, hidden_states: True,
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe.forward_mega_moe",
        lambda layer, hidden_states, forward_batch: expected,
    )

    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True)
    )
    assert moe.forward(torch.zeros(2, 3), forward_batch) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
