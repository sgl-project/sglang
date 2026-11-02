"""Kimi-K3 TP8 KDA recurrent launch on gfx950: the narrow tile for n <= 4 requests."""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

if not (torch.version.hip and torch.cuda.is_available() and is_gfx95_supported()):
    pytest.skip("requires AMD gfx95", allow_module_level=True)

from sglang.kernels.ops.attention.linear import (  # noqa: E402
    fused_sigmoid_gating_recurrent as recurrent_module,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (  # noqa: E402
    _select_recurrent_launch_config,
    fused_sigmoid_gating_delta_rule_update,
)

H = HV = 12
K = V = 128
SLOTS, RING_LEN = 8, 16


def test_only_small_kimi_k3_kda_batches_take_the_narrow_tile():
    for n in (1, 4):
        for target_verify in (True, False):
            assert _select_recurrent_launch_config(
                n=n, h=H, hv=HV, k=K, v=V, is_kda=True, target_verify=target_verify
            ) == (8, 1)
    assert _select_recurrent_launch_config(
        n=5, h=H, hv=HV, k=K, v=V, is_kda=True, target_verify=True
    ) == (32, 1)
    assert _select_recurrent_launch_config(
        n=1, h=H, hv=HV, k=K, v=V, is_kda=False, target_verify=True
    ) == (32, 1)
    assert _select_recurrent_launch_config(
        n=1, h=8, hv=8, k=K, v=V, is_kda=True, target_verify=True
    ) == (32, 1)


def _inputs(n: int, t: int) -> dict:
    torch.manual_seed(2025)
    tokens = n * t
    return dict(
        A_log=torch.randn(HV, device="cuda") * 0.1,
        a=torch.randn(1, tokens, HV * K, dtype=torch.bfloat16, device="cuda"),
        dt_bias=torch.randn(HV * K, device="cuda") * 0.1,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=torch.randn(1, tokens, H, K, dtype=torch.bfloat16, device="cuda"),
        k=torch.randn(1, tokens, H, K, dtype=torch.bfloat16, device="cuda"),
        v=torch.randn(1, tokens, HV, V, dtype=torch.bfloat16, device="cuda"),
        b=torch.randn(1, tokens, HV, dtype=torch.bfloat16, device="cuda"),
        initial_state_source=(torch.randn(SLOTS, HV, V, K, device="cuda") * 0.05).to(
            torch.bfloat16
        ),
        initial_state_indices=torch.randperm(SLOTS, device="cuda")[:n].to(torch.int32),
        cu_seqlens=torch.arange(0, tokens + 1, t, dtype=torch.int32, device="cuda"),
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )


def _run(monkeypatch, launch_config, inputs: dict, rings: dict) -> tuple:
    monkeypatch.setattr(
        recurrent_module, "_select_recurrent_launch_config", launch_config
    )
    state = inputs["initial_state_source"].clone()
    rings = {name: torch.zeros_like(buf) for name, buf in rings.items()}
    out = fused_sigmoid_gating_delta_rule_update(
        **{**inputs, "initial_state_source": state}, **rings
    )
    return out, state, rings


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("t", [4, 8])
def test_narrow_verify_launch_is_bit_exact(n: int, t: int, monkeypatch):
    """Verify output and ReplaySSM rings match the BV=32 launch bit for bit."""
    inputs = dict(
        _inputs(n=n, t=t), disable_state_update=True, cache_steps=t, cache_ring=True
    )
    rings = dict(
        replayssm_rawv=torch.empty(SLOTS, HV, RING_LEN, V, dtype=torch.bfloat16),
        replayssm_rawk=torch.empty(SLOTS, H, RING_LEN, K, dtype=torch.bfloat16),
        replayssm_g=torch.empty(SLOTS, HV, RING_LEN, K, dtype=torch.float32),
        replayssm_beta=torch.empty(SLOTS, HV, RING_LEN, dtype=torch.float32),
    )
    rings = {name: buf.cuda() for name, buf in rings.items()}
    tuned = _run(
        monkeypatch=monkeypatch,
        launch_config=_select_recurrent_launch_config,
        inputs=inputs,
        rings=rings,
    )
    default = _run(
        monkeypatch=monkeypatch,
        launch_config=lambda *args, **kwargs: (32, 1),
        inputs=inputs,
        rings=rings,
    )
    assert torch.equal(tuned[0], default[0])
    for name in rings:
        assert torch.equal(tuned[2][name], default[2][name]), name


@pytest.mark.parametrize("n", [1, 4])
def test_narrow_decode_launch_is_bit_exact(n: int, monkeypatch):
    """Decode output and the updated recurrent state match the BV=32 launch bit for bit."""
    inputs = _inputs(n=n, t=1)
    tuned = _run(
        monkeypatch=monkeypatch,
        launch_config=_select_recurrent_launch_config,
        inputs=inputs,
        rings={},
    )
    default = _run(
        monkeypatch=monkeypatch,
        launch_config=lambda *args, **kwargs: (32, 1),
        inputs=inputs,
        rings={},
    )
    assert torch.equal(tuned[0], default[0])
    assert torch.equal(tuned[1], default[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
