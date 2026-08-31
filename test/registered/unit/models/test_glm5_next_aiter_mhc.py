import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.communicator_mhc import MHCState
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _fake_aiter_mhc_module(*, mhc_pre=None, mhc_post=None):
    aiter_module = ModuleType("aiter")
    ops_module = ModuleType("aiter.ops")
    mhc_module = ModuleType("aiter.ops.mhc")
    if mhc_pre is not None:
        mhc_module.mhc_pre = mhc_pre
    if mhc_post is not None:
        mhc_module.mhc_post = mhc_post
    aiter_module.ops = ops_module
    ops_module.mhc = mhc_module
    return {
        "aiter": aiter_module,
        "aiter.ops": ops_module,
        "aiter.ops.mhc": mhc_module,
    }


def test_aiter_mhc_gate_is_rocm_gfx95_only():
    with (
        patch.object(mhc, "is_hip", return_value=True),
        patch.object(mhc, "is_gfx95_supported", return_value=True),
        patch.object(mhc, "get_bool_env_var", return_value=True),
        patch.object(mhc, "_AITER_MHC_RUNTIME_DISABLED", False),
    ):
        assert mhc._use_aiter_mhc()

    for is_hip, is_gfx95, use_aiter in (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ):
        with (
            patch.object(mhc, "is_hip", return_value=is_hip),
            patch.object(mhc, "is_gfx95_supported", return_value=is_gfx95),
            patch.object(mhc, "get_bool_env_var", return_value=use_aiter),
            patch.object(mhc, "_AITER_MHC_RUNTIME_DISABLED", False),
        ):
            assert not mhc._use_aiter_mhc()


def test_aiter_mhc_pre_dispatch_forwards_norm_contract():
    residual = torch.randn(2, 4, 8)
    fn = torch.randn(24, 32)
    hc_scale = torch.randn(3)
    hc_base = torch.randn(24)
    norm_weight = torch.randn(8)
    expected = (
        torch.randn(2, 4, 1),
        torch.randn(2, 4, 4),
        torch.randn(2, 8),
    )
    fake_pre = MagicMock(return_value=expected)

    with (
        patch.dict(sys.modules, _fake_aiter_mhc_module(mhc_pre=fake_pre)),
        patch.object(mhc, "_use_aiter_mhc", return_value=True),
        patch.object(mhc, "_AITER_MHC_ACTIVE_LOGGED", False),
    ):
        actual = mhc._mhc_pre_dispatch(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-5,
            hc_sinkhorn_eps=1e-5,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            norm_weight=norm_weight,
            norm_eps=1e-4,
        )

    assert actual[0] is expected[0]
    assert actual[1] is expected[1]
    assert actual[2] is expected[2]
    assert actual[3] is True
    fake_pre.assert_called_once_with(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-5,
        1e-5,
        2.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-4,
    )


def test_aiter_mhc_post_dispatch_uses_preallocated_output():
    x = torch.randn(2, 8)
    residual = torch.randn(2, 4, 8)
    post = torch.randn(2, 4, 1)
    comb = torch.randn(2, 4, 4)

    def fake_post(out, x_arg, residual_arg, post_arg, comb_arg):
        assert x_arg is x
        assert residual_arg is residual
        assert post_arg is post
        assert comb_arg is comb
        out.copy_(residual + 1)

    with (
        patch.dict(sys.modules, _fake_aiter_mhc_module(mhc_post=fake_post)),
        patch.object(mhc, "_use_aiter_mhc", return_value=True),
    ):
        actual = mhc._mhc_post_dispatch(x, residual, post, comb)

    torch.testing.assert_close(actual, residual + 1)


class _AddOneNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return x + 1


def test_mhc_state_uses_optional_fused_attention_to_mlp_boundary():
    hidden_states = torch.randn(2, 8)
    residual = torch.randn(2, 32)
    next_hidden_states = torch.randn(2, 8)
    next_residual = torch.randn(2, 32)
    next_h_res = torch.randn(2, 16)
    next_h_post = torch.randn(2, 4)
    fused = MagicMock(
        return_value=(
            next_hidden_states,
            next_residual,
            next_h_res,
            next_h_post,
            False,
        )
    )
    initial_h_res = torch.randn(2, 16)
    initial_h_post = torch.randn(2, 4)
    state = MHCState(
        hc_mult=4,
        hc_attn_pre=MagicMock(),
        hc_ffn_pre=MagicMock(side_effect=AssertionError("unfused pre called")),
        hc_post=MagicMock(side_effect=AssertionError("unfused post called")),
        hc_attn_to_mlp=fused,
        h_res=initial_h_res,
        h_post=initial_h_post,
    )
    norm = _AddOneNorm(8)

    actual_hidden_states, actual_residual = state.attn_to_mlp(
        hidden_states, residual, norm
    )

    torch.testing.assert_close(actual_hidden_states, next_hidden_states + 1)
    assert actual_residual is next_residual
    assert state.h_res is next_h_res
    assert state.h_post is next_h_post
    fused.assert_called_once()
    call_args = fused.call_args.args
    assert call_args[0] is hidden_states
    assert call_args[1] is residual
    assert call_args[2] is initial_h_res
    assert call_args[3] is initial_h_post
    torch.testing.assert_close(call_args[4], norm.weight)
    assert call_args[5] == norm.variance_epsilon


def test_glm_aiter_mhc_boundary_preserves_communicator_shapes():
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = nn.Parameter(torch.randn(24, 32))
    layer.hc_ffn_scale = nn.Parameter(torch.randn(3))
    layer.hc_ffn_base = nn.Parameter(torch.randn(24))

    hidden_states = torch.randn(2, 8)
    residual = torch.randn(2, 32)
    h_res = torch.randn(2, 16)
    h_post = torch.randn(2, 4)
    next_residual = torch.randn(2, 4, 8)
    next_hidden_states = torch.randn(2, 8)
    next_h_post = torch.randn(2, 4)
    next_h_res = torch.randn(2, 4, 4)

    with (
        patch(
            "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc.apply_mhc_post_pre_boundary",
            return_value=(
                next_residual,
                next_hidden_states,
                next_h_post,
                next_h_res,
                True,
            ),
        ) as fused,
        patch.object(glm5_next, "_GLM_AITER_FUSED_MHC_LOGGED", False),
    ):
        actual = layer.hc_attn_to_mlp(
            hidden_states,
            residual,
            h_res,
            h_post,
            torch.ones(8),
            1e-6,
        )

    actual_hidden, actual_residual, actual_h_res, actual_h_post, norm_fused = actual
    assert actual_hidden is next_hidden_states
    torch.testing.assert_close(actual_residual, next_residual.reshape(2, 32))
    torch.testing.assert_close(actual_h_res, next_h_res.reshape(2, 16))
    torch.testing.assert_close(actual_h_post, next_h_post)
    assert norm_fused
    assert fused.call_args.kwargs["fn_transpose"] is True
    assert fused.call_args.kwargs["residual"].shape == (2, 4, 8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
