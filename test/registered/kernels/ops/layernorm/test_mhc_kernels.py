from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@pytest.fixture
def stated_tp_group():
    """Provide a TP-group placeholder for kernels with mocked symmetric memory."""
    from sglang.srt.runtime_context import get_parallel

    with get_parallel().override(tp_group=None):
        yield


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 6, 8, 17, 32, 64])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm, stated_tp_group
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    # Disable symmetric-memory allocation for this single-process kernel test.
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
        if use_norm
        else None
    )
    norm_eps = 1e-6 if use_norm else None

    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_repeat = 2

    residual_ref = post_ref = comb_ref = layer_ref = None
    if num_tokens > 0:
        residual_ref = mhc_post(x, residual, post_prev, comb_prev)
        post_ref, comb_ref, layer_ref = mhc_pre(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    residual_out, post_out, comb_out, layer_out = mhc_fused_post_pre(
        x,
        residual,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    if hidden_size == 4096 and num_tokens in (0, 1, 6, 17):
        _check_glm_boundary(
            x,
            residual,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            use_norm=use_norm,
        )

    torch.cuda.synchronize()
    if num_tokens == 0:
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

    assert residual_ref is not None
    assert post_ref is not None
    assert comb_ref is not None
    assert layer_ref is not None
    assert residual_out.shape == residual_ref.shape
    assert post_out.shape == post_ref.shape
    assert comb_out.shape == comb_ref.shape
    assert layer_out.shape == layer_ref.shape

    torch.testing.assert_close(residual_out, residual_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_atol = 2e-2 if use_norm else 2e-3
    layer_rtol = 2e-2 if use_norm else 2e-3
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_atol, rtol=layer_rtol)


def _check_glm_boundary(x, residual, post, comb, fn, scale, base, *, use_norm):
    from sglang.srt.environ import envs
    from sglang.srt.layers.communicator_mhc import MHCState
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = torch.nn.Parameter(fn)
    layer.hc_ffn_scale = torch.nn.Parameter(scale)
    layer.hc_ffn_base = torch.nn.Parameter(base)
    norm = RMSNorm(x.shape[-1], eps=1e-6).to(x) if use_norm else None
    states = [
        MHCState(
            hc_mult=4,
            hc_attn_pre=layer.hc_attn_pre,
            hc_ffn_pre=layer.hc_ffn_pre,
            hc_post=layer.hc_post,
            hc_ffn_post_pre=callback,
            h_res=comb.flatten(1),
            h_post=post.flatten(1),
        )
        for callback in (None, layer.hc_ffn_post_pre)
    ]
    # Literal, not derived from the cutoff constant: deriving it makes this a
    # mirror that stays green when the cutoff moves. None is the empty batch,
    # which attn_to_mlp short-circuits before reaching the callback.
    fused_expected = {1: True, 6: True, 17: False}[x.shape[0]] if x.shape[0] else None
    with envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True):
        if x.shape[0] > 0:
            declined = (
                layer.hc_ffn_post_pre(
                    hidden_states=x,
                    residual=residual.flatten(1),
                    h_res=comb.flatten(1),
                    h_post=post.flatten(1),
                    out_norm_weight=None,
                    out_norm_eps=None,
                )
                is None
            )
            assert declined is not fused_expected, (
                f"num_tokens={x.shape[0]} fused={not declined}, "
                f"expected fused={fused_expected}"
            )
        outputs = [s.attn_to_mlp(x, residual.flatten(1), norm) for s in states]
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    torch.testing.assert_close(states[0].h_res, states[1].h_res, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(states[0].h_post, states[1].h_post, atol=1e-3, rtol=1e-3)
    # The next combine must consume the FFN mixing matrices, not attention's.
    torch.testing.assert_close(
        states[0].mlp_combine(x, outputs[0][1]),
        states[1].mlp_combine(x, outputs[1][1]),
        atol=2e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize("num_tokens", [32, 33, 64, 128, 257, 4096, 4097, 8192])
def test_hopper_compensated_mhc(num_tokens):
    from sglang.srt.environ import envs
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper compensated mHC dispatch")
    torch.manual_seed(192 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        config=SimpleNamespace(model_type="deepseek_v41"),
        hc_pre_from_prev_sublayer=True,
        hc_attn_fn=w,
        hc_ffn_fn=-w,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is not None
        assert layer._hc_attn_tf32_parts is None
        actual = DeepseekV4DecoderLayer._hc_mix_stats(layer, x, w, scale, base)
    xd = x.flatten(1).double()
    z = (xd @ w.double().T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6)
    expected_pre = torch.sigmoid(z[:, :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    comb = (z[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for result, expected in zip(actual, (expected_pre, expected_post, comb)):
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False):
        fallback = DeepseekV4DecoderLayer._hc_mix_stats(layer, x, w, scale, base)
        for result, expected in zip(fallback, (expected_pre, expected_post, comb)):
            torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is None


@pytest.mark.parametrize("num_tokens", [1, 4, 64, 4096])
def test_hopper_combine_norm(num_tokens):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper combine/norm dispatch")
    torch.manual_seed(39 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(num_tokens, 4, device="cuda")
    norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    layer = SimpleNamespace(
        hc_mult=4, config=SimpleNamespace(model_type="deepseek_v41")
    )
    actual = DeepseekV4DecoderLayer._hc_combine(layer, x, pre, norm)
    expected = norm(mhc.hc_combine(x.flatten(1), pre, 4, x.dtype))
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
