from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")
register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")


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
    from sglang.srt.layers.layer_boundary import MHCState
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
    # which update_and_read_ffn_input short-circuits before reaching the callback.
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
        outputs = [
            s.update_and_read_ffn_input(x, residual.flatten(1), norm) for s in states
        ]
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    torch.testing.assert_close(states[0].h_res, states[1].h_res, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(states[0].h_post, states[1].h_post, atol=1e-3, rtol=1e-3)
    # The next combine must consume the FFN mixing matrices, not attention's.
    torch.testing.assert_close(
        states[0].apply_post(x, outputs[0][1]),
        states[1].apply_post(x, outputs[1][1]),
        atol=2e-3,
        rtol=2e-2,
    )


def _hopper_mhc_layer(norm, w, scale, base):
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        config=SimpleNamespace(model_type="deepseek_v41"),
        hc_pre_from_prev_sublayer=True,
        hc_attn_fn=w,
        hc_ffn_fn=-w,
        hc_attn_scale=scale,
        hc_ffn_scale=scale,
        hc_attn_base=base,
        hc_ffn_base=base,
        hc_cfg=model_mhc.HcConfig(4, 20, 1e-6, 1e-6, 5120, True, False),
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    layer._init_hyper_connections = lambda: (
        DeepseekV4DecoderLayer._init_hyper_connections(layer)
    )
    return layer


def _mhc_mix_inputs(num_tokens):
    torch.manual_seed(192 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    return x, w, scale, base


def _mhc_mix_reference(x, w, scale, base):
    xd = x.flatten(1).double()
    z = (xd @ w.double().T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-6)
    pre = torch.sigmoid(z[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    comb = (z[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return pre, post, comb


@pytest.mark.parametrize("num_tokens", [32, 33, 64, 128, 257, 4096, 4097, 8192])
def test_hopper_compensated_mhc(num_tokens):
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper compensated mHC dispatch")
    x, w, scale, base = _mhc_mix_inputs(num_tokens)
    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    layer = _hopper_mhc_layer(norm, w, scale, base)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is not None
        assert layer._hc_attn_tf32_parts is None
        actual = model_mhc.mix_stats(layer.attn_hc, x)
    reference = _mhc_mix_reference(x, w, scale, base)
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False):
        fallback = model_mhc.mix_stats(layer.attn_hc, x)
        for result, expected in zip(fallback, reference):
            torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        assert layer._hc_attn_bf16_parts is None


@pytest.mark.parametrize("num_tokens", [32, 100, 256, 512, 1024])
def test_deepgemm_prenorm_dual_matches_two_calls(num_tokens):
    from unittest.mock import patch

    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
        pytest.skip("SM10x only")
    torch.manual_seed(7 + num_tokens)
    x = torch.randn(num_tokens, 20480, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    high = w.bfloat16().float()
    low = (w - high).contiguous()
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    args = (x, (high, low), scale, base, 20, 1e-6, 1e-6)
    dual = mhc.hc_mix_stats_sinkhorn_deepgemm(*args)
    not_sm10 = SimpleNamespace(
        is_sm=lambda _: False, num_sms=mhc.current_arch().num_sms
    )
    with patch.object(mhc, "current_arch", return_value=not_sm10):
        two_calls = mhc.hc_mix_stats_sinkhorn_deepgemm(*args)
    for a, b in zip(dual, two_calls):
        assert torch.equal(a, b)


@pytest.mark.parametrize("num_tokens", [32, 33, 256, 257, 4095, 4096])
def test_blackwell_compensated_mhc(num_tokens):
    """Blackwell takes the BF16x3 projection up to 256 rows (not on SM107) and
    for 4096-65536 rows, and the DeepGEMM projection in between."""
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import get_device_sm, is_sm100_supported

    if not is_sm100_supported():
        pytest.skip("Blackwell compensated mHC dispatch")
    bf16x3 = num_tokens >= 4096 or (num_tokens <= 256 and get_device_sm() != 107)
    unused = (
        "hc_mix_stats_sinkhorn_deepgemm" if bf16x3 else "hc_mix_stats_sinkhorn_bf16x3"
    )
    x, w, scale, base = _mhc_mix_inputs(num_tokens)
    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    layer = _hopper_mhc_layer(norm, w, scale, base)
    with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        if layer._hc_attn_bf16_parts is None:
            pytest.skip("DeepGEMM tf32_hc_prenorm_gemm is unavailable")
        with patch.object(mhc, unused, side_effect=AssertionError(unused)):
            actual = model_mhc.mix_stats(layer.attn_hc, x)
    if bf16x3:
        for result, expected in zip(actual, _mhc_mix_reference(x, w, scale, base)):
            torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("num_tokens", [1, 4, 64, 4096])
def test_hopper_combine_norm(num_tokens):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper combine/norm dispatch")
    torch.manual_seed(39 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(num_tokens, 4, device="cuda")
    norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    hc = model_mhc.HcSubLayer(
        model_mhc.HcConfig(4, 20, 1e-6, 1e-6, 5120, True, False),
        None,
        None,
        None,
        norm,
    )
    actual = model_mhc.combine(hc, model_mhc.HcState(x, pre))
    expected = norm(mhc.mhc_combine(x.flatten(1), pre, 4, x.dtype))
    torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=1e-5)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
def test_hopper_mhc_stats_stream_graph(num_tokens):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.utils import is_sm90_supported

    if not is_sm90_supported():
        pytest.skip("Hopper mHC overlap")
    torch.manual_seed(934 + num_tokens)
    x = torch.randn(num_tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="cuda") * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda")
    pre = torch.rand(num_tokens, 4, device="cuda")
    norm = RMSNorm(5120, eps=1e-6).cuda().bfloat16()
    layer = _hopper_mhc_layer(norm, w, scale, base)
    layer.hc_stats_stream = torch.cuda.Stream()
    DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert model_mhc.use_stats_stream(
        layer.hc_cfg, SimpleNamespace(forward_mode=ForwardMode.DECODE), x
    )
    stream = layer.hc_stats_stream

    def run(stats_stream):
        # The layer forks the side stream right before each combine; the combine
        # itself no longer takes a stream.
        model_mhc.fork_stats_stream(stats_stream)
        combined = model_mhc.combine(layer.attn_hc, model_mhc.HcState(x, pre))
        coefficients = model_mhc.mix_stats(layer.attn_hc, x, stats_stream)
        if stats_stream is not None:
            torch.cuda.current_stream().wait_stream(stats_stream)
        # Read on the consumer stream after the join, as hc_post does.
        return combined.clone(), *(v.clone() for v in coefficients)

    run(None)
    run(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(stream)
    for _ in range(3):
        x.normal_()
        pre.uniform_()
        expected = run(None)
        graph.replay()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_mhc_refresh_without_deepgemm_prenorm():
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.layers.deep_gemm_wrapper import configurer
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.runtime_context import override_platform

    norm = SimpleNamespace(weight=torch.ones(5120, device="cuda", dtype=torch.bfloat16))
    w = torch.randn(24, 20480, device="cuda")
    scale = torch.ones(3, device="cuda")
    base = torch.zeros(24, device="cuda")
    layer = _hopper_mhc_layer(norm, w, scale, base)
    layer._can_fuse_attn_mhc = layer._can_fuse_ffn_mhc = True
    with (
        override_platform(is_sm100=True, is_sm90=False),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        patch.object(configurer, "ENABLE_JIT_DEEPGEMM", True),
        patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}),
    ):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert layer.attn_hc.fn is w
    assert layer.ffn_hc.fn is layer.hc_ffn_fn
    assert layer.attn_hc.tf32_parts is None
    assert layer.attn_hc.bf16_parts is None
    assert "_can_fuse_attn_mhc" not in layer.__dict__
    assert "_can_fuse_ffn_mhc" not in layer.__dict__


@pytest.mark.parametrize(
    "replicated,add_shared,has_shared,expected",
    [
        (False, False, True, 16.0),
        (True, False, True, 2.0),
        (True, True, True, 9.0),
        (True, True, False, 2.0),
    ],
)
def test_moe_post_reduces_before_adding_replicated_shared(
    monkeypatch, replicated, add_shared, has_shared, expected
):
    from sglang.srt.layers import moe
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.models import deepseek_v4_mhc as model_mhc

    routed = torch.ones(2, 3)
    shared = torch.full((2, 3), 7.0) if has_shared else None
    pieces = model_mhc.MoEOutput(routed, shared, None, 1.0, replicated)
    partial = routed if replicated or shared is None else routed + shared
    monkeypatch.setattr(model_mhc.MoEOutput, "get_merged", lambda self: partial)
    monkeypatch.setattr(model_mhc, "is_deferred_finalize", lambda value: False)
    monkeypatch.setattr(
        moe_utils, "should_add_replicated_moe_output", lambda: add_shared
    )
    reduced_inputs = []

    def reduce(value):
        reduced_inputs.append(value.clone())
        return value * 2

    monkeypatch.setattr(moe, "post_experts_all_reduce", reduce)
    monkeypatch.setattr(model_mhc, "_compute_triplet", lambda *args: None)
    monkeypatch.setattr(
        model_mhc, "_post_fusion", lambda hc, value, residual, triplet, *args: value
    )
    actual = model_mhc.run_moe_post(
        None, pieces, None, stats_stream=None, next=None, world_size=2
    )
    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], partial)
    torch.testing.assert_close(actual, torch.full((2, 3), expected))


_EPS = 1e-6
_SINKHORN_ITERS = 20
_POST_MULT = 2.0


@pytest.fixture
def single_process(monkeypatch, stated_tp_group):
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    yield


def _mhc_pre_reference(residual, fn, hc_scale, hc_base, *, eps_outside_sqrt):
    hc_mult = residual.shape[1]
    x = residual.flatten(1).float()
    mixes = x @ fn.t()
    mean_sq = x.pow(2).mean(-1, keepdim=True)
    if eps_outside_sqrt:
        rms = 1.0 / (mean_sq.sqrt() + _EPS)
    else:
        rms = torch.rsqrt(mean_sq + _EPS)
    mixes = mixes * rms
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + _EPS
    post = _POST_MULT * torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]).view(
        -1, hc_mult, hc_mult
    )
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + _EPS
    comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + _EPS)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(1)
    return post, comb, layer_input


def _eps_outside_sqrt_inputs(num_tokens, hc_mult=4, hidden=4096):
    torch.manual_seed(0)
    mix = hc_mult * (2 + hc_mult)
    # A small residual puts mean_sq near eps, where the two eps placements differ.
    residual = (
        torch.randn(num_tokens, hc_mult, hidden, device="cuda") * 1e-3
    ).bfloat16()
    fn = torch.randn(mix, hc_mult * hidden, device="cuda") * 0.01
    hc_scale = torch.rand(3, device="cuda")
    hc_base = torch.randn(mix, device="cuda") * 0.1
    return residual, fn, hc_scale, hc_base


def _run_mhc_pre_eps_outside_sqrt(residual, fn, hc_scale, hc_base, **norm):
    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=_EPS,
        hc_pre_eps=_EPS,
        hc_sinkhorn_eps=_EPS,
        hc_post_mult_value=_POST_MULT,
        sinkhorn_repeat=_SINKHORN_ITERS,
        eps_outside_sqrt=True,
        **norm,
    )


def _assert_post_uses_eps_outside_sqrt(post, inputs):
    ref_post, _, _ = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    dsv4_post, _, _ = _mhc_pre_reference(*inputs, eps_outside_sqrt=False)
    # Tolerance covers the kernel's split-K fp32 GEMM order.
    torch.testing.assert_close(post.squeeze(-1), ref_post, rtol=2e-3, atol=2e-3)
    assert (post.squeeze(-1) - dsv4_post).abs().max() > 1e-2


@pytest.mark.parametrize("num_tokens", [7, 300, 3000])
def test_mhc_pre_eps_outside_sqrt(num_tokens, single_process):
    inputs = _eps_outside_sqrt_inputs(num_tokens)
    post, comb, layer_input = _run_mhc_pre_eps_outside_sqrt(*inputs)
    _, ref_comb, ref_input = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    _assert_post_uses_eps_outside_sqrt(post, inputs)
    torch.testing.assert_close(comb, ref_comb, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(layer_input.float(), ref_input, rtol=2e-2, atol=1e-4)


@pytest.mark.parametrize("num_tokens", [7, 300])
def test_mhc_pre_eps_outside_sqrt_fused_norm(num_tokens, single_process):
    inputs = _eps_outside_sqrt_inputs(num_tokens)
    hidden = inputs[0].shape[-1]
    norm_weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).bfloat16()
    post, _, layer_input = _run_mhc_pre_eps_outside_sqrt(
        *inputs, norm_weight=norm_weight, norm_eps=_EPS
    )
    _assert_post_uses_eps_outside_sqrt(post, inputs)
    _, _, ref_input = _mhc_pre_reference(*inputs, eps_outside_sqrt=True)
    ref = torch.nn.functional.rms_norm(ref_input, (hidden,), norm_weight.float(), _EPS)
    # The reference norms the fp32 layer input; the kernel scales a bf16 stash.
    torch.testing.assert_close(layer_input.float(), ref, rtol=2e-2, atol=2e-2)


def _hc_head_reference(x, fn, scale, base, norm_eps, hc_eps, eps_outside_sqrt):
    flat = x.flatten(1).float()
    mean_sq = flat.square().mean(-1, keepdim=True)
    if eps_outside_sqrt:
        rsqrt = 1.0 / (mean_sq.sqrt() + norm_eps)
    else:
        rsqrt = torch.rsqrt(mean_sq + norm_eps)
    pre = torch.sigmoid(torch.nn.functional.linear(flat, fn) * rsqrt * scale + base)
    return torch.sum((pre + hc_eps).unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


@pytest.mark.parametrize("eps_outside_sqrt", [False, True])
def test_fused_hc_head(eps_outside_sqrt):
    from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head

    torch.manual_seed(0)
    num_tokens, hc, hidden = 33, 4, 4096
    x = torch.randn(num_tokens, hc, hidden, device="cuda").bfloat16()
    fn = torch.randn(hc, hc * hidden, device="cuda") * 0.1
    scale = torch.ones(1, device="cuda")
    base = torch.randn(hc, device="cuda") * 0.1
    # A large eps puts a quarter or more of the outputs of the two formulas
    # outside the tolerance of each other.
    norm_eps = 0.5
    out = fused_hc_head(
        x,
        fn,
        scale,
        base,
        norm_eps=norm_eps,
        hc_eps=1e-6,
        eps_outside_sqrt=eps_outside_sqrt,
    )
    ref = _hc_head_reference(x, fn, scale, base, norm_eps, 1e-6, eps_outside_sqrt)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
