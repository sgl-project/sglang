import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.layernorm.hc_mix_stats_deepgemm import (
    hc_mix_stats_sinkhorn_deepgemm,
    split_tf32_hc_weight,
)
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Compensated mHC path targets datacenter Blackwell",
)
EPS = 1e-6


def inputs(m, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, 20480), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((24, 20480), device="cuda", dtype=torch.float32) * 0.02
    scale = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    base = torch.randn(24, device="cuda", dtype=torch.float32) * 0.2
    return x, w, scale, base


def reference(x, w, scale, base):
    x, w, scale, base = (v.double() for v in (x, w, scale, base))
    mixes = (x @ w.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4)
    comb = torch.softmax(comb, dim=-1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    return pre, post, comb


@pytest.mark.parametrize("m", [0, 128, 384, 2049, 4096, 16384, 32768, 65536])
@pytest.mark.parametrize("seed", [0, 42])
def test_compensated_coefficients_match_fp64(m, seed):
    x, w, scale, base = inputs(m, seed)
    parts = split_tf32_hc_weight(w)
    assert torch.equal(parts[0] + parts[1], w)
    got = hc_mix_stats_sinkhorn_deepgemm(x, parts, scale, base, 20, EPS, EPS)
    expected = reference(x, w, scale, base)
    for actual, ref in zip(got, expected):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual.double(), ref, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("m", [384, 4096])
def test_graph_replay_reads_updated_input(m):
    x, w, scale, base = inputs(m, 13)
    parts = split_tf32_hc_weight(w)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            hc_mix_stats_sinkhorn_deepgemm(x, parts, scale, base, 20, EPS, EPS)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hc_mix_stats_sinkhorn_deepgemm(x, parts, scale, base, 20, EPS, EPS)
    # Scaling alone almost cancels under RMS normalization and would not expose
    # a replay that accidentally kept using the capture-time activations.
    x.normal_()
    graph.replay()
    expected = reference(x, w, scale, base)
    for actual, ref in zip(captured, expected):
        torch.testing.assert_close(actual.double(), ref, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("m", [1, 6, 64, 127])
def test_original_sinkhorn_path_without_residual_matches_fp64(m):
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn

    x, w, scale, base = inputs(m, 7)
    got = hc_mix_stats_sinkhorn(x, w, scale, base, 4, 20, EPS, EPS)
    expected = reference(x, w, scale, base)
    for actual, ref in zip(got, expected):
        torch.testing.assert_close(actual.double(), ref, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "m,invariant,use_fast",
    [
        (6, False, False),
        (64, False, False),
        (127, False, False),
        (128, False, True),
        (384, True, False),
        (384, False, True),
        (2049, True, False),
        (2049, False, True),
    ],
)
def test_model_dispatch_preserves_invariant_and_small_rows(m, invariant, use_fast):
    x, w, scale, base = inputs(m, 1)
    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_attn_fn = torch.nn.Parameter(w)
    layer.hc_ffn_fn = torch.nn.Parameter(w.clone())
    layer._hc_attn_tf32_parts = split_tf32_hc_weight(w)
    layer.hc_mult, layer.hc_sinkhorn_iters = 4, 20
    layer.rms_norm_eps = layer.hc_eps = EPS
    target = "sglang.kernels.ops.layernorm.hc_mix_stats_deepgemm.hc_mix_stats_sinkhorn_deepgemm"
    with (
        patch(
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            return_value=invariant,
        ),
        patch(target, wraps=hc_mix_stats_sinkhorn_deepgemm) as fast,
    ):
        layer._hc_mix_and_combine(
            x.view(m, 4, 5120), layer.hc_attn_fn, scale, base, None, lambda v: v
        )
    assert fast.called == use_fast


@pytest.mark.parametrize("enabled", [False, True])
def test_weight_cache_honors_deepgemm_availability(enabled):
    _, w, _, _ = inputs(0, 3)
    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_pre_from_prev_sublayer = True
    layer.hc_attn_fn = torch.nn.Parameter(w)
    layer.hc_ffn_fn = torch.nn.Parameter(w.clone())
    layer.input_layernorm = torch.nn.LayerNorm(5120, device="cuda")
    layer.post_attention_layernorm = torch.nn.LayerNorm(5120, device="cuda")
    with (
        patch(
            "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
            enabled,
        ),
        envs.SGLANG_DSV41_COMPENSATED_MHC.override(True),
    ):
        layer.refresh_mhc_norm_weight_cache()
    if enabled:
        assert torch.equal(sum(layer._hc_attn_tf32_parts), w)
        assert torch.equal(sum(layer._hc_ffn_tf32_parts), w)
    else:
        assert layer._hc_attn_tf32_parts is None
        assert layer._hc_ffn_tf32_parts is None

    # The default opt-out also avoids allocating derived weight buffers.
    with envs.SGLANG_DSV41_COMPENSATED_MHC.override(False):
        layer.refresh_mhc_norm_weight_cache()
    assert layer._hc_attn_tf32_parts is None
    assert layer._hc_ffn_tf32_parts is None


@pytest.mark.parametrize("m", [128, 384, 4096])
def test_fused_compensation_preserves_epilogue_bits(m):
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel

    torch.manual_seed(1)
    hi = torch.randn(16, m, 24, device="cuda")
    lo = torch.randn_like(hi) * 0.001
    sq = torch.rand(16, m, device="cuda") * 1280
    scale = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    base = torch.randn(24, device="cuda")

    def reduce(partial, residual=None):
        pre = torch.empty(m, 4, device="cuda")
        post = torch.empty_like(pre)
        comb = torch.empty(m, 4, 4, device="cuda")
        _hc_mix_reduce_sinkhorn_kernel[(m,)](
            partial,
            sq,
            scale,
            base,
            pre,
            post,
            comb,
            m,
            1.0 / 20480,
            EPS,
            MIX=24,
            HC=4,
            NUM_SLICES=16,
            ITERS=20,
            EPS=EPS,
            part_mix_residual_ptr=residual,
            num_warps=1,
        )
        return pre, post, comb

    expected = reduce(hi + lo)
    actual = reduce(hi, lo)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
