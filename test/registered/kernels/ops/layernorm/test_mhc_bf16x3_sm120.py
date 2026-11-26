"""SM120 reaches the compensated mHC projection through the explicit opt-in only.

arg_groups/model_hook.py turns SGLANG_OPT_DEEPGEMM_HC_PRENORM off on SM120 behind an
``is_set()`` guard, so a user can still opt in. The load-time gate in deepseek_v4.py
then decides whether that opt-in builds anything. These tests pin both halves of the
contract: SM120 takes the pure-Triton bf16x3 route and never the DeepGEMM tf32 one,
the row band is the only thing that selects it, and the coefficients match an fp64
mirror of the reduce arithmetic.

Registered on ``1-gpu-small``, which scripts/ci/runner_configs.yml maps to a 5090, so
the SM120 branch executes in CI instead of skipping on a mismatched card.
"""

from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as kernels_mhc
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 5120
HC_MULT = 4
SINKHORN_ITERS = 20
RMS_EPS = 1e-6
HC_EPS = 1e-6
MIX = (2 + HC_MULT) * HC_MULT  # 24
K = HC_MULT * HIDDEN  # 20480

pytestmark = pytest.mark.skipif(
    get_device_sm() not in (120, 121), reason="SM12x compensated mHC dispatch"
)


def _sm120_platform():
    """State the architecture explicitly rather than trusting the runner's card."""
    from sglang.srt.runtime_context import override_platform

    return override_platform(
        is_sm90=False, is_sm100=False, is_sm120=True, is_blackwell=True
    )


def _layer(norm, w, scale, base):
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
        hc_cfg=model_mhc.HcConfig(
            HC_MULT, SINKHORN_ITERS, HC_EPS, RMS_EPS, HIDDEN, True, False
        ),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=SINKHORN_ITERS,
        rms_norm_eps=RMS_EPS,
        hc_eps=HC_EPS,
    )
    layer._init_hyper_connections = lambda: (
        DeepseekV4DecoderLayer._init_hyper_connections(layer)
    )
    return layer


def _material(num_tokens, seed):
    torch.manual_seed(seed + num_tokens)
    x = torch.randn(num_tokens, HC_MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(MIX, K, device="cuda", dtype=torch.float32) * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(MIX, device="cuda")
    norm = SimpleNamespace(
        weight=torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    )
    return x, w, scale, base, _layer(norm, w, scale, base)


def _fp64_triplet(x, w, scale, base, chunk=1024):
    """Mirror _hc_mix_reduce_sinkhorn_kernel in fp64, as the Hopper test does.

    Chunked over rows because the reduce is row-independent, and an 8192-row fp64
    mirror would want 1.3 GB of scratch -- more than the kernel under test uses.
    """
    hc = HC_MULT
    rows = x.flatten(1)
    wd = w.double()
    pre_out, post_out, comb_out = [], [], []
    for start in range(0, rows.shape[0], chunk):
        xd = rows[start : start + chunk].double()
        z = (xd @ wd.T) * torch.rsqrt(xd.square().mean(-1, keepdim=True) + RMS_EPS)
        pre_out.append(torch.sigmoid(z[:, :hc] * scale[0] + base[:hc]) + HC_EPS)
        post_out.append(
            2 * torch.sigmoid(z[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        )
        comb = (z[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
        comb = (comb - comb.amax(-1, keepdim=True)).exp()
        comb = comb / comb.sum(-1, keepdim=True) + HC_EPS
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
        for _ in range(SINKHORN_ITERS - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
            comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
        comb_out.append(comb)
    return torch.cat(pre_out), torch.cat(post_out), torch.cat(comb_out)


def test_opt_in_builds_bf16_parts_and_skips_deepgemm():
    """With DeepGEMM present but unverified here, SM120 must not take the tf32 route."""
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.layers.deep_gemm_wrapper import configurer
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    _, w, scale, base, layer = _material(1, seed=11)
    deep_gemm = SimpleNamespace(tf32_hc_prenorm_gemm=lambda *args: None)
    with (
        _sm120_platform(),
        envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True),
        patch.object(configurer, "ENABLE_JIT_DEEPGEMM", True),
        patch.dict("sys.modules", {"deep_gemm": deep_gemm}),
    ):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)

    parts = layer._hc_attn_bf16_parts
    assert parts is not None, "the SM120 opt-in built nothing"
    assert len(parts) == 3
    for part in parts:
        assert part.shape == (MIX, K)
        assert part.dtype == torch.bfloat16
    assert layer._hc_attn_tf32_parts is None, (
        "SM120 was routed into the DeepGEMM tf32 branch"
    )
    assert layer.attn_hc.bf16_parts is parts
    assert layer.attn_hc.tf32_parts is None

    # The default stays off: without the opt-in the weight triple is never built,
    # so its resident cost is not paid by workloads that never ask for it.
    with _sm120_platform(), envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(False):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
    assert layer._hc_attn_bf16_parts is None
    assert layer.attn_hc.bf16_parts is None


@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_in_band_rows_use_the_compensated_projection(monkeypatch, num_tokens):
    """Rows in [4096, 65536] take bf16x3 and must match fp64 arithmetic."""
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    x, w, scale, base, layer = _material(num_tokens, seed=19)
    calls = []
    real = kernels_mhc.hc_mix_stats_sinkhorn_bf16x3

    def spy(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels_mhc, "hc_mix_stats_sinkhorn_bf16x3", spy)
    with _sm120_platform(), envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        actual = model_mhc.mix_stats(layer.attn_hc, x)

    assert calls, "in-band rows did not reach the compensated projection"
    weight_parts = calls[0]
    assert weight_parts is layer.attn_hc.bf16_parts
    # Same tolerances the Hopper test holds itself to, so a miss here is a signal
    # about the SM120 lowering rather than slack the test absorbed.
    for result, expected in zip(actual, _fp64_triplet(x, w, scale, base)):
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_out_of_band_rows_stay_on_the_partial_path(monkeypatch, num_tokens):
    """Decode and small prefill keep the split-K partial kernel and stay correct."""
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    x, w, scale, base, layer = _material(num_tokens, seed=23)

    def refuse(*args, **kwargs):
        raise AssertionError(
            f"num_tokens={num_tokens} entered the compensated projection, which "
            "only applies to hopper_medium rows or [4096, 65536]"
        )

    partial = []
    real_partial = kernels_mhc.hc_mix_stats_sinkhorn

    def spy_partial(*args, **kwargs):
        partial.append(args[1])
        return real_partial(*args, **kwargs)

    monkeypatch.setattr(kernels_mhc, "hc_mix_stats_sinkhorn_bf16x3", refuse)
    monkeypatch.setattr(kernels_mhc, "hc_mix_stats_sinkhorn", spy_partial)
    with _sm120_platform(), envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True):
        DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        actual = model_mhc.mix_stats(layer.attn_hc, x)

    assert partial, "out-of-band rows did not take the partial path"
    assert partial[0] is layer.attn_hc.fn
    for result, expected in zip(actual, _fp64_triplet(x, w, scale, base)):
        torch.testing.assert_close(result.double(), expected, rtol=1e-5, atol=2e-6)


def test_split_bf16_hc_weight_recovers_the_fp32_projection():
    """Three parts are only worth it if they reconstruct the fp32 weight.

    Compared as a ratio, not against an absolute threshold: bf16 error scales with
    the weight magnitude, so an absolute bound on the single-part error would move
    with the checkpoint and say nothing about the split.
    """
    torch.manual_seed(7)
    w = torch.randn(MIX, K, device="cuda", dtype=torch.float32) * 0.01
    high, middle, low = kernels_mhc.split_bf16_hc_weight(w)
    wd = w.double()
    split_error = (high.double() + middle.double() + low.double() - wd).abs().max()
    single_error = (w.bfloat16().double() - wd).abs().max()
    assert split_error < 1e-6, split_error.item()
    assert single_error > 100 * split_error, (split_error.item(), single_error.item())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
