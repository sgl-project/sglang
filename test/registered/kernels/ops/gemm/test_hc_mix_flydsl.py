import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.hc_mix_flydsl import (
    _MAX_ROWS,
    flydsl_hc_mix_supported,
    flydsl_hc_mix_weights_supported,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=50, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires ROCm gfx950",
)
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950",
    reason="Requires gfx950",
)

# Qwen3.8-Flash-Next's gated hyper-connection.
HC, HS, LOWRANK, EPS = 4, 2560, 320, 1e-6
K = HC * HS


def test_gated_residual_dispatch():
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    # The pinned AITER may predate the kernel; production gates on it the same way.
    hc_mix = pytest.importorskip("aiter.ops.flydsl.hc_mix").hc_mix

    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(hidden_size=2560, hc_lowrank=320)
    layer = GatedResidual(cfg, use_combine=False).cuda().bfloat16()
    x = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with patch("aiter.ops.flydsl.hc_mix.hc_mix", wraps=hc_mix) as call:
            result, (_, normed, _) = layer.mix(x)
            assert call.call_count == 1
        d, u = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
        assert layer._mix_flydsl_ok
        with patch(
            "sglang.srt.layers.hc_mix_flydsl._aiter_hc_mix_available",
            return_value=False,
        ):
            assert not flydsl_hc_mix_weights_supported(d, u, 4)
        t = torch.nn.functional.silu(normed.float() @ d.float().T / 4).bfloat16()
        ref = (
            ((t.float() @ u.float().T).sigmoid() * normed.float())
            .view(4, 4, 2560)
            .mean(1)
            .bfloat16()
        )
        torch.testing.assert_close(result, ref, atol=0.004, rtol=0.01)
        packed = layer._mix_flydsl_weights
        layer.mix(x)
        assert layer._mix_flydsl_weights is packed
        d.mul_(0.5)
        layer.mix(x)
        assert layer._mix_flydsl_weights is not packed
        assert flydsl_hc_mix_supported(x.repeat(32, 1), True)
        for bad_x in [x[:0], x.repeat(33, 1), x.half(), x[:, ::2]]:
            assert not flydsl_hc_mix_supported(bad_x, True)
        assert not flydsl_hc_mix_supported(x, False)
        assert not flydsl_hc_mix_weights_supported(d, u, 5)
        # K only needs whole 64-wide tiles, not the old 2048 multiple.
        k = 64 * 33
        assert flydsl_hc_mix_weights_supported(d[:, :k], u[:k], 4)
        assert not flydsl_hc_mix_weights_supported(d[:, : k - 32], u[: k - 32], 4)
        assert not flydsl_hc_mix_weights_supported(d[:16], u[:, :16], 4)
        assert not flydsl_hc_mix_weights_supported(d.half(), u.half(), 4)


@pytest.mark.parametrize("rows", [32, 96, 128])
def test_gated_residual_dispatch_many_rows(rows):
    # Speculative verify passes carry requests x draft tokens rows.
    if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("Requires gfx950")
    # The pinned AITER may predate the kernel; production gates on it the same way.
    hc_mix = pytest.importorskip("aiter.ops.flydsl.hc_mix").hc_mix

    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(hidden_size=2560, hc_lowrank=320)
    layer = GatedResidual(cfg, use_combine=False).cuda().bfloat16()
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with patch("aiter.ops.flydsl.hc_mix.hc_mix", wraps=hc_mix) as call:
            result, (_, normed, _) = layer.mix(x)
            assert call.call_count == 1
        d, u = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
        t = torch.nn.functional.silu(normed.float() @ d.float().T / 4).bfloat16()
        ref = (
            ((t.float() @ u.float().T).sigmoid() * normed.float())
            .view(rows, 4, 2560)
            .mean(1)
            .bfloat16()
        )
        torch.testing.assert_close(result, ref, atol=0.004, rtol=0.01)


def test_inference_tensor_weight_cache():
    from sglang.srt.layers.hc_mix_flydsl import weight_cache_key

    with torch.inference_mode():
        w = torch.ones(4)
        assert weight_cache_key(w, w)[1] is None


def _make_site(enabled, seed=0):
    """A GatedResidual as Qwen4Exp builds it, with SGLANG_USE_HIP_HC_PREFILL_TRITON on
    or off. The same seed gives the same weights."""
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    with envs.SGLANG_USE_HIP_HC_PREFILL_TRITON.override(enabled):
        site = GatedResidual(cfg).cuda().bfloat16()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in site.named_parameters():
            # The norm's (1 + w) is folded into the projection weights: give it a
            # spread that a dropped or misplaced fold cannot hide in.
            scale = 0.5 if name.startswith("hc_norm") else 0.05
            param.copy_(scale * torch.randn(param.shape, generator=gen, device="cuda"))
    return site


def _make_inputs(rows, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn((rows, HC, HS), generator=gen, device="cuda", dtype=torch.bfloat16)
    # Branch scales 1, 2, 4, 8 (exact in BF16): a rstd taken from another branch shows.
    h *= (2.0 ** torch.arange(HC, device="cuda")).to(torch.bfloat16)[:, None]
    y = torch.randn((rows, HS), generator=gen, device="cuda", dtype=torch.bfloat16)
    return h.view(rows, K), y


def _reference(site, h, y):
    """The site in FP32 from its parameters: mixed [M, HS], inject logits [M, HC] and
    combined [M, K]."""
    rows = h.shape[0]
    hf = h.float().view(rows, HC, HS)
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    normed = (hf * rstd).reshape(rows, K) * (1.0 + site.hc_norm.weight.float())
    t = torch.nn.functional.silu(
        normed @ site.input_mix_weight_down.weight.float().T / HC
    )
    gate = torch.sigmoid(t @ site.input_mix_weight_up.weight.float().T)
    mixed = (gate * normed).view(rows, HC, HS).mean(1)
    logits = normed @ site.block_inject_weight.weight.float().T
    inject = 2 * torch.sigmoid(logits / HC)
    combined = (hf + inject[..., None] * y.float()[:, None, :]).reshape(rows, K)
    return mixed, logits, combined


def _errors(actual, ref):
    """Max and RMS error, relative to the reference's max and RMS."""
    diff = (actual.float() - ref).abs()
    return (
        (diff.max() / ref.abs().max()).item(),
        (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
    )


def _assert_within_twice_unfused_error(actual, unfused, ref):
    (max_err, rms_err), (unfused_max, unfused_rms) = (
        _errors(actual, ref),
        _errors(unfused, ref),
    )
    assert max_err <= 2 * unfused_max, (max_err, unfused_max)
    assert rms_err <= 2 * unfused_rms, (rms_err, unfused_rms)


def _spy_split_k_down(monkeypatch):
    """The row counts of the split-K down launches from now on."""
    from sglang.kernels.ops.gemm import hc_prefill_triton

    launched = []
    split_k_down = hc_prefill_triton._split_k_down

    def spy(hidden, folded_weight, partial):
        launched.append(hidden.shape[0])
        return split_k_down(hidden, folded_weight, partial)

    monkeypatch.setattr(hc_prefill_triton, "_split_k_down", spy)
    return launched


@requires_gfx950
@pytest.mark.parametrize(
    "rows", [_MAX_ROWS + 1, 160, 176, 208, 255, 300, 511, 512, 513, 1024, 1025, 16384]
)
def test_prefill_mix_and_combine_match_reference(monkeypatch, rows):
    # Up to 512 rows (a one-row tail after the 64-row tiles at 129, ragged at 255 and
    # 511, full at 512, verify batches of 4 rows a request among them): the split-K down
    # kernel. 513 is the first row of the bmm path; 1024 and 1025 are the last and the
    # first row of its two up-kernel launch configs (1025 a one-row tail after full
    # blocks); 16384 is the served prefill chunk.
    launched = _spy_split_k_down(monkeypatch)
    site, unfused = _make_site(enabled=True), _make_site(enabled=False)
    h, y = _make_inputs(rows)
    with torch.no_grad():
        want_mixed, want_logits, want_combined = _reference(site, h, y)
        mixed, residuals = site.mix(h)
        combined = site.combine(y, residuals)
        also_combined, next_normed = site.combine_and_normalize(
            y, residuals, unfused.hc_norm
        )
        unfused_mixed, unfused_residuals = unfused.mix(h)
        unfused_combined = unfused.combine(y, unfused_residuals)
        # The unfused path's inject logits: the GEMV its combine kernel forms from the
        # normed residual in BF16.
        unfused_logits = torch.nn.functional.linear(
            unfused_residuals[1].float(), unfused.block_inject_weight.weight.float()
        )
    # The prefill path leaves no normed residual behind.
    assert residuals[1] is None and unfused_residuals[1] is not None
    assert launched == ([rows] if rows <= 512 else [])
    _assert_within_twice_unfused_error(mixed, unfused_mixed, want_mixed)
    _assert_within_twice_unfused_error(residuals[2], unfused_logits, want_logits)
    _assert_within_twice_unfused_error(combined, unfused_combined, want_combined)
    # The inject logits are not gate partials, and the path prepares no next norm.
    assert next_normed is None and torch.equal(also_combined, combined)


@requires_gfx950
@pytest.mark.parametrize("rows", [176, 300])
def test_split_k_down_writes_each_slice_once(rows):
    from sglang.kernels.ops.gemm import hc_prefill_triton

    splits = HC * hc_prefill_triton._SPLITS_PER_BRANCH
    partial_cols = (LOWRANK + HC + 15) // 16 * 16
    gen = torch.Generator(device="cuda").manual_seed(6)
    h = torch.randn((rows, K), generator=gen, device="cuda", dtype=torch.bfloat16)
    weight = (
        0.05 * torch.randn((partial_cols, K), generator=gen, device="cuda")
    ).bfloat16()
    # NaN marks what the kernel leaves unwritten.
    partial = torch.full(
        (splits, rows, partial_cols), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    hc_prefill_triton._split_k_down(hidden=h, folded_weight=weight, partial=partial)
    want = torch.einsum(
        "msk,nsk->smn",
        h.double().view(rows, splits, -1),
        weight.double().view(partial_cols, splits, -1),
    )
    torch.testing.assert_close(partial.double(), want, rtol=2**-8, atol=1e-3)


@requires_gfx950
@pytest.mark.parametrize("rows", [176, 300])
def test_up_kernel_is_bitwise_the_same_on_either_grid(rows):
    from sglang.kernels.ops.gemm import hc_prefill_triton

    gen = torch.Generator(device="cuda").manual_seed(5)

    def randn(*shape, scale=1.0):
        return scale * torch.randn(
            shape, generator=gen, device="cuda", dtype=torch.bfloat16
        )

    t, h, up = randn(rows, LOWRANK), randn(rows, K), randn(K, LOWRANK, scale=0.05)
    norm_weight = randn(K, scale=0.5)
    rstd = 0.5 + torch.rand((rows, HC), generator=gen, device="cuda")
    row_major, xcd_ordered = (
        torch.empty((rows, HS), device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    for out, xcd_order in ((row_major, False), (xcd_ordered, True)):
        hc_prefill_triton._launch_up(
            t=t,
            up=up,
            hidden=h,
            rstd=rstd,
            norm_weight=norm_weight,
            mixed=out,
            hc=HC,
            xcd_order=xcd_order,
        )
    assert torch.equal(xcd_ordered, row_major)


@requires_gfx950
def test_rejects_operands_the_kernels_cannot_read():
    from sglang.kernels.ops.gemm.hc_prefill_triton import (
        hc_prefill_combine,
        hc_prefill_weights_supported,
    )

    site = _make_site(enabled=True)
    down, up, norm_weight = (
        site.input_mix_weight_down.weight,
        site.input_mix_weight_up.weight,
        site.hc_norm.weight,
    )
    assert hc_prefill_weights_supported(down, up, norm_weight, HC)
    # The up kernel reads the up weight and the norm weight row-major.
    column_major_up = up.T.contiguous().T
    assert not hc_prefill_weights_supported(down, column_major_up, norm_weight, HC)
    strided_norm_weight = torch.empty(2 * K, device="cuda", dtype=torch.bfloat16)[::2]
    assert not hc_prefill_weights_supported(down, up, strided_norm_weight, HC)

    def combine(hs):
        return hc_prefill_combine(
            torch.zeros((4, HC * hs), device="cuda", dtype=torch.bfloat16),
            torch.zeros((4, hs), device="cuda", dtype=torch.bfloat16),
            torch.zeros((4, HC), device="cuda", dtype=torch.float32),
            HC,
        )

    # The combine's 512-column blocks are not masked: a hidden size they do not tile
    # would leave its tail columns unwritten.
    assert combine(HS).shape == (4, K)
    with pytest.raises(ValueError):
        combine(HS + 64)


@requires_gfx950
@pytest.mark.parametrize(
    "enabled, rows, deterministic",
    [
        (False, _MAX_ROWS + 1, False),
        (True, _MAX_ROWS, False),
        (True, _MAX_ROWS + 1, True),
    ],
    ids=["flag-off", "flydsl-row-limit", "deterministic-inference"],
)
def test_declines_to_the_unfused_path(monkeypatch, enabled, rows, deterministic):
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    if deterministic:
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_exec",
            lambda: SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=True)
            ),
        )
    site = _make_site(enabled)
    with torch.no_grad():
        _, residuals = site.mix(_make_inputs(rows)[0])
    assert residuals[1] is not None
    if not enabled:
        assert _unsupported_derived_weight_cache_error(site) is None


@requires_gfx950
def test_folded_weights_follow_inplace_updates_and_updates_are_refused():
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    site = _make_site(enabled=True)
    h, y = _make_inputs(_MAX_ROWS + 1)
    with torch.no_grad():
        site.mix(h)
        site.hc_norm.weight.mul_(-1.0)
        mixed, residuals = site.mix(h)
        want_mixed, _, _ = _reference(site, h, y)
    assert residuals[1] is None
    assert _errors(mixed, want_mixed)[0] < 1e-2
    # Loaders write through param.data, which the cache cannot see.
    assert (
        "SGLANG_USE_HIP_HC_PREFILL_TRITON"
        in _unsupported_derived_weight_cache_error(site)
    )


@requires_gfx950
def test_graph_replay_matches_eager_without_recompiling(monkeypatch):
    import triton

    site = _make_site(enabled=True)

    def run(h, y):
        mixed, residuals = site.mix(h)
        return mixed, site.combine(y, residuals)

    # One row count per kernel variant: the split-K path and the bmm path's two up-kernel
    # launch configs.
    capture_rows = (300, 800, 4096)
    with torch.no_grad():
        for rows in capture_rows:
            # The eager warm-up that precedes a capture builds the folded weights.
            h, y = _make_inputs(rows)
            run(h, y)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = run(h, y)
            new_h, new_y = _make_inputs(rows, seed=7)
            h.copy_(new_h)
            y.copy_(new_y)
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(outputs, run(new_h, new_y), strict=True):
                assert torch.equal(got, want), rows
        compiled = []
        monkeypatch.setattr(
            triton.knobs.runtime,
            "jit_post_compile_hook",
            lambda **compile_info: compiled.append(compile_info["repr"]),
        )
        # Each differs in M % 16 from its variant's capture row count.
        for rows in (208, 785, 6001):
            run(*_make_inputs(rows))
        torch.cuda.synchronize()
    assert not compiled, compiled


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
