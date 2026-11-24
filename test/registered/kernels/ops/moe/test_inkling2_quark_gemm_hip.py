"""Grouped GEMM over packed Quark expert weights (gfx950)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.inkling2_quark_kernel_test_utils import (
    TOP_K,
    build_route,
    onehot_check,
    routed_case,
    run_gemm,
    synth_mxfp4,
    synth_uint4,
    triton_gap,
)

register_amd_ci(est_time=90, suite="base-b-test-1-gpu-small-amd-mi35x")

GFX95 = is_gfx95_supported()
TRITON_GAP = triton_gap("dot_scaled", "join") if GFX95 else ""
pytestmark = [
    pytest.mark.skipif(not GFX95, reason="needs an AMD gfx95x GPU"),
    pytest.mark.skipif(bool(TRITON_GAP), reason=TRITON_GAP),
]

N = 2304
KS = (2304, 1152)  # w13: K = hidden; w2: K = intermediate
ONEHOT_REPS = 20  # a miscompile that depends on timing shows in some launches only
TOL = 1e-6  # relative L2 error of the FP32 output against float64
SCALES = {"bf16": torch.bfloat16, "f32": torch.float32}

if GFX95:  # a failing import must fail the file on the machines that run it
    from sglang.kernels.ops.moe import inkling2_quark_gemm_hip as G

    def _configs(fmt):
        unique = []
        for _, cfg in G._TABLES[fmt]:
            if cfg not in unique:
                unique.append(cfg)
        return unique

    MX_CONFIGS = _configs(G.MXFP4)
    U4_CONFIGS = _configs(G.UINT4_AWQ)
else:
    MX_CONFIGS = U4_CONFIGS = []


def cfg_id(cfg):
    return f"{cfg.bm}x{cfg.bn}x{cfg.bk}s{cfg.split_k}w{cfg.warps}t{cfg.stages}"


def check_exact(bad, total, what):
    assert bad == 0, (
        f"{bad} of {total} dequantized weights differ from the oracle: {what}"
    )


def _fixtures(experts, seed):
    """Packed tensors of both formats at both Ks (UINT4 with BF16 and F32 scales)."""
    data = {}
    for k in KS:
        data["mxfp4", k] = synth_mxfp4(experts, N, k, seed=seed + k, device="cuda")
        for name, dtype in SCALES.items():
            data["uint4", k, name] = synth_uint4(
                experts, N, k, seed=seed + 2 + k, scale_dtype=dtype, device="cuda"
            )
    return data


@pytest.fixture(scope="module")
def onehot():
    return _fixtures(2, 11)


@pytest.fixture(scope="module")
def routed():  # 16 experts: top-8 routing needs at least 8
    return _fixtures(16, 21)


def case(fixture, fmt_name, k, scale="bf16"):
    """(format id, packed tensors) of one format at K; scale picks the UINT4 scales."""
    if fmt_name == "mxfp4":
        return G.MXFP4, fixture["mxfp4", k]
    return G.UINT4_AWQ, fixture["uint4", k, scale]


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("cfg", MX_CONFIGS, ids=cfg_id)
def test_onehot_mxfp4_every_table_config(onehot, k, cfg):
    bad, total = onehot_check(
        G.MXFP4, onehot["mxfp4", k], k, N, cfg, experts=(0, 1), reps=ONEHOT_REPS
    )
    check_exact(bad, total, f"MXFP4 K={k} {tuple(cfg)}")


@pytest.mark.parametrize("scale", list(SCALES))
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("cfg", U4_CONFIGS, ids=cfg_id)
def test_onehot_uint4_every_table_config(onehot, k, scale, cfg):
    bad, total = onehot_check(
        G.UINT4_AWQ,
        onehot["uint4", k, scale],
        k,
        N,
        cfg,
        experts=(0, 1),
        reps=ONEHOT_REPS,
    )
    check_exact(bad, total, f"UINT4_AWQ {scale} scales K={k} {tuple(cfg)}")


@pytest.mark.parametrize("start", [16, 48])
@pytest.mark.parametrize("kind", ["w13", "w2"])
@pytest.mark.parametrize("fmt_name", ["mxfp4", "uint4"])
def test_expert_parallel_slice(routed, fmt_name, kind, start):
    """This rank holds 16 of 64 experts: only the rows routed to them are computed."""
    fmt, data = case(routed, fmt_name, 2304 if kind == "w13" else 1152)
    for routing in ("product", "launcher"):
        r = routed_case(
            fmt,
            data,
            kind,
            97,
            num_experts=64,
            start=start,
            seed=start,
            routing=routing,
        )
        assert 0 < r["local_rows"] < r["rows"] and r["rel_l2"] <= TOL, (routing, r)


@pytest.mark.parametrize("fmt_name", ["mxfp4", "uint4"])
def test_schedule_is_rebuilt_when_the_route_has_another_block_m(routed, fmt_name):
    """A route built for BM 16 used by a GEMM that runs with BM 64 (and the reverse):
    the launcher must not reuse the schedule of the wrong tile height."""
    fmt, data = case(routed, fmt_name, 1152)
    for gemm_bm, route_bm in ((64, 16), (16, 64)):
        cfg = G.QConfig(gemm_bm, 128, 128, 1, 4, 2)
        r = routed_case(
            fmt, data, "w2", 100, num_experts=16, config=cfg, seed=gemm_bm,
            route_block_m=route_bm,
        )  # fmt: skip
        assert r["rel_l2"] <= TOL and r["local_rows"] == r["rows"], (
            gemm_bm,
            route_bm,
            r,
        )


@pytest.mark.parametrize(
    "fmt_name, k, requested, expected_splits",
    [
        ("mxfp4", 2304, 3, 3),  # BK 256: 9 tiles
        ("mxfp4", 2304, 6, 3),  # a requested 6 is clamped to a divisor of 9
        ("mxfp4", 2304, 2, 1),
        ("mxfp4", 1152, 9, 9),  # BK 256 is halved to 128: 9 tiles
        ("uint4", 2304, 9, 9),  # BK 64: 36 tiles
        ("uint4", 1152, 3, 3),
    ],
)
def test_split_k_slabs(routed, fmt_name, k, requested, expected_splits):
    """split_k > 1 returns one FP32 slab of partial sums per K range."""
    fmt, data = case(routed, fmt_name, k)
    if fmt_name == "mxfp4":
        cfg = G.QConfig(16, 128, 256, requested, 4, 2)
    else:
        cfg = G.QConfig(16, 128, 64, requested, 2, 2)
    r = routed_case(
        fmt,
        data,
        "w2",
        40,
        num_experts=16,
        config=cfg,
        seed=requested,
        out_dtype=torch.bfloat16,
    )
    assert r["splits"] == expected_splits, r
    if expected_splits > 1:
        assert r["dtype"] == torch.float32  # slabs stay FP32 whatever out_dtype says
        assert r["rel_l2"] <= TOL and max(r["slab_rel_l2"]) <= TOL, r
    else:
        assert r["dtype"] == torch.bfloat16 and r["rel_l2"] <= 4e-3, (
            r
        )  # BF16 rounding only


def test_no_routed_rows_gives_an_empty_result(routed):
    ids = torch.zeros(0, TOP_K, dtype=torch.int32, device="cuda")
    route = build_route("product", ids, 0, 16, 16, 16)
    x = torch.zeros(0, 2304, dtype=torch.bfloat16, device="cuda")
    y = run_gemm(
        G.MXFP4, routed["mxfp4", 2304], x, route, N, TOP_K, torch.float32, 0,
        num_experts=16,
    )  # fmt: skip
    assert y.shape == (1, 0, N)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
