"""The Hopper NVFP4 W4A16 MoE op against an FP32 reference with exact NVFP4 dequantization.

Bounds: relative L2 over all local routes < 1% and every single route < 2%; the op lands at about 0.4% and 0.5% (BF16
rounding of the activations, the intermediate and the output). A route that is skipped, written to another row, or
computed with a dropped K chunk, the wrong expert or the wrong scale is off by 9% (one of 112 chunks) to 100%, so the
per-route bound catches a single bad route. Per-route calls fill the output with a sentinel first and check that rows
of non-local routes hold zeros.
"""

import math
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.nvfp4_w4a16_sm90 import nvfp4_w4a16_sm90_moe
from sglang.kernels.ops.quantization.nvfp4_sm90_repack import (
    Nvfp4Sm90Experts,
    repack_modelopt_nvfp4_moe_for_sm90,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="needs an sm_90 GPU",
)

E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
# name: (hidden, intermediate, local experts, global experts, top_k)
SHAPES = {
    "deepseek_ep8": (7168, 2048, 32, 256, 8),
    "qwen3_30b": (
        2048,
        768,
        128,
        128,
        8,
    ),  # I / 64 = 12 chunks, not a multiple of the 4-stage ring
    "qwen3_next_ep8": (2048, 512, 64, 512, 10),
    "eplb36": (4096, 1536, 36, 288, 8),
    # Per-GPU expert widths that are not a multiple of 128: Qwen3.8-Flash-Next (640 wide) at TP2, TP4 and TP8, the
    # whole 640, DeepSeek-V4.1-Flash (2304 wide) at TP8 and TP4, and a DeepSeek-R1-sized layer 64 columns past 2048.
    "q38_tp2": (2560, 320, 64, 64, 10),
    "q38_tp4": (2560, 160, 64, 64, 10),
    "q38_tp8": (2560, 80, 64, 64, 10),
    "q38_ep": (2560, 640, 64, 64, 10),
    "q38_tp4_full": (2560, 160, 512, 512, 10),
    # A lone half block: 64 columns (Qwen3-Next's 512 at TP8) and 48, padded to 64.
    "w64": (2048, 64, 64, 64, 10),
    "w48": (2048, 48, 64, 64, 10),
    "dsv41_tp8": (5120, 288, 48, 48, 6),
    "dsv41_tp4": (5120, 576, 48, 48, 6),
    "r1_2112": (7168, 2112, 32, 256, 8),
    # Clamped SwiGLU layers (swiglu_limit 10): GLM-5.3-Flash at TP4 and TP8.
    "glm53f_tp4": (4096, 512, 36, 288, 8),
    "glm53f_tp8": (4096, 256, 36, 288, 8),
    # Large experts where decode_fused rotates its intermediate rows through only 2 shared-memory buffers when one
    # expert takes every token: Muse Spark (EP8) at 4 tokens, Phi-3.5-MoE at 3.
    "muse": (4096, 4096, 32, 256, 8),
    "phi35": (4096, 6400, 16, 16, 2),
}
SENTINEL = 7.0
_LAYERS = {}


def _dequant(w, sf, gs=None):
    tab = torch.tensor(E2M1, dtype=torch.float32, device=w.device)
    v = torch.stack((tab[(w & 15).long()], tab[(w >> 4).long()]), dim=-1).flatten(-2)
    v = v * sf.float().repeat_interleave(16, dim=-1)
    return v if gs is None else v * gs.float()


def _checkpoint(E, H, I, case, seed):
    """ModelOpt-layout experts. case: normal; subnormal (half the block scales are E4M3 subnormals); small (global
    scale 1, all block scales subnormal); gateup (each expert's up global scale is its gate scale times 0.1 .. 10);
    loud (gate and up 20x larger than normal, so a SwiGLU clamp of 10 binds on a large share of elements); loudgateup
    (loud, with each expert's up global scale its gate scale times 0.5 .. 2)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ck = {}
    for name, (n, k) in (("w13", (2 * I, H)), ("w2", (H, I))):
        ck[name + "_weight"] = torch.randint(
            0, 256, (E, n, k // 2), dtype=torch.uint8, generator=g, device="cuda"
        )
        u = torch.rand((E, n, k // 16), generator=g, device="cuda")
        if case == "subnormal":
            sf = torch.exp2(-9.0 + 6.0 * u)
        elif case == "small":
            sf = torch.exp2(-9.0 + 3.0 * u)
        else:
            sf = torch.exp2(6.5 + u).clamp(8.0, 448.0)
        sf = sf.to(torch.float8_e4m3fn)
        gs = 1.0 / math.sqrt(8.6 * float(sf.float().pow(2).mean()) * k)
        gs = torch.full((E,), 1.0 if case == "small" else gs, device="cuda")
        if case in ("normal", "gateup"):
            gs = gs * torch.exp(0.5 * torch.randn((E,), generator=g, device="cuda"))
        if case in ("loud", "loudgateup") and name == "w13":
            gs = gs * 20.0
        ck[name + "_weight_scale"] = sf
        ck[name + "_weight_scale_2"] = gs
    gate = ck["w13_weight_scale_2"]
    spread = {"gateup": 3.3, "loudgateup": 1.0}.get(case)
    up = (
        gate * torch.exp2(torch.empty_like(gate).uniform_(-spread, spread, generator=g))
        if spread
        else gate
    )
    ck["w13_weight_scale_2"] = torch.stack((gate, up), dim=1).contiguous()
    return ck


def _layer(shape, case="normal"):
    if (shape, case) not in _LAYERS:
        if len(_LAYERS) > 3:
            _LAYERS.clear()
            torch.cuda.empty_cache()
        H, I, E, _, _ = SHAPES[shape]
        ck = _checkpoint(E, H, I, case, seed=len(shape) * 7 + len(case))
        _LAYERS[(shape, case)] = (ck, repack_modelopt_nvfp4_moe_for_sm90(**ck))
    return _LAYERS[(shape, case)]


def _routing(T, topk, E, Eg, kind="skew", seed=0, remote="minus_one"):
    """Distinct expert ids per token, local experts 0 .. E - 1 the most popular (the busiest EP rank). kind: skew, hot
    (half of the tokens route to local expert 3), empty (every other local expert gets no route), remote (no local
    route). remote: other ranks' routes as -1 (SGLang's local id map) or as their global id."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pop = (
        torch.sort(torch.randn(Eg, generator=g, device="cuda"), descending=True).values
        * 1.2
    )
    s = pop - torch.log(
        -torch.log(torch.rand((T, Eg), generator=g, device="cuda").clamp_min(1e-20))
    )
    if kind == "hot":
        s[: T // 2 + 1, 3] = float("inf")
    elif kind == "empty":
        s[:, 1:E:2] = float("-inf")
    elif kind == "remote":
        s[:, :E] = float("-inf")
    elif kind == "shared":  # every token routes to local expert 3
        s[:, 3] = float("inf")
    ids = torch.topk(s, topk, dim=1).indices.to(torch.int32)
    if remote == "minus_one":
        ids = torch.where(ids < E, ids, -1)
    return ids.contiguous()


def _reference(ck, x, ids, E, swiglu_limit=None, clamped=None, variant=None):
    """FP32 per-route outputs [R, H] and the local route mask; gate and up rows each with their own global scale. With
    swiglu_limit L the expert computes SiLU(min(gate, L)) * clamp(up, -L, L), and clamped (a list) collects per expert
    the share of gate and of up values the clamp changed."""
    flat = ids.reshape(-1).long()
    local = (flat >= 0) & (flat < E)
    out = torch.zeros((flat.numel(), x.shape[1]), dtype=torch.float32, device=x.device)
    I = ck["w2_weight"].shape[2] * 2
    gs13 = ck["w13_weight_scale_2"]
    for e in range(E):
        sel = (local & (flat == e)).nonzero().squeeze(1)
        if sel.numel() == 0:
            continue
        w13 = _dequant(ck["w13_weight"][e], ck["w13_weight_scale"][e])
        h = x.float()[sel // ids.shape[1]] @ w13.T
        gate, up = h[:, :I] * gs13[e, 0], h[:, I:] * gs13[e, 1]
        if swiglu_limit is not None:
            if clamped is not None:
                clamped.append(
                    (
                        float((gate > swiglu_limit).float().mean()),
                        float((up.abs() > swiglu_limit).float().mean()),
                    )
                )
            up = up.clamp(-swiglu_limit, swiglu_limit)
            if (
                variant == "after_silu"
            ):  # a wrong clamp, for tests that must tell it apart
                act = torch.nn.functional.silu(gate).clamp(max=swiglu_limit)
            elif (
                variant == "both_sides"
            ):  # a wrong clamp, for tests that must tell it apart
                act = torch.nn.functional.silu(gate.clamp(-swiglu_limit, swiglu_limit))
            else:
                act = torch.nn.functional.silu(gate.clamp(max=swiglu_limit))
            inter = act * up
        else:
            inter = torch.nn.functional.silu(gate) * up
        out[sel] = (
            inter
            @ _dequant(
                ck["w2_weight"][e], ck["w2_weight_scale"][e], ck["w2_weight_scale_2"][e]
            ).T
        )
    return out, local


def _assert_close(y, ref, rows):
    d, r = y[rows].float() - ref[rows], ref[rows]
    if r.numel() == 0:
        return
    assert bool(torch.isfinite(d).all())
    rel = float(d.norm() / r.norm())
    worst = float((d.norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)).max())
    assert rel < 0.01 and worst < 0.02, (rel, worst)


def _check(
    shape,
    T,
    kind="skew",
    case="normal",
    variant="auto",
    ctas=0,
    remote="minus_one",
    swiglu_limit=None,
):
    H, I, E, Eg, topk = SHAPES[shape]
    ck, experts = _layer(shape, case)
    x = torch.randn(
        (T, H), generator=torch.Generator(device="cuda").manual_seed(T), device="cuda"
    ).bfloat16()
    ids = _routing(T, topk, E, Eg, kind, seed=T, remote=remote)
    y = torch.full((T * topk, H), SENTINEL, dtype=torch.bfloat16, device="cuda")
    kw = {} if swiglu_limit is None else {"swiglu_limit": swiglu_limit}
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant, ctas=ctas, **kw)
    clamped = []
    ref, local = _reference(ck, x, ids, E, swiglu_limit, clamped)
    _assert_close(y, ref, local)
    assert bool((y[~local] == 0).all()), "a row of a non-local route is not zero"
    return ck, x, ids, ref, local, clamped


@pytest.mark.parametrize("T", [1, 7, 8, 9, 33, 255, 753, 2047, 2048, 4096, 8191, 16384])
@pytest.mark.parametrize("shape", ["deepseek_ep8", "qwen3_30b"])
def test_token_counts(shape, T):
    """Every token count from 1 to 16384 on the auto kernel choice. Fails on: a stride that uses H for I (DeepSeek
    has H 7168 != I 2048), an off-by-one in an expert's last token tile, an expert the planner drops, the one-CTA
    kernel's unrolled producer loop taken when GEMM2's 12 chunks do not fill whole ring passes (Qwen3-30B)."""
    _check(shape, T)


@pytest.mark.parametrize(
    "variant,T,ctas",
    [
        ("pair", 9, 0),
        ("pair", 753, 0),
        ("single", 16384, 0),
        ("single", 753, 2),
        ("pair", 2048, 3),
    ],
)
def test_variants_and_ctas(variant, T, ctas):
    """Each kernel outside the range auto picks it for, and with 2 or 3 CTAs so every CTA wraps its 8-slot tile
    queue. Fails on: both CTAs of a pair taking the same row block, a queue parity error on wrap, a claim counter that
    skips or repeats a tile."""
    _check("deepseek_ep8", T, variant=variant, ctas=ctas)


@pytest.mark.parametrize(
    "kind,remote",
    [
        ("hot", "minus_one"),
        ("empty", "minus_one"),
        ("remote", "minus_one"),
        ("skew", "global"),
    ],
)
def test_routing_kinds(kind, remote):
    """One expert taking half of the tokens, empty experts, no local route, other ranks' routes as global ids.
    Fails on: a planner that shifts tile offsets past zero-route experts, -1 or a global id read as local, a hang
    or a write when no tile exists."""
    for T in (33, 4096):
        _check("deepseek_ep8", T, kind=kind, remote=remote)


@pytest.mark.parametrize("case", ["subnormal", "small", "gateup"])
def test_scale_cases(case):
    """E4M3 subnormal block scales, global scale 1 with all-subnormal block scales, and gate and up global scales
    that differ. Fails on: a scale decode that flushes subnormals, a repack that keeps only the gate's global scale,
    a fold applied as gate / up."""
    for T in (9, 4096):
        _check("deepseek_ep8", T, case=case)


@pytest.mark.parametrize("shape", ["qwen3_next_ep8", "eplb36"])
def test_other_shapes(shape):
    """Top 10 and a non-power-of-two local expert count. Fails on: top 8 hard-coded in the gather (route / 8), an
    expert table sized for a power of two."""
    for T in (1, 255, 4096):
        _check(shape, T)


@pytest.mark.parametrize("T", [1, 9, 40, 100, 753, 4096])
@pytest.mark.parametrize(
    "shape",
    [
        "q38_tp2",
        "q38_tp4",
        "q38_tp8",
        "q38_ep",
        "dsv41_tp8",
        "dsv41_tp4",
        "r1_2112",
        "w64",
        "w48",
    ],
)
def test_expert_widths(shape, T):
    """Per-GPU expert widths of 48, 64, 80, 160, 288, 320, 576 and 2112 (and 640, which the old op took, as a
    control) at every MMA width (test_expert_widths_cover_every_tile_width checks the token counts reach each one).
    64, 160, 320, 576 and 2112 end GEMM1 on a half-height row block and give GEMM2 an odd number of K chunks, 48 and
    64 have nothing but that half block and one K chunk, and 48, 80 and 288 also pad up to 48 zero columns. Fails on:
    the old 128-column refusal, a half block that writes its missing 64 columns into the next row of the intermediate,
    a dropped or doubled last K chunk, padding that is not exactly zero, a half block's weights read at a full block's
    stride (every later expert reads the wrong bytes)."""
    _check(shape, T)


def _tile_widths(ids, E):
    """The MMA widths the op's planner gives the token tiles of this routing (as plan_block and mx_w do)."""
    counts = torch.bincount(ids[(ids >= 0) & (ids < E)].long(), minlength=E).tolist()
    widths = set()
    for c in counts:
        if c == 0:
            continue
        pn = (c + 127) // 128
        tsz = ((c + pn - 1) // pn + 31) // 32 * 32
        for t in range(pn):
            nv = min(tsz, c - t * tsz)
            widths.add(
                (nv + 7) // 8 * 8
                if nv <= 16
                else 32
                if nv <= 32
                else 64
                if nv <= 64
                else 128
            )
    return widths


def test_expert_widths_cover_every_tile_width():
    """test_expert_widths' token counts give every new width tiles on each consumer path: mma.sync (8 or 16 tokens),
    the narrow wgmma paths (32 and 64) and the full one (128)."""
    for shape in ("q38_tp2", "q38_tp4", "dsv41_tp8", "r1_2112", "w64", "w48"):
        _, _, E, Eg, topk = SHAPES[shape]
        widths = set()
        for T in (1, 9, 40, 100, 753, 4096):
            widths |= _tile_widths(_routing(T, topk, E, Eg, seed=T), E)
        assert widths & {8, 16} and {32, 64, 128} <= widths, (shape, sorted(widths))


@pytest.mark.parametrize(
    "shape,variant,T,ctas",
    [
        ("q38_tp2", "single", 4096, 2),
        ("q38_tp4", "single", 753, 3),
        ("dsv41_tp4", "auto", 16384, 0),
        ("r1_2112", "auto", 16384, 0),
    ],
)
def test_expert_widths_queue_wrap_and_pair_fallback(shape, variant, T, ctas):
    """The new widths with 2 or 3 CTAs, so every CTA wraps its tile queue across half blocks and one-chunk stages, and
    at 16384 tokens, where auto picks CTA pairs for full-block shapes and must stay on one CTA per tile here. Fails on:
    a queue or ring parity error after a one-chunk stage, the pair kernel taken on a half block."""
    _check(shape, T, variant=variant, ctas=ctas)


def test_expert_widths_full_qwen38_layer():
    """Qwen3.8-Flash-Next at TP4 with its real expert count and top 10: 512 experts, 160 columns each."""
    for T in (1, 64, 2048):
        _check("q38_tp4_full", T)


def test_pair_kernel_refuses_a_half_block():
    _, experts = _layer("q38_tp2")
    x = torch.randn((64, 2560), device="cuda").bfloat16()
    ids = _routing(64, 10, 64, 64)
    with pytest.raises(Exception, match="CTA pairs"):
        nvfp4_w4a16_sm90_moe(x, ids, experts, variant="pair")


def test_repack_size_matches_the_checkpoint_up_to_64_columns():
    """A width that is a multiple of 64 repacks into exactly the checkpoint's bytes, and 80 and 288 add only the zero
    columns up to the next multiple of 64. Fails on: padding to 128, which would waste a fifth of the bytes at 320."""
    for shape in ("q38_tp2", "q38_tp4", "q38_tp8", "dsv41_tp8", "dsv41_tp4"):
        H, I, E, _, _ = SHAPES[shape]
        ck, experts = _layer(shape)
        ckpt = sum(
            ck[n].numel() * ck[n].element_size()
            for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
        )
        packed = experts.w13.numel() + experts.w2.numel()
        assert packed * I == ckpt * ((I + 63) // 64 * 64), (shape, packed, ckpt)


@pytest.mark.parametrize("T", [1, 9, 100, 4096])
@pytest.mark.parametrize("shape", ["glm53f_tp4", "glm53f_tp8", "dsv41_tp8"])
def test_swiglu_limit(shape, T):
    """swiglu_limit 10 on GLM-5.3-Flash and DeepSeek-V4.1-Flash shapes, with gate and up large enough that the clamp
    binds on a large share of elements, against the clamped FP32 reference. Fails on: no clamp (the unclamped
    reference is checked to be far from the clamped one), up clamped on one side only."""
    ck, x, ids, ref, local, clamped = _check(shape, T, case="loud", swiglu_limit=10.0)
    gate_share = sum(g for g, _ in clamped) / len(clamped)
    up_share = sum(u for _, u in clamped) / len(clamped)
    assert gate_share > 0.2 and up_share > 0.4, (gate_share, up_share)
    unclamped, _ = _reference(ck, x, ids, SHAPES[shape][2])
    d = (unclamped[local] - ref[local]).norm() / ref[local].norm()
    assert float(d) > 0.5, float(d)


@pytest.mark.parametrize("T", [9, 4096])
@pytest.mark.parametrize("shape", ["glm53f_tp8", "q38_tp2"])
def test_swiglu_limit_where_the_clamp_shape_matters(shape, T):
    """A limit of 1 on normal-sized gate and up, where SiLU(gate) is far from linear: clamping the gate from below as
    well, or clamping after SiLU instead of before, moves the output by several percent here (at a limit of 10 both
    stay inside the error bound). q38_tp2 runs the clamp in a half-block layer. Fails on: either of those, and on the
    clamp missing in the half block's copy-out."""
    ck, x, ids, ref, local, clamped = _check(shape, T, swiglu_limit=1.0)
    E = SHAPES[shape][2]
    gate_share = sum(g for g, _ in clamped) / len(clamped)
    up_share = sum(u for _, u in clamped) / len(clamped)
    assert gate_share > 0.1 and up_share > 0.2, (gate_share, up_share)
    rows = ref[local]
    for wrong in ("both_sides", "after_silu"):
        alt = _reference(ck, x, ids, E, 1.0, variant=wrong)[0][local]
        assert float((alt - rows).norm() / rows.norm()) > 0.03, wrong


@pytest.mark.parametrize("T", [9, 4096])
def test_swiglu_limit_with_unequal_gate_and_up_scales(T):
    """The repack folds up / gate into GEMM2's scale, so the up values GEMM1 stages carry the gate's scale, and the
    clamp must still bind on the true up value. Fails on: the staged up value clamped at the plain limit."""
    _check("glm53f_tp8", T, case="loudgateup", swiglu_limit=10.0)


def test_half_block_clamped_layer_under_cuda_graph_and_reload():
    """The half-block, clamped kernel captured in a CUDA graph, replayed with new routings and inputs, then after an
    in-place reload of different weights. Fails on: a host-side branch on the routing in the new kernel choice, a
    reload that misses the up / gate ratio (the clamp keeps the old weights' limit)."""
    H, I, E, Eg, topk = SHAPES["q38_tp2"]
    ck1 = _checkpoint(E, H, I, "normal", 21)
    ck2 = _checkpoint(E, H, I, "loudgateup", 22)
    experts = repack_modelopt_nvfp4_moe_for_sm90(**ck1)
    T = 300
    x = torch.randn((T, H), device="cuda").bfloat16()
    ids = _routing(T, topk, E, Eg, seed=4)
    w = torch.rand((T, topk), device="cuda")
    out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
    run = lambda: nvfp4_w4a16_sm90_moe(
        x, ids, experts, topk_weights=w, out=out, swiglu_limit=1.0
    )
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for rep, ck in enumerate((ck1, ck2)):
        if rep == 1:
            assert repack_modelopt_nvfp4_moe_for_sm90(**ck, out=experts) is experts
        x.copy_(torch.randn((T, H), device="cuda").bfloat16())
        ids.copy_(_routing(T, topk, E, Eg, kind="hot", seed=30 + rep))
        graph.replay()
        eager = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, swiglu_limit=1.0)
        torch.cuda.synchronize()
        assert float((out.float() - eager.float()).abs().max()) <= 1e-2 * float(
            eager.float().abs().max()
        )
        ref, local = _reference(ck, x, ids, E, 1.0)
        ref_sum = (ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)).sum(
            1
        )
        _assert_close(out, ref_sum, slice(None))


def test_swiglu_limit_far_above_every_value_is_bitwise_the_plain_op():
    _, experts = _layer("q38_tp4")
    for T in (9, 753):
        x = torch.randn((T, 2560), device="cuda").bfloat16()
        ids = _routing(T, 10, 64, 64, seed=T)
        a = nvfp4_w4a16_sm90_moe(x, ids, experts)
        b = nvfp4_w4a16_sm90_moe(x, ids, experts, swiglu_limit=1e30)
        assert torch.equal(a, b)


def test_weighted_sum():
    """The combined output against the reference weighted sum, with other ranks' routes as -1 and a routed scaling
    factor. Fails on: a combine that reads rows of non-local routes (they hold NaN here), a weight applied to the
    wrong slot, the scale dropped."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (1, 300, 4096):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=5 + T)
        w = torch.rand((T, topk), device="cuda")
        ref, local = _reference(ck, x, ids, E)
        ref_sum = (ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)).sum(
            1
        ) * 2.5
        # The op's [R, H] workspaces reuse these freed blocks, so a read of a non-local route's row gives NaN.
        poison = [
            torch.full((T * topk, H), float("nan"), dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        del poison
        out = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, scale=2.5)
        _assert_close(out, ref_sum, slice(None))


def test_cuda_graph_capture_and_replay():
    """Capture at three token counts (the one-block plan, and each GEMM kernel), replay with fresh routings and
    inputs, compare with eager calls and the reference. Fails on: a host-side plan or sync (capture fails, or replay
    reuses the captured routing), a buffer outside the graph's pool, meta not zeroed per call (the second replay
    claims no tiles)."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    for T in (3, 753, 4096):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=1)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, out=out)
        for rep, kind in enumerate(("skew", "hot", "empty")):
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(_routing(T, topk, E, Eg, kind=kind, seed=77 + rep))
            graph.replay()
            eager = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w)
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E)
            ref_sum = (
                ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
            ).sum(1)
            _assert_close(out, ref_sum, slice(None))
            assert float((out.float() - eager.float()).abs().max()) <= 1e-2 * float(
                eager.float().abs().max()
            )


def test_reload_in_place_keeps_graphs_valid():
    """Repack new weights into the same buffers and replay a graph captured before; the output must follow the new
    weights. Fails on: a repack that allocates new buffers on reload (the captured tensor maps keep reading the old
    weights) or leaves stale bytes."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck1, ck2 = _checkpoint(E, H, I, "normal", 11), _checkpoint(E, H, I, "gateup", 12)
    experts = repack_modelopt_nvfp4_moe_for_sm90(**ck1)
    T = 255
    x = torch.randn((T, H), device="cuda").bfloat16()
    ids = _routing(T, topk, E, Eg, seed=3)
    y = torch.empty((T * topk, H), dtype=torch.bfloat16, device="cuda")
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvfp4_w4a16_sm90_moe(x, ids, experts, out=y)
    assert repack_modelopt_nvfp4_moe_for_sm90(**ck2, out=experts) is experts
    graph.replay()
    torch.cuda.synchronize()
    ref, local = _reference(ck2, x, ids, E)
    _assert_close(y, ref, local)


def test_argument_checks():
    """Bad shapes and dtypes raise before any kernel runs. Fails on: a missing check (the call reads out of
    bounds)."""
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    _, ex = _layer("deepseek_ep8")
    x = torch.randn((16, H), device="cuda").bfloat16()
    ids = _routing(16, topk, E, Eg)
    bad = [
        (x.half(), ids, ex),
        (x[:, :4096].contiguous(), ids, ex),
        (x, ids.long(), ex),
        (x, ids[:8], ex),
        (x, ids, Nvfp4Sm90Experts(ex.w13[:16], ex.w2, ex.gs13, ex.gs2, ex.gsr, H, I)),
        (x, ids, Nvfp4Sm90Experts(ex.w13, ex.w2, ex.gs13, ex.gs2, ex.gsr, H, I + 128)),
    ]
    for args in bad:
        with pytest.raises(Exception):
            nvfp4_w4a16_sm90_moe(*args)


_SPIN = r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void __launch_bounds__(1024, 1) spin(volatile int* flag, int* started) {
  extern __shared__ char s[];
  if (threadIdx.x == 0) {
    s[0] = 1;
    atomicAdd_system(started, 1);
    while (*flag == 0) __nanosleep(2000);
  }
  __syncthreads();
}
void launch(torch::Tensor flag, torch::Tensor started, int64_t blocks) {
  cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024);
  spin<<<(int)blocks, 1024, 160 * 1024, at::cuda::getCurrentCUDAStream()>>>(
      (volatile int*)flag.data_ptr<int>(), started.data_ptr<int>());
}
"""


@pytest.mark.parametrize(
    "variant,T,free",
    [("single", 753, 16), ("pair", 753, 16)]
    + [("decode_pdl", T, free) for T in (1, 2) for free in (16, 66, 92)],
)
def test_progress_while_another_kernel_holds_sms(variant, T, free):
    """Another stream's kernel holds all but 16 SMs (an overlapped side stream, MPS, a green context); the op must
    finish on the SMs it has. Fails on: a persistent kernel that gives CTA b tile b statically, so the resident CTAs
    spin on GEMM2 tiles whose GEMM1 tiles belong to CTAs that are still queued. decode_fused is a cooperative launch
    of one CTA per SM and only starts once every SM is free, so it is left out here (see _decode_variant). decode_pdl
    runs with 16, 66 and 92 SMs free. Fails on (decode_pdl): GEMM2 consumer groups sharing ring slots, where a group
    reads a slot's parity from two fills back while the previous fill is still in flight (at 1 token with 66 or 92 SMs
    free it faulted with a device exception)."""
    from torch.utils.cpp_extension import load_inline

    spin = load_inline(
        "nvfp4_sm90_test_spin",
        cpp_sources="void launch(torch::Tensor, torch::Tensor, int64_t);",
        cuda_sources=_SPIN,
        functions=["launch"],
    )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    x = torch.randn((T, H), device="cuda").bfloat16()
    ids = _routing(T, topk, E, Eg, seed=3)
    y = torch.empty((T * topk, H), dtype=torch.bfloat16, device="cuda")
    nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant)
    torch.cuda.synchronize()
    flag = torch.zeros(1, dtype=torch.int32).pin_memory()
    started = torch.zeros(1, dtype=torch.int32).pin_memory()
    safety = threading.Timer(
        60, lambda: flag.fill_(1)
    )  # never leave the blocker spinning
    safety.start()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            spin.launch(flag, started, sms - free)
        t0 = time.time()
        while int(started[0]) < sms - free and time.time() - t0 < 10:
            time.sleep(0.01)
        assert int(started[0]) == sms - free, "the blocker did not become resident"
        done = torch.cuda.Event()
        with torch.cuda.stream(torch.cuda.Stream()):
            nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant=variant)
            done.record()
        t0 = time.time()
        while not done.query() and time.time() - t0 < 10:
            time.sleep(0.01)
        finished = done.query()
    finally:
        flag.fill_(1)
        safety.cancel()
    torch.cuda.synchronize()
    assert finished, f"no progress on the {free} free SMs within 10s"
    ref, local = _reference(ck, x, ids, E)
    _assert_close(y, ref, local)


# ---- the decode kernels (1 to 4 tokens): one launch ("decode_fused") and two launches ("decode_pdl") ----


def _whole_blocks(shape):
    """The decode kernels take only padded widths of whole 128-column blocks (no half block)."""
    return (SHAPES[shape][1] + 63) // 64 % 2 == 0


def _fused_fits(shape, T):
    """decode_fused's shared-memory bound (Phi-3.5-MoE's 6400-wide experts fit up to 3 tokens)."""
    H, I = SHAPES[shape][:2]
    return (43008 + T * (2 * H + 16) + 1023) // 1024 * 1024 + 2 * 36864 + 2 * T * (
        2 * I + 16
    ) <= 232448


_DECODE_CASES = (
    [
        ("decode_fused", s, T)
        for s in SHAPES
        if _whole_blocks(s)
        for T in (1, 2, 3, 4)
        if _fused_fits(s, T)
    ]
    + [
        ("decode_pdl", s, T)
        for s in ("qwen3_30b", "qwen3_next_ep8")
        for T in (1, 2, 3, 4)
    ]
    + [("decode_pdl", s, T) for s in ("deepseek_ep8", "eplb36") for T in (1, 2)]
)


@pytest.mark.parametrize("variant,shape,T", _DECODE_CASES)
def test_decode_token_counts(variant, shape, T):
    """Each decode kernel at every token count it serves, per-route outputs. Fails on: a tagged word written to the
    wrong intermediate column (16-row GEMM1 items), a stale word taken as current, a slot's rows read from another
    slot's shared-memory rows, a GEMM2 group whose slots share a stage but not a column offset, rotating row buffers
    refilled before every consumer warp is done (DeepSeek at 4 tokens), a two-launch GEMM2 reading the intermediate
    before its GEMM1 grid finished."""
    _check(shape, T, variant=variant)


@pytest.mark.parametrize("variant", ["decode_fused", "decode_pdl"])
@pytest.mark.parametrize(
    "kind,remote",
    [
        ("hot", "minus_one"),
        ("empty", "minus_one"),
        ("remote", "minus_one"),
        ("skew", "global"),
        ("remote", "global"),
    ],
)
def test_decode_routing_kinds(variant, kind, remote):
    """One expert taking most routes, empty experts, no local route at all. Fails on: a hang when no slot exists, a
    slot with more than one route read with the wrong row stride."""
    for T in (1, 2):
        _check("qwen3_30b", T, kind=kind, remote=remote, variant=variant)


@pytest.mark.parametrize("variant", ["decode_fused", "decode_pdl"])
def test_decode_weighted_sum_repeat_and_graph(variant):
    """The weighted sum against the reference, then a CUDA graph replayed with new routings: every result must match
    the reference and an eager call bitwise. Eager calls, warm-up and capture share one stream, so the graph runs on
    the eager calls' live epoch, which must advance by one per call, replays included. Fails on: an epoch that does
    not advance (a later call takes an earlier call's words), weights or scales staged from the wrong route or slot, a
    host-side routing decision."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    H, I, E, Eg, topk = SHAPES["qwen3_30b"]
    ck, experts = _layer("qwen3_30b")
    s = torch.cuda.Stream()

    def epoch():
        ws = op._fused_ws[(torch.cuda.current_device(), s.cuda_stream)][0]
        return int(ws[:1].view(torch.int32)[1])

    for T in (1, 2):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=5 + T)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")
        with torch.cuda.stream(s):
            for _ in range(
                3
            ):  # warm-up on the capture stream: the graph starts on a running epoch
                nvfp4_w4a16_sm90_moe(
                    x, ids, experts, topk_weights=w, out=out, scale=2.5, variant=variant
                )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            nvfp4_w4a16_sm90_moe(
                x, ids, experts, topk_weights=w, out=out, scale=2.5, variant=variant
            )
        torch.cuda.synchronize()
        e0 = epoch() if variant == "decode_fused" else None
        for rep in range(4):
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(
                _routing(T, topk, E, Eg, kind=("skew", "hot")[rep % 2], seed=40 + rep)
            )
            torch.cuda.synchronize()
            with torch.cuda.stream(s):
                graph.replay()
                eager = nvfp4_w4a16_sm90_moe(
                    x, ids, experts, topk_weights=w, scale=2.5, variant=variant
                )
            torch.cuda.synchronize()
            if e0 is not None:
                assert e0 > 1 and epoch() == e0 + 2 * (rep + 1), (e0, epoch())
            ref, local = _reference(ck, x, ids, E)
            ref_sum = (
                ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
            ).sum(1) * 2.5
            _assert_close(out, ref_sum, slice(None))
            assert torch.equal(out, eager)


def test_decode_dispatch():
    """auto takes the one-launch kernel for large experts and the two-launch one for small ones, only up to
    DECODE_MAX_TOKENS tokens, and gives the same result as the variant it picks. Fails on: a dispatch that sends a
    shape to a kernel that refuses it, or past the token range."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    assert op._decode_variant(1, 8, 32, 7168, 2048) == "decode_fused"
    assert op._decode_variant(2, 16, 256, 6144, 256) == "decode_pdl"
    assert op._decode_variant(3, 24, 32, 7168, 2048) == "auto"
    for shape in ("deepseek_ep8", "qwen3_30b"):
        H, I, E, Eg, topk = SHAPES[shape]
        ck, experts = _layer(shape)
        x = torch.randn((2, H), device="cuda").bfloat16()
        ids = _routing(2, topk, E, Eg, seed=11)
        w = torch.rand(
            (2, topk), device="cuda"
        )  # the weighted sum (decode_pdl's per-route sums use atomics)
        a = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w)
        b = nvfp4_w4a16_sm90_moe(
            x,
            ids,
            experts,
            topk_weights=w,
            variant=op._decode_variant(2, 2 * topk, E, H, I),
        )
        assert torch.equal(a, b)


# ---- the decode kernels with swiglu_limit, and widths that end on a half block ----

_DECODE_VARIANTS = ["decode_fused", "decode_pdl"]


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
@pytest.mark.parametrize("shape", ["glm53f_tp4", "glm53f_tp8"])
@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_decode_swiglu_limit(variant, shape, T):
    """swiglu_limit 10 in both decode kernels on GLM-5.3-Flash's TP4 and TP8 shapes, gate and up loud enough that the
    clamp binds on a large share of elements, against the clamped FP32 reference. Fails on: a decode kernel that
    applies plain SwiGLU (the unclamped reference is checked to be far from the clamped one), up clamped on one side
    only, the limit not reaching the kernel."""
    ck, x, ids, ref, local, clamped = _check(
        shape, T, case="loud", variant=variant, swiglu_limit=10.0
    )
    gate_share = sum(g for g, _ in clamped) / len(clamped)
    up_share = sum(u for _, u in clamped) / len(clamped)
    assert gate_share > 0.2 and up_share > 0.4, (gate_share, up_share)
    unclamped, _ = _reference(ck, x, ids, SHAPES[shape][2])
    assert float((unclamped[local] - ref[local]).norm() / ref[local].norm()) > 0.5


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
@pytest.mark.parametrize("T", [1, 2])
def test_decode_swiglu_limit_where_the_clamp_shape_matters(variant, T):
    """A limit of 1 on normal-sized gate and up in both decode kernels. Fails on: the gate clamped from below too, or
    the clamp applied after SiLU (each moves the output by several percent here)."""
    shape = "glm53f_tp8"
    ck, x, ids, ref, local, clamped = _check(
        shape, T, variant=variant, swiglu_limit=1.0
    )
    gate_share = sum(g for g, _ in clamped) / len(clamped)
    up_share = sum(u for _, u in clamped) / len(clamped)
    assert gate_share > 0.1 and up_share > 0.2, (gate_share, up_share)
    rows = ref[local]
    for wrong in ("both_sides", "after_silu"):
        alt = _reference(ck, x, ids, SHAPES[shape][2], 1.0, variant=wrong)[0][local]
        assert float((alt - rows).norm() / rows.norm()) > 0.03, wrong


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
@pytest.mark.parametrize("T", [1, 2])
def test_decode_swiglu_limit_with_unequal_gate_and_up_scales(variant, T):
    """Up carries the gate's global scale inside the decode kernels too (GEMM2's scale holds up / gate), so the up
    clamp must bind at lim / |gsr|. Fails on: up clamped at the plain limit."""
    _check("glm53f_tp8", T, case="loudgateup", variant=variant, swiglu_limit=10.0)


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
def test_decode_swiglu_limit_far_above_every_value_is_bitwise_the_plain_kernel(variant):
    """With a limit no value reaches, the clamped decode kernels match the plain ones bitwise, on the weighted sum
    (decode_pdl's per-route path adds partial sums with shared atomics, so its order and its last bit vary run to
    run). Fails on: a clamp that changes the order or precision of the SwiGLU arithmetic."""
    H, I, E, Eg, topk = SHAPES["glm53f_tp8"]
    _, experts = _layer("glm53f_tp8")
    for T in (1, 2):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=T)
        w = torch.rand((T, topk), device="cuda")
        a = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, variant=variant)
        b = nvfp4_w4a16_sm90_moe(
            x, ids, experts, topk_weights=w, variant=variant, swiglu_limit=1e30
        )
        assert torch.equal(a, b)


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
def test_decode_clamped_weighted_sum_under_cuda_graph(variant):
    """The clamped decode kernel captured in a CUDA graph with the weighted sum, replayed with new routings, against
    the clamped reference and an eager call bitwise. Fails on: the clamp lost on replay."""
    H, I, E, Eg, topk = SHAPES["glm53f_tp8"]
    ck, experts = _layer("glm53f_tp8", "loud")
    for T in (1, 2):
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=7 + T)
        w = torch.rand((T, topk), device="cuda")
        out = torch.empty((T, H), dtype=torch.bfloat16, device="cuda")

        def run(o):
            return nvfp4_w4a16_sm90_moe(
                x,
                ids,
                experts,
                topk_weights=w,
                out=o,
                variant=variant,
                swiglu_limit=10.0,
            )

        run(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(out)
        for rep in range(3):
            x.copy_(torch.randn((T, H), device="cuda").bfloat16())
            ids.copy_(
                _routing(T, topk, E, Eg, kind=("skew", "hot")[rep % 2], seed=50 + rep)
            )
            graph.replay()
            eager = run(None)
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E, 10.0)
            ref_sum = (
                ref.view(T, topk, H) * (w * local.view(T, topk)).unsqueeze(-1)
            ).sum(1)
            _assert_close(out, ref_sum, slice(None))
            assert torch.equal(out, eager)


@pytest.mark.parametrize("variant", _DECODE_VARIANTS)
@pytest.mark.parametrize("shape", ["q38_tp2", "dsv41_tp8", "w64"])
def test_decode_kernels_refuse_a_half_block(variant, shape):
    """The decode kernels read whole 256-row GEMM1 blocks, so a layer whose padded width ends on a half block must
    raise instead of reading past the expert's rows. Fails on: a width check that misses the half block."""
    H, I, E, Eg, topk = SHAPES[shape]
    _, experts = _layer(shape)
    x = torch.randn((1, H), device="cuda").bfloat16()
    ids = _routing(1, topk, E, Eg)
    with pytest.raises(Exception, match="half block"):
        nvfp4_w4a16_sm90_moe(x, ids, experts, variant=variant)


@pytest.mark.parametrize(
    "shape", ["q38_tp2", "q38_tp4", "dsv41_tp8", "q38_tp8", "glm53f_tp4", "glm53f_tp8"]
)
@pytest.mark.parametrize("T", [1, 2])
def test_decode_dispatch_half_blocks_and_clamp(shape, T):
    """auto at 1 and 2 tokens with swiglu_limit: half-block widths (Qwen3.8 TP2 and TP4, DeepSeek-V4.1-Flash TP8) stay
    on the grouped kernels, whole-block ones (Qwen3.8 TP8 padded to 128, GLM-5.3-Flash TP4 and TP8) take decode_pdl,
    and every result matches the clamped FP32 reference and the pinned kernel bitwise. Fails on: a half-block layer
    dispatched to a decode kernel (it raises), a clamped layer that loses its clamp on the decode path."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    H, I, E, Eg, topk = SHAPES[shape]
    pick = op._decode_variant(T, T * topk, E, H, (I + 63) // 64 * 64)
    assert pick == ("decode_pdl" if _whole_blocks(shape) else "auto"), pick
    ck, x, ids, ref, local, clamped = _check(shape, T, swiglu_limit=1.0)
    assert sum(g for g, _ in clamped) > 0, "the clamp never binds"
    experts = _layer(shape)[1]
    w = torch.rand(
        (T, topk), device="cuda"
    )  # the weighted sum: deterministic on every kernel
    a = nvfp4_w4a16_sm90_moe(x, ids, experts, topk_weights=w, swiglu_limit=1.0)
    b = nvfp4_w4a16_sm90_moe(
        x,
        ids,
        experts,
        topk_weights=w,
        swiglu_limit=1.0,
        variant="single" if pick == "auto" else pick,
    )
    assert torch.equal(a, b)


# ---- decode_fused: two row buffers, the shared-memory bound, the workspace under graphs, the epoch wrap ----


def _fused_row_buffers(T, H, I, routes, mmax):
    """decf_k's choice of intermediate row buffers (0: one row per local route, no rotation)."""
    rowi = 2 * I + 16
    oring = (43008 + T * (2 * H + 16) + 1023) // 1024 * 1024
    avail = 232448 - oring - 2 * 36864
    if routes * rowi > avail or (routes * (I // 4) + 511) // 512 > 16:
        return min(16, avail // (mmax * rowi))
    return 0


@pytest.mark.parametrize("shape,T", [("muse", 4), ("phi35", 3)])
def test_decode_fused_with_two_row_buffers(shape, T):
    """Every token routes to one expert, so decode_fused rotates its intermediate rows through 2 buffers, fewer than
    its 3 fetcher pairs. Fails on: a pair that refills a buffer while the consumers still read it (a parity wait that
    cannot tell phase c from c - 2 passes two phases early), seen as mixed rows in GEMM2."""
    H, I, E, Eg, topk = SHAPES[shape]
    ck, experts = _layer(shape)
    for rep in range(12):
        ids = _routing(T, topk, E, Eg, kind="shared", seed=60 + rep)
        local = int(((ids >= 0) & (ids < E)).sum())
        assert _fused_row_buffers(T, H, I, local, T) == 2, local
        x = torch.randn((T, H), device="cuda").bfloat16()
        y = torch.full((T * topk, H), SENTINEL, dtype=torch.bfloat16, device="cuda")
        nvfp4_w4a16_sm90_moe(x, ids, experts, out=y, variant="decode_fused")
        w = torch.rand((T, topk), device="cuda")
        o = nvfp4_w4a16_sm90_moe(
            x, ids, experts, topk_weights=w, variant="decode_fused"
        )
        ref, lm = _reference(ck, x, ids, E)
        _assert_close(y, ref, lm)
        ref_sum = (ref.view(T, topk, H) * (w * lm.view(T, topk)).unsqueeze(-1)).sum(1)
        _assert_close(o, ref_sum, slice(None))


def test_decode_dispatch_respects_decode_fused_shared_memory():
    """auto takes decode_fused only where its shared memory fits, by the same bound the kernel checks. Mixtral-8x7B
    at TP1 (I 14336) fits at 1 token and not at 2, where auto keeps the grouped kernels; widths 12160 and 12288 sit on
    either side of the bound at 2 tokens. Fails on: auto sending a 2-token Mixtral call to decode_fused, which raises,
    or the Python bound drifting from the kernel's."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    H, E, topk = 4096, 8, 2
    assert op._decode_variant(1, topk, E, H, 14336) == "decode_fused"
    assert op._decode_variant(2, 2 * topk, E, H, 14336) == "auto"
    for I in (12160, 12288, 14336):
        ck = _checkpoint(E, H, I, "normal", seed=I)
        experts = repack_modelopt_nvfp4_moe_for_sm90(**ck)
        for T in (1, 2):
            x = torch.randn((T, H), device="cuda").bfloat16()
            ids = _routing(T, topk, E, E, seed=T)
            ref, local = _reference(ck, x, ids, E)
            _assert_close(nvfp4_w4a16_sm90_moe(x, ids, experts), ref, local)
            fits = op._decode_fused_fits(T, H, I)
            assert fits == (T == 1 or I == 12160), (I, T)
            if fits:
                _assert_close(
                    nvfp4_w4a16_sm90_moe(x, ids, experts, variant="decode_fused"),
                    ref,
                    local,
                )
            else:
                with pytest.raises(Exception, match="shared memory"):
                    nvfp4_w4a16_sm90_moe(x, ids, experts, variant="decode_fused")
        del ck, experts
        torch.cuda.empty_cache()


def test_decode_fused_graphs_captured_without_warm_up():
    """Two graphs captured on a fresh stream with no eager call before, every cached workspace then filled with words
    that carry the epoch the next call would use, and the second graph replayed first. Fails on: a workspace made
    inside the first capture (zeroed only by that graph's replays) that the second graph reuses, so its first replay
    takes stale words as current."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    s = torch.cuda.Stream()
    graphs = {}
    for T in (
        2,
        1,
    ):  # the larger first, so the second capture could reuse the first one's buffer
        x = torch.randn((T, H), device="cuda").bfloat16()
        ids = _routing(T, topk, E, Eg, seed=70 + T)
        out = torch.empty((T * topk, H), dtype=torch.bfloat16, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            nvfp4_w4a16_sm90_moe(x, ids, experts, out=out, variant="decode_fused")
        graphs[T] = (g, x, ids, out)
    torch.cuda.synchronize()
    for chain in op._fused_ws.values():
        while chain is not None:
            ws = chain[0]
            st = ws[:1].view(torch.int32)
            k = int(st[1])
            st[0], st[1] = 0, k
            ws[2:] = (
                (k + 1) << 32
            ) | 0x3F803F80  # BF16 1.0 pairs tagged with the next epoch
            chain = chain[1]
    torch.cuda.synchronize()
    for T in (1, 2):
        g, x, ids, out = graphs[T]
        with torch.cuda.stream(s):
            g.replay()
        torch.cuda.synchronize()
        ref, local = _reference(ck, x, ids, E)
        _assert_close(out, ref, local)


def test_decode_fused_epoch_wrap():
    """The call that takes the last 32-bit epoch zeroes every tagged word and the next call starts at epoch 1. The
    words are filled with values tagged epoch 1 first, as a word last written 2^32 - 1 calls ago would be. Fails on: a
    wrap that keeps those words, so the call after it takes them as current."""
    import sglang.kernels.ops.moe.nvfp4_w4a16_sm90 as op

    H, I, E, Eg, topk = SHAPES["deepseek_ep8"]
    ck, experts = _layer("deepseek_ep8")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.randn((2, H), device="cuda").bfloat16()
        ids = _routing(2, topk, E, Eg, seed=80)
        nvfp4_w4a16_sm90_moe(x, ids, experts, variant="decode_fused")
        torch.cuda.synchronize()
        ws = op._fused_ws[(torch.cuda.current_device(), s.cuda_stream)][0]
        st = ws[:1].view(torch.int32)
        st[0], st[1] = 0, -2  # the last call took epoch 2^32 - 2
        ws[2:] = (1 << 32) | 0x3F803F80
        for rep, (want_epoch, zeroed) in enumerate(((0, True), (1, False))):
            ids = _routing(2, topk, E, Eg, seed=81 + rep)
            y = nvfp4_w4a16_sm90_moe(x, ids, experts, variant="decode_fused")
            torch.cuda.synchronize()
            ref, local = _reference(ck, x, ids, E)
            _assert_close(y, ref, local)
            assert int(st[1]) == want_epoch and int(st[0]) == 0, (
                int(st[0]),
                int(st[1]),
            )
            if zeroed:
                assert bool((ws[2:] == 0).all()), "the wrap left tagged words"
