"""Inkling2's ROCm kernels against float64 / torch references, and their run-to-run
determinism (two ROCm Triton miscompilations showed up only as nondeterminism)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the ROCm kernels and dispatch Inkling2 runs on MI35x
register_amd_ci(est_time=50, suite="base-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires a ROCm GPU",
)

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def _rel(a, b):
    return (
        (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)
    ).item()


def _dequant(w, s):
    if w.dtype == torch.uint8:
        lut = torch.tensor(
            E2M1 + [-v for v in E2M1], device=w.device, dtype=torch.float64
        )
        v = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
        return v * s.double().repeat_interleave(16, -1)
    return w.double() * torch.exp2(s.double() - 127).repeat_interleave(32, -1)


def _sum_splits(partials, dtype):
    """expert_gemm's split-K slabs added in split order, as its consumers sum them."""
    total = partials[0].float()
    for part in partials[1:]:
        total = total + part
    return total.to(dtype)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize(
    "tokens,nexp,num_experts",
    [(1, 16, 64), (13, 16, 64), (300, 16, 64), (512, 48, 384)],
)
def test_expert_gemm(fmt, tokens, nexp, num_experts):
    """NVFP4: products of e2m1, e4m3 scales and NVFP4-fake-quantized activations are
    exact in FP16, so the FP32 accumulator equals float64 up to summation order, from
    one token to the large-batch tile (4,096 rows over 384 experts, MTP verify above
    c16); it repeats bitwise."""
    from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
    from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90

    torch.manual_seed(tokens)
    n, k, top_k, start = 512, 1024, 8, nexp
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w = (torch.randn(nexp, n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s = torch.randint(
            120, 130, (nexp, n, k // 32), device="cuda", dtype=torch.uint8
        )
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = (
        experts_sm90.fake_quantize(x, experts_hip.activation_dtype(w.dtype))[0]
        if fmt == "nvfp4"
        else experts_sm90.fake_quantize_mxfp8(x)
    )
    ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).int()
    ids[0, 0] = start  # at least one row for this rank
    rows = ids.numel()
    cfg = experts_hip.default_config(rows, num_experts, fmt == "nvfp4")
    route = experts_hip.route(ids.flatten(), start, nexp, num_experts, block_m=cfg.bm)

    def run():
        partials = experts_hip.expert_gemm(
            xq, w, s, route, n, top_k, torch.float32, rows, num_experts=num_experts
        )
        return _sum_splits(partials, torch.float32)

    out = run()
    flat = ids.flatten().long()
    local = ((flat >= start) & (flat < start + nexp)).nonzero().flatten()
    wd = _dequant(w, s)
    xr = xq.double()[local // top_k]
    ref = torch.stack([xr[i] @ wd[flat[r] - start].T for i, r in enumerate(local)])
    assert _rel(out[local], ref) < (1e-7 if fmt == "nvfp4" else 1e-6)
    assert torch.equal(run()[local], out[local])


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 40])
def test_consumers_sum_split_k_partials(fmt, tokens):
    """The SwiGLU and combine kernels sum expert_gemm's split-K partials in split order:
    bitwise equal to summing them first, with several splits and with one."""
    from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as marlin_sm90
    from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
    from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90

    torch.manual_seed(tokens)
    nexp, n2, k, top_k, start, caps = 16, 512, 1024, 8, 8, (7.0, 7.0)
    if fmt == "nvfp4":
        w13 = torch.randint(
            0, 256, (nexp, n2, k // 2), device="cuda", dtype=torch.uint8
        )
        s13 = (torch.rand(nexp, n2, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
        w2 = torch.randint(0, 256, (nexp, k, n2 // 4), device="cuda", dtype=torch.uint8)
        s2 = (torch.rand(nexp, k, n2 // 32, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w13 = (torch.randn(nexp, n2, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s13 = torch.randint(
            120, 130, (nexp, n2, k // 32), device="cuda", dtype=torch.uint8
        )
        w2 = (torch.randn(nexp, k, n2 // 2, device="cuda") * 8).to(torch.float8_e4m3fn)
        s2 = torch.randint(
            120, 130, (nexp, k, n2 // 64), device="cuda", dtype=torch.uint8
        )
    nvfp4 = fmt == "nvfp4"
    act = experts_hip.activation_dtype(w13.dtype)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 2
    if nvfp4:
        xq, x_amax = experts_sm90.fake_quantize(x, act)
    else:
        xq = experts_sm90.fake_quantize_mxfp8(x)
    ids = (
        torch.stack(
            [torch.randperm(2 * nexp, device="cuda")[:top_k] for _ in range(tokens)]
        ).int()
        + start // 2
    )  # some rows routed to other ranks
    flat, rows = ids.flatten(), ids.numel()
    gammas = torch.rand(rows, device="cuda")
    route = experts_hip.route(flat, start, nexp, 64)
    out_dtype = torch.float32 if nvfp4 else torch.bfloat16
    w_amax = torch.rand(nexp, device="cuda") + 0.5
    denominator = torch.tensor([3.0], device="cuda")

    def run(reduce):
        def gemm(*args):
            partials = experts_hip.expert_gemm(*args, num_experts=64)
            return _sum_splits(partials, out_dtype) if reduce else partials

        acc = gemm(xq, w13, s13, route, n2, top_k, out_dtype, rows)
        if nvfp4:
            hidden, h_amax = marlin_sm90.swiglu_fake_quantize(
                acc, flat, gammas, (x_amax, w_amax, denominator), start, nexp, top_k,
                caps, act,
            )  # fmt: skip
            scales = (h_amax, w_amax, denominator)
        else:
            hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(
                acc, flat, gammas, start, nexp, caps
            )
            h_amax, scales = None, (None, None, None)
        down = gemm(hidden, w2, s2, route, k, 1, out_dtype, rows)
        sums = marlin_sm90.source_partial(
            down, flat, scales, start, nexp, top_k, round_bf16=not nvfp4
        )
        return hidden, h_amax, sums

    ref, fused = run(True), run(False)
    # Rows routed to other ranks are left unset; compare this rank's rows.
    local = (flat >= start) & (flat < start + nexp)
    bits = torch.int16 if ref[0].element_size() == 2 else torch.int8
    assert torch.equal(ref[0][local].view(bits), fused[0][local].view(bits))
    if nvfp4:
        assert torch.equal(ref[1][local], fused[1][local])
    assert torch.equal(ref[2].view(torch.int32), fused[2].view(torch.int32))


@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("local_heavy", [False, True])
@pytest.mark.parametrize("tokens", [1, 4])
def test_fused_route_matches_separate_kernels(rank, with_bias, local_heavy, tokens):
    """route_topk_metadata (up to four tokens, ROCm decode) is bitwise route_topk
    followed by route_metadata: weights, ids and every routing metadata tensor
    (local_heavy routes all eight ids to this rank's experts, filling the schedule)."""
    from sglang.kernels.ops.moe.inkling2_route import route_metadata, route_topk
    from sglang.kernels.ops.moe.inkling2_route_hip import route_topk_metadata

    experts, num_local, block_m = 384, 48, 16
    one = torch.ones((), device="cuda")
    start = rank * num_local
    for seed in range(2):
        torch.manual_seed(
            1000 * rank + 100 * with_bias + 10 * local_heavy + seed + 7 * tokens
        )
        partials = torch.randn(5, tokens, experts, device="cuda") * 3
        if local_heavy:
            partials[0, :, start : start + num_local] += 30
        bias = torch.randn(experts, device="cuda") * 0.1 if with_bias else None
        _, ids, weights = route_topk(partials, bias, 8, 8.0, one)
        meta = route_metadata(
            ids.flatten(), weights.flatten(), start, num_local, experts, block_m
        )
        f_weights, f_ids, f_meta, quantized = route_topk_metadata(
            partials, bias, 8.0, one, start, num_local, block_m
        )
        assert quantized is None
        assert torch.equal(f_ids, ids)
        assert torch.equal(f_weights.view(torch.int32), weights.view(torch.int32))
        for a, b in zip(f_meta, meta):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize(
    "quant,out_dtype",
    [("nvfp4", torch.bfloat16), ("nvfp4", torch.float16), ("mxfp8", torch.bfloat16)],
)
@pytest.mark.parametrize("width", [4608, 1000])
@pytest.mark.parametrize("tokens", [1, 4])
def test_fused_route_quantizes_activations(quant, out_dtype, width, tokens):
    """route_topk_metadata's extra programs fake-quantize the token's activations
    bitwise as fake_quantize / fake_quantize_mxfp8, with the routing unchanged."""
    from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90
    from sglang.kernels.ops.moe.inkling2_route_hip import route_topk_metadata

    one = torch.ones((), device="cuda")
    for seed in range(3):
        torch.manual_seed(seed)
        partials = torch.randn(5, tokens, 384, device="cuda") * 3
        bias = torch.randn(384, device="cuda") * 0.1
        x = (torch.randn(tokens, width, device="cuda") * (seed + 1)).to(torch.bfloat16)
        x[0, seed * 7] = 0.0
        plain = route_topk_metadata(partials, bias, 8.0, one, 48, 48, 16)
        fused = route_topk_metadata(
            partials, bias, 8.0, one, 48, 48, 16, x, quant, out_dtype
        )
        for a, b in zip(
            (plain[0], plain[1], *plain[2]), (fused[0], fused[1], *fused[2])
        ):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        xq, x_amax = fused[3]
        if quant == "nvfp4":
            ref, ref_amax = experts_sm90.fake_quantize(x, out_dtype)
            assert torch.equal(x_amax.view(torch.int32), ref_amax.view(torch.int32))
        else:
            ref = experts_sm90.fake_quantize_mxfp8(x)
            assert x_amax is None
        assert xq.dtype == ref.dtype
        assert torch.equal(xq.view(torch.int16), ref.view(torch.int16))


def test_router_matches_exact_sigmoid_selection():
    """route_topk (the gate's top-k at every row count) selects the experts of torch's
    sigmoid plus bias, with renorm_topk_logits_scaled's weights for them."""
    from sglang.kernels.ops.moe.inkling2_route import route_topk
    from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled

    torch.manual_seed(0)
    one = torch.ones((), device="cuda")
    for m in (1, 7, 64, 512):
        logits = torch.randn(m, 384, device="cuda") * 3
        bias = torch.randn(384, device="cuda") * 0.1
        ids_ref = (torch.sigmoid(logits) + bias).topk(8, dim=-1).indices
        _, ids, w = route_topk(logits[None], bias, 8, 8.0, one)
        ids = ids.long()
        assert torch.equal(ids.sort(-1).values, ids_ref.sort(-1).values)
        w_ref, _ = renorm_topk_logits_scaled(logits, ids, 0, 8.0, one)
        assert (w - w_ref).abs().max().item() < 1e-6


def test_dynamic_conv_projection_partials():
    """On ROCm the global layers' dynamic conv projections (4 outputs per rank) take
    the paired split-K partials path, which uses_k_groups otherwise leaves to
    hipBLASLt; the summed partials match a float64 reference."""
    from sglang.kernels.ops.gemm.inkling2_linear import (
        linear_k_group_partials_pair,
        uses_k_groups,
    )

    for m in (1, 7, 64, 128):
        torch.manual_seed(m)
        x, x2 = (torch.randn(2, m, 4608, device="cuda") * 2).bfloat16()
        w, w2 = (torch.randn(2, 4, 4608, device="cuda") * 0.02).bfloat16()
        assert not uses_k_groups(x, w)
        assert uses_k_groups(x, w, partials=True)
        partials = linear_k_group_partials_pair(x, w, x2, w2)
        assert partials.shape == (2, 5, m, 4)
        for p, a, b in zip(partials, (x, x2), (w, w2)):
            ref = a.double() @ b.double().T
            assert (p.double().sum(0) - ref).abs().max().item() < 1e-3
    assert not uses_k_groups(
        torch.randn(129, 4608, device="cuda").bfloat16(), w, partials=True
    )


@pytest.mark.parametrize(
    "quant,out_dtype",
    [("nvfp4", torch.bfloat16), ("nvfp4", torch.float16), ("mxfp8", torch.bfloat16)],
)
@pytest.mark.parametrize("tokens", [2, 128])
def test_route_metadata_quantize_matches_separate(quant, out_dtype, tokens):
    """route_metadata_quantize is bitwise route_metadata followed by the rows' fake
    quantization (fake_quantize / fake_quantize_mxfp8), for 16 to 1024 ids."""
    from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from sglang.kernels.ops.moe.inkling2_route_hip import route_metadata_quantize

    torch.manual_seed(tokens + len(quant))
    ids = torch.stack(
        [torch.randperm(384, device="cuda")[:8] for _ in range(tokens)]
    ).int()
    weights = torch.rand(tokens, 8, device="cuda")
    x = (torch.randn(tokens, 4608, device="cuda") * 2).bfloat16()
    for start in (0, 144):
        meta = route_metadata(ids.flatten(), weights.flatten(), start, 48, 384, 16)
        f_meta, (xq, x_amax) = route_metadata_quantize(
            ids.flatten(), weights.flatten(), start, 48, 384, 16, x, quant, out_dtype
        )
        for a, b in zip(f_meta, meta):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        if quant == "nvfp4":
            ref, ref_amax = experts_sm90.fake_quantize(x, out_dtype)
            assert torch.equal(x_amax.view(torch.int32), ref_amax.view(torch.int32))
        else:
            ref = experts_sm90.fake_quantize_mxfp8(x)
            assert x_amax is None
        assert torch.equal(xq.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("gather", [8, 1])
@pytest.mark.parametrize("tokens", [1, 32, 256, 1024])
def test_expert_gemm_mfma_order(monkeypatch, fmt, tokens, gather):
    """Weights in MFMA order: 16-row tiles run the Gluon kernel and the others the
    Triton tile on that order, both bitwise the checkpoint layout's partials in
    every default_config band (8-8,192 rows over 64 experts), and run to run."""
    from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
    from sglang.kernels.ops.moe import inkling2_experts_sm90 as experts_sm90

    if not experts_hip.MFMA_ORDER:
        pytest.skip("MFMA-order experts need gfx950 and Gluon")
    gluon = experts_hip.experts_gluon
    torch.manual_seed(tokens + gather)
    # w2's 2,304-wide K splits into 3-tile loops (the unrolled NVFP4 path) at <= 64 rows.
    k = 1024 if gather == 8 else 2304
    nexp, n, top_k, start, num_experts = 16, 512, 8, 16, 64
    nvfp4 = fmt == "nvfp4"
    if nvfp4:
        w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w = (torch.randn(nexp, n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s = torch.randint(
            120, 130, (nexp, n, k // 32), device="cuda", dtype=torch.uint8
        )
    rows = tokens * top_k
    x = torch.randn(rows // gather, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = (
        experts_sm90.fake_quantize(x, experts_hip.activation_dtype(w.dtype))[0]
        if nvfp4
        else experts_sm90.fake_quantize_mxfp8(x)
    )
    ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).int()
    cfg = experts_hip.default_config(rows, num_experts, nvfp4, mfma_order=True)
    route = experts_hip.route(ids.flatten(), start, nexp, num_experts, block_m=cfg.bm)
    calls = []
    kernel = gluon.grouped_gemm
    monkeypatch.setattr(gluon, "grouped_gemm", lambda *a: calls.append(a) or kernel(*a))
    args = (route, n, gather, torch.float32, rows)
    plain = experts_hip.expert_gemm(xq, w, s, *args, num_experts=num_experts)
    ordered = gluon.mfma_order_weights(w), gluon.mfma_order_scales(s)
    order = [
        experts_hip.expert_gemm(
            xq,
            *ordered,
            *args,
            num_experts=num_experts,
            mfma_order=True,
        )  # fmt: skip
        for _ in range(4)
    ]
    assert len(calls) == 4 * (cfg.bm == 16)
    flat = ids.flatten()
    local = ((flat >= start) & (flat < start + nexp)).nonzero().flatten()
    assert local.numel() > 0
    assert all(torch.equal(plain[:, local], o[:, local]) for o in order)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
