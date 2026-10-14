"""Check the mono FFN launch's numerics and graph replay against SGLang's ops on gfx950 (torchrun, TP2 or TP4)."""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.kernels.ops.mono.deepseek_v41_hip.spec import NUMERICS

HIDDEN, HC, MIX = 5120, 4, 24
E, TOPK, SCALE, LIMIT = 384, 6, 1.5, 10.0
INTER, PAD = 2304, 128
WIDTHS = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48)  # every decode graph width <= 48
RMS_EPS, HC_EPS = 1e-20, 1e-6
MIX_RTOL = NUMERICS.bound("ffn.mix.rel")
FP4 = [
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


def cos(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


def bf16_ulps(a, b):
    """(max difference in bf16 ulps, share of elements that differ) of two bf16 tensors."""
    a, b = a.float(), b.float()
    _, e = torch.frexp(torch.maximum(a.abs(), b.abs()))
    ulps = (a - b).abs() / torch.ldexp(torch.ones_like(a), e - 8)
    return ulps.max().item(), (a != b).float().mean().item()


def hc_post_fp64(a, res, post, comb):
    """fp64 hc_post of the reduced bf16 attention output ``a``: (each element, its largest term's |.|)."""
    terms = torch.cat(
        [
            (post.double()[:, :, None] * a.double()[:, None, :])[..., None],
            (comb.double()[..., None] * res.double()[:, :, None, :]).permute(
                0, 2, 3, 1
            ),
        ],
        -1,
    )
    return terms.sum(-1), terms.abs().amax(-1)  # fmt: skip


def term_ulps(out, ref, big):
    """Max |out - ref| in bf16 ulps of ``big``: a near-cancelling sum has no meaningful ulp of its own."""
    _, e = torch.frexp(big)
    return (
        ((out.double() - ref).abs() / torch.ldexp(torch.ones_like(big), e - 8))
        .max()
        .item()
    )


# ---------------------------------------------------------------- MX helpers (the launch's rules)
def code_ceil(v):
    """Biased exponent of the smallest power of two >= v (v > 0), clamped to [lo, 254]."""
    bits = v.float().contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()


def mxfp8_qdq(x, rule):
    """Per-32 MXFP8 quantize-dequantize: "aiter" (routed experts) or "vllm" (shared expert) scale rule."""
    g = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    if rule == "aiter":
        code = code_ceil(amax.clamp_min(1e-10) * (1.0 / 448)).clamp(1, 254)
    else:
        code = code_ceil(amax.clamp_min(torch.finfo(torch.float32).tiny) / 448.0).clamp(
            0, 254
        )
    inv = torch.exp2(127.0 - code.float())
    q = (g * inv).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q / inv).reshape(x.shape)


def deq_fp4(packed, scale):
    """[N, K / 2] packed e2m1 (even index low nibble), [N, K / 32] E8M0 -> [N, K] f32."""
    lut = torch.tensor(FP4, device=packed.device)
    p = packed.view(torch.uint8).long()
    v = torch.stack([lut[p & 15], lut[p >> 4]], -1).flatten(-2)
    return v * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(
        32, -1
    )


def deq_fp8_block(w, scale):
    """[N, K] e4m3 with [N / 32, K / 32] E8M0 blocks -> f32."""
    s = torch.exp2(scale.view(torch.uint8).float() - 127)
    return w.float() * s.repeat_interleave(32, 0).repeat_interleave(32, 1)


def bf(x):
    return x.to(torch.bfloat16).float()


# ---------------------------------------------------------------- weights
def make_weights(tp, rank, dev, seed=0):
    """One layer's tensors on this rank in SGLang's loaded layout, plus the copies the golden model reads."""
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    g = torch.Generator(device=dev).manual_seed(1000 + 7919 * seed)
    gs = torch.Generator(device=dev).manual_seed(2001 + rank + 7919 * seed)
    inter = INTER // tp
    padded = -(-inter // PAD) * PAD
    u8 = torch.uint8

    def rnd(*shape, s=1.0, gen=g):
        return torch.randn(*shape, generator=gen, device=dev) * s

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=gs, device=dev, dtype=u8)

    def e8m0(*shape, gen=gs):
        return torch.randint(118, 123, shape, generator=gen, device=dev, dtype=u8)

    w = dict(
        hc_fn=rnd(MIX, HC * HIDDEN, s=0.01),
        hc_scale=rnd(3, s=0.5).abs() + 0.5,
        hc_base=rnd(MIX, s=0.1),
        norm=(1 + rnd(HIDDEN, s=0.1)).to(torch.bfloat16),
        gate_w=rnd(E, HIDDEN, s=0.02).to(torch.bfloat16),
        # SGLang keeps the router bias in bf16
        bias=rnd(E, s=0.02).to(torch.bfloat16),
    )
    # routed experts: [gate; up] each padded to ``padded`` rows, w2's K padded alike (zero weights)
    w13 = torch.zeros(E, 2 * padded, HIDDEN // 2, dtype=u8, device=dev)
    s13 = torch.zeros(E, 2 * padded, HIDDEN // 32, dtype=u8, device=dev)
    for half in (0, padded):
        w13[:, half : half + inter] = nib(E, inter, HIDDEN // 2)
        s13[:, half : half + inter] = e8m0(E, inter, HIDDEN // 32)
    w2 = torch.zeros(E, HIDDEN, padded // 2, dtype=u8, device=dev)
    s2 = torch.zeros(E, HIDDEN, padded // 32, dtype=u8, device=dev)
    w2[..., : inter // 2] = nib(E, HIDDEN, inter // 2)
    s2[..., : inter // 32] = e8m0(E, HIDDEN, inter // 32)
    w.update(w13_raw=w13, s13_raw=s13, w2_raw=w2, s2_raw=s2)
    fp4 = torch.float4_e2m1fn_x2
    # Fp8MoEMethod's gfx950 processing (SGLANG_USE_AITER_MOE_GU_ITLV, not AITER_FORCE_A8W4)
    w["w13"] = shuffle_weight(w13.view(fp4), is_guinterleave=True, gate_up=True)
    w["w13_s"] = shuffle_scale(s13.view(-1, s13.shape[-1]), E, True, True)
    w["w2"] = shuffle_weight(w2.view(fp4), is_guinterleave=True, gate_up=False)
    w["w2_s"] = shuffle_scale(s2.view(-1, s2.shape[-1]), E, True, False)
    # the shared expert: this rank's shards of the checkpoint's FP8, 32 x 32 E8M0 blocks
    sgu = torch.randn(2 * inter, HIDDEN, generator=gs, device=dev).clamp(-6, 6)
    w.update(
        sgu=sgu.to(torch.float8_e4m3fn),
        sgu_s=e8m0(2 * inter // 32, HIDDEN // 32),
        sw2=torch.randn(HIDDEN, inter, generator=gs, device=dev)
        .clamp(-6, 6)
        .to(torch.float8_e4m3fn),
        sw2_s=e8m0(HIDDEN // 32, inter // 32),
    )
    return w, inter, padded


def mono_weights(w):
    from sglang.kernels.ops.mono.deepseek_v41_hip.runner import FFNWeights

    return FFNWeights(
        hc_ffn_fn=w["hc_fn"], hc_ffn_scale=w["hc_scale"], hc_ffn_base=w["hc_base"],
        ffn_norm=w["norm"], gate_w=w["gate_w"], bias=w["bias"].float(), w13=w["w13"],
        w13_s=w["w13_s"].view(torch.uint8), w2=w["w2"], w2_s=w["w2_s"].view(torch.uint8),
        sgu=w["sgu"], sgu_s=w["sgu_s"], sw2=w["sw2"], sw2_s=w["sw2_s"],
    )  # fmt: skip


# ---------------------------------------------------------------- golden MoE (this rank's partial)
def routing(x, w):
    logits = x.float() @ w["gate_w"].float().T
    sc = F.softplus(logits).sqrt()
    ids = (sc + w["bias"].float()).topk(TOPK, dim=1).indices
    wt = sc.gather(1, ids)
    return ids, wt * (SCALE / wt.sum(1, keepdim=True))


def golden_partial(x, w, inter, padded):
    """This rank's MoE partial of bf16 rows ``x``: routed (top-k order, bf16 at every add) + shared."""
    ids, wts = routing(x, w)
    xr, xs = mxfp8_qdq(x, "aiter"), mxfp8_qdq(x, "vllm")
    per_k = torch.zeros(
        x.shape[0], TOPK, HIDDEN, device=x.device
    )  # a row's contributions, top-k order
    for e in ids.unique().tolist():
        rows, ks = (ids == e).nonzero(as_tuple=True)
        h = xr[rows] @ deq_fp4(w["w13_raw"][e], w["s13_raw"][e]).T
        gate, up = h[:, :padded], h[:, padded:]
        mid = bf(F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT))
        y = mxfp8_qdq(mid, "aiter") @ deq_fp4(w["w2_raw"][e], w["s2_raw"][e]).T
        per_k[rows, ks] = bf(y * wts[rows, ks, None])
    routed = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for k in range(TOPK):
        routed = bf(routed + per_k[:, k])
    hs = xs @ deq_fp8_block(w["sgu"], w["sgu_s"]).T
    smid = bf(
        F.silu(bf(hs[:, :inter].clamp(max=LIMIT)))
        * bf(hs[:, inter:].clamp(-LIMIT, LIMIT))
    )
    shared = bf(mxfp8_qdq(smid, "vllm") @ deq_fp8_block(w["sw2"], w["sw2_s"]).T)
    return ids, bf(routed + shared)


# ---------------------------------------------------------------- the test
def seam_inputs(M, tp, rank, dev):
    """The attention seam's outputs (alike on every rank) and this rank's wo_b partial."""
    g = torch.Generator(device=dev).manual_seed(11 + M)
    res = torch.randn(M, HC, HIDDEN, generator=g, device=dev)
    res = (res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True))).to(torch.bfloat16)
    comb = torch.rand(M, HC, HC, generator=g, device=dev) + 0.1
    for _ in range(20):
        comb = comb / comb.sum(-1, keepdim=True)
        comb = comb / comb.sum(-2, keepdim=True)
    post = torch.sigmoid(torch.randn(M, HC, generator=g, device=dev)) * 2
    pre = torch.sigmoid(torch.randn(M, HC, generator=g, device=dev)) + 1e-6
    gp = torch.Generator(device=dev).manual_seed(100 + 7 * rank + M)
    part = (torch.randn(M, HIDDEN, generator=gp, device=dev) * 0.5 / tp).to(
        torch.bfloat16
    )
    return part, res, post.contiguous(), comb.contiguous(), pre.contiguous()


def reduced(part):
    """Return the attention all-reduce as an fp32 sum in rank order, as the kernel sums."""
    parts = [torch.empty_like(part) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, part)
    a = parts[0].float()
    for p in parts[1:]:
        a = a + p.float()
    return a.to(torch.bfloat16)


def sglang_ffn_seam(part, res, post, comb, pre, w):
    """Return SGLang's decode chain for the same step: all-reduce, gfx950 boundary kernel and norm."""
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
        rmsnorm_with_sinkhorn,
    )

    res2, y, co = hc_boundary_fused_deferred(
        reduced(part), res, post, comb, pre, w["hc_fn"], w["hc_scale"], w["hc_base"],
        HC, 20, RMS_EPS, HC_EPS,
    )  # fmt: skip
    _, normed = rmsnorm_with_sinkhorn(y, w["norm"], RMS_EPS, co, fake_quant=False)
    return res2, normed, co.post, co.comb, co.pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=500)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.mono.deepseek_v41_hip.layer import scratch_layout
    from sglang.kernels.ops.mono.deepseek_v41_hip.runner import DSV41MonoRunner

    w, inter, padded = make_weights(tp, rank, dev)
    mw = mono_weights(w)
    mw.check(tp)
    runner = DSV41MonoRunner(tp, rank, cpu, dev)
    fails, steps = [], []
    for M in WIDTHS:
        part, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        out, res_o, post_o, comb_o, pre_o = runner.ffn(mw, part, res, post, comb, pre)
        res2, normed_ref, post2, comb2, pre2 = sglang_ffn_seam(
            part, res, post, comb, pre, w
        )
        off = scratch_layout(M, tp)["normed"][0]
        normed = (
            runner.scratch(M)[off : off + M * HIDDEN * 2]
            .view(torch.bfloat16)
            .view(M, HIDDEN)
        )
        ids, gpart = golden_partial(normed, w, inter, padded)
        gold = gpart.float()
        dist.all_reduce(gold)
        gold = gold.to(torch.bfloat16)
        ids_ref, _ = routing(normed_ref, w)
        same = (ids.sort(1).values == ids_ref.sort(1).values).all(1)
        tok = F.cosine_similarity(out.float(), gold.float(), dim=1)
        res_ulps, res_share = bf16_ulps(res_o, res2)
        ref64, big = hc_post_fp64(reduced(part), res, post, comb)
        err, err_sgl = term_ulps(res_o, ref64, big), term_ulps(res2, ref64, big)
        checks = {
            "residual exact": torch.equal(res_o, res2),
            # the same fp32 math in another order: error in ulps of each element's largest term vs an fp64 hc_post
            "residual err <= SGLang's + 0.5 ulp, <= 0.01% differ": err <= err_sgl + 0.5
            and res_share <= 1e-4,
            "post": torch.allclose(post_o.view(M, HC), post2, rtol=MIX_RTOL, atol=1e-5),
            "comb": torch.allclose(comb_o, comb2, rtol=MIX_RTOL, atol=1e-5),
            "pre": torch.allclose(pre_o, pre2, rtol=MIX_RTOL, atol=1e-5),
            "normed": cos(normed, normed_ref) > NUMERICS.bound("ffn.normed.cosine")
            and rel(normed, normed_ref) < NUMERICS.bound("ffn.normed.rel"),
            "moe": cos(out, gold) > NUMERICS.bound("ffn.moe.cosine")
            and rel(out, gold) < NUMERICS.bound("ffn.moe.rel"),
            # a top-6 near-tie may route a token differently from the golden model's own logits
            "moe per token": (tok > 0.999).float().mean().item() >= 0.95,
            "routing same": same.float().mean().item() >= 0.9,
            "ranks agree": True,
        }
        ref0 = out.clone()
        dist.broadcast(ref0, 0)
        checks["ranks agree"] = torch.equal(ref0, out)
        stats = (
            f"res max|d| {(res_o.float() - res2.float()).abs().max().item():.2e} "
            f"({res_ulps:.2f} ulp, {100 * res_share:.4f}% differ; vs fp64 {err:.2f} / SGLang {err_sgl:.2f} "
            f"ulp of the largest term), "
            f"normed cos {cos(normed, normed_ref):.6f} rel {rel(normed, normed_ref):.4f}, "
            f"moe cos {cos(out, gold):.6f} rel {rel(out, gold):.4f} min-token cos {tok.min().item():.5f}, "
            f"routing same {same.float().mean().item():.3f}"
        )
        bad = [k for k, v in checks.items() if not v and k != "residual exact"]
        if rank == 0:
            print(f"M={M:2d} {'PASS' if not bad else 'FAIL ' + ','.join(bad)}"
                  f" (residual exact: {checks['residual exact']}) {stats}", flush=True)  # fmt: skip
        fails += [f"M={M} {k}" for k in bad]
        outs = runner._outs(M, res)
        steps.append((part, res, post, comb, pre, outs))

    # the hand-off protocol: every width in one graph, back to back, bit for bit against eager
    def run_all():
        for part, res, post, comb, pre, outs in steps:
            runner.ffn(mw, part, res, post, comb, pre, outs=outs)

    run_all()
    torch.cuda.synchronize()
    ref = [t.clone() for s in steps for t in s[5]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad_replays = 0
    for i in range(args.replays):
        graph.replay()
        if i % 50 == 49 or i == args.replays - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[5]]
            bad_replays += sum(not torch.equal(a, b) for a, b in zip(got, ref))
    if rank == 0:
        print(f"replay check: {args.replays} replays of widths {WIDTHS} in one graph, outputs sampled "
              f"every 50 replays: {bad_replays} mismatching checks (stress: test/manual/kernels/mono/stress_deepseek_v41_hip.py)",
              flush=True)  # fmt: skip
    if bad_replays:
        fails.append("replay")
    finish(fails, dev)


def finish(fails, dev):
    """Print this rank's failures, then exit nonzero on every rank if any rank failed."""
    if fails:
        print(f"rank {dist.get_rank()} FAILED: {fails}", flush=True)
    bad = torch.tensor([int(bool(fails))], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print("ALL PASS" if not bad.item() else "FAILED", flush=True)
    dist.destroy_process_group()
    if bad.item():
        sys.exit(1)


if __name__ == "__main__":
    main()
