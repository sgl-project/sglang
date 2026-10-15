# SPDX-License-Identifier: Apache-2.0
"""Check DeepSeek-V4.1's ffn span against SGLang's ops on gfx950 at every production width, TP2 and TP4: the
spec's numerics (max_abs / max_rel per output), graph replay of every width back to back, no allocation across a
capture and its replays, and a digest of every output for the old-against-new comparison.

Run directly: ``python3 test_dsv41_ffn.py`` relaunches itself under torchrun with 2 and 4 GPUs."""

import hashlib
import json
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_amd_ci(est_time=900, suite="nightly-amd-8-gpu-mi35x", nightly=True)

HIDDEN, HC, MIX = 5120, 4, 24
E, TOPK, SCALE, LIMIT = 384, 6, 1.5, 10.0
INTER, PAD = 2304, 128
RMS_EPS, HC_EPS = 1e-20, 1e-6
MAX_ROWS = 48
REPLAYS = 500
# widths no capture reaches, kept to cover the kernels' odd shapes
EXTRA_WIDTHS = (3, 5, 7, 17, 31, 33, 47)


def production_widths(extra=EXTRA_WIDTHS):
    """The widths attach builds (runtime.widths_for) for the cookbook's decode graphs (--cuda-graph-max-bs-decode
    64, SGLang's capture sizes) without and with DSpark (6 tokens a verify step), plus ``extra``."""
    from sglang.experimental.monokernels.runtime import widths_for
    from sglang.srt.arg_groups.cuda_graph_hook import (
        generate_decode_cuda_graph_batch_sizes,
    )

    out = set(extra)
    for algorithm, tokens in ((None, None), ("DSPARK", 6)):
        cfg = types.SimpleNamespace(
            disable_cuda_graph_padding=False, speculative_algorithm=algorithm
        )
        sizes = generate_decode_cuda_graph_batch_sizes(cfg, 64)
        for widths in widths_for(sizes, tokens, MAX_ROWS).values():
            out.update(widths)
    return sorted(out)


def init_dist():
    """This torchrun rank's (rank, TP size, device, CPU group); the process group once per process."""
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, tp, torch.device("cuda", torch.cuda.current_device())


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
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        MonoLayerWeights,
    )

    return MonoLayerWeights(
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


def digest(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def over_bounds(values):
    """The ``<span>.<output>.<metric>`` values above the spec's bounds (a bound of None is not checked)."""
    from sglang.experimental.monokernels.hip.deepseek_v41.spec import NUMERICS

    return [
        f"{k} {v:.3g} > {NUMERICS.bounds[k]:.3g}"
        for k, v in sorted(values.items())
        if NUMERICS.bounds.get(k) is not None and v > NUMERICS.bounds[k]
    ]


def metric_lines(tp, M, outputs, refs):
    """``METRIC`` json lines: the RFC metrics of each ffn output against its reference (spec.measure); and the
    values above the spec's bounds."""
    from sglang.experimental.monokernels.spec import measure

    lines, values = [], {}
    for name, out in outputs.items():
        m = measure(out.float(), refs[name].float())
        lines.append(json.dumps({"tp": tp, "rows": M, "key": f"ffn.{name}", **m}))
        values.update({f"ffn.{name}.{k}": v for k, v in m.items()})
    return lines, over_bounds(values)


def run_ffn():
    """Every check on this rank; True when every rank passed."""
    rank, tp, dev = init_dist()
    cpu = dist.new_group(backend="gloo")
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (
        scratch_layout,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        DSV41MonoLayer,
    )

    widths = production_widths()
    w, inter, padded = make_weights(tp, rank, dev)
    mw = mono_weights(w)
    mw.check(tp)
    runner = DSV41MonoLayer(tp, rank, cpu, dev, deadline_s=60.0)
    runner.allocate(widths)
    builds = runner.build(widths, k1=False)
    if rank == 0:
        print(f"WIDTHS {widths}", flush=True)
        print(f"BUILD seconds per width {builds}", flush=True)
    fails, steps = [], []
    for M in widths:
        part, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        out, res_o, post_o, comb_o, pre_o = (
            t.clone()
            for t in runner.ffn(mw, part, res, post, comb, pre, layer_id=M % 2)
        )
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
            "post": torch.allclose(post_o.view(M, HC), post2, rtol=1e-4, atol=1e-5),
            "comb": torch.allclose(comb_o, comb2, rtol=1e-4, atol=1e-5),
            "pre": torch.allclose(pre_o, pre2, rtol=1e-4, atol=1e-5),
            "normed": cos(normed, normed_ref) > 0.9999
            and rel(normed, normed_ref) < 0.012,
            "moe": cos(out, gold) > 0.9999 and rel(out, gold) < 0.015,
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
            outputs = {
                "out": out,
                "residual": res_o,
                "post": post_o.view(M, HC),
                "comb": comb_o,
                "pre": pre_o,
            }
            refs = {
                "out": gold,
                "residual": res2,
                "post": post2,
                "comb": comb2,
                "pre": pre2,
            }
            lines, over = metric_lines(tp, M, outputs, refs)
            for line in lines:
                print("METRIC", line, flush=True)
            if over:
                print(f"M={M:2d} above the spec's bounds: {over}", flush=True)
            fails += [f"M={M} {k}" for k in over]
            print(
                f"DIGEST ffn tp{tp} M={M} {digest(out, res_o, post_o, comb_o, pre_o)}",
                flush=True,
            )
        fails += [f"M={M} {k}" for k in bad]
        # the outputs of each width, copied out of the runner's alternating output sets inside the graph
        dst = [torch.empty_like(t) for t in (out, res_o, post_o, comb_o, pre_o)]
        steps.append((M, part, res, post, comb, pre, dst))

    # the hand-off protocol: every width in one graph, back to back, bit for bit against eager
    def run_all():
        for i, (M, part, res, post, comb, pre, dst) in enumerate(steps):
            got = runner.ffn(mw, part, res, post, comb, pre, layer_id=i)
            for d, g in zip(dst, got):
                d.copy_(g)

    run_all()
    torch.cuda.synchronize()
    ref = [t.clone() for s in steps for t in s[6]]

    def captured_growth(fn):
        """(bytes the allocator grew during a capture of fn, the graph)."""
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(dev) - before, g

    # the same copies without the launches: what torch's capture itself allocates
    control, _ = captured_growth(lambda: [d.copy_(d) for s in steps for d in s[6]])
    capture, graph = captured_growth(run_all)
    after_capture = torch.cuda.memory_allocated(dev)
    bad_replays = 0
    for i in range(REPLAYS):
        graph.replay()
        if i % 50 == 49 or i == REPLAYS - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[6]]
            bad_replays += sum(not torch.equal(a, b) for a, b in zip(got, ref))
    torch.cuda.synchronize()
    replays = torch.cuda.memory_allocated(dev) - after_capture
    if rank == 0:
        print(f"replay check: {REPLAYS} replays of widths {widths} in one graph, outputs sampled "
              f"every 50 replays: {bad_replays} mismatching checks", flush=True)  # fmt: skip
        print(f"ALLOCATOR capture {capture} bytes (control graph {control}), {REPLAYS} replays {replays} "
              f"bytes", flush=True)  # fmt: skip
    if bad_replays:
        fails.append("replay")
    if capture > control or replays:
        fails.append(
            f"allocator grew: capture {capture} (control {control}), replays {replays}"
        )
    return finish(fails, dev)


def finish(fails, dev):
    """Print this rank's failures; True when no rank failed."""
    if fails:
        print(f"rank {dist.get_rank()} FAILED: {fails}", flush=True)
    bad = torch.tensor([int(bool(fails))], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print("ALL PASS" if not bad.item() else "FAILED", flush=True)
    return not bad.item()


def test_ffn_span():
    assert run_ffn(), "the ffn span failed a check on some rank; see the log"


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4), timeout=1800)
