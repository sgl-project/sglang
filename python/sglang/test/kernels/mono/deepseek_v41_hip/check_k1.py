"""Check that K1 stays bitwise equal to SGLang's decode chain for the seam and the SWA KV write on gfx950,
in each variant: the owed post folded in, a settled residual (after an Engram write), the first
layer (copy 0 of the broadcast embedding), and an index source's normed input and qr."""

import argparse
import math
import os

import torch
import torch.distributed as dist

from sglang.kernels.ops.mono.deepseek_v41_hip.spec import NUMERICS
from sglang.test.kernels.mono.deepseek_v41_hip.check_ffn import (
    HC,
    HC_EPS,
    HIDDEN,
    RMS_EPS,
    finish,
    make_weights,
    mono_weights,
    seam_inputs,
)

RECORD, BLOCK, PAGES, POSITIONS = 584, 64, 64, 131072
Q_RANK, NOPE, HEAD = 1280, 448, 512
# the spec's bounds; measured: mixes <= 5.4e-7 (TP2 / TP4 synthetic, TP4 server)
MIX_TOL = NUMERICS.bound("k1.mix.max_abs")
# K1 rounds q as SGLang does but sums wq_b in another order: bf16 flips only
Q_COS, Q_REL = NUMERICS.bound("k1.q.cosine"), NUMERICS.bound("k1.q.rel")
WIDTHS = "1,2,3,4,5,7,8,12,16,17,24,31,32,33,40,47,48"
# (name, fold, index, first): the variants, each against its SGLang chain
VARIANTS = (
    ("fold", True, False, False),
    ("settled", False, False, False),
    ("first", False, False, True),
    ("index", True, True, False),
)
# the widths the variants other than fold run at (fold runs at every width)
VARIANT_WIDTHS = (1, 6, 12, 24, 48)
QR_REL = NUMERICS.bound("k1.qr.rel")
# SGLang's wqkv_a configs K1 is matched to: the GEMV per M bucket up to 32 rows, the split-K tile above
GEMV = {1: "w16s1r16t16k", 2: "w16s1r16t16k", 4: "w16s1r16t16k", 8: "w8s2r16t16k", 16: "w8s2r16t16k",
        32: "w8s1r16t32k"}  # fmt: skip
LARGE_M = (64, 64, 256, 4, 5)


def attn_weights(heads, dev, g):
    """K1's weights of one layer, wide-range: e4m3 [N, K] with varied E8M0 32 x 32 block codes."""

    def fp8(n, k, lo, hi):
        w = (torch.randn(n, k, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return w, torch.randint(lo, hi, (n // 32, k // 32), generator=g, device=dev).to(
            torch.uint8
        )

    wqkv, wqkv_s = fp8(1792, HIDDEN, 116, 126)
    wqb, wqb_s = fp8(heads * 512, 1280, 121, 122)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=dev)
    kv_norm = (torch.randn(512, generator=g, device=dev).abs() * 2 + 0.05).to(
        torch.bfloat16
    )
    return dict(
        wqkv=wqkv, wqkv_s=wqkv_s, q_norm=q_norm, kv_norm=kv_norm, wqb=wqb, wqb_s=wqb_s
    )


def rope_table(dev):
    j = torch.arange(32, device=dev, dtype=torch.float64)
    ang = torch.arange(POSITIONS, device=dev, dtype=torch.float64)[
        :, None
    ] * 10000.0 ** (-j / 32)
    return torch.cat([ang.cos(), ang.sin()], -1).float().contiguous()


def check_wqkv_route(M):
    """Return whether SGLang's wqkv_a config at M is the one K1 implements."""
    from sglang.kernels.ops.mono.deepseek_v41_hip.attention.plan import (
        wqkv_order,
    )
    from sglang.kernels.ops.quantization import mxfp8_native_amd_gfx95 as mx

    if M <= 32:
        b = mx._m_bucket(M)
        got, want = mx._select_config(b, 1792, HIDDEN), mx._GemvConfig.parse(GEMV[b])
        return got == want and wqkv_order(M)[0] == want.waves, f"GEMV {got}"
    got = tuple(mx._large_m_tile(M, 1792, HIDDEN))
    return got == LARGE_M and wqkv_order(M)[0] == LARGE_M[4], f"tile {got}"


def query_ref(qkv, a, cos_sin, pos):
    """fp64 query from SGLang's bf16 wqkv_a output: q_norm, MXFP8, wq_b, bf16, GPT-J RoPE of dims 448.."""
    x = qkv[:, :Q_RANK].float()
    y = (
        (
            x
            * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + RMS_EPS)
            * a["q_norm"].float()
        )
        .bfloat16()
        .float()
    )
    g = y.view(-1, Q_RANK // 32, 32)
    mant, e = torch.frexp(
        g.abs().amax(-1, keepdim=True).clamp_min(1.1754943508222875e-38) / 448
    )
    code = (e - (mant == 0.5).int()).clamp(-127, 127).float()
    x8 = (g * torch.exp2(-code)).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).double() * torch.exp2(code).double()
    w = a["wqb"].double() * torch.exp2(a["wqb_s"].double() - 127).repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    o = (x8.view(-1, Q_RANK) @ w.T).bfloat16().double().view(len(pos), -1, HEAD)
    cs = cos_sin[pos].double()
    c, s = cs[:, None, :32], cs[:, None, 32:]
    v0, v1 = o[..., NOPE::2], o[..., NOPE + 1 :: 2]
    o[..., NOPE::2], o[..., NOPE + 1 :: 2] = v0 * c - v1 * s, v0 * s + v1 * c
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default=WIDTHS)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
        rmsnorm_with_sinkhorn,
    )
    from sglang.kernels.ops.mono.deepseek_v41_hip.layer import (
        MonoBuild,
        build_mono_k1,
        scratch_layout,
    )
    from sglang.kernels.ops.mono.deepseek_v41_hip.runner import DSV41MonoRunner
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        prepare_mxfp8_native_weight,
    )

    w, _, _ = make_weights(tp, rank, dev, seed=0)
    m = mono_weights(w)
    del w
    r = DSV41MonoRunner(tp, rank, cpu, dev)
    heads = r.q.shape[1]
    g = torch.Generator(device=dev).manual_seed(11)
    a = attn_weights(heads, dev, g)
    m.ffn_norm.copy_(
        (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp().to(m.ffn_norm.dtype)
    )
    cos_sin = rope_table(dev)
    freqs = torch.complex(cos_sin[:, :32], cos_sin[:, 32:])  # SGLang's freqs_cis
    w_sh, w_su = prepare_mxfp8_native_weight(
        a["wqkv"], torch.exp2(a["wqkv_s"].float() - 127), (32, 32)
    )
    stride = math.ceil(BLOCK * RECORD / 576) * 576
    edges = torch.tensor(
        [BLOCK - 1, BLOCK, PAGES * BLOCK - 1], device=dev
    )  # page edges, the last slot

    def records(buf, s):
        """Return the 584 bytes of slots ``s``: 448 NOPE e4m3, 64 bf16 rope, then 8 scale bytes (7, pad)."""
        page, off = s // BLOCK, s % BLOCK
        col = torch.cat([off[:, None] * 576 + torch.arange(576, device=dev),
                         BLOCK * 576 + off[:, None] * 8 + torch.arange(8, device=dev)], 1)  # fmt: skip
        return buf[page[:, None].expand_as(col), col]

    def inputs(M):
        _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        res = (res * (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp()).to(
            res.dtype
        )
        x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        pos = torch.randint(0, POSITIONS, (M,), generator=g, device=dev)
        pos[: min(M, 2)] = torch.tensor([POSITIONS - 1, 65536], device=dev)[: min(M, 2)]
        perm = torch.randperm(PAGES * BLOCK, generator=g, device=dev)
        slot = torch.cat([edges, perm[~torch.isin(perm, edges)]])[:M]
        return res, x, post, comb, pre, pos, slot

    fails = []
    widths = [int(v) for v in args.widths.split(",")]
    runs = [
        (M, v)
        for M in widths
        for v in VARIANTS
        if v[0] == "fold" or M in VARIANT_WIDTHS
    ]
    for M, (vname, fold, index, first) in runs:
        ok, route = check_wqkv_route(M)
        k1 = build_mono_k1(MonoBuild(M, tp, fold=fold, index=index))
        lay = scratch_layout(M, tp)
        sc = r.scratch(M)
        static = [t.clone() for t in inputs(M)]
        swa = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
        res, x, post, comb, pre, pos, slot = static
        if first:
            pre = r.first_pre[:M]
        # a settled residual: no post owed, the seam reads only the residual
        xk, postk, combk = (x, post, comb) if fold else (res, res, res)

        def launch():
            r.epoch[0] += (
                1  # the FFN launch moves the epoch; K1 alone must not see the last launch's tags
            )
            k1(
                res.data_ptr(), xk.data_ptr(), postk.data_ptr(), combk.data_ptr(), pre.data_ptr(),
                m.hc_ffn_fn.data_ptr(), m.hc_ffn_scale.data_ptr(), m.hc_ffn_base.data_ptr(),
                m.ffn_norm.data_ptr(),
                r.res_mid.data_ptr(), r.post_a.data_ptr(), r.comb_a.data_ptr(), r.pre_a.data_ptr(),
                a["wqkv"].data_ptr(), a["wqkv_s"].data_ptr(), a["q_norm"].data_ptr(), a["kv_norm"].data_ptr(),
                a["wqb"].data_ptr(), a["wqb_s"].data_ptr(), cos_sin.data_ptr(), pos.data_ptr(),
                slot.data_ptr(), swa.data_ptr(), swa.stride(0), BLOCK, 0, 0, 0, 0, 0, 0, r.q.data_ptr(),
                r.normed.data_ptr(), r.qr.data_ptr(), sc.data_ptr(),
                r.peer.local, r.peer.addresses.data_ptr(), r.rank, r.epoch.data_ptr(),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        diff = dict(res=0, x8=0, x8s=0, nope=0, scale=0, rope=0, replay=0, normed=0)
        mix, q_cos, q_rel, qr_rel = 0.0, 1.0, 0.0, 0.0
        for _ in range(args.trials):
            for t, v in zip(static, inputs(M)):
                t.copy_(v)
            swa.fill_(255)
            launch()
            torch.cuda.synchronize()
            o = lay["x8"][0]
            x8 = sc[o : o + M * HIDDEN].clone()
            o = lay["x8s"][0]
            x8s = (
                sc[o : o + M * HIDDEN // 32 * 4]
                .view(torch.int32)
                .view(M, HIDDEN // 32)
                .to(torch.uint8)
            )
            got = records(swa, slot.long())
            eager = [
                t[:M].clone() for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a)
            ] + [got.clone()]
            # the same step replayed: every output and record bitwise equal to eager
            for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a):
                t.fill_(float("nan"))
            swa.fill_(255)
            graph.replay()
            torch.cuda.synchronize()
            replayed = [
                t[:M] for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a)
            ] + [records(swa, slot.long())]
            diff["replay"] += sum(
                int((u.view(torch.uint8) != v.view(torch.uint8)).sum())
                for u, v in zip(eager, replayed)
            )

            # SGLang's boundary: no x when nothing is owed, no pre_prev on the first
            # layer (the collapse is copy 0)
            ro, y, co = hc_boundary_fused_deferred(
                x if fold else None,
                res,
                post if fold else None,
                comb if fold else None,
                None if first else pre,
                m.hc_ffn_fn,
                m.hc_ffn_scale,
                m.hc_ffn_base,
                HC,
                20,
                RMS_EPS,
                HC_EPS,
            )
            ro = res if ro is None else ro
            y = res[:, 0, :].contiguous() if y is None else y
            xq, normed = rmsnorm_with_sinkhorn(
                y, m.ffn_norm, RMS_EPS, co, emit_fp8=True
            )
            pre_r, post_r, comb_r = co.tensors()
            qkv = mxfp8_native_blockscaled_linear(
                xq.q[:M], w_sh, w_su, input_scale=xq.scale[:M]
            )
            ref = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
            fused_k_norm_rope_flashmla(
                qkv[:, Q_RANK:].contiguous(),
                a["kv_norm"],
                RMS_EPS,
                freqs,
                pos,
                slot.int(),
                ref,
                BLOCK,
            )
            want = records(ref, slot.long())

            diff["res"] += int(
                (eager[1].view(torch.int16) != ro.view(torch.int16)).sum()
            )
            diff["x8"] += int(
                (x8 != xq.q[:M].contiguous().view(torch.uint8).flatten()).sum()
            )
            diff["x8s"] += int(
                (
                    x8s != xq.scale[:M].contiguous().view(torch.uint8).reshape(M, -1)
                ).sum()
            )
            # byte 583 pads the 7 scales: no reader, and SGLang's store leaves it as it was
            for name, cols in (
                ("nope", slice(0, 448)),
                ("rope", slice(448, 576)),
                ("scale", slice(576, 583)),
            ):
                diff[name] += int((got[:, cols] != want[:, cols]).sum())
            for k1_t, ref_t in (
                (eager[4], pre_r),
                (eager[2], post_r),
                (eager[3], comb_r),
            ):
                mix = max(mix, (k1_t.reshape(ref_t.shape) - ref_t).abs().max().item())
            if index:
                # an index source's outputs, its compressor and indexer's inputs
                diff["normed"] += int(
                    (r.normed[:M].view(torch.int16) != normed.view(torch.int16)).sum()
                )
                xr = qkv[:, :Q_RANK].float()
                qr_ref = (
                    xr * torch.rsqrt(xr.pow(2).mean(1, keepdim=True) + RMS_EPS)
                    * a["q_norm"].float()
                ).bfloat16().float()  # fmt: skip
                qr_rel = max(
                    qr_rel,
                    ((r.qr[:M].float() - qr_ref).norm() / qr_ref.norm()).item(),
                )
            qr, qk = (
                query_ref(qkv, a, cos_sin, pos.long()).flatten(),
                eager[0].double().flatten(),
            )
            q_cos = min(
                q_cos, torch.nn.functional.cosine_similarity(qk, qr, dim=0).item()
            )
            q_rel = max(q_rel, ((qk - qr).norm() / qr.norm()).item())
        del graph
        bad = ([] if ok else ["wqkv route"]) + [k for k, v in diff.items() if v]
        bad += (["mix"] if mix > MIX_TOL else []) + (
            ["q"] if q_cos < Q_COS or q_rel > Q_REL else []
        )
        bad += ["qr"] if qr_rel > QR_REL else []
        if rank == 0:
            print(f"M={M:2d} {vname:7s} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} | qr rel {qr_rel:.1e} | wqkv_a {route} | differing: "
                  f"{diff} | pre/post/comb max|d| {mix:.2e} (bound {MIX_TOL:.0e}) | q cos {q_cos:.7f} "
                  f"rel {q_rel:.2e}", flush=True)  # fmt: skip
        fails += [f"M={M} {vname} {b}" for b in bad]
    finish(fails, dev)


if __name__ == "__main__":
    main()
