"""Check K2 (the attention back fused into the FFN launch) stage by stage on gfx950
(torchrun, TP2 or TP4): its XO (combine, inverse RoPE, bf16) against a torch attention,
its X8B against wo_a of its own XO, the wo_b partial it pushed against wo_b of its own X8B,
its output bit for bit against the FFN launch fed that partial; an index source's K2,
which builds the key table itself, and a window-only K2 bit for bit against the K2 of
the same keys; then its graph replay."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.experimental.monokernels.hip.deepseek_v41.spec import NUMERICS
from sglang.test.monokernels.deepseek_v41.check_ffn import (
    HIDDEN,
    bf16_ulps,
    cos,
    deq_fp8_block,
    finish,
    make_weights,
    mono_weights,
    mxfp8_qdq,
    rel,
    seam_inputs,
)

HEAD_DIM, NOPE, ROPE = 512, 448, 64
DATA, SCALE_BYTES, REC = 576, 8, 584
WINDOW, TOPK = 128, 512
O_RANK = 1024
SWA_BLOCK, COMP_BLOCK = 64, 128
SWA_PAGES, COMP_PAGES = 48, 32
POSITIONS = 4096
WIDTHS = (1, 6, 12, 24, 48)


def page_bytes(block):
    """A V4 page: ``block`` data rows, then ``block`` scale rows, padded to 576 B."""
    return -(-block * REC // DATA) * DATA


def make_cache(pages, block, g, dev):
    """A paged V4 cache of random records: (raw bytes [pages, page bytes], nope f32, rope f32)."""
    nope = (torch.randn(pages, block, NOPE, generator=g, device=dev) * 64).clamp(
        -448, 448
    )
    nope = nope.to(torch.float8_e4m3fn)
    rope = torch.randn(pages, block, ROPE, generator=g, device=dev).to(torch.bfloat16)
    # 7 UE8M0 codes (a 64-value tile each) and a pad byte
    codes = torch.randint(
        118, 126, (pages, block, 7), generator=g, device=dev, dtype=torch.uint8
    )
    scales = torch.cat([codes, torch.zeros_like(codes[..., :1])], -1)
    raw = torch.zeros(pages, page_bytes(block), dtype=torch.uint8, device=dev)
    data = torch.cat(
        [nope.view(torch.uint8), rope.view(torch.uint8)], -1
    )  # [pages, block, 576]
    raw[:, : block * DATA] = data.reshape(pages, -1)
    raw[:, block * DATA : block * REC] = scales.reshape(pages, -1)
    k_nope = nope.float() * torch.exp2(codes.float() - 127).repeat_interleave(64, -1)
    keys = torch.cat([k_nope, rope.float()], -1).reshape(pages * block, HEAD_DIM)
    return raw, keys


def key_lists(M, g, dev, pad_rows=()):
    """Each token's window slots and top-k compressed slots (-1: an invalid one), and the K2 key table."""
    kt = torch.full((M, WINDOW + TOPK), -1, dtype=torch.int32, device="cpu")
    klen = torch.zeros(2 * M, dtype=torch.int32, device="cpu")
    lists = []
    for t in range(M):
        if t in pad_rows:
            lists.append(([], []))
            continue
        nswa = int(torch.randint(1, WINDOW + 1, (1,), generator=g))
        ntopk = int(torch.randint(0, TOPK + 1, (1,), generator=g))
        swa = torch.randperm(SWA_PAGES * SWA_BLOCK, generator=g)[:nswa].tolist()
        comp = torch.randperm(COMP_PAGES * COMP_BLOCK, generator=g)[:ntopk].tolist()
        if ntopk > 2:
            comp[ntopk // 2] = -1  # an invalid top-k slot reads the zero record, masked
        for k, c in enumerate(comp):
            kt[t, k] = -1 if c < 0 else c | -(2**31)
        for j, s in enumerate(swa):
            kt[t, ntopk + j] = s
        klen[t], klen[M + t] = ntopk + nswa, ntopk
        lists.append((swa, comp))
    return kt.to(dev), klen.to(dev), lists


def source_keys(M, lists, dev):
    """The key table's sources, as K1 and an index source's K2 read them: SWA slots
    (-1 on a pad row), window slots and lengths, top-k slots and lengths."""
    slot = torch.full((M,), -1, dtype=torch.int64)
    swa_idx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    topk = torch.full((M, TOPK), -1, dtype=torch.int32)
    swa_lens = torch.zeros(M, dtype=torch.int32)
    topk_lens = torch.zeros(M, dtype=torch.int32)
    for t, (swa, comp) in enumerate(lists):
        if not swa:
            continue  # a pad row
        slot[t] = swa[0]
        swa_idx[t, : len(swa)] = torch.tensor(swa, dtype=torch.int32)
        topk[t, : len(comp)] = torch.tensor(comp, dtype=torch.int32)
        swa_lens[t], topk_lens[t] = len(swa), len(comp)
    keys = tuple(x.to(dev) for x in (swa_idx, swa_lens, topk, topk_lens))
    return slot.to(dev), keys


def window_only(M, lists, dev):
    """The key table of the same windows without top-k rows (a compress ratio 0 layer)."""
    kt = torch.full((M, WINDOW + TOPK), -1, dtype=torch.int32)
    klen = torch.zeros(2 * M, dtype=torch.int32)
    for t, (swa, _) in enumerate(lists):
        kt[t, : len(swa)] = torch.tensor(swa, dtype=torch.int32)
        klen[t] = len(swa)
    return kt.to(dev), klen.to(dev)


def attention_xo(q, swa_keys, comp_keys, lists, sink, pos, cos_sin):
    """[M, H * 512] XO as K2 computes it: attention with the sink, bf16 o, inverse GPT-J
    RoPE of dims 448.., bf16 (exact softmax: K2 rounds p and partials to bf16)."""
    M, H, _ = q.shape
    o = torch.zeros(M, H, HEAD_DIM, device=q.device)
    for t, (swa, comp) in enumerate(lists):
        rows = [comp_keys[c] for c in comp if c >= 0] + [swa_keys[s] for s in swa]
        if not rows:
            continue
        k = torch.stack(rows).bfloat16().float()
        s = (q[t].float() @ k.T) * HEAD_DIM**-0.5  # [H, keys]
        m = torch.maximum(s.amax(1), sink)
        p = torch.exp(s - m[:, None])
        o[t] = (p @ k) / (p.sum(1) + torch.exp(sink - m))[:, None]
    o = o.bfloat16().float()
    cs = cos_sin[pos]  # [M, 64]: 32 cos, then 32 sin
    c, sn = cs[:, None, :32], cs[:, None, 32:]
    e, od = o[..., NOPE::2], o[..., NOPE + 1 :: 2]
    r = torch.stack([e * c + od * sn, od * c - e * sn], -1).flatten(-2)
    return torch.cat([o[..., :NOPE], r], -1).reshape(M, H * HEAD_DIM).bfloat16().float()


def pushed_partials(runner, rows, tp):
    """[tp, rows, 5120] bf16: every rank's wo_b partial as the last K2 pushed it into this
    rank's ATTN region (bf16 pairs next to their tags, ``seam.attn_region_pair``)."""
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (
        peer_half_bytes,
    )

    ep = int(runner.epoch[0].item()) - 1  # the K2 launch's epoch (run_ffn moved it on)
    base = (ep & 1) * peer_half_bytes(tp)
    words = runner.peer.bytes[base : base + tp * rows * HIDDEN // 2 * 8].view(
        torch.int32
    )
    return (
        words.view(tp, rows, HIDDEN // 2, 2)[..., 0].contiguous().view(torch.bfloat16)
    )


def mx_region(scratch, layout, name, rows, cols):
    """A scratch region of MXFP8 rows (e4m3 words, then an i32 E8M0 code a 32) dequantized."""
    qo, so = layout[name][0], layout[name + "s"][0]
    q = (
        scratch[qo : qo + rows * cols]
        .view(torch.float8_e4m3fn)
        .view(rows, cols)
        .float()
    )
    code = scratch[so : so + rows * cols // 32 * 4].view(torch.int32).view(rows, -1)
    return (q.view(rows, -1, 32) * torch.exp2(code.float() - 127)[..., None]).view(
        rows, cols
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=200)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.layer import (
        scratch_layout,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
        K2Weights,
    )

    d = Dims(tp)
    w, _, _ = make_weights(tp, rank, dev)
    mw = mono_weights(w)
    mw.check(tp)
    runner = DSV41MonoRunner(tp, rank, cpu, dev)

    # the KV caches are alike on every rank; the heads, wo_a and wo_b are this rank's
    gc = torch.Generator(device=dev).manual_seed(5)
    swa_raw, swa_keys = make_cache(SWA_PAGES, SWA_BLOCK, gc, dev)
    comp_raw, comp_keys = make_cache(COMP_PAGES, COMP_BLOCK, gc, dev)
    gr = torch.Generator(device=dev).manual_seed(77 + rank)
    o_rows = d.groups * O_RANK

    def e8m0(*shape):
        return torch.randint(
            118, 124, shape, generator=gr, device=dev, dtype=torch.uint8
        )

    woa = (torch.randn(o_rows, d.group_k, generator=gr, device=dev) * 4).clamp(
        -448, 448
    )
    wob = (torch.randn(HIDDEN, o_rows, generator=gr, device=dev) * 4).clamp(-448, 448)
    woa = (woa.to(torch.float8_e4m3fn), e8m0(o_rows // 32, d.group_k // 32))
    wob = (wob.to(torch.float8_e4m3fn), e8m0(HIDDEN // 32, o_rows // 32))
    freqs = torch.outer(
        torch.arange(POSITIONS, device=dev).float(),
        1.0 / 10000 ** (torch.arange(32, device=dev).float() / 32),
    )
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()
    sink = torch.randn(d.heads, generator=gr, device=dev)
    aw = K2Weights(
        woa=woa[0], woa_s=woa[1], wob=wob[0], wob_s=wob[1], sink=sink, cos_sin=cos_sin
    )
    aw.check(d)

    fails, steps = [], []
    for M in WIDTHS:
        gk = torch.Generator().manual_seed(
            1000 + M
        )  # alike on every rank: the step's metadata
        pad = (M - 1,) if M >= 6 else ()
        kt, klen, lists = key_lists(M, gk, dev, pad_rows=pad)
        pos = torch.randint(0, POSITIONS, (M,), generator=gk).to(dev)
        _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        q = (
            torch.randn(M, d.heads, HEAD_DIM, generator=gr, device=dev) * 0.05
        ).bfloat16()
        runner.q[:M] = q
        runner.res_mid[:M], runner.post_a[:M] = res, post
        runner.comb_a[:M], runner.pre_a[:M] = comb, pre
        runner.kt[: M * (WINDOW + TOPK)] = kt.reshape(-1)
        runner.klen[: 2 * M] = klen
        out, res_o, post_o, comb_o, pre_o = runner.k2(
            aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK
        )
        torch.cuda.synchronize()
        lay, sc = scratch_layout(M, tp), runner.scratch(M)
        live = torch.tensor([t not in pad for t in range(M)], device=dev)
        # XO: the attention, combine and inverse RoPE
        o = lay["xo"][0]
        xo = sc[o : o + M * d.q_rows * 2].view(torch.bfloat16).view(M, d.q_rows).float()
        xo_ref = attention_xo(q, swa_keys, comp_keys, lists, sink, pos, cos_sin)
        row_cos = F.cosine_similarity(xo[live], xo_ref[live], dim=1)
        # X8B: wo_a of K2's own XO
        x8b = mx_region(sc, lay, "x8b", M, o_rows)
        wa = deq_fp8_block(*woa).reshape(d.groups, O_RANK, -1)
        ya = torch.einsum("tgk,grk->tgr", xo.view(M, d.groups, -1), wa)
        x8b_ref = mxfp8_qdq(ya.bfloat16().float().reshape(M, -1), "vllm")
        # wo_b: the partial K2 pushed (before the FFN launch below overwrites the region)
        pushed = pushed_partials(runner, M, tp)
        own = pushed[rank]
        part_ref = (x8b @ deq_fp8_block(*wob).T).bfloat16()
        part_ulps, part_share = bf16_ulps(own[live], part_ref[live])
        gathered = [torch.empty_like(own) for _ in range(tp)]
        dist.all_gather(gathered, own.contiguous())
        # the output: bit for bit the FFN launch fed K2's own partial
        ref = runner.ffn(mw, own.contiguous(), res, post, comb, pre)
        same = all(
            torch.equal(a, b) for a, b in zip(ref, (out, res_o, post_o, comb_o, pre_o))
        )
        checks = {
            "finite": all(
                torch.isfinite(x).all().item()
                for x in (out, res_o, post_o, comb_o, pre_o)
            ),
            # bf16 p / partials against an exact softmax, then MXFP8 rounding flips: ~1% rel
            "xo": cos(xo[live], xo_ref[live]) > NUMERICS.bound("k2.attn_out.cosine")
            and row_cos.min().item() > 0.999,
            "pad rows read nothing": bool((xo[~live] == 0).all()),
            "x8b": rel(x8b, x8b_ref) < NUMERICS.bound("k2.wo_a.rel"),
            # the GEMV's summation order: few elements apart (a near-cancelling sum by more ulps)
            "wo_b partial": rel(own[live], part_ref[live])
            < NUMERICS.bound("k2.wo_b.rel")
            and part_share <= 0.02,
            "pushes arrive": all(
                torch.equal(g, pushed[src]) for src, g in enumerate(gathered)
            ),
            "ffn stages": same,
        }
        outs = (out, res_o, post_o, comb_o, pre_o)
        # an index source's K2 builds the table from K1's slots and the indexer's
        # top-k: the same table, the same outputs bit for bit
        slot, keys = source_keys(M, lists, dev)
        runner.kt.fill_(-7)
        runner.klen.fill_(-7)
        got = runner.k2(
            aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK,
            slot=slot, keys=keys, index=True,
        )  # fmt: skip
        torch.cuda.synchronize()
        checks["index source"] = (
            all(torch.equal(a, b) for a, b in zip(got, outs))
            and torch.equal(runner.kt[: M * (WINDOW + TOPK)], kt.reshape(-1))
            and torch.equal(runner.klen[: 2 * M], klen)
        )
        # a window-only layer's K2 (no compressed cache) against the full K2 on a
        # table of window keys only
        kt0, klen0 = window_only(M, lists, dev)
        pair = []
        for comp in (comp_raw, None):
            runner.kt[: M * (WINDOW + TOPK)] = kt0.reshape(-1)
            runner.klen[: 2 * M] = klen0
            pair.append(runner.k2(aw, mw, M, pos, swa_raw, SWA_BLOCK, comp, COMP_BLOCK))
        torch.cuda.synchronize()
        checks["window only"] = all(torch.equal(a, b) for a, b in zip(*pair))
        ref0 = out.clone()
        dist.broadcast(ref0, 0)
        checks["ranks agree"] = torch.equal(ref0, out)
        bad = [k for k, v in checks.items() if not v]
        if rank == 0:
            print(
                f"M={M:2d} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} | "
                f"xo cos {cos(xo[live], xo_ref[live]):.6f} rel {rel(xo[live], xo_ref[live]):.4f} "
                f"min-row {row_cos.min().item():.5f} | x8b rel {rel(x8b, x8b_ref):.5f} | "
                f"wo_b partial rel {rel(own[live], part_ref[live]):.2e}, {100 * part_share:.3f}% differ "
                f"(max {part_ulps:.0f} ulp) | "
                f"ffn stages bit for bit {same}",
                flush=True,
            )
        fails += [f"M={M} {k}" for k in bad]
        steps.append((M, pos, kt, klen, q, res, post, comb, pre, runner._outs(M, res)))
        steps.append(
            (M, pos, (slot, keys), None, q, res, post, comb, pre, runner._outs(M, res))
        )

    # the hand-offs under replay: every width's K2 in one graph, bit for bit against eager
    def run_all():
        for M, pos, kt, klen, q, res, post, comb, pre, outs in steps:
            runner.q[:M].copy_(q)
            runner.res_mid[:M].copy_(res)
            runner.post_a[:M].copy_(post)
            runner.comb_a[:M].copy_(comb)
            runner.pre_a[:M].copy_(pre)
            if klen is None:  # an index source's K2: it builds the table
                slot, keys = kt
                runner.k2(
                    aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK,
                    outs=outs, slot=slot, keys=keys, index=True,
                )  # fmt: skip
                continue
            runner.kt[: M * (WINDOW + TOPK)].copy_(kt.reshape(-1))
            runner.klen[: 2 * M].copy_(klen)
            runner.k2(
                aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK, outs=outs
            )

    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for s in steps for t in s[-1]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad_replays = 0
    for i in range(args.replays):
        graph.replay()
        if i % 50 == 49 or i == args.replays - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[-1]]
            bad_replays += sum(not torch.equal(a, b) for a, b in zip(got, eager))
    if rank == 0:
        print(f"replay check: {args.replays} replays of widths {WIDTHS} in one graph: "
              f"{bad_replays} mismatching checks", flush=True)  # fmt: skip
    if bad_replays:
        fails.append("replay")
    finish(fails, dev)


if __name__ == "__main__":
    main()
