# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode layer's FFN launch, for a layer whose attention vLLM runs: a
push of vLLM's unreduced wo_b output to every rank (``seam.stage_push``) + the
FFN seam (the TP sum folded in) + the MoE (``stages.moe``, its all-reduce
in-kernel too).

Tags: the launch's hand-offs carry 2 e + 2 (e: the epoch, moved on by the
launch's CTA 0 at its end). Peer regions alternate by e's parity; every rank
runs the same launches, so the epochs agree across ranks.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from .common.epoch import EPOCH_MARKS, epoch_begin, epoch_end, gstore, kernel_symbol
from .common.ops import CM_DEV, peer_bases, rsrc, traced
from .common.plan import BLOCKS, THREADS, WAVES, cdiv, first_task, key_tuple
from .common.sync import Mailbox, publish, sreg
from .sources import SOURCES
from .stages import moe, seam
from .stages.dims import Dims as StageDims
from .stages.moe_shape import SORT_NETS, MoeBuild, route_shape

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
HIDDEN = seam.HIDDEN
SLICES = seam.SLICES  # 160
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTERS = ("ugq", "dq")
# the epoch buffer, in words: the epoch, a mark a CTA (epoch.EPOCH_MARKS), then on
# lines of their own the MoE's counters
EPOCH_COUNTERS = -(-(EPOCH_MARKS + BLOCKS) // 64) * 64
EPOCH_WORDS = EPOCH_COUNTERS + COUNTER_WORDS


@dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(tokens=key.tokens, tp=key.tp)


def _end(sizes) -> int:
    """The end of regions of these byte sizes, laid out in order 256-aligned."""
    off = 0
    for n in sizes:
        off = -(-off // 256) * 256 + n
    return off


def _attn_bytes(s: int, tp: int) -> tuple:
    """Byte sizes of the attention launches of vllm-project/vllm#60397, which
    this package does not include. They stay reserved so that every FFN
    scratch offset matches the tested layout."""
    heads, groups = 64 // tp, 8 // tp
    q_rows, o_rows = heads * 512, groups * 1024
    chunk = 64 if heads // 16 == 2 and s * cdiv(640, 64) <= 256 else 128
    n = s * cdiv(640, chunk)
    front = _end(
        [s * HIDDEN, s * HIDDEN // 8, 16 * s, 16 * s * 1792, s * 1280, s * 160, 8 * s]
    )
    back = _end(
        [n * heads * 1024, n * heads * 8, n * 8, s * q_rows, s * q_rows // 8,
         s * heads, s * o_rows, s * o_rows // 8, o_rows // 4]
    )  # fmt: skip
    return front, back


def scratch_layout(s: int, tp: int) -> dict:
    """Every region of step width s's launch -> (byte offset, bytes), disjoint:
    the seam's, the MoE's regions, the normed rows and their flags. Each width
    has a scratch of its own (``DSV41MonoLayer.scratch``): under another width's
    layout a mailbox pair's tag word could be plain data, which a poll may take
    for a current tag. The MoE's counters, whose slots outlive a step width,
    are in the epoch buffer."""
    front, back = _attn_bytes(s, tp)
    out: dict = {}
    off = 0

    def place(regions):
        nonlocal off
        base = off
        for name, (o, n) in regions.items():
            assert name not in out, name
            out[name] = (base + o, n)
        off = max(off, base + max(o + n for o, n in regions.values()))
        off = -(-off // 256) * 256

    place({"attn_front": (0, front)})
    place(seam.scratch_layout(s))
    place({"attn_back": (0, back)})
    regions = moe.scratch_layout(_moe_key(MonoBuild(s, tp))).items()
    mr = {n: v for n, v in regions if n not in COUNTERS}
    lo = min(o for o, _ in mr.values())
    place({n: (o - lo, n_) for n, (o, n_) in mr.items()})
    place({"normed": (0, s * HIDDEN * 2), "xrdy_moe": (s * HIDDEN * 2 + 256, s * 8)})
    return out


def scratch_bytes(s: int = MAX_TOKENS, tp: int = 2) -> int:
    return max(o + n for o, n in scratch_layout(s, tp).values())


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention's partials, then the MoE's."""
    return seam.attn_peer_bytes(MAX_TOKENS, tp) + moe.peer_bytes(MAX_TOKENS, tp)


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


@traced
def store_normed(normed, t, col, ys, live):
    """The FFN seam's norm output -> the MoE input row (bf16, device scope:
    this launch reads it)."""
    if live:
        for j in range_constexpr(8):
            bo.buffer_store(
                ys[j].to(fx.BFloat16),
                rsrc(normed),
                t * HIDDEN + col + j,
                cache_modifier=CM_DEV,
            )


def _seam_lds(s):
    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
        pl: fx.Array[fx.Float32, s * seam.COLS, 16]

    return SeamLds


@traced
def run_ffn(key: MonoBuild, lds, a: dict, amb, bases, own, rank, ep, part=None):
    """The FFN launch on this CTA: the FFN seam (the attention's TP partials --
    ``part`` pushed to every rank's ATTN region first -- summed and folded into
    the residual), the MoE and its all-reduce into ``a["out"]``, then the
    launch pair's epoch moved on."""
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    GATE0 = SLICES
    NORM0 = SLICES + s
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    sl = lds.seam.peek()
    scratch = a["scratch"]
    seam_args = (
        "res_in",
        "post_in",
        "comb_in",
        "pre_in",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "res_out",
        "post_out",
        "comb_out",
        "pre_out",
    )
    ca = {
        "S": s,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "pend_lds": sl.pl.ptr,
        "put": amb.put,
        "put_bf": amb.put_bf,
        "put_words": amb.put_words,
        "poll": amb.poll,
        "peer_addr": lambda p: bases[p],
        "rank": rank,
        "sym": own,
        "args": {n: a[n] for n in seam_args} | {"norm_w": a["ffn_w"]},
        "d": StageDims(tp),
    }
    normed = scratch + fx.Int64(layout["normed"][0])
    for region in ("lin", "pmix"):
        ca[region] = sreg(scratch, layout[region][0], region)
    mlayout = {n: layout[n] for n in moe.scratch_layout(mkey) if n not in COUNTERS}
    mlayout["xrdy"] = layout["xrdy_moe"]
    margs = moe.moe_args(
        normed,
        a["gate_w"],
        a["bias"],
        a["w13"],
        a["w13_s"],
        a["w2"],
        a["w2_s"],
        a["sgu"],
        a["sgu_s"],
        a["sw2"],
        a["sw2_s"],
        a["out"],
    )
    cb = moe.moe_context(
        s, rs, lds.moe.peek(), amb, bases, mlayout, scratch, margs, rank, own
    )
    cb["x_cm"], cb["x_ready"] = CM_DEV, True
    cb["tag"] = ep & 255
    for i, region in enumerate(COUNTERS):
        cb[region] = sreg(a["epoch"], 4 * (EPOCH_COUNTERS + i * moe.TAGS), region)
    for task in range(first_task(bid, 0), SLICES, BLOCKS):
        if const_expr(part is not None):
            seam.stage_push(ca, task, part)
        seam.stage_reduce(ca, task)
        seam.stage_slice(ca, task)
    for t in range(first_task(bid, GATE0), s, BLOCKS):
        seam.stage_gate(ca, t)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        seam.stage_norm(ca, t, lambda *v: store_normed(normed, *v))
        publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
    gpu.barrier()

    # ---- the MoE, its all-reduce into ``out``
    moe.run_moe(cb, mkey, bid, 0, 0)

    def reset():
        # the MoE counter slot 128 launch pairs ahead: its last use long
        # done, its next use far off
        slot = (ep + 128) & 255
        for region in COUNTERS:
            gstore(cb[region].value + fx.Int64(slot * 4), fx.Int32(0), words=1)

    epoch_end({"bid": bid, "tid": tid}, a["epoch"], ep, reset)


# ---------------------------------------------------------------- the FFN launch


def build_mono_ffn(key: MonoBuild):
    """The FFN launch of a layer whose attention vLLM ran (``part``: this rank's
    unreduced wo_b output, bf16 [S, HIDDEN]): ``run_ffn``, the partial pushed
    to every rank first. It moves the epoch on at its end."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    MoeLds = moe.moe_smem(s, rs, False)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class FfnLds:
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_ffn", s=s, tp=tp)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kffn(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):
        _ = keyed
        lds = fx.SharedAllocator().allocate(FfnLds)
        ep = _epoch(epoch)
        par = fx.Int64(ep & 1) * fx.Int64(half)
        bases = [b + par for b in peer_bases(peers, tp)]
        epoch_begin({"bid": fx.block_idx.x, "tid": fx.thread_idx.x}, epoch, ep)
        a = dict(
            res_in=res_in,
            post_in=post_in,
            comb_in=comb_in,
            pre_in=pre_in,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            ffn_w=ffn_w,
            res_out=res_out,
            post_out=post_out,
            comb_out=comb_out,
            pre_out=pre_out,
            gate_w=gate_w,
            bias=bias,
            w13=w13,
            w13_s=w13_s,
            w2=w2,
            w2_s=w2_s,
            sgu=sgu,
            sgu_s=sgu_s,
            sw2=sw2,
            sw2_s=sw2_s,
            out=out,
            scratch=scratch,
            epoch=epoch,
        )
        amb = Mailbox((ep << 1) + 2)
        run_ffn(key, lds, a, amb, bases, sym + par, rank, ep, part=part)

    @flyc.jit
    def launch(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            part,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
