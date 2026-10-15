# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build K1 (attention seam and front) and the FFN launch (seam, MoE and TP all-reduce)."""

from dataclasses import dataclass
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sglang.experimental.monokernels.hip.common.epoch import (
    EPOCH_MARKS,
    epoch_begin,
    epoch_end,
    gstore,
    kernel_symbol,
)
from sglang.experimental.monokernels.hip.common.guard import register_sites
from sglang.experimental.monokernels.hip.common.mx import clamp_fp8
from sglang.experimental.monokernels.hip.common.ops import (
    CM_DEV,
    butterfly,
    fp8_pack4,
    peer_bases,
    rsrc,
    traced,
)
from sglang.experimental.monokernels.hip.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    key_tuple,
)
from sglang.experimental.monokernels.hip.common.sync import (
    Mailbox,
    WaitGuard,
    publish,
    sreg,
)

from .attention import front as mb_front
from .attention.plan import Dims, front_scratch
from .sites import WAIT_SITES
from .sources import SOURCES
from .stages import moe, seam
from .stages.dims import Dims as StageDims
from .stages.moe_shape import SORT_NETS, MoeBuild, route_shape

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
HIDDEN = seam.HIDDEN
X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
SLICES = seam.SLICES  # 160
# K1 and the MoE share the MXFP8 x region (plain words in both); other names get one region each
SHARED_REGIONS = ("x8", "x8s")
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTERS = ("ugq", "dq")
# epoch buffer words: the epoch, one mark per CTA, then the MoE counters on their own lines
EPOCH_COUNTERS = -(-(EPOCH_MARKS + BLOCKS) // 64) * 64
EPOCH_WORDS = EPOCH_COUNTERS + COUNTER_WORDS
register_sites(WAIT_SITES)


@dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int
    # a fault-injection build stalls wait site ``fault_site`` from launch ``fault_after`` on; "" in normal builds
    fault_site: str = ""
    fault_after: int = 0

    @property
    def fault(self) -> Optional[tuple]:
        return (self.fault_site, self.fault_after) if self.fault_site else None


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(tokens=key.tokens, tp=key.tp)


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(key.tokens, key.tp)


def scratch_layout(s: int, tp: int) -> dict:
    """Return each scratch region of width s -> (byte offset, bytes), without overlap."""
    d = Dims(tp)
    out: dict = {}
    off = 0

    def place(regions):
        nonlocal off
        base = off
        for name, (o, n) in regions.items():
            if name in out and name not in SHARED_REGIONS:
                raise ValueError(f"scratch region {name} placed twice")
            out[name] = (base + o, n)
        off = max(off, base + max(o + n for o, n in regions.values()))
        off = -(-off // 256) * 256

    place(front_scratch(s, d))
    place(seam.scratch_layout(s))
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


def _check_width(s: int) -> None:
    if not 1 <= s <= MAX_TOKENS:
        raise ValueError(f"{s} rows; the launches serve 1 to {MAX_TOKENS}")


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- K1


@traced
def store_x8(c, t, col, ys, live):
    """Store the seam's norm output at ``col`` as MXFP8 X8 / X8S for the front."""
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = moe.vllm_mx_code(butterfly(amax, (1, 2), fx.max))
    mul = ((254 - code) << 23).bitcast(fx.Float32)
    w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
    w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
    if live:
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["x8"]),
            t * X_WORDS + col // 4,
            cache_modifier=CM_DEV,
        )
        if c["tid"] % 4 == 0:
            bo.buffer_store(
                code, rsrc(c["x8s"]), t * X_GROUPS + col // 32, cache_modifier=CM_DEV
            )


@traced
def publish_x8(c, t):
    """Token t's two XRDY flags (one a wqkv K half) once its X8 / X8S landed."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < 2:
        c["mb"].put(c["xrdy"], 2 * t + c["tid"], fx.Int32(1))
    gpu.barrier()


def build_mono_k1(key: MonoBuild):
    s, tp = key.tokens, key.tp
    _check_width(s)
    layout = scratch_layout(s, tp)
    fkey = _front_key(key)
    FrontLds = mb_front.front_lds(s, Dims(tp))
    NORM0 = SLICES  # the norms on the CTAs past the slices: the wqkv waits on them
    GATE0 = SLICES + s
    if GATE0 + s > BLOCKS:
        raise ValueError(
            f"K1 at {s} rows needs {GATE0 + s} CTAs; the launch has {BLOCKS}"
        )

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_k1", s=s, tp=tp)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        q: Int64,
        scratch: Int64,
        epoch: Int64,
    ):
        _ = keyed
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(K1Lds)
        sl = lds.seam.peek()
        ep = _epoch(epoch)
        tag = (ep << 1) + 1
        guard = WaitGuard(epoch, ep, tp, key.fault)
        fargs = {
            "x": 0,
            "x_stride": 0,
            "wqkv": wqkv,
            "wqkv_s": wqkv_s,
            "qn_w": qn_w,
            "kvn_w": kvn_w,
            "wqb": wqb,
            "wqb_s": wqb_s,
            "cos_sin": cos_sin,
            "pos": pos,
            "slot": slot,
            "swa": swa,
            "swa_stride": swa_stride,
            "swa_block": swa_block,
            "q": q,
        }
        flayout = {n: layout[n] for n in front_scratch(s, Dims(tp))}
        c = mb_front.front_context(
            fkey, lds.front.peek(), fargs, scratch, tag, flayout, guard
        )
        amb = Mailbox(tag, guard)
        tid = fx.thread_idx.x
        ca = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rl": sl.rl.ptr,
            "fl": sl.fl.ptr,
            "red": sl.red.ptr,
            "put": amb.put,
            "put_bf": amb.put_bf,
            "put_words": amb.put_words,
            "poll": amb.poll,
            "args": {
                "res_in": res_in,
                "pend": pend,
                "post_in": post_in,
                "comb_in": comb_in,
                "pre_in": pre_in,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "norm_w": attn_w,
                "res_out": res_out,
                "post_out": post_out,
                "comb_out": comb_out,
                "pre_out": pre_out,
            },
            "d": StageDims(tp),
            # SGLang's Triton boundary + rmsnorm arithmetic (seam.slice_row, stage_norm)
            "sgl": True,
        }
        for region in ("lin", "pmix"):
            ca[region] = sreg(scratch, layout[region][0], region)

        def seam_slice():
            for task in range(first_task(bid, 0), SLICES, BLOCKS):
                seam.stage_slice(ca, task)
            gpu.barrier()

        def seam_rest():
            for t in range(first_task(bid, NORM0), s, BLOCKS):
                seam.stage_norm(ca, t, lambda *v: store_x8(c, *v))
                publish_x8(c, t)
            for t in range(first_task(bid, GATE0), s, BLOCKS):
                seam.stage_gate(ca, t)
            gpu.barrier()

        mb_front.run_front(
            c, fkey, bid, x8_given=True, before_wqkv=seam_slice, after_wqkv=seam_rest
        )

    @flyc.jit
    def launch(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        q: Int64,
        scratch: Int64,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            attn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            wqkv,
            wqkv_s,
            qn_w,
            kvn_w,
            wqb,
            wqb_s,
            cos_sin,
            pos,
            slot,
            swa,
            swa_stride,
            swa_block,
            q,
            scratch,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- the FFN launch


@traced
def store_normed(normed, t, col, ys, live):
    """Store the FFN seam's norm output as the MoE input row (bf16, device scope)."""
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
    """Run the FFN seam, the MoE, its all-reduce into ``a["out"]``, and the epoch update."""
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
    moe.run_moe(cb, mkey, bid)

    def reset():
        # the MoE counter slot 128 launch pairs ahead: its last use is long done
        slot = (ep + 128) & 255
        for region in COUNTERS:
            gstore(cb[region].value + fx.Int64(slot * 4), fx.Int32(0), words=1)

    epoch_end({"bid": bid, "tid": tid}, a["epoch"], ep, reset, amb.guard)


def build_mono_ffn(key: MonoBuild):
    """Build the FFN launch: push this rank's wo_b partial, then ``run_ffn``."""
    s, tp = key.tokens, key.tp
    _check_width(s)
    rs = route_shape(_moe_key(key))
    if rs.experts // 64 not in SORT_NETS:
        raise ValueError(f"no sort network for {rs.experts} experts")
    MoeLds = moe.moe_smem(s, rs)
    half = peer_half_bytes(tp)
    if SLICES + 2 * s > BLOCKS:
        raise ValueError(
            f"the FFN launch at {s} rows needs {SLICES + 2 * s} CTAs; it has {BLOCKS}"
        )
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
        amb = Mailbox((ep << 1) + 2, WaitGuard(epoch, ep, tp, key.fault))
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
