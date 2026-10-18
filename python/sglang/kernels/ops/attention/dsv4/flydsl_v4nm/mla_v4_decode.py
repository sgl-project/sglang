# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter#5882 (https://github.com/ROCm/aiter/pull/5882),
# commit 03359c43df (03359c43df40abf91ac38a2040c9b8e284d92297),
# file aiter/ops/flydsl/mla_v4_decode.py.
# Unchanged except for import paths and sglang formatting (isort, ruff format).
# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin

"""FlyDSL DeepSeek-V4 fp8 MLA decode (v4 nm contract) on gfx950.

flydsl_mla_decode_fwd_v4_nm: drop-in for aiter.mla.mla_decode_fwd_v4_nm (bf16 output).
flydsl_mla_v4_decode_fused: the same attention + inverse RoPE + wo_a mxfp8 quant.

The split plan is keyed by the batch size, head count, max_seqlen_q and a stream hint,
never by kv lengths (graph-capture safe). The merge scratch is owned here, one per
(device, stream).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import flydsl.expr as fx
import torch
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

from .kernels.mla_v4_decode_gfx950 import (
    DV,
    MAX_GROUPS,
    MAX_S,
    ROPE,
    ROWB,
    WS_HDR,
    V4Cfg,
    build_mla_v4_decode,
)
from .mla_v4_grouped import GROUPED_MAX_MSQ

__all__ = [
    "flydsl_mla_decode_fwd_v4_nm",
    "flydsl_mla_v4_decode_fused",
    "flydsl_mla_v4_decode_supported",
    "flydsl_mla_v4_grouped_pays",
]

# (H, max_seqlen_q) pairs the asm v4 nm kernel ships
SUPPORTED = ((16, 1), (16, 2), (16, 4), (32, 1), (64, 1), (128, 1))

# Split plans measured on MI355X (cold caches, vs the best asm split count + merge).
# H16 msq1 with a stream hint: T_pad upper bound -> (CSA, HCA) splits (HCA past
# T_pad 40: _plan_h16's key-balanced grid)
_POLICY_H16 = {
    8: (18, 18),
    16: (12, 12),
    24: (9, 9),
    32: (8, 8),
    40: (6, 6),
    48: (4, 4),
    64: (4, 4),
    80: (3, 3),
    128: (2, 2),
    160: (3, 1),
    256: (1, 1),
}
# below _POLICY_Q16: B upper bound -> (16-row blocks per CTA, splits) of K SPREAD plans
_POLICY_WIDE = {
    (16, 2): ((1, 1, 32), (None, 1, 16)),
    (16, 4): ((None, 1, 16),),
    (32, 1): ((1, 1, 24), (None, 1, 16)),
    (64, 1): ((None, 1, 16),),
    (128, 1): ((None, 1, 16),),
}
# stream-length bodies (V4Cfg.SUB): B upper bound -> longest short stream (K, S = 1);
# (H, msq) -> ((B max, B min, ((RB, S, longest stream), ...)), ...) mid-length streams
_SHORT_KEYS = ((8, 192), (16, 256), (64, 768), (128, 256))
# Q16 mains: 32-row ones take the short body up to _Q16_SHORT_KEYS keys; 64-row ones
# past B 16 at S >= 3 run streams of <= max(_Q16_RB2_KEYS / S, _Q16_RB2_MIN) keys on
# unsplit 32-row Q16 CTAs (at S = 2 the joint kernel spills: long streams pay more)
_Q16_SHORT_KEYS = 160
_Q16_RB2_KEYS, _Q16_RB2_MIN = 2304, 448
_POLICY_MID = {
    (16, 2): ((1, 1, ((1, 16, 2048),)), (16, 9, ((1, 8, 4096),))),
    (16, 4): ((8, 5, ((1, 8, 3072),)), (16, 9, ((1, 4, 2048),))),
    (32, 1): ((16, 9, ((1, 8, 4096),)),),
    (64, 1): ((8, 5, ((1, 8, 3072),)), (16, 9, ((1, 4, 2048),))),
}
# smallest B of each pair on Q16 CTAs
_POLICY_Q16 = {(16, 2): 9, (16, 4): 5, (32, 1): 9, (64, 1): 5, (128, 1): 2}
# grids a little past half the GPU run 2 splits: largest grid, in 1/32 of the CUs, for
# Q16 (RB, NRG) and for the H16 K layout
_PAST_HALF = {(4, 1): 18, (2, 1): 19}
_PAST_HALF_K = 19
# Q16 split count -> streams of at most this many 32-key tiles run unsplit (SHORT)
_Q16_SHORT_TILES = {2: 24, 3: 10}
# unsplit Q16 grids of fewer CTAs than CUs: split 1 of the longest streams of > _IDLE_F
# tiles on the idle CUs (V4Cfg.IDLE), past them on _IDLE_X2 more CTAs per XCD for
# streams of > _IDLE_W tiles; larger grids of two row groups (up to 512 sequences):
# splits 1-3 of streams of > _IDLE_WL tiles on _IDLE_XL CTAs per XCD past the grid
_IDLE_X2, _IDLE_F, _IDLE_W = 4, 32, 96
_IDLE_XL, _IDLE_WL = 16, 64
# PV16R key window: (lo, hi at B <= CUs, hi above)
_PV16_LO, _PV16_HI = 256, (4096, 2048)
# grouped HCA: streams of <= 17 32-key tiles on the K body, longer ones on RQ
_GROUPED_RQ_TILES = 18
# H16 bf16 without a hint: most splits per sequence; longest unsplit stream; shortest
# key-balanced SK run in 64-key tiles (grid of one, two CTAs per CU)
_H16_MAX_S = 18
_H16_SHORT_KEYS = 128
_H16_SKT = (8, 12)
# SK grids of at most this many splits per sequence: V4Cfg.TAIL where it adds some
_H16_TAIL_SKD = 8
# 64-row Q16 with fewer splits than this per sequence: V4Cfg.TAIL where it adds some
_TAIL_S = 16


@cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def flydsl_mla_v4_decode_supported(
    num_heads: int, max_seqlen_q: int = 1, grouped: bool = False
) -> bool:
    """True if the FlyDSL v4 decode covers (num_heads, max_seqlen_q) on this GPU;
    grouped: with q_kv_bounds (H = 16, max_seqlen_q <= 8)."""
    try:
        if get_gfx() != "gfx950":
            return False
    except Exception:  # noqa: BLE001
        return False
    if grouped:
        return num_heads == 16 and 1 <= max_seqlen_q <= GROUPED_MAX_MSQ
    return (num_heads, max_seqlen_q) in SUPPORTED


def flydsl_mla_v4_grouped_pays(
    num_heads: int,
    num_draft: int,
    num_requests: int,
    hint: str | None = "hca",
    epi: str = "invrope_mxfp8",
) -> bool:
    """Whether a DSpark verify step should share one stream per request (grouped
    q_kv_bounds) on its HCA / SWA-only layers (MI355X, DSv4-Pro verify step)."""
    if num_heads != 16 or not 1 <= num_draft <= GROUPED_MAX_MSQ:
        return False
    if epi == "bf16" and hint in (None, "hca"):
        return num_requests >= 32
    return num_requests >= 64


def _plan(
    H: int,
    msq: int,
    B: int,
    epi: str = "bf16",
    hint: str | None = None,
    cus: int = 256,
    grouped: bool = False,
) -> V4Cfg:
    """Variant for B sequences. hint: "csa" | "hca" | "swa" (SWA-only layer) | None."""
    cfg = _plan_for(H, msq, max(1, B), epi, hint, cus, grouped)
    if cfg.ws_need(B)[0] > MAX_GROUPS:
        # more merge groups than protocol slots: run unsplit
        cfg = dataclasses.replace(
            cfg,
            S=1,
            ADAPT=0,
            MINT=1,
            SUB=(),
            SPREAD=False,
            SHORT=0,
            SHORTM=False,
            SHORTF=0,
            OWN=False,
            SK=0,
            SKMEAN=False,
            IDLE=(),
        )
    return cfg


def _plan_for(H, msq, B, epi, hint, cus, grouped) -> V4Cfg:
    if grouped:
        return _plan_grouped(msq, B, epi, hint, cus)
    if H == 16 and msq == 1:
        return _plan_h16(B, epi, hint, cus)
    if B >= _POLICY_Q16[(H, msq)]:
        return _with_subs(_q16_plan(H, msq, B, epi, cus), B, cus)
    for bmax, rb, s in _POLICY_WIDE[(H, msq)]:
        if bmax is None or B <= bmax:
            break
    cfg = V4Cfg(H=H, MSQ=msq, RB=rb, S=s, EPI=epi, KC=True, SPREAD=True)
    return _with_subs(cfg, B, cus)


def _plan_h16(B, epi, hint, cus) -> V4Cfg:
    tpad = -(-B // 8) * 8
    adapt = {}
    if hint == "swa":
        s = 1
    elif epi == "bf16" and hint is None:
        s = min(_H16_MAX_S, max(1, cus // tpad))
        if 16 <= tpad <= 2 * cus:
            # ragged batches: the key-balanced grid (V4Cfg.SK; past cus sequences two
            # CTAs per CU)
            sk = cus * -(-tpad // cus)
            cfg = _sk_plan(B, epi, cus, 16 * cus // sk, SK=sk, SKT=_H16_SKT[sk > cus])
            if 2 * tpad > cus and tpad <= cus * _PAST_HALF_K // 32:
                cfg = dataclasses.replace(cfg, SKD=2, SKT=14, SKMEAN=True, SHORT=4)
            return cfg
        if tpad > cus:
            s = 2
            adapt = {"ADAPT": cus, "MINT": 16}
        elif tpad < 16:
            short = V4Cfg(H=16, RB=1, EPI=epi, KC=True)
            adapt = {"SUB": ((_H16_SHORT_KEYS, short),)}
    else:
        s = 1
        for tb, v in _POLICY_H16.items():
            if tpad <= tb:
                s = v[0 if hint == "csa" else 1]
                break
        if hint == "hca" and tpad > 40:
            # ragged HCA batches (a few long sessions set the longest stream): the
            # key-balanced grid; past one sequence per CU, run-time split counts
            if tpad > cus:
                return V4Cfg(H=16, RB=1, S=8, EPI=epi, KC=True, ADAPT=cus, MINT=4)
            return _sk_plan(B, epi, cus, 8, SK=cus, SKT=6)
    return V4Cfg(H=16, RB=1, S=min(s, MAX_S), EPI=epi, KC=True, **adapt)


def _sk_plan(B, epi, cus, S, **kw) -> V4Cfg:
    """H16 key-balanced grid (V4Cfg.SK) of at most max(S, SKD) splits, else SKD
    splits per sequence: cus // T_pad, or cus // B on V4Cfg.TAIL where that is more"""
    tpad = -(-B // 8) * 8
    skd = max(1, min(_H16_MAX_S, cus // tpad))
    tail = skd < cus // B <= _H16_TAIL_SKD
    skd = cus // B if tail else skd
    return V4Cfg(
        H=16,
        RB=1,
        EPI=epi,
        KC=True,
        S=max(S, skd),
        SKD=skd,
        TAIL=tail,
        MINT=3 if 4 * tpad > cus else 1,
        SHORT=_H16_SHORT_KEYS // 64,
        **kw,
    )


def _with_subs(cfg: V4Cfg, B: int, cus: int) -> V4Cfg:
    """cfg + its stream-length bodies where they fit (at most the sequence's CTAs; the
    short body at most one CTA per CU over the batch; under Q16 16-row K or 32-row Q16
    bodies)."""
    if cfg.S == 1 or cfg.XR or cfg.IDLE:
        return cfg
    if cfg.Q16 and cfg.RB == 4 and B > 16 and cfg.S >= 3:
        rb2 = dataclasses.replace(cfg, RB=2, S=1, SHORT=0, PF16=False)
        keys = max(_Q16_RB2_KEYS // cfg.S, _Q16_RB2_MIN)
        # (the main's short streams all take the body)
        return dataclasses.replace(cfg, SHORT=0, SUB=((keys, rb2),))
    subs = []
    keys = next((k for bmax, k in _SHORT_KEYS if B <= bmax), 0)
    if cfg.Q16 and cfg.RB == 2:
        keys = min(keys, _Q16_SHORT_KEYS)
    rbs = (1,) if cfg.Q16 else (1, 2)
    for rb in rbs if keys else ():
        nrg = cfg.H * cfg.MSQ // (16 * rb)
        if nrg <= cfg.NRG * cfg.S and B * nrg <= cus:
            subs.append((keys, _k_body(cfg, rb, 1)))
            break
    for bmax, bmin, mids in _POLICY_MID.get((cfg.H, cfg.MSQ), ()):
        if bmin <= B <= bmax:
            for rb, s, mkeys in mids:
                sub = _k_body(cfg, rb, s)
                fits = rb in rbs and sub.NRG * s <= cfg.NRG * cfg.S and sub != cfg
                if fits and mkeys > keys:
                    subs.append((mkeys, sub))
                    keys = mkeys
            break
    return dataclasses.replace(cfg, SUB=tuple(subs)) if subs else cfg


def _q16_plan(H: int, msq: int, B: int, epi: str, cus: int) -> V4Cfg:
    rb = 4 if H * msq % 64 == 0 else 2
    nrg = H * msq // (16 * rb)
    bpad = -(-B // 8) * 8
    s = max(1, min(MAX_S, cus // (bpad * nrg)))
    xr = nrg > 1 and bpad <= 8
    if nrg > 1 and bpad > 8:
        # row groups on consecutive XCDs where that allows more splits
        s_xr = min(MAX_S, cus // 8 // -(-B * nrg // 8))
        if s_xr > s:
            s, xr = s_xr, True
    # the units past the last complete XCD octet over all XCDs (V4Cfg.TAIL)
    tail_ok = rb == 4 and s < _TAIL_S and 8 < B < cus // 8 * nrg and B % 8
    s_tl = _tail_splits(B, nrg, xr, cus) if tail_ok else 0
    tail, s = s_tl > s, max(s, s_tl)
    mean_short = rb == 4 and nrg == 1 and 2 * bpad == cus
    past_half = (
        s == 1
        and not xr
        and cus < 2 * bpad * nrg
        and bpad * nrg <= cus * _PAST_HALF.get((rb, nrg), 0) // 32
    )
    if past_half:
        s = 2
    if s == 1 and not xr and (bpad * nrg < cus or nrg > 1 and bpad <= 512):
        s, idle = 2, (cus // 8, _IDLE_X2, _IDLE_F, _IDLE_W)
        if bpad * nrg >= cus:
            s, idle = 4, (bpad // 8 * nrg, _IDLE_XL, _IDLE_F, _IDLE_WL)
        return V4Cfg(
            H=H,
            MSQ=msq,
            RB=rb,
            S=s,
            EPI=epi,
            LAYOUT="RD",
            Q16=True,
            PF16=True,
            PFN=cus if bpad * nrg > cus else 0,
            IDLE=idle,
        )
    if s == 1:
        short = 0
    elif mean_short:
        short = 64
    elif past_half:
        short = 1 << 16
    elif bpad <= 8:
        short = 6
    else:
        short = _Q16_SHORT_TILES.get(s, 8)
    cfg = V4Cfg(
        H=H,
        MSQ=msq,
        RB=rb,
        S=s,
        EPI=epi,
        LAYOUT="RD",
        Q16=True,
        SHORT=short,
        SHORTM=(mean_short or past_half) and s > 1,
        SHORTF=16 if past_half else 0,
        PF16=s > 1,
        XR=xr,
        PFN=cus if bpad * nrg > cus else 0,
        TAIL=tail,
    )
    if rb == 4 and nrg == 1 and bpad >= cus and epi == "bf16":
        cfg = dataclasses.replace(
            cfg, PV16R=(_PV16_LO, _PV16_HI[0] if bpad <= cus else _PV16_HI[1])
        )
    return cfg


def _tail_splits(B: int, nrg: int, xr: bool, cus: int) -> int:
    # most splits under V4Cfg.TAIL with at most one CTA per CU on every XCD (one row
    # group: 7/8 of them; a full XCD costs up to 1.25x at 1k-4k keys; XR: at most 12,
    # more cost ~8 % at 200-500 keys)
    units, per = (B * nrg, 1) if xr else (B, nrg)
    cap = cus // 8 if nrg > 1 else cus * 7 // 64
    s = 12 if xr else MAX_S
    while s > 1 and units // 8 * s * per + -(-(units % 8) * s * per // 8) > cap:
        s -= 1
    return s


def _k_body(cfg: V4Cfg, rb: int, s: int) -> V4Cfg:
    return V4Cfg(
        H=cfg.H,
        MSQ=cfg.MSQ,
        RB=rb,
        S=s,
        EPI=cfg.EPI,
        KC=True,
        SPREAD=cfg.SPREAD,
        TAIL=cfg.TAIL,
    )


def _plan_grouped(msq: int, B: int, epi: str, hint: str | None, cus: int) -> V4Cfg:
    """Grouped verify at H = 16: B groups of up to msq drafts sharing one stream."""
    hca = hint in (None, "hca")
    if epi == "bf16" and (B > 32 or (B == 32 and hca)):
        return _grouped_q16(msq, B, hca, cus)
    if B >= 32 and hca:
        # long streams on one 128-row RQ CTA per (group, split), short ones on K
        short = V4Cfg(
            H=16,
            MSQ=GROUPED_MAX_MSQ,
            RB=2 if B >= 64 else 1,
            EPI=epi,
            KC=True,
            MASK="bounds",
        )
        return V4Cfg(
            H=16,
            MSQ=GROUPED_MAX_MSQ,
            RB=GROUPED_MAX_MSQ,
            S=4,
            EPI=epi,
            LAYOUT="RQ",
            MASK="bounds",
            OWN=True,
            SUB=((32 * (_GROUPED_RQ_TILES - 1), short),),
        )
    if B > 32:
        return V4Cfg(
            H=16,
            MSQ=msq + (msq & 1),
            RB=2,
            EPI=epi,
            KC=hint != "swa",
            MASK="bounds",
        )
    s = 1 if hint == "swa" or B > 16 else (2 if B > 8 else 4)
    return V4Cfg(H=16, MSQ=msq, RB=1, S=s, EPI=epi, MASK="bounds")


def _grouped_q16(msq: int, B: int, hca: bool, cus: int) -> V4Cfg:
    """32-row Q16 CTAs (two drafts each), 64-row ones where the 32-row grid exceeds
    one CTA per CU and they need fewer."""
    mp = msq + (msq & 1)
    rb = 2
    nrg = 16 * mp // 32
    bpad = -(-B // 8) * 8
    mp4 = -(-msq // 4) * 4
    if bpad * nrg > cus and 16 * mp4 // 64 < nrg:
        mp, rb, nrg = mp4, 4, 16 * mp4 // 64
    s = max(1, min(4, cus // (bpad * nrg))) if hca else 1
    return V4Cfg(
        H=16,
        MSQ=mp,
        RB=rb,
        S=s,
        LAYOUT="RD",
        MASK="bounds",
        Q16=True,
        SHORT=16 if s > 1 else 0,
        PF16=s > 1,
    )


class _Workspace:
    """Grow-only merge scratch of one (device, stream): protocol slots (hdr, zero
    between launches) + split partials (buf). A new stream takes a spare hdr zeroed
    at an earlier eager launch, so a graph captured on it replays no memset. Old
    buffers stay alive: a captured graph may still reference them."""

    def __init__(self, device):
        self.device = device
        self.hdr = None
        self.buf = None
        self._keep = []

    def ensure(self, nbytes: int):
        if self.hdr is None:
            self.hdr = _HDR_SPARE.pop(self.device.index, None)
            if self.hdr is None:
                self.hdr = torch.zeros(
                    WS_HDR // 4, dtype=torch.int32, device=self.device
                )
        if (
            not torch.cuda.is_current_stream_capturing()
            and self.device.index not in _HDR_SPARE
        ):
            _HDR_SPARE[self.device.index] = _zeroed_hdr(self.device)
        need = -(-nbytes // 4)
        if self.buf is None or self.buf.numel() < need:
            if self.buf is not None:
                self._keep.append(self.buf)
                need = max(need, self.buf.numel() * 2)
            self.buf = torch.empty(need, dtype=torch.int32, device=self.device)
        return self


_WORKSPACES: dict = {}
_HDR_SPARE: dict = {}
_ZERO_STREAMS: dict = {}
_COMPILED: set = set()
_DUMMY: dict = {}


def _zeroed_hdr(device):
    # zeroed on an idle side stream, before any stream can use it
    zs = _ZERO_STREAMS.get(device.index)
    if zs is None:
        zs = _ZERO_STREAMS[device.index] = torch.cuda.Stream(device)
    with torch.cuda.stream(zs):
        hdr = torch.zeros(WS_HDR // 4, dtype=torch.int32, device=device)
    zs.synchronize()
    return hdr


def _workspace(device) -> _Workspace:
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = _Workspace(device)
    return ws


def _launch(
    cfg: V4Cfg,
    B,
    q,
    q_rope,
    kv,
    kv_rope,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sink,
    out,
    out_s0,
    out_s1,
    xs=None,
    positions=None,
    freqs=None,
    bounds=None,
):
    device = q.device
    dmy = _DUMMY.get(device.index)
    if dmy is None:
        dmy = _DUMMY[device.index] = torch.zeros(64, dtype=torch.int64, device=device)
    groups, nbytes = cfg.ws_need(B)
    if groups:
        ws = _workspace(device).ensure(nbytes)
        hdr, wsb = ws.hdr, ws.buf
    else:
        hdr = wsb = dmy
    if cfg not in _COMPILED and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"flydsl_mla_v4_decode: variant {cfg} not compiled; run this batch size "
            "once eagerly before graph capture"
        )
    u8 = torch.uint8
    epi = cfg.EPI == "invrope_mxfp8"
    _run_compiled(
        build_mla_v4_decode(cfg),
        ptr_arg(q.view(u8), fx.Uint8),
        ptr_arg(q_rope, fx.BFloat16),
        ptr_arg(kv.view(u8), fx.Uint8),
        ptr_arg(kv_rope, fx.BFloat16),
        ptr_arg(dmy if qo_indptr is None else qo_indptr, fx.Int32),
        ptr_arg(kv_indptr, fx.Int32),
        ptr_arg(kv_indices, fx.Int32),
        ptr_arg(sink, fx.Float32),
        ptr_arg(out.view(u8) if epi else out, fx.Uint8 if epi else fx.BFloat16),
        ptr_arg(dmy if xs is None else xs.view(u8), fx.Uint8),
        ptr_arg(dmy if positions is None else positions, fx.Int64),
        ptr_arg(dmy if freqs is None else freqs, fx.Float32),
        ptr_arg(wsb, fx.Int32),
        ptr_arg(dmy if bounds is None else bounds, fx.Int32),
        ptr_arg(hdr, fx.Int32),
        B,
        kv.shape[0],
        out_s0,
        out_s1,
        fx.Stream(torch.cuda.current_stream(device)),
    )
    _COMPILED.add(cfg)


def _check_bounds(q_kv_bounds, total_q):
    if q_kv_bounds is not None:
        assert q_kv_bounds.dtype == torch.int32 and q_kv_bounds.is_contiguous()
        assert q_kv_bounds.dim() == 2 and q_kv_bounds.shape[1] == 4
        assert q_kv_bounds.shape[0] >= total_q, (q_kv_bounds.shape, total_q)


def flydsl_mla_decode_fwd_v4_nm(
    q,
    qrope,
    kv_buffer,
    kvrope,
    output,
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    *,
    sink,
    hint=None,
    q_kv_bounds=None,
):
    """aiter.mla.mla_decode_fwd_v4_nm on gfx950: writes the final bf16 attention into
    ``output`` [total_q, H, 512] (rows without keys become NaN).

    hint: "csa" | "hca" | "swa" | None, only selects the split plan.
    q_kv_bounds: optional int32 [total_q, 4] (a0, a1, b0, b1); q row r attends stream
    position j of its sequence iff a0 <= j < a1 or b0 <= j < b1 (grouped verify).
    """
    H = q.shape[1]
    msq = int(max_seqlen_q)
    grouped = q_kv_bounds is not None
    if not flydsl_mla_v4_decode_supported(H, msq, grouped):
        raise RuntimeError(
            f"flydsl_mla_decode_fwd_v4_nm: no variant for num_heads={H}, "
            f"max_seqlen_q={msq}, grouped={grouped}"
        )
    B = qo_indptr.numel() - 1
    assert q.dim() == 3 and q.shape[2] == ROWB and q.element_size() == 1, q.shape
    assert qrope.shape[1:] == (H, ROPE) and qrope.dtype == torch.bfloat16
    assert q.is_contiguous() and qrope.is_contiguous()
    assert kv_buffer.element_size() == 1 and kvrope.dtype == torch.bfloat16
    kv = kv_buffer.reshape(-1, ROWB)
    kv_rope = kvrope.reshape(-1, ROPE)
    assert kv.shape[0] == kv_rope.shape[0]
    assert sink.dtype == torch.float32 and sink.is_contiguous() and sink.numel() == H
    for t in (qo_indptr, kv_indptr, kv_page_indices):
        assert t.dtype == torch.int32 and t.is_contiguous(), t.dtype
    assert output.dtype == torch.bfloat16 and output.shape[1:] == (H, DV)
    assert output.stride(2) == 1
    _check_bounds(q_kv_bounds, q.shape[0])
    cfg = _plan(H, msq, B, "bf16", hint, _num_cus(q.device.index), grouped)
    if B > 0:
        _launch(
            cfg,
            B,
            q,
            qrope,
            kv,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            sink,
            output,
            output.stride(0),
            output.stride(1),
            bounds=q_kv_bounds,
        )
    return output


def flydsl_mla_v4_decode_fused(
    q,
    q_rope,
    kv,
    kv_rope,
    kv_indptr,
    kv_indices,
    sink,
    positions,
    freqs,
    compress_ratio,
    *,
    qo_indptr=None,
    max_seqlen_q: int = 1,
    q_kv_bounds=None,
):
    """DeepSeek-V4 decode tail in one launch: attention + split merge + inverse RoPE of
    dims 448..511 + wo_a mxfp8 quant.

    q / q_rope / kv / kv_rope / sink: as flydsl_mla_decode_fwd_v4_nm; one sequence per
    token (kv_indptr [T + 1]) unless q_kv_bounds + qo_indptr + max_seqlen_q give
    grouped-verify streams. positions [T] int64; freqs [max_pos, 64] fp32 =
    view_as_real(freqs_cis). compress_ratio: 4 CSA, 128 HCA, else SWA-only.
    Returns (xq [T, H*512] fp8_e4m3fn, xs [T, H*4] e8m0); tokens without keys get NaN.
    """
    T, H = q.shape[0], q.shape[1]
    msq = int(max_seqlen_q)
    grouped = q_kv_bounds is not None
    if not (msq == 1 or grouped) or not flydsl_mla_v4_decode_supported(H, msq, grouped):
        raise RuntimeError(
            f"flydsl_mla_v4_decode_fused: no variant for num_heads={H}, "
            f"max_seqlen_q={msq}, grouped={grouped}"
        )
    if grouped:
        assert qo_indptr is not None and qo_indptr.dtype == torch.int32
        B = qo_indptr.numel() - 1
    else:
        B = T
    assert q.shape[2] == ROWB and q_rope.shape[1:] == (H, ROPE)
    assert q.is_contiguous() and q_rope.is_contiguous() and q.element_size() == 1
    assert q_rope.dtype == kv_rope.dtype == torch.bfloat16
    assert kv_indptr.dtype == kv_indices.dtype == torch.int32
    assert sink.dtype == freqs.dtype == torch.float32 and positions.dtype == torch.int64
    assert sink.is_contiguous() and freqs.is_contiguous() and kv.element_size() == 1
    assert freqs.shape[-1] == ROPE
    kv = kv.reshape(-1, ROWB)
    kv_rope = kv_rope.reshape(-1, ROPE)
    assert kv.shape[0] == kv_rope.shape[0]
    xq = torch.empty((T, H * DV), dtype=torch.float8_e4m3fn, device=q.device)
    xs = torch.empty((T, H * DV // 128), dtype=torch.uint8, device=q.device)
    _check_bounds(q_kv_bounds, T)
    hint = {4: "csa", 128: "hca"}.get(compress_ratio, "swa")
    cfg = _plan(H, msq, B, "invrope_mxfp8", hint, _num_cus(q.device.index), grouped)
    if T > 0 and B > 0:
        _launch(
            cfg,
            B,
            q,
            q_rope,
            kv,
            kv_rope,
            None if cfg.MSQ == 1 else qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            xq,
            0,
            0,
            xs,
            positions,
            freqs,
            q_kv_bounds,
        )
    return xq, xs
