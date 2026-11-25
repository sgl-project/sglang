"""SM90 NVFP4 W4A16 EP experts + post-norm (gate: MuseSpark1xFp4Experts.use_fused_w4a16); NVRTC kernels
moe_w4a16.cuh, and moe_w4a16_big.cuh for SP batches from BIG_MIN.
Adapted from https://github.com/gauravjain14/muse-okc-submissions (muse_moe_w4a16; Apache-2.0), which credits
vjeux/okc-2026 and alygg77/okc-muse-spark-kernels.
"""

from __future__ import annotations

import ctypes
import functools
from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

MAX_TOKENS = 2048
BIG_MIN = 2049
# SP batches from here (NT >= 14 instances) take dynamic guests
DG_MIN = 769
SP_MIN = 65  # SP instances exist for NT >= 8 only (batches above 64)
BIG_TOKENS = 8192
# init_runtime raises it to the prefill chunk, assumed a multiple of lcm(128, tp)
_BIG_MAXT = 16384
_BIG_SLOTS = 8 * BIG_TOKENS + 1152
DYNAMIC_GUESTS = 2
_BIG_NE = 32 + DYNAMIC_GUESTS
# A large-M launch has at most BIG_TOKENS / 32 + _BIG_NE tiles of 256 rows; the kernel's
# per-tile done and ready counters (C_DONE, C_READY) are 1024 words apart.
assert BIG_TOKENS % 128 == 0 and BIG_TOKENS // 32 + _BIG_NE <= 1024
_BIG_SMEM = 4 * 38912 + 69632 + 3328 + 1280
_BIG_WIDTHS = (32, 64, 96, 128, 160, 192, 224, 256, 240)
_BIG_CTR = 3400 + 32
_HIDDEN = 4096
_INTER = 4096
_TOP_K = 8
_NUM_LOCAL = 32
_BLOCK = 35840

# (max tokens, moe_body's (RW, NT, NS, MW), dynamic smem)
_CONFIGS = (
    (16, (4, 4, 4, 1), 232448),
    (32, (8, 4, 4, 1), 232448),
    (64, (8, 4, 4, 2), 232448),
    (128, (8, 8, 3, 4), 228352),
    (256, (8, 8, 3, 8), 232448),
    (512, (8, 8, 3, 16), 228352),
    (1024, (8, 8, 3, 32), 229376),
)
_SP_CONFIGS = tuple(c for c in _CONFIGS if SP_MIN <= c[0] <= 512) + (
    (768, (8, 8, 3, 32), 229376),
    (1024, (8, 16, 2, 32), 230400),
    (2048, (8, 14, 2, 64), 232448),
)
_THREADS = 384
_CTR_WORDS = 65536 + 16384 + 64

_P, _I, _U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64


class _Params(ctypes.Structure):
    _fields_ = cu.fields(
        (_P, "x ids topk_w"),
        (_I, "M"),
        (_P, "w13 gs13 w2 gs2 nw1 inter ypart out ctr"),
        (_I, "pmax"),
        (_P, "xs yh scr"),
        (_P * 8, "xsp osp"),
        (_I, "R"),
        (_P, "ptj"),
        (_P, "sw13 sw2 gsrc peer"),
    )


class _BigParams(ctypes.Structure):
    _fields_ = cu.fields(
        (_U64 * 96, "wm"),
        (_U64 * 96, "wsg"),
        (_P, "th tx2 th2 txh thh ids w13 w2 gs13 gs2 nw xt ctr rev ucnt"),
        (_I, "lag M cap first ne R"),
        (_P * 8, "xs os"),
        (_I, "cl pre"),
        (_P, "sw13 sw2 sgs gsrc peer"),
    )


def _wgmma_src(widths):
    """wgmma_rs<N> (m64nNk16 bf16, A from registers) for the kernels' `// @WGMMA_RS@` slot."""
    src = ""
    for n in widths:
        d = ",".join(f"%{i}" for i in range(n // 2))
        a = ",".join(f"%{n // 2 + i}" for i in range(4))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(n // 2))
        src += (
            f"template <> DEVI void wgmma_rs<{n}>(float* d, const u32* a, u64 desc) {{\n"
            f'  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, {{{a}}}, %{n // 2 + 4}, 1, 1, 1, 0;"\n'
            f'    : {outs}\n    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));\n}}\n'
        )
    return src


def is_supported(
    hidden_size: int, intermediate_size: int, num_local_experts: int, top_k: int
) -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return False
    try:
        from cuda.bindings import driver, nvrtc  # noqa: F401
    except ImportError:
        return False
    return (hidden_size, intermediate_size, num_local_experts, top_k) == (
        _HIDDEN,
        _INTER,
        _NUM_LOCAL,
        _TOP_K,
    )


def _compile(name, src):
    return cu.build_module(name, src, [b"--gpu-architecture=sm_90a", b"-std=c++17"])


class _Module:
    def __init__(self):
        sp_cfgs = [(c, m) for _, c, m in _SP_CONFIGS]
        inst = lambda cs: "".join(
            "INST{}({}, {}, {}, {})\n".format(p, *c)
            for p, l in cs
            for c in (x[-2] for x in l)
        )
        src = cu.read_source("moe_w4a16.cuh").replace(
            "// @WGMMA_RS@\n", _wgmma_src(range(24, 136, 8))
        )
        self.module = _compile(
            "moe_w4a16.cu", src + inst((("", _CONFIGS), ("SP", sp_cfgs)))
        )
        self.funcs = {
            c: cu.function(self.module, "moe_" + "_".join(map(str, c)), m)
            for _, c, m in _CONFIGS
        }
        self.funcs.update(
            {
                ("sp",) + c: cu.function(
                    self.module, "moesp_" + "_".join(map(str, c)), m
                )
                for c, m in sp_cfgs
            }
        )
        src = cu.read_source("moe_w4a16_big.cuh").replace(
            "// @WGMMA_RS@\n", _wgmma_src(_BIG_WIDTHS)
        )
        src = (
            "#define PFM_CASES "
            + " ".join(f"PFM_CASE({n})" for n in _BIG_WIDTHS)
            + "\n"
            + src
        )
        self.big_module = _compile("moe_w4a16_big.cu", src)
        self.big = cu.function(self.big_module, "pfm_main", _BIG_SMEM)
        self.fin = cu.function(self.big_module, "moe_w4a16_sp_fin", 65536 + 32)
        self.fin_ctas = 2 * torch.cuda.get_device_properties().multi_processor_count
        self.split = [
            cu.function(self.big_module, k)
            for k in ("moe_w4a16_split_cnt", "moe_w4a16_dsplit")
        ]

    def launch_fin(self, recv, tw, out, R, rows):
        vals = [_P(recv), _P(tw), _P(out), _I(R), _I(rows)]
        cu.launch(
            self.fin,
            min(rows, self.fin_ctas),
            256,
            65536 + 32,
            torch.cuda.current_stream().cuda_stream,
            vals,
        )

    def launch_dsplit(self, ids, cnt, out, gsrc, me, dur):
        s, n = torch.cuda.current_stream().cuda_stream, ids.numel()
        rev, ctr = (
            (_big_workspace(ids.device.index)[k].data_ptr() for k in ("rev", "ctr"))
            if 8 * BIG_MIN <= n <= 8 * _BIG_MAXT
            else (0, 0)
        )
        cu.launch(
            self.split[0],
            (n + 2047) // 2048,
            256,
            0,
            s,
            [
                _P(ids.data_ptr()),
                _P(cnt.data_ptr()),
                _I(n),
                _P(dur.data_ptr()),
                _P(ctr),
            ],
        )
        cu.launch(
            self.split[1],
            (n + 1023) // 1024,
            256,
            0,
            s,
            [_P(t.data_ptr()) for t in (ids, cnt, out, gsrc)]
            + [_I(n), _I(me), _I(n < 8 * BIG_MIN), _P(rev), _P(ctr)],
            [cu.pdl()],
        )


@functools.lru_cache(maxsize=None)
def _module(device_index: int) -> _Module:
    with torch.cuda.device(device_index):
        # driver API loads into torch's primary context
        torch.cuda.current_stream().synchronize()
        return _Module()


@functools.cache
def _workspace(device_index: int) -> dict:
    dev = torch.device("cuda", device_index)
    i32 = dict(dtype=torch.int32, device=dev)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return {
        "sms": sms,
        "ctr": torch.zeros(_CTR_WORDS, **i32),
        "scratch": torch.empty(sms * 64 * 1040 // 4, **i32),
        "ptj": torch.empty(MAX_TOKENS * _TOP_K, **i32),
        "split": torch.tensor([0] * 257 + [1024] * 8, **i32),
    }


def _tensor_map(t: torch.Tensor, box_rows: int) -> bytes:
    rows, cols = t.shape
    if box_rows < 0:
        return cu.tensor_map(
            t.data_ptr(), [64, rows, cols // 64], [cols * 2, 128], [64, -box_rows, 2]
        )
    return cu.tensor_map(t.data_ptr(), [cols, rows], [cols * 2], [64, box_rows])


@functools.lru_cache(maxsize=1024)
def _weight_map(ptr: int, nblocks: int, kblocks: int, groups: int) -> bytes:
    return cu.tensor_map(
        ptr,
        (128, 4, 16, nblocks),
        (512, 2048, _BLOCK),
        (128, kblocks, groups, 1),
        words=True,
    )


@functools.cache
def _big_workspace(device_index: int) -> dict:
    dev = torch.device("cuda", device_index)
    i32 = dict(dtype=torch.int32, device=dev)
    ws = {
        "xt": torch.empty((_BIG_SLOTS, _HIDDEN), dtype=torch.bfloat16, device=dev),
        "h": torch.empty((_BIG_SLOTS, _INTER), dtype=torch.bfloat16, device=dev),
        # per expert a list of up to 8 routes per token (ids may repeat in a token), then ranks
        "rev": torch.empty((_BIG_NE + 1) * _TOP_K * _BIG_MAXT, **i32),
        "ucnt": torch.empty(_BIG_MAXT // 128 * 64, **i32),
        "ctr": torch.zeros(_BIG_CTR, **i32),
    }
    p = _BigParams()
    maps = b"".join(
        _tensor_map(t, n) for t in (ws["xt"], ws["h"]) for n in _BIG_WIDTHS + (-32, -64)
    )
    maps += b"".join(
        _tensor_map(t, n // 2) for t in (ws["xt"], ws["h"]) for n in _BIG_WIDTHS
    )
    ws["maps"] = torch.frombuffer(bytearray(maps), dtype=torch.uint8).to(dev)
    base = ws["maps"].data_ptr()
    nw = len(_BIG_WIDTHS)
    p.tx2, p.th, p.th2, p.txh, p.thh = (
        base + 128 * k for k in (nw, nw + 2, 2 * nw + 2, 2 * nw + 4, 3 * nw + 4)
    )
    for k in ("xt", "ctr", "rev", "ucnt"):
        setattr(p, k, ws[k].data_ptr())
    p.cap = _BIG_SLOTS
    ws["params"] = p
    return ws


def init_runtime(device: torch.device, max_tokens: int) -> None:
    global _BIG_MAXT
    if max_tokens > _BIG_MAXT:
        _BIG_MAXT = max_tokens
        _big_workspace.cache_clear()  # rev / ucnt are sized by _BIG_MAXT
    _module(device.index)
    _workspace(device.index)
    _big_workspace(device.index)


def _local_ids(ids: torch.Tensor) -> torch.Tensor:
    """int32 local ids; wider ids are clamped first so none wraps into [0, _BIG_NE)."""
    if ids.dtype != torch.int32:
        ids = ids.clamp(-1, _BIG_NE)
    return ids.to(torch.int32).contiguous()


def _config(num_tokens: int, sp: bool):
    for bound, cfg, smem in _SP_CONFIGS if sp else _CONFIGS:
        if num_tokens <= bound:
            return cfg, smem
    raise ValueError(num_tokens)


def _launch(
    x,
    local_ids,
    topk_weights,
    out,
    w13,
    gs13,
    w2,
    gs2,
    norm_weight,
    sp=None,
    dg=None,
):
    M = local_ids.shape[0]
    cfg, smem = _config(M, sp is not None)
    rw, nt, _, mw = cfg
    dev = x.device
    ws = _workspace(dev.index)
    pmax = M * _TOP_K
    T = 8 * nt
    xp = 256 * (8 // rw) * 2 + 16
    tiles = _NUM_LOCAL + (pmax + T - 1) // T
    inter = torch.empty(
        tiles * (16 // (8 // rw)) * T * xp // 2, dtype=torch.bfloat16, device=dev
    )
    xs = torch.empty_like(inter)
    ypart = torch.empty(
        (4, pmax, _HIDDEN) if sp is None and mw < 32 else (1,),
        dtype=torch.float32,
        device=dev,
    )
    yh = inter.new_empty((pmax, _HIDDEN) if sp is None else (1,))
    params = _Params(
        x.data_ptr(),
        local_ids.data_ptr(),
        topk_weights.data_ptr(),
        M,
        w13.data_ptr(),
        gs13.data_ptr(),
        w2.data_ptr(),
        gs2.data_ptr(),
        norm_weight.data_ptr(),
        inter.data_ptr(),
        ypart.data_ptr(),
        0 if out is None else out.data_ptr(),
        ws["ctr"].data_ptr(),
        pmax,
        xs.data_ptr(),
        yh.data_ptr(),
        ws["scratch"].data_ptr(),
    )
    if sp is not None:
        shards, recvs, rows = sp
        params.xsp[: len(shards)] = shards
        params.osp[: len(recvs)] = recvs
        params.R, params.ptj = rows, ws["ptj"].data_ptr()
        cfg = ("sp",) + cfg
    if dg is not None:
        params.sw13, params.sw2, params.gsrc, params.peer = (
            t.data_ptr() for t in (*dg.stage[:2], dg.gsrc, dg.peer)
        )
    pdl = sp is None and mw <= 4
    cu.launch(
        _module(dev.index).funcs[cfg],
        ws["sms"],
        _THREADS,
        smem,
        torch.cuda.current_stream().cuda_stream,
        [params],
        [cu.pdl()] if pdl else None,
    )


def _launch_big(ids, w13, gs13, w2, gs2, norm_weight, xs, os_, rows, dg):
    dev = ids.device
    p = _BigParams.from_buffer_copy(_big_workspace(dev.index)["params"])
    for v, (kb, rg) in enumerate(((1, 8), (1, 16), (2, 16))):
        for k, w in enumerate((w13, w2)):
            for m, t in ((p.wm, w), (p.wsg, dg and dg.stage[k])):
                if t is not None:
                    ctypes.memmove(
                        ctypes.addressof(m) + 128 * (2 * v + k),
                        _weight_map(t.data_ptr(), t.numel() // _BLOCK, kb, rg),
                        128,
                    )
    if dg is not None:
        p.sw13, p.sw2, p.sgs, p.gsrc, p.peer = (
            t.data_ptr() for t in (*dg.stage, dg.gsrc, dg.peer)
        )
    p.ids, p.w13, p.w2, p.gs13, p.gs2, p.nw = (
        t.data_ptr() for t in (ids, w13, w2, gs13, gs2, norm_weight)
    )
    p.ne = _NUM_LOCAL if dg is None else _BIG_NE
    p.M, p.R = ids.shape[0], rows
    p.lag = 16 if p.M <= BIG_TOKENS else 6
    p.xs[: len(xs)] = xs
    p.os[: len(os_)] = os_
    p.pre = dg is not None
    p.cl = p.M <= BIG_TOKENS and not p.pre
    for i in range((p.M + BIG_TOKENS - 1) // BIG_TOKENS):
        p.first = i == 0
        cu.launch(
            _module(dev.index).big,
            _workspace(dev.index)["sms"],
            _THREADS,
            _BIG_SMEM,
            torch.cuda.current_stream().cuda_stream,
            [p],
            [cu.cluster_dim(2)] + [cu.pdl()] * (i > 0),
        )


def nvfp4_w4a16_moe_with_post_norm(
    hidden_states: torch.Tensor,
    local_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_global_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    out=None,
) -> torch.Tensor:
    """``local_ids`` -1: remote."""
    x = hidden_states.contiguous()
    ids = _local_ids(local_ids)
    tw = topk_weights.to(torch.float32).contiguous()
    M = x.shape[0]
    if out is None:
        out = torch.empty((M, _HIDDEN), dtype=torch.bfloat16, device=x.device)
    if M > 1024:
        for s in range(0, M, 1024):
            _launch(
                x[s : s + 1024],
                ids[s : s + 1024],
                tw[s : s + 1024],
                out[s : s + 1024],
                w13,
                w13_global_scale,
                w2,
                w2_global_scale,
                norm_weight,
            )
    elif M:
        _launch(
            x,
            ids,
            tw,
            out,
            w13,
            w13_global_scale,
            w2,
            w2_global_scale,
            norm_weight,
        )
    return out


class DynamicGuests(NamedTuple):
    """stage: (w13, w2, global scales) this batch's guests are pulled into; gsrc: their experts; peer: every rank's weight addresses."""

    stage: tuple
    gsrc: torch.Tensor
    peer: torch.Tensor


def nvfp4_w4a16_moe_sp(
    local_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_global_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    shard: torch.Tensor,
    shard_ptrs,
    recv_ptrs,
    dg: Optional[DynamicGuests] = None,
) -> None:
    """Route j of token t (row t % R of rank t // R's ``shard``) -> row j * R + t % R of that rank's recv.

    Below BIG_MIN tokens a token's ids must be distinct (the small-M kernel writes one recv
    row per token and expert); the large-M kernel also takes repeated ids."""
    ids = _local_ids(local_ids)
    tw = topk_weights.to(torch.float32).contiguous()
    M, R = ids.shape[0], shard.shape[0]
    assert (
        M == R * len(shard_ptrs) <= _BIG_MAXT
        and len(recv_ptrs) == len(shard_ptrs) <= 8
        and shard.shape[1] == _HIDDEN
    )
    if SP_MIN <= M < BIG_MIN:
        _launch(
            shard,
            ids,
            tw,
            None,
            w13,
            w13_global_scale,
            w2,
            w2_global_scale,
            norm_weight,
            (shard_ptrs, recv_ptrs, R),
            dg=dg,
        )
        return
    _launch_big(
        ids,
        w13,
        w13_global_scale,
        w2,
        w2_global_scale,
        norm_weight,
        shard_ptrs,
        recv_ptrs,
        R,
        dg,
    )


def sp_dynamic_guest_ids(
    topk_ids: torch.Tensor, rank: int, gsrc: torch.Tensor, dur: torch.Tensor
) -> torch.Tensor:
    """Local ids of the gathered ``topk_ids`` with this batch's guests at 32 + j."""
    ids = topk_ids.to(torch.int32).contiguous()
    out, d = torch.empty_like(ids), ids.device.index
    _module(d).launch_dsplit(ids, _workspace(d)["split"], out, gsrc, rank, dur)
    return out


def nvfp4_w4a16_moe_sp_finalize(
    recv: torch.Tensor, topk_weights: torch.Tensor, rank: int, rows: int
) -> torch.Tensor:
    """Post-norm + top-k sum of ``recv``."""
    tw = topk_weights[rank * rows : (rank + 1) * rows].to(torch.float32).contiguous()
    out = recv.new_empty((rows, _HIDDEN))
    _module(recv.device.index).launch_fin(
        recv.data_ptr(), tw.data_ptr(), out.data_ptr(), rows, rows
    )
    return out


def _decoded(pair: int) -> int:
    """FP4 codes the kernels' decode() reads from a stored word pair."""
    w0, w1 = pair & 0xFFFFFFFF, pair >> 32
    rotl = lambda x, n: (x << n | x >> (32 - n)) & 0xFFFFFFFF
    f = (
        (w0 >> 8 & 255) | (w1 & 255) << 8 | (w0 >> 24) << 16 | (w1 >> 16 & 255) << 24
    )  # __byte_perm(w0, w1, 0x6341)
    regs = (
        w0,
        w0 << 2 ^ f,
        w0 << 6,
        rotl(w0, 12),
        w1,
        w1 << 4,
        rotl(w1, 10),
        rotl(w1, 14) ^ f,
    )
    codes = 0
    for n in range(16):
        o = regs[n // 2] >> 16 * (n % 2)
        codes |= ((o >> 6 & 7) | (o >> 15 & 1) << 3) << 4 * n
    return codes


@functools.cache
def _tables(device: torch.device) -> torch.Tensor:
    """enc[i, b]: the stored word pair (int64) that decodes to byte b at code bits 8i .. 8i + 7, 0 elsewhere."""
    # decode() is linear over GF(2): Gauss-Jordan on the 64 one-bit pairs
    rows = [[_decoded(1 << k), 1 << k] for k in range(64)]
    for i in range(64):
        p = next(r for r in rows[i:] if r[0] >> i & 1)
        rows.remove(p)
        rows.insert(i, p)
        for r in rows:
            if r is not p and r[0] >> i & 1:
                r[0] ^= p[0]
                r[1] ^= p[1]
    byte = torch.arange(256, dtype=torch.int64)
    enc = torch.zeros(8, 256, dtype=torch.int64)
    for k, (_, pair) in enumerate(rows):
        enc[k // 8] ^= (byte >> k % 8 & 1) * (pair - (pair >> 63 << 64))
    return enc.to(device)


def scales_supported(*scales: torch.Tensor) -> bool:
    """6-bit encoding holds E4M3 scales 0 or in [2.0, 448]."""
    return all(
        bool((((b >= 0x40) & (b <= 0x7E)) | (b == 0)).all())
        for b in (s.view(torch.uint8) for s in scales)
    )


def _repack_matrix(
    packed: torch.Tensor, sf: torch.Tensor, gate_up: bool
) -> torch.Tensor:
    R = packed.shape[0]
    dev = packed.device
    enc = _tables(dev)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(R, 4096)
    sb = sf.view(torch.uint8)
    # zero scale zeroes its block
    codes = codes.masked_fill((sb == 0).repeat_interleave(16, dim=1), 0)
    if gate_up:
        j = torch.arange(R // 16, device=dev).view(-1, 1)
        r = torch.arange(16, device=dev).view(1, 16)
        rows = torch.where(r < 8, 8 * j + r, R // 2 + 8 * j + (r - 8)).flatten()
        codes, sb = codes[rows], sb[rows]
    c = (
        codes.view(R // 16, 2, 8, 16, 4, 4, 4, 2, 2)
        .permute(3, 0, 4, 2, 5, 6, 7, 1, 8)
        .reshape(-1, 16)
        .long()
    )
    pairs = c.new_zeros(c.shape[0])
    for i in range(8):
        pairs ^= enc[i][c[:, 2 * i] | c[:, 2 * i + 1] << 4]
    del c
    s = (
        sb.reshape(R // 16, 2, 8, 16, 4, 4)
        .permute(3, 0, 2, 5, 4, 1)
        .reshape(16, R // 32, 2, 32, 4, 2)
        .long()
    )
    f = torch.where(s == 0, torch.zeros_like(s), s - 0x40)
    P = lambda rg, kk: (f[:, :, rg, :, kk, 0], f[:, :, rg, :, kk, 1])
    (al, ah), (bl, bh), (cl, ch), (dl, dh) = P(0, 0), P(1, 0), P(0, 1), P(1, 1)
    (el, eh), (fl, fh), (gl, gh), (hl, hh) = P(0, 2), P(1, 2), P(0, 3), P(1, 3)
    w0 = (
        (al << 4) | (ah << 20) | (bl << 10) | (bh << 26) | (gl & 15) | ((gh & 15) << 16)
    )
    w1 = (
        (cl << 4) | (ch << 20) | (dl << 10) | (dh << 26) | (hl & 15) | ((hh & 15) << 16)
    )
    w2 = (
        (el << 4)
        | (eh << 20)
        | (fl << 10)
        | (fh << 26)
        | (gl >> 4)
        | ((hl >> 4) << 2)
        | ((gh >> 4) << 16)
        | ((hh >> 4) << 18)
    )
    s6 = torch.stack((w0, w1, w2), -1).view(torch.int32).view(-1, 2)[:, 0]
    s6 = s6.contiguous().view(torch.uint8).view(16, R // 256, 3072)
    return torch.cat([pairs.view(torch.uint8).view(16, R // 256, 32768), s6], -1)


@torch.no_grad()
def repack_nvfp4_experts(
    weight: torch.Tensor, weight_scale: torch.Tensor, gate_up: bool
) -> torch.Tensor:
    """-> [E, 16, R/256, 35840] uint8, lossless if ``scales_supported``."""
    E, R = weight.shape[:2]
    out = torch.empty(
        (E, 16, R // 256, _BLOCK), dtype=torch.uint8, device=weight.device
    )
    for e in range(E):
        out[e] = _repack_matrix(weight[e], weight_scale[e], gate_up)
    return out
