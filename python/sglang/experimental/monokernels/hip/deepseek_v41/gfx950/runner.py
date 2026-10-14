# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runner and buffers of the mono decode launches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.experimental.monokernels.hip.common import guard
from sglang.experimental.monokernels.hip.common.plan import BLOCKS

from .attention.plan import DATA, HEAD_DIM, KEYS, Q_RANK, TOPK, WINDOW, Dims
from .layer import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS

__all__ = [
    "BLOCKS",
    "MAX_TOKENS",
    "DSV41MonoRunner",
    "K1Weights",
    "K2Weights",
    "FFNWeights",
]

HC = 4
RECORD = DATA + 8  # a token's KV record: 576 B of data, then 8 B of scales
_LOG2E = 1.4426950408889634
DEFAULT_DEADLINE_S = (
    60.0  # a wait's deadline until the framework sets one from the watchdog
)


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    import inspect

    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _check_tensors(owner, tag: str, want: dict) -> None:
    """Raise unless each named tensor has the expected shape and dtype and is contiguous."""
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        if not (
            tuple(t.shape) == shape
            and t.dtype == dtype
            and t.is_contiguous()
            and t.is_cuda
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
            )


def check_rows(tag: str, rows: int, device, want: dict) -> None:
    """Raise unless each tensor is contiguous on ``device`` with ``rows`` rows of (row size, dtype)."""
    for name, (t, size, dtype) in want.items():
        if not (
            t.dtype == dtype
            and t.numel() == rows * size
            and t.is_contiguous()
            and t.device == device
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype} {t.device}, want {rows} x {size} {dtype}"
            )


def check_cache(tag: str, cache: torch.Tensor, block: int) -> None:
    """Raise unless ``cache`` is uint8 [pages, page bytes] with ``block`` records a page."""
    if not (
        cache.dtype == torch.uint8
        and cache.dim() == 2
        and cache.stride(1) == 1
        and cache.stride(0) >= block * RECORD
    ):
        raise RuntimeError(
            f"{tag}: {cache.dtype} {tuple(cache.shape)} {cache.stride()}, "
            f"want uint8 pages of {block} x {RECORD} B"
        )


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DSv4.1 mono decode: {what} first reached inside a CUDA graph capture"
        )


@dataclass
class FFNWeights:
    """One layer's FFN-launch tensors on this rank, in SGLang's loaded layout."""

    hc_ffn_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_ffn_scale: torch.Tensor  # [3] f32
    hc_ffn_base: torch.Tensor  # [24] f32
    ffn_norm: torch.Tensor  # [5120] bf16
    gate_w: torch.Tensor  # [384, 5120] bf16
    bias: torch.Tensor  # [384] f32 (e_score_correction_bias)
    w13: torch.Tensor  # [384, 2 inter, 2560] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [384, 5120, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, 5120] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, 160] E8M0
    sw2: torch.Tensor  # shared down [5120, inter] e4m3
    sw2_s: torch.Tensor

    def check(self, tp: int) -> None:
        """One rank's MoE tensors in SGLang's loaded layout."""
        d, fp4, e4m3, u8 = (
            MoeDims(tp),
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.uint8,
        )
        want = {
            "gate_w": ((EXPERTS, HIDDEN), torch.bfloat16),
            "bias": ((EXPERTS,), torch.float32),
            "w13": ((EXPERTS, 2 * d.inter, HIDDEN // 2), fp4),
            "w13_s": ((EXPERTS * 2 * d.inter, HIDDEN // 32), u8),
            "w2": ((EXPERTS, HIDDEN, d.inter // 2), fp4),
            "w2_s": ((EXPERTS * HIDDEN, d.down_scale_cols), u8),
            "sgu": ((2 * d.sh_inter, HIDDEN), e4m3),
            "sgu_s": ((2 * d.sh_inter // 32, HIDDEN // 32), u8),
            "sw2": ((HIDDEN, d.sh_inter), e4m3),
            "sw2_s": ((HIDDEN // 32, d.sh_inter // 32), u8),
        }
        _check_tensors(self, "MoE", want)


@dataclass
class K1Weights:
    """One layer's attention-seam and attention-front tensors on this rank, as K1 reads them."""

    hc_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_scale: torch.Tensor  # [3] f32
    hc_base: torch.Tensor  # [24] f32
    norm: torch.Tensor  # [5120] input_layernorm
    wqkv: torch.Tensor  # [1792, 5120] e4m3, row-major: wq_a then wkv_a
    wqkv_s: torch.Tensor  # [56, 160] E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wqb: torch.Tensor  # [heads * 512, 1280] e4m3, row-major
    wqb_s: torch.Tensor  # [heads * 16, 40] E8M0
    cos_sin: torch.Tensor  # [positions, 64] f32: 32 cos, then 32 sin

    def check(self, d: Dims) -> None:
        """One rank's attention-front tensors in K1's layout."""
        e4m3, u8, f32, bf = (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.float32,
            torch.bfloat16,
        )
        want = {
            "hc_fn": ((24, HC * HIDDEN), f32),
            "hc_scale": ((3,), f32),
            "hc_base": ((24,), f32),
            "wqkv": ((1792, HIDDEN), e4m3),
            "wqkv_s": ((1792 // 32, HIDDEN // 32), u8),
            "q_norm": ((1280,), bf),
            "kv_norm": ((512,), bf),
            "wqb": ((d.heads * HEAD_DIM, 1280), e4m3),
            "wqb_s": ((d.heads * HEAD_DIM // 32, 1280 // 32), u8),
        }
        _check_tensors(self, "K1", want)
        if not (
            self.norm.shape == (HIDDEN,)
            and self.norm.is_contiguous()
            and self.norm.is_cuda
        ):
            raise RuntimeError(f"K1 norm: {tuple(self.norm.shape)}")
        cs = self.cos_sin
        if not (
            cs.dtype == f32
            and cs.dim() == 2
            and cs.shape[1] == 64
            and cs.is_contiguous()
        ):
            raise RuntimeError(
                f"K1 cos_sin: {tuple(cs.shape)} {cs.dtype}, want [positions, 64] f32"
            )


@dataclass
class K2Weights:
    """One layer's attention-back tensors on this rank, as K2 reads them."""

    woa: torch.Tensor  # [G * 1024, 4096] e4m3, row-major
    woa_s: torch.Tensor  # [G * 32, 128] E8M0 (32 x 32 blocks)
    wob: torch.Tensor  # [5120, G * 1024] e4m3, row-major
    wob_s: torch.Tensor  # [160, G * 32] E8M0
    sink: torch.Tensor  # [heads] f32, this rank's
    cos_sin: torch.Tensor  # [positions, 64] f32: 32 cos, then 32 sin

    def check(self, d: Dims) -> None:
        """One rank's attention-back tensors in K2's layout."""
        g, e4m3, u8 = d.groups, torch.float8_e4m3fn, torch.uint8
        want = {
            "woa": ((g * 1024, d.group_k), e4m3),
            "woa_s": ((g * 32, d.group_k // 32), u8),
            "wob": ((HIDDEN, g * 1024), e4m3),
            "wob_s": ((HIDDEN // 32, g * 32), u8),
            "sink": ((d.heads,), torch.float32),
        }
        _check_tensors(self, "K2", want)
        cs = self.cos_sin
        if not (cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == 64):
            raise RuntimeError(f"K2 cos_sin: {tuple(cs.shape)} {cs.dtype}")


class DSV41MonoRunner:
    """The mono decode runner of one TP rank (``group``: a gloo group for the peer handle exchange)."""

    def __init__(self, tp: int, rank: int, group, device: torch.device | str = "cuda"):
        _ensure_writable_flydsl_cache()
        from sglang.experimental.monokernels.hip.common.peer import PeerBuffer

        self.tp, self.rank = tp, rank
        self.d = Dims(tp)
        dev = torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self.device = dev
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, the guard block (deadline, error words), a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        self.set_deadline(DEFAULT_DEADLINE_S)
        self.q = torch.zeros(
            MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )
        # K1's outputs: the attention seam's residual and mixes
        self.res_mid = torch.zeros(
            MAX_TOKENS, HC, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        # K2's key table, written by K1 (by K2 on an index source): a slot per
        # key, then (kv_len, ntopk) per token
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        # a window-only layer's top-k: none (K1 reads the table, ntopk is 0)
        self.no_topk = torch.zeros(MAX_TOKENS, TOPK, dtype=torch.int32, device=dev)
        self.no_topk_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=dev)
        # the first layer's incoming pre mix: copy 0 of the broadcast embedding
        self.first_pre = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.first_pre[:, 0] = 1.0
        # an index source's K1 outputs, its compressor and indexer's inputs
        self.normed = torch.zeros(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.qr = torch.zeros(MAX_TOKENS, Q_RANK, dtype=torch.bfloat16, device=dev)
        # the record an absent key reads: zero data and scales, masked anyway
        self.zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        self.qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        # two parity halves, then the error flags: rank r writes word r of every
        # rank's flags when its wait fails, and every pending wait checks them
        half = peer_half_bytes(tp)
        self.peer = PeerBuffer(2 * half + guard.FLAG_BYTES, group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._kernels: dict = {}

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline (below the server's watchdog timeout)."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def scratch(self, tokens: int) -> torch.Tensor:
        """Return this width's scratch, allocated before graph capture."""
        buf = self._scratch.get(tokens)
        if buf is None:
            no_capture(f"step width {tokens}")
            buf = torch.zeros(
                scratch_bytes(tokens, self.tp), dtype=torch.uint8, device=self.device
            )
            self._scratch[tokens] = buf
        return buf

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def k1_kernel(self, tokens: int, fold: bool = True, index: bool = False):
        """K1 at this width; it writes K2's key table unless the layer is an index
        source (whose top-k comes after K1)."""
        key = (tokens, "k1", fold, index)
        if key not in self._kernels:
            no_capture(f"K1 width {tokens}")
            self._kernels[key] = build_mono_k1(
                MonoBuild(
                    tokens,
                    self.tp,
                    kt=not index,
                    fold=fold,
                    index=index,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                )
            )
        return self._kernels[key]

    def precompile(self, tokens: int, k1=(), k2=(), ffn: bool = False) -> None:
        """Allocate this width's scratch and compile its launches without running
        them: K1 for each (fold, index) in ``k1``, K2 for each (compressed,
        index) in ``k2``, and the FFN launch."""
        self.scratch(tokens)
        kernels = [self.k1_kernel(tokens, *v) for v in sorted(k1)]
        kernels += [self.k2_kernel(tokens, *v) for v in sorted(k2)]
        if ffn:
            kernels.append(self.ffn_kernel(tokens))
        for launch in kernels:
            compile_only(launch)

    def ffn_kernel(self, tokens: int):
        key = (tokens, "ffn")
        if key not in self._kernels:
            no_capture(f"FFN launch width {tokens}")
            self._kernels[key] = build_mono_ffn(
                MonoBuild(
                    tokens,
                    self.tp,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                )
            )
        return self._kernels[key]

    def k2_kernel(self, tokens: int, compressed: bool = True, index: bool = False):
        key = (tokens, "k2", compressed, index)
        if key not in self._kernels:
            no_capture(f"K2 width {tokens}")
            self._kernels[key] = build_mono_k2(
                MonoBuild(
                    tokens,
                    self.tp,
                    kt=True,
                    index=index,
                    compressed=compressed,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                )
            )
        return self._kernels[key]

    @staticmethod
    def _outs(M: int, residual: torch.Tensor) -> tuple:
        dev = residual.device
        return (
            torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.empty_like(residual),
            torch.empty(M, HC, 1, dtype=torch.float32, device=dev),
            torch.empty(M, HC, HC, dtype=torch.float32, device=dev),
            torch.empty(M, HC, dtype=torch.float32, device=dev),
        )

    def ffn(
        self,
        w: FFNWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
    ):
        """Run the FFN launch: attention TP reduce, FFN seam, MoE and all-reduce."""
        M = part.shape[0]
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "part": (part, HIDDEN, bf),
                "residual": (residual, HC * HIDDEN, bf),
                "post": (post_mix, HC, f32),
                "comb": (res_mix, HC * HC, f32),
                "pre": (pre_mix, HC, f32),
            },
        )
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        self.ffn_kernel(M)(
            part.data_ptr(),
            residual.data_ptr(),
            post_mix.data_ptr(),
            res_mix.data_ptr(),
            pre_mix.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return outs

    def k1(
        self,
        w: K1Weights,
        x: torch.Tensor
        | None,  # [M, 5120] bf16: the previous layer's pending MoE output
        residual: torch.Tensor,  # [M, 4, 5120] bf16: its pending residual
        post: torch.Tensor | None,  # [M, 4, 1] f32: its pending FFN post mix
        comb: torch.Tensor | None,  # [M, 4, 4] f32
        pre: torch.Tensor,  # [M, 4] f32: the incoming pre mix (its FFN pre mix)
        pos: torch.Tensor,  # [>= M] int64 positions
        slot: torch.Tensor,  # [>= M] int64 SWA cache slots
        swa: torch.Tensor,  # the layer's SWA cache: uint8 [pages, page bytes]
        swa_block: int,
        keys: tuple,  # int32 [>= M, ...]: window slots, window lengths, top-k slots, top-k lengths
        index: bool = False,
    ):
        """Run K1: the pending post, this layer's attention seam and the attention
        front; it writes the step's SWA KV records and K2's key table. With ``x``
        None the residual arrives settled and no post is folded in (the first
        layer, after an Engram write). ``index``: an index source, whose top-k
        comes after K1: K1 writes its normed input and qr (``normed``, ``qr``)
        for the indexer and leaves the key table to K2. Returns (q, residual,
        post, comb, pre) views of the runner's buffers."""
        M = residual.shape[0]
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        bf, f32 = torch.bfloat16, torch.float32
        fold = x is not None
        want = {
            "residual": (residual, HC * HIDDEN, bf),
            "pre": (pre, HC, f32),
        }
        if fold:
            want |= {
                "x": (x, HIDDEN, bf),
                "post": (post, HC, f32),
                "comb": (comb, HC * HC, f32),
            }
        check_rows("K1", M, self.device, want)
        if not fold:  # unread: the seam takes the residual as it is
            x = post = comb = residual
        for name, t in (("positions", pos), ("slots", slot)):
            if not (t.dtype == torch.int64 and t.numel() >= M and t.is_contiguous()):
                raise RuntimeError(f"K1 {name}: {tuple(t.shape)} {t.dtype}")
        check_cache("K1 SWA cache", swa, swa_block)
        for name, t, width in zip(
            ("window slots", "window lengths", "top-k slots", "top-k lengths"),
            keys,
            ((WINDOW,), None, (TOPK,), None),
        ):
            if not (
                t is not None
                and t.dtype == torch.int32
                and t.is_contiguous()
                and t.shape[0] >= M
                and (width is None or tuple(t.shape[1:]) == width)
            ):
                raise RuntimeError(
                    f"K1 {name}: "
                    + ("None" if t is None else f"{tuple(t.shape)} {t.dtype}")
                )
        swa_idx, swa_lens, topk, topk_lens = (t.data_ptr() for t in keys)
        self.k1_kernel(M, fold, index)(
            residual.data_ptr(),
            x.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            pre.data_ptr(),
            w.hc_fn.data_ptr(),
            w.hc_scale.data_ptr(),
            w.hc_base.data_ptr(),
            w.norm.data_ptr(),
            self.res_mid.data_ptr(),
            self.post_a.data_ptr(),
            self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            w.wqkv.data_ptr(),
            w.wqkv_s.data_ptr(),
            w.q_norm.data_ptr(),
            w.kv_norm.data_ptr(),
            w.wqb.data_ptr(),
            w.wqb_s.data_ptr(),
            w.cos_sin.data_ptr(),
            pos.data_ptr(),
            slot.data_ptr(),
            swa.data_ptr(),
            swa.stride(0),
            swa_block,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            self.q.data_ptr(),
            self.normed.data_ptr(),
            self.qr.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return (
            self.q[:M],
            self.res_mid[:M],
            self.post_a[:M],
            self.comb_a[:M],
            self.pre_a[:M],
        )

    def k2(
        self,
        aw: K2Weights,
        w: FFNWeights,
        rows: int,
        pos: torch.Tensor,  # [>= rows] int64, K1's positions
        swa: torch.Tensor,  # the layer's SWA cache: uint8 [pages, page bytes]
        swa_block: int,
        comp: torch.Tensor | None,  # its compressed cache, the same layout
        comp_block: int,
        outs: tuple | None = None,
        slot: torch.Tensor | None = None,  # an index source's: K1's slots and keys
        keys: tuple | None = None,
        index: bool = False,
    ):
        """Run K2 on K1's q and attention-seam outputs: the attention back, its TP
        reduce, the FFN seam, the MoE and its all-reduce. ``comp`` None: a
        window-only layer. ``index``: an index source; K2 first builds the key
        table from ``keys``, whose top-k the indexer selected after K1."""
        M = rows
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        residual = self.res_mid[:M]
        post, comb, pre = self.post_a[:M], self.comb_a[:M], self.pre_a[:M]
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "K2",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        check_cache("K2 SWA cache", swa, swa_block)
        compressed = comp is not None
        if compressed:
            check_cache("K2 compressed cache", comp, comp_block)
        else:  # unread: the window's keys only
            comp, comp_block = swa, swa_block
        if not (pos.dtype == torch.int64 and pos.numel() >= M and pos.is_contiguous()):
            raise RuntimeError(f"K2 positions: {tuple(pos.shape)} {pos.dtype}")
        if index:
            if slot is None or keys is None:
                raise RuntimeError("K2 of an index source needs K1's slots and keys")
            table = (slot, *keys)
        else:  # unread
            table = (self.klen, self.klen, self.klen, self.no_topk, self.no_topk_lens)
        self.k2_kernel(M, compressed, index)(
            self.q.data_ptr(),
            swa.data_ptr(),
            swa.stride(0),
            swa_block,
            comp.data_ptr(),
            comp.stride(0),
            comp_block,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            *(t.data_ptr() for t in table),
            pos.data_ptr(),
            aw.sink.data_ptr(),
            self.qk_scale,
            aw.cos_sin.data_ptr(),
            aw.woa.data_ptr(),
            aw.woa_s.data_ptr(),
            aw.wob.data_ptr(),
            aw.wob_s.data_ptr(),
            self.zero_rec.data_ptr(),
            residual.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            pre.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return outs
