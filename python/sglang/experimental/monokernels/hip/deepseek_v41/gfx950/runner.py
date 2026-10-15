# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runner and buffers of the DeepSeek-V4.1 monokernels on one TP rank.

Every buffer is allocated when the runner is made or in ``allocate``, both at attach, before KV sizing; no launch
allocates. ``build`` compiles every reachable width at attach without running it (the KV pool K1 writes to does
not exist yet); the first launch of a width loads its compiled code from FlyDSL's cache in the capture warm-up."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.experimental.monokernels.hip.common.plan import BLOCKS

from .attention.plan import HEAD_DIM, Dims
from .layer import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    peer_half_bytes,
    scratch_bytes,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS

__all__ = [
    "BLOCKS",
    "HC",
    "MAX_TOKENS",
    "DSV41MonoLayer",
    "MonoLayerWeights",
    "check_rows",
]

HC = 4
# launch argument counts, for compile-only builds (every argument is a pointer or a scalar)
_K1_ARGS = 28
_FFN_ARGS = 29


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


# FlyDSL's compile-only switch (flydsl.utils.env.compile.compile_only), read on every launch
_COMPILE_ONLY = "COMPILE_ONLY"


@contextlib.contextmanager
def _compile_only():
    """FlyDSL compiles and caches a launch without running it."""
    old = os.environ.get(_COMPILE_ONLY)
    os.environ[_COMPILE_ONLY] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(_COMPILE_ONLY, None)
        else:
            os.environ[_COMPILE_ONLY] = old


def _compile(launch, nargs: int) -> None:
    """Compile and cache a launch without running it, through FlyDSL's compile API in its compile-only mode. The
    mode is checked first: a launch that ran would dereference the null arguments."""
    import flydsl.compiler as flyc
    from flydsl.utils import env as flydsl_env

    with _compile_only():
        if not flydsl_env.compile.compile_only:
            raise RuntimeError(
                "FlyDSL is not in compile-only mode; a build would launch with null arguments"
            )
        flyc.compile(launch, *[0] * nargs)


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


@dataclass
class MonoLayerWeights:
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


class DSV41MonoLayer:
    """The monokernel runner of one TP rank. The constructor is collective (peer memory and the guard)."""

    def __init__(
        self,
        tp: int,
        rank: int,
        group,
        device,
        deadline_s: float,
        fault=None,
        mirror_ptr: int = 0,
    ):
        _ensure_writable_flydsl_cache()
        from sglang.experimental.monokernels.hip.common.guard import Guard
        from sglang.experimental.monokernels.hip.common.peer_memory import PeerBuffer

        self.tp, self.rank, self.fault = tp, rank, fault
        self.d = Dims(tp)
        dev = torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self.device = dev
        bf, f32 = torch.bfloat16, torch.float32
        # [epoch, -, guard block address (2 words), a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.q = torch.zeros(MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=bf, device=dev)
        # K1's outputs: the attention seam's residual and mixes
        self.res_mid = torch.zeros(MAX_TOKENS, HC, HIDDEN, dtype=bf, device=dev)
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=f32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=f32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=f32, device=dev)
        # K1's int64 positions and SWA slots of the step, copied by the first K1 layer of a forward
        self.pos64 = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=dev)
        self.slot64 = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=dev)
        # two sets of FFN outputs: layer i writes set i % 2, and layer i + 1 consumes it before it writes the
        # other set (the pending mHC post of the HIP boundary)
        self.outs = [
            (
                torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf, device=dev),
                torch.zeros(MAX_TOKENS, HC, HIDDEN, dtype=bf, device=dev),
                torch.zeros(MAX_TOKENS, HC, 1, dtype=f32, device=dev),
                torch.zeros(MAX_TOKENS, HC, HC, dtype=f32, device=dev),
                torch.zeros(MAX_TOKENS, HC, dtype=f32, device=dev),
            )
            for _ in range(2)
        ]
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self.guard = Guard(tp, rank, group, dev, deadline_s, mirror_ptr)
        self.guard.bind(self.epoch)
        self._scratch: dict = {}
        self._kernels: dict = {}

    def persistent_bytes(self) -> dict:
        """Every persistent buffer and its size (bytes)."""
        named = {
            "epoch": self.epoch,
            "q": self.q,
            "res_mid": self.res_mid,
            "post_a": self.post_a,
            "comb_a": self.comb_a,
            "pre_a": self.pre_a,
            "pos64": self.pos64,
            "slot64": self.slot64,
            "guard_block": self.guard.block,
        }
        out = {k: t.numel() * t.element_size() for k, t in named.items()}
        out["ffn_outs"] = sum(
            t.numel() * t.element_size() for s in self.outs for t in s
        )
        out["peer"] = self.peer.bytes.numel()
        out["stop"] = self.guard.stop.bytes.numel()
        return out

    def allocate(self, widths) -> dict:
        """Each width's scratch (bytes per width)."""
        for w in widths:
            if not 1 <= w <= MAX_TOKENS:
                raise ValueError(
                    f"width {w}; the launches serve 1 to {MAX_TOKENS} rows"
                )
            self._scratch[w] = torch.zeros(
                scratch_bytes(w, self.tp), dtype=torch.uint8, device=self.device
            )
        return {w: t.numel() for w, t in self._scratch.items()}

    def build(self, widths, k1: bool) -> dict:
        """Compile the FFN launch (and K1) at each width; seconds per width."""
        times = {}
        for w in widths:
            t0 = time.perf_counter()
            key = MonoBuild(w, self.tp, *(self.fault or ("", 0)))
            ffn = build_mono_ffn(key)
            _compile(ffn, _FFN_ARGS)
            self._kernels[w, "ffn"] = ffn
            if k1:
                k = build_mono_k1(key)
                _compile(k, _K1_ARGS)
                self._kernels[w, "k1"] = k
            times[w] = round(time.perf_counter() - t0, 2)
        return times

    def kernel(self, rows: int, kind: str):
        k = self._kernels.get((rows, kind))
        if k is None:
            raise RuntimeError(f"no {kind} launch built for {rows} rows")
        return k

    def scratch(self, rows: int) -> torch.Tensor:
        buf = self._scratch.get(rows)
        if buf is None:
            raise RuntimeError(f"no scratch allocated for {rows} rows")
        return buf

    def ffn(
        self,
        w: MonoLayerWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        layer_id: int,
    ):
        """Run the FFN launch: attention TP reduce, FFN seam, MoE and all-reduce. Returns slices of the
        layer's output set (MoE output, residual, post, comb, pre)."""
        M = part.shape[0]
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
        out, res_out, post_out, comb_out, pre_out = (
            t[:M] for t in self.outs[layer_id % 2]
        )
        self.kernel(M, "ffn")(
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
        return out, res_out, post_out, comb_out, pre_out
