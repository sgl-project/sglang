# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runner and buffers of the mono decode launches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention.plan import HEAD_DIM, Dims
from .common.plan import BLOCKS
from .layer import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    peer_half_bytes,
    scratch_bytes,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS

__all__ = ["BLOCKS", "MAX_TOKENS", "DSV41MonoLayer", "MonoLayerWeights"]

HC = 4


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


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DSv4.1 mono decode: {what} first reached inside a CUDA graph capture"
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
    """The mono decode runner of one TP rank (``group``: a gloo group for the peer handle exchange)."""

    def __init__(self, tp: int, rank: int, group, device: torch.device | str = "cuda"):
        _ensure_writable_flydsl_cache()
        from .common.peer_memory import PeerBuffer

        self.tp, self.rank = tp, rank
        self.d = Dims(tp)
        dev = torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self.device = dev
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
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
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._kernels: dict = {}

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

    def ffn_kernel(self, tokens: int):
        key = (tokens, "ffn")
        if key not in self._kernels:
            no_capture(f"FFN launch width {tokens}")
            self._kernels[key] = build_mono_ffn(MonoBuild(tokens, self.tp))
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
        w: MonoLayerWeights,
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
