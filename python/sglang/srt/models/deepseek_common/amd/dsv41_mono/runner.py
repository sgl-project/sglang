# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash mono decode layer: host side.

One ``DSV41MonoLayer`` a TP rank runs every mono layer of a decode step (M <= 48
rows): a layer whose attention vLLM runs (``ffn``) takes one persistent launch
(``layer``) from its unreduced attention output to its outputs at the next
attention seam -- the MoE's reduced output, the residual after the FFN seam and
that seam's mixes -- with both TP all-reduces inside the kernel (symmetric peer
memory). Every argument is a device pointer: the launches can be captured in a
HIP graph. The kernels move their mailbox epoch on themselves; every rank runs
the same launches, so the ranks' epochs agree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

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
    """Aiter points FlyDSL at its bundled, read-only cache; new kernels need a
    writable one."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _check_tensors(owner, tag: str, want: dict) -> None:
    """Each named tensor of ``owner`` has the (shape, dtype) the kernels read,
    contiguous: their buffer loads are unbounded, so a mismatch would read
    garbage rather than fault."""
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous(), (
            f"{tag} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
        )


@dataclass
class MonoLayerWeights:
    """One layer's FFN-launch tensors on this rank, in vLLM's loaded layout."""

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
        """The MoE's tensors: 384 experts, TP-sharded intermediates (the routed
        one padded as the loader pads it), AITER's A8W4 layout, 32 x 32 E8M0
        blocks for the shared expert."""
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
    """The mono decode runner of one TP rank (``group``: its TP group, for the
    peer-memory handle exchange; a gloo / CPU group)."""

    def __init__(self, tp: int, rank: int, group, device: torch.device | str = "cuda"):
        _ensure_writable_flydsl_cache()
        from .common.peer_memory import PeerBuffer

        self.tp, self.rank = tp, rank
        dev = self.device = torch.device(device)
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._kernels: dict = {}

    def scratch(self, tokens: int) -> torch.Tensor:
        """Step width ``tokens``'s scratch: a width's layout has memory of its own
        (``layer.scratch_layout``). Allocated at the width's first step, which is
        eager: vLLM runs every graph's batch eagerly before capturing it."""
        buf = self._scratch.get(tokens)
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"DSv4.1 mono decode: step width {tokens} first reached inside "
                    "a CUDA graph capture"
                )
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
        """-> (out, residual, post_mix, res_mix, pre_mix) of a layer whose
        attention vLLM ran: the attention's TP reduce, the FFN seam, the MoE and
        its all-reduce in one launch."""
        M = part.shape[0]
        assert self.supports(M), M
        for t_ in (part, residual, post_mix, res_mix, pre_mix):
            assert t_.is_contiguous()
        assert part.dtype == residual.dtype == torch.bfloat16
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
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
