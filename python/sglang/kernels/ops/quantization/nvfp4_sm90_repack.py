"""Repack ModelOpt NVFP4 MoE weights into the layout of the Hopper NVFP4 W4A16 MoE op (ops/moe/nvfp4_w4a16_sm90).

Input, per layer (the parameters ``ModelOptNvFp4FusedMoEMethod.create_weights`` registers, gate rows first):
    w13_weight          uint8          [E, 2 I, H / 2]  two E2M1 codes per byte, element 2 i in the low nibble
    w13_weight_scale    float8_e4m3fn  [E, 2 I, H / 16] one block scale per 16 values along k
    w13_weight_scale_2  float32        [E, 2] (gate, up), [E, 1] or [E]
    w2_weight, w2_weight_scale, w2_weight_scale_2: [E, H, I / 2], [E, H, I / 16], [E]
Output: one copy of each GEMM's weights with the checkpoint's own size when I is a multiple of 64 (512B of codes and
64B of block scales per 16 x 64 unit), plus one FP32 global scale per expert per GEMM. Any I that is a multiple of 16
(the NVFP4 block along w2's K) is accepted: the op works on I rounded up to 64, and the repack writes zeros (zero codes
under zero block scales, exactly 0) into the up to 48 extra columns. The repack is lossless, subnormal E4M3 block
scales included. Gate and up global scales that differ are exact too: the up scale multiplies GEMM2's input by one factor
per expert, so the pack folds up / gate into GEMM2's scale.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

# Bit positions of the code bits (b0, b1, b2, sign) of A pair m's first and second value in the kernel's decode4.
_POS = [
    [(6, 7, 8, 15), (22, 23, 24, 31)],
    [(3, 4, 5, 12), (19, 20, 21, 28)],
    [(0, 1, 2, 9), (16, 17, 18, 25)],
    [(30, 26, 27, 29), (14, 10, 11, 13)],
]


class Nvfp4Sm90Experts(msgspec.Struct, frozen=True):
    """One MoE layer's local experts in the op's layout."""

    w13: torch.Tensor  # uint8 [E, Ip / 64 * H / 64 * 8, 576]: see w13_units
    w2: torch.Tensor  # uint8 [E, H / 256, Ip / 64, 16, 576]
    gs13: torch.Tensor  # float32 [E]: the gate global scale (GEMM1 epilogue)
    gs2: torch.Tensor  # float32 [E]: w2's global scale times up / gate (GEMM2 epilogue)
    gsr: (
        torch.Tensor
    )  # float32 [E]: up / gate (1 where gate is 0), which scales a SwiGLU clamp on up
    hidden: int
    intermediate: (
        int  # the checkpoint's I; the op computes on padded_intermediate(I) columns
    )

    @property
    def padded(self) -> int:
        return padded_intermediate(self.intermediate)


def padded_intermediate(intermediate: int) -> int:
    """The intermediate width the op computes on: I rounded up to GEMM2's 64-wide K chunk."""
    return (intermediate + 63) // 64 * 64


def w13_units(hidden: int, intermediate: int) -> int:
    """16 x 64 units per expert of GEMM1's weights. Rows come in blocks of 256 (128 intermediate columns), each stored
    as H / 64 chunks of 16 units. When the padded I is an odd multiple of 64, the last block holds only its first
    128 rows (one 64-column half), stored as H / 64 chunks of 8 units, so no bytes are spent on the missing half."""
    return padded_intermediate(intermediate) // 64 * (hidden // 64) * 8


def kperm_index(n: int, device=None) -> torch.Tensor:
    """K order inside every 64-wide chunk: position 16 kk + 8 half + 2 q + j holds column 16 q + 4 kk + 2 half + j,
    so the 16 values lane q multiplies in one chunk share one block scale."""
    p = torch.arange(n, device=device)
    c, r = p // 64 * 64, p % 64
    kk, half, q, j = r // 16, (r // 8) % 2, (r // 2) % 4, r % 2
    return c + 16 * q + 4 * kk + 2 * half + j


def _scale_code(v: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes -> the lossless code 8 E + m with scale = 1.m * 2^(E - 9) (subnormals normalized); 0 for 0."""
    v = v.to(torch.int32)
    e, m = (v >> 3) & 15, v & 7
    p = torch.where(
        m >= 4, 2, torch.where(m >= 2, 1, 0)
    )  # subnormal m * 2^-9 = 1.x * 2^(p - 9)
    code = torch.where(e == 0, p * 8 + ((m << (3 - p)) & 7), (e + 2) * 8 + m)
    return torch.where(v == 0, 0, code).to(torch.uint8)


def _decode4_layout(words: torch.Tensor) -> torch.Tensor:
    """int32 words with pair m's codes at bits 4 m (first) and 16 + 4 m (second) -> decode4's bit layout."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for m in range(4):
        for h in range(2):
            code = (w >> (16 * h + 4 * m)) & 15
            for b, pos in enumerate(_POS[m][h]):
                out |= ((code >> b) & 1) << pos
    return torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)


def _pack_one(w: torch.Tensor, sf: torch.Tensor, out: torch.Tensor) -> None:
    """One expert: codes [N, K / 2] and scale bytes [N, K / 16], rows in tile order -> out [N / (16 U), K / 64, U, 576]
    for blocks of U units of 16 rows. Each 16-row x 64-k unit holds per lane and k16 step one word of codes, then the
    unit's 64 block scales."""
    N, K = w.shape[0], 2 * w.shape[1]
    G, C, U = out.shape[0], K // 64, out.shape[2]
    codes = torch.stack((w & 0x0F, w >> 4), dim=-1).reshape(N, K)
    # A block whose scale is 0 is zero whatever its codes. Zero them so the scale code 0 (read as 2^-9) is exact.
    codes = codes.masked_fill((sf == 0).repeat_interleave(16, dim=-1), 0)
    codes = codes.index_select(1, kperm_index(K, codes.device))
    v = codes.view(G, U, 2, 8, C, 4, 2, 4, 2)  # [G, tile, rr, g, c, kk, half, q, j]
    v = v.permute(0, 4, 1, 3, 7, 5, 8, 6, 2)  # [G, c, tile, g, q, kk, j, half, rr]
    nib = (v[..., 0] | (v[..., 1] << 4)).reshape(G, C, U, 512)
    s = (
        _scale_code(sf)
        .view(G, U, 2, 8, C, 4)
        .permute(0, 4, 1, 3, 5, 2)
        .reshape(G, C, U, 64)
    )
    out[..., :512] = (
        _decode4_layout(nib.contiguous().view(torch.int32))
        .view(torch.uint8)
        .view(G, C, U, 512)
    )
    out[..., 512:] = s


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    """t [n, ...] -> [rows, ...] with zero rows appended (t itself when n == rows)."""
    if t.shape[0] == rows:
        return t
    return torch.cat((t, t.new_zeros((rows - t.shape[0],) + t.shape[1:])))


def _pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    if t.shape[1] == cols:
        return t
    return torch.cat((t, t.new_zeros((t.shape[0], cols - t.shape[1]))), dim=1)


def _check_block_scales(sf: torch.Tensor, name: str) -> None:
    b = sf.view(torch.uint8)
    if bool((b & 0x80).any()):
        raise ValueError(f"{name}: negative E4M3 block scales are not supported")
    if bool(((b & 0x7F) == 0x7F).any()):
        raise ValueError(f"{name}: NaN E4M3 block scales")


def repack_modelopt_nvfp4_moe_for_sm90(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    out: Optional[Nvfp4Sm90Experts] = None,
) -> Nvfp4Sm90Experts:
    """Repack one layer's local experts, into ``out`` when given (weight reload: captured CUDA graphs keep reading
    the same buffers). One expert at a time bounds the temporaries to one expert's size."""
    E, N13, H2 = w13_weight.shape
    H, I = 2 * H2, N13 // 2
    if (
        w13_weight_scale.dtype != torch.float8_e4m3fn
        or w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise ValueError("block scales must be float8_e4m3fn")
    if H % 256 or I % 16 or I == 0 or N13 % 2:
        raise ValueError(
            f"hidden must be a multiple of 256 and intermediate of 16, got hidden {H}, intermediate {I}"
        )
    Ip = padded_intermediate(I)
    if (
        tuple(w13_weight_scale.shape) != (E, 2 * I, H // 16)
        or tuple(w2_weight.shape) != (E, H, I // 2)
        or tuple(w2_weight_scale.shape) != (E, H, I // 16)
    ):
        raise ValueError(
            f"inconsistent NVFP4 MoE shapes: w13 {tuple(w13_weight.shape)}, w13 scale {tuple(w13_weight_scale.shape)}, "
            f"w2 {tuple(w2_weight.shape)}, w2 scale {tuple(w2_weight_scale.shape)}"
        )
    if w13_weight_scale_2.numel() not in (E, 2 * E):
        raise ValueError(
            f"w13_weight_scale_2 must be [E], [E, 1] or [E, 2], got {tuple(w13_weight_scale_2.shape)}"
        )
    gs = w13_weight_scale_2.float().reshape(E, -1)
    gate, up = gs[:, 0], gs[:, -1]
    g2 = w2_weight_scale_2.float().reshape(E)
    _check_block_scales(w13_weight_scale, "w13_weight_scale")
    _check_block_scales(w2_weight_scale, "w2_weight_scale")
    dev = w13_weight.device
    if out is None:
        out = Nvfp4Sm90Experts(
            w13=torch.empty((E, w13_units(H, I), 576), dtype=torch.uint8, device=dev),
            w2=torch.empty(
                (E, H // 256, Ip // 64, 16, 576), dtype=torch.uint8, device=dev
            ),
            gs13=torch.empty(E, dtype=torch.float32, device=dev),
            gs2=torch.empty(E, dtype=torch.float32, device=dev),
            gsr=torch.empty(E, dtype=torch.float32, device=dev),
            hidden=H,
            intermediate=I,
        )
    elif (out.hidden, out.intermediate, out.gs13.shape[0]) != (H, I, E):
        raise ValueError("out holds a different layer shape")
    # GEMM1 rows: each 256-row block covers 128 intermediate columns as two halves of 64 gate rows then the same 64 up
    # rows (a last half block only the first half), and the intermediate columns run in the K order of GEMM2's packed
    # weights, so GEMM1 writes the intermediate in GEMM2's input order. Padded columns are zero rows of w13 and zero
    # columns of w2.
    ki = kperm_index(Ip, dev).view(Ip // 64, 64)
    rows = torch.stack((ki, Ip + ki), dim=1).reshape(-1)
    full = Ip // 128 * (H // 64) * 16  # units of the full 256-row blocks
    C1 = H // 64
    for e in range(E):
        w13e = w13_weight[e].view(2, I, H // 2)
        s13e = w13_weight_scale[e].view(torch.uint8).view(2, I, H // 16)
        w13e = torch.cat([_pad_rows(w13e[j], Ip) for j in range(2)]).index_select(
            0, rows
        )
        s13e = torch.cat([_pad_rows(s13e[j], Ip) for j in range(2)]).index_select(
            0, rows
        )
        dst = out.w13[e]
        if full:
            _pack_one(
                w13e[: Ip // 128 * 256],
                s13e[: Ip // 128 * 256],
                dst[:full].view(-1, C1, 16, 576),
            )
        if Ip % 128:
            _pack_one(w13e[-128:], s13e[-128:], dst[full:].view(1, C1, 8, 576))
        _pack_one(
            _pad_cols(w2_weight[e], Ip // 2),
            _pad_cols(w2_weight_scale[e].view(torch.uint8), Ip // 16),
            out.w2[e],
        )
    # An expert whose gate scale is 0 outputs exactly 0 either way.
    safe_gate = torch.where(gate != 0, gate, torch.ones_like(gate))
    out.gs13.copy_(gate)
    out.gs2.copy_(g2 * torch.where(gate != 0, up / safe_gate, torch.zeros_like(gate)))
    out.gsr.copy_(torch.where(gate != 0, up / safe_gate, torch.ones_like(gate)))
    return out
