"""Gather exact selected KV values for small TP-local MegaAttention decode."""

import torch
import triton
import triton.language as tl

from .dequant_k_cache import _e2m1_code_to_fp32


@triton.jit
def _gather(
    MAIN,
    EXTRA,
    MI,
    EI,
    ML,
    EL,
    OUT,
    INDICES,
    LENGTHS,
    NK: tl.constexpr,
    NE: tl.constexpr,
    MP: tl.constexpr,
    EP: tl.constexpr,
    MS: tl.constexpr,
    ES: tl.constexpr,
    MIS: tl.constexpr,
    EIS: tl.constexpr,
    MI1: tl.constexpr,
    EI1: tl.constexpr,
    HAS_ML: tl.constexpr,
    HAS_EL: tl.constexpr,
    MAIN_TOKENS,
    EXTRA_TOKENS,
):
    row = tl.program_id(0)
    token, slot = row // (NK + NE), row % (NK + NE)
    d = tl.arange(0, 512)
    main_length = tl.load(ML + token) if HAS_ML else NK
    extra_length = tl.load(EL + token) if HAS_EL else NE
    main_padded = tl.maximum(((main_length + 63) // 64) * 64, 64)
    if slot < main_padded:
        loc = tl.load(MI + token * MIS + slot * MI1)
        valid = (slot < main_length) & (loc >= 0) & (loc < MAIN_TOKENS)
        base = (tl.maximum(loc, 0) // MP).to(tl.int64) * MS
        offset = tl.maximum(loc, 0) % MP
        bits = tl.load(MAIN + base + offset * 512 + d, valid, 0)
        exponent = tl.load(MAIN + base + MP * 512 + offset * 16 + d // 32, valid, 0).to(
            tl.int32
        )
        scale = tl.where(exponent == 0, 0x00400000, exponent << 23).to(
            tl.float32, bitcast=True
        )
        value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    else:
        extra_slot = slot - main_padded
        loc = tl.load(EI + token * EIS + extra_slot * EI1, extra_slot < NE, -1)
        valid = (extra_slot < extra_length) & (loc >= 0) & (loc < EXTRA_TOKENS)
        base = (tl.maximum(loc, 0) // EP).to(tl.int64) * ES
        offset = tl.maximum(loc, 0) % EP
        bits = tl.load(EXTRA + base + offset * 256 + d // 2, valid, 0)
        sf = tl.load(EXTRA + base + EP * 256 + offset * 32 + d // 16, valid, 0)
        code = (bits >> ((d % 2) * 4)) & 15
        value = _e2m1_code_to_fp32(code) * sf.to(tl.float8e4nv, bitcast=True).to(
            tl.float32
        )
    tl.store(OUT + row * 512 + d, value.to(tl.bfloat16))
    tl.store(INDICES + row, tl.where(valid, row, -1))
    if slot == 0:
        tl.store(LENGTHS + token, main_padded + extra_length)


class MegaGatherPlan:
    def __init__(self, tokens: int, main_topk: int, extra_topk: int, device):
        self.tokens = tokens
        self.main_topk = main_topk
        self.extra_topk = extra_topk
        rows = tokens * (main_topk + extra_topk)
        self.values = torch.empty((rows, 1, 512), dtype=torch.bfloat16, device=device)
        self.indices = torch.empty(
            (tokens, 1, main_topk + extra_topk), dtype=torch.int32, device=device
        )
        self.lengths = torch.empty(tokens, dtype=torch.int32, device=device)

    def forward(
        self,
        q,
        main,
        extra,
        main_indices,
        extra_indices,
        main_lengths,
        extra_lengths,
        positions,
        cos_sin_cache,
        softmax_scale,
        attn_sink,
    ):
        if main.shape[-1] != 528 or extra.shape[-1] != 288:
            raise ValueError(
                "Gathered Mega decode requires V41 main and V41_FP4 extra caches"
            )
        main_bytes, extra_bytes = main.view(torch.uint8), extra.view(torch.uint8)
        _gather[(self.values.shape[0],)](
            main_bytes,
            extra_bytes,
            main_indices,
            extra_indices,
            main_lengths,
            extra_lengths,
            self.values,
            self.indices,
            self.lengths,
            self.main_topk,
            self.extra_topk,
            main.shape[1],
            extra.shape[1],
            main_bytes.stride(0),
            extra_bytes.stride(0),
            main_indices.stride(0),
            extra_indices.stride(0),
            main_indices.stride(1),
            extra_indices.stride(1),
            main_lengths is not None,
            extra_lengths is not None,
            main.shape[0] * main.shape[1],
            extra.shape[0] * extra.shape[1],
            num_warps=4,
        )
        data, scale, _, _ = (
            torch.ops.sgl_kernel.flash_mla_mega_prefill_unpacked.default(
                q,
                self.values,
                self.indices,
                positions,
                cos_sin_cache,
                softmax_scale,
                attn_sink,
                self.lengths,
            )
        )
        return data, scale
