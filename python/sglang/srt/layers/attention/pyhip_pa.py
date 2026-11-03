# SPDX-License-Identifier: MIT
# Helpers also copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Ported from tingqli/pyhip at f55f5493aae559f3d4090ff5ac645380243d1d99:
# src/pyhip/ops/mha/flydsl/mha_pa_bf16_942.py and required _common.py helpers;
# rocdl_aux and its dependencies from src/pyhip/codegen/flydsl/helpers.py.
# Upstream license follows.
# The MIT License (MIT)
# Copyright © 2026 <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""gfx942 BF16 paged attention for the MiMo full-prefill adapter.

This module contains the upstream D128/D192 BM256/BN64 kernel and its helper
closure. D256 and linear-attention APIs are outside this port. The SGLang
adapter retains its narrower MiMo geometry guard. Only Torch and FlyDSL are
required; importing this module does not load or modify the PyHIP package.
"""

import functools
import math
import os
from numbers import Real

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

_GFX94X_BITS = {
    # bit 0：缓存/一致性策略的一部分；对原子指令，还表示返回操作前的内存值。
    #       不要简单理解为“总是绕过某级缓存”。
    "sc0": 0,
    # bit 1：non-temporal（非时间局部性）提示，表示数据预计不会很快再次使用。
    "nt": 1,
    # bit 3：buffer 寻址的 swizzle（交错）标志；影响地址组织，
    #       不是对寄存器中的数据执行 ds_swizzle。
    "swz": 3,
    # bit 4：与 sc0 一起指定缓存/一致性策略。
    #       两个位的实际作用还取决于指令和同步范围。
    "sc1": 4,
    # bit 31：向 LLVM 后端标记此 buffer 操作为 volatile；
    #         在最终机器指令选择前会被移除，不是硬件缓存策略位。
    "volatile_op": 31,
}


@functools.cache
def _raw_ptr_buffer_load_op_aux_type_is_ir_value():
    try:
        # introduced by https://github.com/llvm/llvm-project/pull/198875
        policy = ir.Attribute.parse("#rocdl<gfx942_cache_policy sc0|nt>")
    except Exception as e:
        return True
    return False


def rocdl_aux(spec=""):
    """将 gfx942/gfx950 缓存策略名称转换为 i32 value 或 MLIR I32Attr"""
    if isinstance(spec, int):
        value = spec
    else:
        names = spec.strip().lower().split()
        bits = 0
        for name in names:
            if name not in _GFX94X_BITS:
                raise ValueError(f"unknown cache policy flag: {name!r}")
            bits |= 1 << _GFX94X_BITS[name]
        value = bits
    if _raw_ptr_buffer_load_op_aux_type_is_ir_value():
        return fx.Int32(value).ir_value()

    return ir.IntegerAttr.get(fx.Int32.ir_type, value)


_MAX_FLOOR = -3.4028234663852886e38


def _softmax_scale(softmax_scale, default):
    """Validated host softmax scale; default when None."""
    if isinstance(softmax_scale, (torch.Tensor, bool)) or not isinstance(
        softmax_scale, (Real, type(None))
    ):
        raise TypeError("softmax_scale must be a host real scalar or None")
    scale = default if softmax_scale is None else float(softmax_scale)
    # The kernels multiply masked -inf scores by it: an FP32 zero would give NaN.
    scaled = scale * math.log2(math.e)
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError("softmax_scale * log2(e) must be finite and positive in FP32")
    return scale


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _min(a, b):
    return (a < b).select(a, b)


def _pin_i32(value):
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(value).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _pin(values, chunk=8):
    values = fx.Vector(values)
    dtype = values.dtype
    words = values.bitcast(fx.Int32) if dtype.width < 32 else values
    result = []
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, min(start + chunk, words.numel))],
            words.dtype,
        )
        tied = fx.Vector(
            llvm.inline_asm(
                part.ir_value().type,
                [part.ir_value()],
                "",
                "=v,0",
                has_side_effects=True,
            )
        )
        result.extend(tied[i] for i in range(tied.numel))
    return fx.Vector.from_elements(result, words.dtype).bitcast(dtype)


def _join(a, b):
    a, b = fx.Vector(a), fx.Vector(b)
    return fx.Vector.from_elements(
        [a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.Float32
    )


def _exp(value):
    return fx.Float32(
        llvm.call_intrinsic(
            fx.Float32.ir_type,
            "llvm.amdgcn.exp2.f32",
            [fx.Float32(value).ir_value()],
            [],
            [],
        )
    )


def _intrinsic(result, name, *args):
    return result(
        llvm.call_intrinsic(
            result.ir_type, name, [fx.Float32(a).ir_value() for a in args], [], []
        )
    )


def _maximum(a, b):
    # Opaque to LLVM's MFMA hazard checks; the D256 kernels use _maxnum.
    return fx.Float32(
        llvm.inline_asm(
            fx.Float32.ir_type,
            [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
            "v_max_f32 $0, $1, $2",
            "=v,v,v",
            has_side_effects=False,
        )
    )


def _pin_s64(value):
    return fx.Int64(
        llvm.inline_asm(
            fx.Int64.ir_type,
            [fx.Int64(value).ir_value()],
            "",
            "=s,0",
            has_side_effects=True,
        )
    )


def _advance_max(old, new):
    return fx.Float32(
        llvm.inline_asm(
            fx.Float32.ir_type,
            [old.ir_value(), new.ir_value()],
            "v_mov_b32 $0, $2",
            "=v,0,v",
            has_side_effects=True,
        )
    )


def _pack_bf16(values):
    # gfx942 has no native packed FP32 -> BF16 instruction; software RNE for P and O (NaN not quieted).
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    # One opaque v_perm_b32 per pair (element i in the low half); rocdl.perm_b32 cost BF16 942 up to 28%.
    selector = fx.Int32(0x07060302)
    words = [
        fx.Uint32(
            llvm.inline_asm(
                fx.Int32.ir_type,
                [rounded[i + 1].ir_value(), rounded[i].ir_value(), selector.ir_value()],
                "v_perm_b32 $0, $1, $2, $3",
                "=v,v,v,s",
                has_side_effects=True,
            )
        )
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _align8():
    """The next instruction starts 8-byte aligned (at most one s_nop of padding).

    gfx942 fetches 32-byte blocks: in a dense run of 8-byte VALU/MFMA instructions that starts at an address
    = 4 (mod 8), every fourth one straddles two blocks and the wave idles ~4 cycles before it (ATT). The parity
    of a run otherwise follows from all code before it. It only costs where the run is issue-bound.
    """
    rocdl.sched_barrier(0)
    llvm.inline_asm(None, [], ".p2align 3", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _wait(*, vmcnt=63, expcnt=7, lgkmcnt=63):
    # gfx94x s_waitcnt encoding; an omitted counter is not waited on.
    rocdl.s_waitcnt(
        (vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)
    )


def _schedule(pairs, count, group, exp=False):
    # Interleave one MFMA with `count` VALU (0x002) or transcendental (0x400) ops.
    for _ in range(pairs):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(0x400 if exp else 0x002, count, group)


def _read_page_g2r(table, index):
    """s_load_dword of one page id; _page_ready makes it usable."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(table) + index))
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [address.ir_value()],
            "s_load_dword $0, $1, 0",
            "=s,s,~{memory}",
            has_side_effects=True,
        )
    )


def _page_ready(value):
    # A tied '=s,0' result lets register allocation copy the still-pending
    # SMEM result BEFORE the wait when page32/page64 aliases do not coalesce.
    # Keep the first read of that register inside the assembly, AFTER waiting.
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [value.ir_value()],
            "s_waitcnt lgkmcnt(0)\ns_mov_b32 $0, $1",
            "=s,s",
            has_side_effects=True,
        )
    )


def _buffer(tensor, size_bytes):
    """Raw buffer descriptor; loads at or past size_bytes return zero, stores are dropped."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"),
        pointer,
        fx.Int16(0).ir_value(),
        fx.Int64(size_bytes).ir_value(),
        fx.Int32(0x27000).ir_value(),
    )


def _buffer_words(resource, voffset, soffset=0):
    """One 16-byte buffer_load (g2r) at byte offsets voffset + soffset."""
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(
            ir.VectorType.get([4], fx.Int32.ir_type),
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(soffset).ir_value(),
            aux=rocdl_aux(0),
        )
    )


def _read_address(address, immediate=0):
    """One ds_read_b128 (s2r) at a precomputed LDS byte address; asynchronous until lgkmcnt."""
    return fx.Vector(
        llvm.inline_asm(
            ir.VectorType.get([4], fx.Int32.ir_type),
            [address.ir_value()],
            f"ds_read_b128 $0, $1 offset:{immediate}",
            "=v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _max3(a, b, c):
    return fx.Float32(
        llvm.inline_asm(
            fx.Float32.ir_type,
            [a.ir_value(), b.ir_value(), c.ir_value()],
            "v_max3_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            has_side_effects=False,
        )
    )


def _mfma_hazard_wait():
    # Inline-asm readers of fresh MFMA accumulators get no wait states from LLVM's hazard recognizer.
    # 11 = NumPasses(8) + 3 for the 32x32 MFMAs of this kernel on gfx942.
    rocdl.sched_barrier(0)
    llvm.inline_asm(None, [], "s_nop 10", "", has_side_effects=True)
    rocdl.sched_barrier(0)


def _row_max_local(values):
    # Independent MAX3 chains avoid the 32-deep VALU dependency chain.
    p = [_max3(values[i], values[i + 1], values[i + 2]) for i in range(0, 30, 3)]
    a = _max3(_max3(p[0], p[1], p[2]), _max3(p[3], p[4], p[5]), _max3(p[6], p[7], p[8]))
    b = _max3(p[9], values[30], values[31])
    return _max3(a, b, fx.Float32(_MAX_FLOOR))


def _row_max(values):
    value = _row_max_local(values)
    return _maximum(value, value.shuffle_xor(32, 64))


def _row_sum_local(values):
    partials = [values[i] + values[i + 1] for i in range(0, 32, 2)]
    for width in (8, 4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _row_sum(values):
    value = _row_sum_local(values)
    return value + value.shuffle_xor(32, 64)


def _center_part(values, scale, maximum, begin, end):
    """S' = S * scale - m (one FMA) for scores begin..end-1; the rest pass through."""
    result = []
    for i in range(32):
        value = values[i]
        if begin <= i < end:
            value = fx.Float32(
                llvm.inline_asm(
                    fx.Float32.ir_type,
                    [value.ir_value(), scale.ir_value(), maximum.ir_value()],
                    "v_fma_f32 $0, $1, $2, -$3",
                    "=v,v,v,v",
                    has_side_effects=False,
                )
            )
        result.append(value)
    return _pin(fx.Vector.from_elements(result, fx.Float32), 32)


def _reround(values, maximum):
    """FMA-centered S' = S * scale - m (32 scores) rounded to a multiple of ulp(m).

    That is round(S * scale) - m (see _scale), up to ties and the binade edge of m. The
    FMA alone leaves S' of the row max up to ulp(m) / 2 off zero.
    """
    # (x + c) - c with c = 1.5 * 2^exponent(m) rounds x to a multiple of ulp(m). In place:
    # this runs in cold branches of kernels near their VGPR limit.
    c = (
        (fx.Float32(maximum).bitcast(fx.Int32) & fx.Int32(0x7F800000))
        | fx.Int32(0x00400000)
    ).bitcast(fx.Float32)
    values = fx.Vector(values)
    return fx.Vector.from_elements(
        [
            fx.Float32(
                llvm.inline_asm(
                    fx.Float32.ir_type,
                    [values[i].ir_value(), c.ir_value()],
                    "v_add_f32 $0, $0, $2\nv_sub_f32 $0, $0, $2",
                    "=v,0,v",
                    has_side_effects=False,
                )
            )
            for i in range(32)
        ],
        fx.Float32,
    )


@flyc.jit
def _reround_if(values, maximum, exact):
    """_reround when the ballot mask exact is nonzero."""
    values = fx.Vector(values)
    if fx.Int64(exact) != fx.Int64(0):
        values = _reround(values, maximum)
    return values


def _exp_part(values, begin, end):
    """P = exp2(S') for scores begin..end-1; the rest pass through."""
    return fx.Vector.from_elements(
        [_exp(values[i]) if begin <= i < end else values[i] for i in range(32)],
        fx.Float32,
    )


BM, BN, THREADS = 256, 64, 512
PADDED_CHUNK = 1040
V_SLOT = 16640  # one DV128 V tile: 16 padded 1040-byte chunks
LOG2E = math.log2(math.e)
# P to BF16: RNE like O, or ties away from zero ("1": one VALU less per value in QK-high). Read at import.
_P_BF16_RTA = os.environ.get("AITER_FLYDSL_MHA_P_BF16_RTA", "0").lower() in (
    "1",
    "true",
)
# m: lazy, or "1": also moved (to row max + 1) when a new row max carries about l / 8 or more, so that the P of
# such a dominant key is exactly 1/2 in BF16. Read at import.
_EXACT_MAX = os.environ.get("AITER_FLYDSL_MHA_EXACT_MAX", "0").lower() in ("1", "true")


def _stage_end_aligned():
    """_stage_end, then the next stage starts 8-byte aligned (_align8; D128 main loop).

    Unaligned, the parity of each D128 loop copy followed from all code before it, about 1.6% per affected copy;
    aligned, it follows from the stage alone, and D128's dense S3 runs stay aligned. D192's S3 is misaligned this
    way (+1.5%), so it keeps _stage_end.
    """
    _stage_end()
    _align8()


def _row_sum_tail(pairs):
    """_row_sum_local after its first level: the sum of 16 pair sums, in the same order."""
    for width in (8, 4, 2, 1):
        pairs = [pairs[2 * i] + pairs[2 * i + 1] for i in range(width)]
    return pairs[0]


def _interleave(count, group, *, exps=0, valus=0, salus=0):
    """`count` MFMAs, each followed by up to `exps` transcendental, `valus` VALU and `salus` SALU ops."""
    for _ in range(count):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        for mask, ops in ((0x400, exps), (0x002, valus), (0x004, salus)):
            if ops:
                rocdl.sched_group_barrier(mask, ops, group)


def _cross32_async(value, address):
    # Issue in a memory stage; consume only after its explicit LGKM wait and
    # compiler/CTA boundary. This shuffle does not allocate LDS storage.
    return fx.Float32(
        llvm.inline_asm(
            fx.Float32.ir_type,
            [address.ir_value(), value.ir_value()],
            "ds_bpermute_b32 $0, $1, $2",
            "=v,v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _pack_probability(values):
    if not _P_BF16_RTA:
        return _pack_bf16(values).bitcast(fx.BFloat16)
    bits = fx.Vector(values).bitcast(fx.Uint32) + fx.Uint32(0x8000)
    selector = fx.Int32(0x07060302)
    words = [
        fx.Int32(
            llvm.inline_asm(
                fx.Int32.ir_type,
                [bits[i + 1].ir_value(), bits[i].ir_value(), selector.ir_value()],
                "v_perm_b32 $0, $1, $2, $3",
                "=v,v,v,s",
                has_side_effects=True,
            )
        )
        for i in range(0, 32, 2)
    ]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _first_max(scores, scale):
    return _maximum(_row_max(scores) * scale, fx.Float32(_MAX_FLOOR)) + 1.0


def _next_max(local_max, scale, row_max, row_sum):
    """Ballot of the rows whose m moves, and the new m: lazily a row takes its max + 1 once that exceeds m + 7;
    with _EXACT_MAX the whole wave moves to max(m, max + 1), also once some 2^(max - m) passes about l / 8.
    """
    if _EXACT_MAX:
        # Moves when max - m > med3(e - 3, -1, 7) (+3 folded into the FMA), l in [2^(e-1), 2^e) incl. P(t-1):
        # a new max (m = old max + 1) whose P passes l / 8..l / 4, or any max past m + 7.
        candidate = _intrinsic(fx.Float32, "llvm.fma.f32", local_max, scale, 3.0)
        exponent = _intrinsic(fx.Int32, "llvm.amdgcn.frexp.exp.i32.f32", row_sum).to(
            fx.Float32
        )
        threshold = _intrinsic(fx.Float32, "llvm.amdgcn.fmed3.f32", exponent, 2.0, 10.0)
        ballot = fx.Int64(
            rocdl.ballot(fx.Int64.ir_type, (candidate - row_max > threshold).ir_value())
        )
        return ballot, (ballot != fx.Int64(0)).select(
            _maximum(row_max, candidate - 2.0), row_max
        )
    candidate = local_max * scale
    ballot = fx.Int64(
        rocdl.ballot(fx.Int64.ir_type, (candidate > row_max + 7.0).ir_value())
    )
    return ballot, (candidate > row_max + 7.0).select(candidate + 1.0, row_max)


def _read_tile_pages_g2r(table, tile, last, last_page, page_size):
    """Page ids (first, second) of a KV tile: BN64 spans two page32 pages; page128 holds two tiles."""
    # Clamp every speculative SMEM request, including the second page.
    first = _min(tile, last) * BN // page_size
    page0 = _read_page_g2r(table, first)
    page1 = (
        _read_page_g2r(table, _min(first + 1, last_page)) if page_size == 32 else page0
    )
    return page0, page1


def _read_global_g2r(tensor, offset):
    # A genuine 64-bit GLOBAL pointer; never replace V with a raw descriptor.
    pointer = fx.get_iter(tensor) + fx.Int64(offset)
    source = fx.make_view(pointer, fx.make_layout(8, 1))
    fragment = fx.make_rmem_tensor(8, fx.BFloat16)
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16), source, fragment)
    return fragment.load().bitcast(fx.Int32)


def _read_s2r(storage, base, immediate=0):
    address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + base))
    return fx.Vector(
        llvm.inline_asm(
            ir.VectorType.get([4], fx.Int32.ir_type),
            [address.ir_value()],
            f"ds_read_b128 $0, $1 offset:{immediate}",
            "=v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _store_r2s(address, words, rounds, immediate=0):
    """ds_write_b128 of `rounds` 16-byte words at a precomputed LDS address, 8320 bytes apart."""
    for i in range(rounds):
        part = fx.Vector.from_elements([words[i * 4 + j] for j in range(4)], fx.Int32)
        llvm.inline_asm(
            ir.Type.parse("!llvm.void"),
            [address.ir_value(), part.ir_value()],
            f"ds_write_b128 $0, $1 offset:{immediate + i * 8320}",
            "v,v,~{memory}",
            has_side_effects=True,
        )


# ---- DV128 hot path: per-lane addresses are prepared once, stages only issue ----


def _k_g2r_offset(lane_offset, page0, page1, tile, half_page, hk, dq, page_size):
    page = half_page.select(page1, page0) if page_size == 32 else page0
    return _pin_i32(
        lane_offset + page * (hk * page_size * dq * 2) + (tile * BN % page_size) * 16
    )


def _k_g2r_bases(base, page, tile, hk, dq, page_size):
    """64-bit SGPR address of each DQ/64 part (8 columns) of a page64/page128 K tile."""
    first = base + fx.Int64(page) * (hk * page_size * dq * 2)
    if page_size == 128:
        first = first + fx.Int64(tile & 1) * (BN * 16)
    return [_pin_s64(first + i * page_size * 128) for i in range(dq // 64)]


def _global_words(address, lane_offset):
    """One 16-byte global_load_dwordx4 at a 64-bit SGPR address plus a VGPR byte offset."""
    return fx.Vector(
        llvm.inline_asm(
            ir.VectorType.get([4], fx.Int32.ir_type),
            [lane_offset.ir_value(), address.ir_value()],
            "global_load_dwordx4 $0, $1, $2",
            "=v,v,s,~{memory}",
            has_side_effects=True,
        )
    )


def _read_k_g2r(resource, address, lane_offset, dq, page_size):
    """K tile into staging VGPRs.

    Page32 tiles span two pages split between the lane halves: one 32-bit buffer over the pool with
    the divergent page in VOFFSET (never a scalar waterfall). Larger pages: `address` holds the
    _k_g2r_bases, so the pool may exceed 4 GiB.
    """
    if page_size == 32:
        parts = [
            _buffer_words(resource, address, i * page_size * 128)
            for i in range(dq // 64)
        ]
    else:
        parts = [_global_words(base, lane_offset) for base in address]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _v_g2r_bases(base, page0, page1, tile, hk, page_size):
    first = (
        base
        + fx.Int64(page0) * (hk * page_size * 256)
        + fx.Int64(tile * BN % page_size) * 256
    )
    second = (
        base + fx.Int64(page1) * (hk * page_size * 256)
        if page_size == 32
        else first + 8192
    )
    return _pin_s64(first), _pin_s64(second)


def _read_v_g2r(first, second, lane_offset):
    # Full 64-bit SGPR address plus lane-relative VGPR byte offset. This is
    # GLOBAL, not BUFFER, and does not truncate the tensor's GPU base.
    parts = [_global_words(address, lane_offset) for address in (first, second)]
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _read_k_s2r(address, half, dq):
    """Keys 32 * half .. 32 * half + 31 as the QK A operand, from the pinned LDS read address."""
    parts = [_read_address(address, half * 512 + d * 2080) for d in range(dq // 16)]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(
        fx.make_rmem_tensor(fx.make_layout((32, dq), (1, 32)), fx.BFloat16)
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_v_s2r(address, half, slot=0):
    """V dims 64 * half .. 64 * half + 63 of ring slot `slot` as the PV A operand."""
    parts = [
        _read_address(address, slot * V_SLOT + half * 1040 + n * 512 + k * 4160)
        for n in range(2)
        for k in range(4)
    ]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(fx.make_layout((8, 4, 2), (1, 8, 32)), fx.BFloat16)
    frag.store(words.bitcast(fx.BFloat16))
    return frag


# ---- Cooperative tile copies (LSE/V192 path and the D128 prologue) ----


def _read_k_tile_g2r(resource, bases, tid, page0, page1, tile, hk, dq, page_size):
    """Cooperative K tile g2r; page64/page128 read from the tile's _k_g2r_bases."""
    if dq == 192:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    if page_size != 32:
        # Atom tid + i * THREADS is token tid % BN of column tid // BN + 8i: the bases add the 8i.
        return _read_k_g2r(
            resource,
            bases,
            (tid // BN) * (page_size * 16) + (tid % BN) * 16,
            dq,
            page_size,
        )
    parts = []
    for i in range(dq // 64):
        atom = tid + i * THREADS
        token, column = atom % BN, atom // BN
        page = (token < 32).select(page0, page1)
        within = token % page_size + (tile * BN) % page_size
        offset = column * page_size * 16 + within * 16
        # Different halves of a wave copy different pages. Their page
        # offset belongs in VOFFSET, not scalar SOFFSET (which waterfalls).
        parts.append(_buffer_words(resource, offset + page * (hk * page_size * dq * 2)))
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _read_v_tile_g2r(
    tensor, tid, page0, page1, tile, hkv, hk, dv, page_size, bounded=False
):
    if bounded:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    parts = []
    for i in range(dv // 64):
        atom = tid + i * THREADS
        token, column = (atom // dv) * 8, atom % dv
        page = (token < 32).select(page0, page1) if page_size == 32 else page0
        within = token % page_size + (tile * BN) % page_size
        offset = (
            (fx.Int64(page) * hk + fx.Int64(hkv)) * (page_size * dv)
            + fx.Int64(within // 8) * (dv * 8)
            + fx.Int64(column) * 8
        )
        parts.append(_read_global_g2r(tensor, offset))
    return fx.Vector.from_elements([p[j] for p in parts for j in range(4)], fx.Int32)


def _store_tile_r2s(storage, tid, words, slot_offset, rounds, key=False, bounded=False):
    if bounded:
        tid = _pin_i32(fx.Int32(gpu.thread_id("x"))) & 511
    atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    for i in range(rounds):
        byte = (tid + i * THREADS) * 16
        chunk, within = byte // 1024, byte % 1024
        if key:
            within = within ^ ((within & 256) >> 2)
        offset = slot_offset + chunk * PADDED_CHUNK + within
        pointer = fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, storage.memspace, 16),
            fx.get_iter(storage) + offset,
        )
        src = fx.make_rmem_tensor(4, fx.Int32)
        src.store(
            fx.Vector.from_elements([words[4 * i + j] for j in range(4)], fx.Int32)
        )
        fx.copy(atom, src, fx.make_view(pointer, fx.make_layout(4, 1)))


def _mma():
    # BF16 atom K8 consumes four values/lane. Pair its reduction steps for
    # b128 copies, rather than reinterpreting the FP8 atom's eight values.
    return fx.make_tiled_mma(
        fx.make_mma_atom(rocdl.MFMA(32, 32, 8, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
        (None, None, fx.make_layout((4, 2, 2), (1, 8, 4))),
    )


def _read_q_g2r(resource, row_offset, lane, dq):
    parts = [
        _buffer_words(resource, row_offset + (lane >> 5) * 16 + d * 32)
        for d in range(dq // 16)
    ]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((BM, dq), (1, BM)), fx.BFloat16)
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_k_tile_s2r(storage, base, half, dq, dv, full=False):
    """Keys 32 * half .. + 31 from the K slot; D192/V192 without `full` read only the first K64 chunk."""
    width = 64 if (dq == 192 or dv == 192) and not full else dq
    parts = [
        _read_s2r(storage, base, half * 512 + d * 2080) for d in range(width // 16)
    ]
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = _mma().make_fragment_A(
        fx.make_rmem_tensor(fx.make_layout((32, width), (1, 32)), fx.BFloat16)
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


def _read_v_tile_s2r(storage, base, slot, half, dq, dv):
    width = dv // 2
    parts = []
    for n in range(width // 32):
        for k in range(BN // 16):
            byte = (2 * k * dv + half * width + n * 32) * 16
            parts.append(
                _read_s2r(
                    storage,
                    ((dq + slot * dv) // 8) * PADDED_CHUNK + base,
                    (byte // 1024) * PADDED_CHUNK + byte % 1024,
                )
            )
    words = fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)
    frag = fx.make_rmem_tensor(
        fx.make_layout((8, 4, width // 32), (1, 8, 32)), fx.BFloat16
    )
    frag.store(words.bitcast(fx.BFloat16))
    return frag


@flyc.jit
def _v_tail(values, remaining):
    words = fx.Vector(values).bitcast(fx.Int32)
    if remaining < BN:
        limit = _pin_i32(remaining - ((fx.Int32(gpu.thread_id("x")) >> 5) & 1) * 8)
        masks = []
        for i in fx.range_constexpr(16):
            token = (i // 4) * 16 + (i % 4) * 2
            mask = (limit > token).select(fx.Int32(0xFFFF), fx.Int32(0))
            mask = mask | (limit > token + 1).select(fx.Int32(-65536), fx.Int32(0))
            masks.append(mask)
        words = fx.Vector.from_elements(
            [words[i] & masks[i % 16] for i in range(words.numel)], fx.Int32
        )
    return words.bitcast(fx.BFloat16)


def _qk(q, k, storage, base, half, dq, dv, full=False):
    mma = _mma()
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, BM), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    if (dq == 192 or dv == 192) and not full:
        # Full D192 K + Q + previous P + O exceeds the two-wave VGPR
        # budget. Consume contiguous K64 chunks in the SAME reduction order;
        # each retired operand frees its registers before the next LDS read.
        for chunk in range(dq // 64):
            if chunk:
                rocdl.sched_barrier(0)
                k = _read_k_tile_s2r(storage, base + chunk * 8320, half, 64, dv)
                _wait(lgkmcnt=0)
                # Inline LDS reads are opaque to LLVM's wait insertion. A
                # scheduler fence prevents the following MFMAs from moving
                # above this explicit wait (verified in the failed K64 ISA).
                rocdl.sched_barrier(0)
            q_part = fx.make_view(
                fx.get_iter(q) + chunk * 32,
                fx.make_layout((4, 1, (2, 4)), (1, 0, (4, 8))),
            )
            fx.gemm(mma, acc, k, q_part, acc, traversal_order="kmn")
    else:
        fx.gemm(mma, acc, k, q, acc, traversal_order="kmn")
    return acc.load()


def _pv(p, v, output, dv):
    mma = _mma()
    prob = mma.make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((BM, BN), (1, BM)), fx.BFloat16)
    )
    prob.store(p)
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((dv // 2, BM), (1, dv // 2)), fx.Float32)
    )
    acc.store(output)
    operand = fx.make_view(
        fx.get_iter(v), fx.make_layout((4, dv // 64, (2, 4)), (1, 32, (4, 8)))
    )
    fx.gemm(mma, acc, operand, prob, acc, traversal_order="mnk")
    return acc.load()


@flyc.jit
def _mask(scores, tile, row, q_len, kv_len, causal: fx.Constexpr[bool]):
    values = fx.Vector(scores)
    lane_half = (fx.Int32(gpu.thread_id("x")) >> 5) & 1
    if fx.const_expr(causal):
        if row - (fx.Int32(gpu.thread_id("x")) & 31) + kv_len - q_len < (tile + 1) * BN:
            bound = _pin_i32(
                _min(kv_len - 1, kv_len - q_len + row) - tile * BN - lane_half * 8
            )
            values = fx.Vector.from_elements(
                [
                    (bound >= fx.Int32((i // 8) * 16 + i % 8)).select(
                        values[i], fx.Float32(float("-inf"))
                    )
                    for i in range(32)
                ],
                fx.Float32,
            )
    else:
        if kv_len - tile * BN < BN:
            # Materialize the lane coordinate only in the rare tail branch.
            # Otherwise LICM carries a per-lane bound with a VALU decrement
            # at the next iteration's memory-stage entry.
            tail_lane = _pin_i32(fx.Int32(gpu.thread_id("x")))
            bound = _pin_i32(kv_len - 1 - tile * BN - ((tail_lane >> 5) & 1) * 8)
            values = fx.Vector.from_elements(
                [
                    (bound >= fx.Int32((i // 8) * 16 + i % 8)).select(
                        values[i], fx.Float32(float("-inf"))
                    )
                    for i in range(32)
                ],
                fx.Float32,
            )
    return values


@flyc.jit
def _rescale(o0, o1, row_sum, s_cur, old_max, new_max, ballot, exact):
    """Rescale of O and l when m moved (_next_max), plus the exact rule for S'(t).

    The FMA S * scale - m is up to ulp(m) / 2 off the rounded m. Once some |m| >= 2^22
    (checked when m moves, then sticky), S'(t) is also rounded to a multiple of ulp(m).
    """
    o0, o1, row_sum, s_cur = (
        fx.Vector(o0),
        fx.Vector(o1),
        fx.Float32(row_sum),
        fx.Vector(s_cur),
    )
    ballot, exact = fx.Int64(ballot), fx.Int64(exact)
    if ballot != fx.Int64(0):
        correction = _exp(fx.Float32(old_max) - fx.Float32(new_max))
        o0, o1, row_sum = o0 * correction, o1 * correction, row_sum * correction
        big = fx.absf(fx.Float32(new_max)) >= 2.0**22
        exact = exact | fx.Int64(rocdl.ballot(fx.Int64.ir_type, big.ir_value()))
    # A separate branch: merged into the one above, V192 (spilling) lost 5%.
    s_cur = _reround_if(s_cur, new_max, exact)
    return o0, o1, row_sum, s_cur, exact


def _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, heads, dv):
    """O * inv to BF16, C-shuffled through LDS (r2s b64, s2r b128), then b128 row stores (r2g).

    Reuses 32 KiB of LDS for one 64-column slice at a time (V192 has a third slice).
    """
    atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
    outputs = _join(o0, o1)
    for half in fx.range_constexpr(dv // 64):
        for n in fx.range_constexpr(2):
            for group in fx.range_constexpr(4):
                val = fx.Vector.from_elements(
                    [
                        outputs[half * 32 + n * 16 + group * 4 + i] * inv
                        for i in range(4)
                    ],
                    fx.Float32,
                )
                words = _pack_bf16(val)
                col = n * 32 + group * 8 + ((out_tid >> 5) & 1) * 4
                element = (out_row * 64 + col) ^ ((out_row & 15) * 4)
                address = fx.Int32(fx.ptrtoint(fx.get_iter(storage) + element * 2))
                llvm.inline_asm(
                    ir.Type.parse("!llvm.void"),
                    [address.ir_value(), words.ir_value()],
                    "ds_write_b64 $0, $1",
                    "v,v,~{memory}",
                    has_side_effects=True,
                )
        _wait(lgkmcnt=0)
        _stage_end()
        for i in fx.range_constexpr(4):
            element = out_tid * 8 + i * THREADS * 8
            read_row, read_col = element // 64, element % 64
            element = element ^ ((read_row & 14) * 4)
            words = _read_s2r(storage, element * 2)
            _wait(lgkmcnt=0)
            rocdl.sched_barrier(0)
            src = fx.make_rmem_tensor(8, fx.BFloat16)
            src.store(
                fx.Vector.from_elements(
                    [
                        ((read_row & 1) == 0).select(words[j], words[j ^ 2])
                        for j in range(4)
                    ],
                    fx.Int32,
                ).bitcast(fx.BFloat16)
            )
            offset = read_row * (heads * dv) + half * 64 + read_col
            fx.copy(
                atom,
                src,
                fx.make_view(fx.get_iter(obuf) + offset, fx.make_layout(8, 1)),
            )
        _stage_end()


@flyc.jit
def _body_v128(
    Q,
    K,
    V,
    O,
    LSE,
    QS,
    KS,
    VS,
    table,
    storage,
    q0,
    q_len,
    kv_len,
    head,
    qb,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NP: fx.Int32,
    DQ: fx.Constexpr[int],
    DV: fx.Constexpr[int],
    QROW: fx.Constexpr[int],
    QHEAD: fx.Constexpr[int],
    PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    STAGGER: fx.Constexpr[bool],
):
    """DV128 without LSE (Dq 128/192): query rows 256 * qb .. + 255 of one head over KV tiles t.

    LDS: one K slot of (Dq / 8) * 1040 bytes and a two-slot V ring (V(t) in slot t & 1).
    Page ids run two tiles ahead of QK, K one tile ahead (g2r in S4, r2s in S6 of phase
    t - 1, after both groups read K(t - 1)) and V one tile ahead (GLOBAL g2r in S0, r2s
    in S2). Phase t runs QK and the row max of tile t and, one tile behind, the
    exponentials, row sum and PV of tile t - 1. Even stages are memory (VMEM/LDS),
    odd stages compute (MFMA/VALU); waves 4-7 (STAGGER) run one stage behind waves 0-3.
    Row sums/maxima of the two lane halves meet through ds_bpermute in memory stages.

    stage  phase t; cur = t & 1, prev = (t - 1) & 1
    S0     s2r K(t) keys 0-31; s_load pages(t+2); g2r V(t)
    S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-23
    S2     V(t) lands; r2s V(t) -> [cur]; s2r K(t) keys 32-63; page64/128: K(t+1) page bases
    S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, lane row sum, P to BF16
    S4     exchange row sums; s2r V(t-1) dims 0-63 [prev]; g2r K(t+1)
    S5     PV: O dims 0-63 += P V      softmax: l += row sum; mask S(t), lane row max
    S6     exchange row maxima; s2r V(t-1) dims 64-127 [prev]; K(t+1) lands; r2s K(t+1)
    S7     PV: O dims 64-127 += P V    softmax: new m, S'(t); rescale O and l

    D128 offloads the longest stage, S3: it keeps only the first level of the lane row sum,
    S5 finishes it, S6 exchanges it with the row maxima and S7 adds it to l before the new m
    (same additions in the same order).

    D192 gives S1 one exp per MFMA and spreads S3's VALU over its MFMAs, masks S(t) in S4 and
    ends S7 with its barrier before the cold rescale branches: their taken skips, like the mask's,
    used to delay the other group's next compute stage.

    Softmax in log2 units: S' = S * scale - m (one FMA), P = exp2(S'), l = sum(P) and
    O = sum(P V); the output is O / l * v_descale. m starts at the first row max + 1 and
    moves to a later row max + 1 only when that exceeds m + 7, rescaling O and l
    (AITER_FLYDSL_MHA_EXACT_MAX=1: also when a new max carries about l / 8, _next_max). The
    FMA is up to ulp(m) / 2 off the rounded m: once some |m| >= 2^22 (checked when m
    moves, then sticky), a cold branch of _rescale also rounds S'(t) to a multiple of
    ulp(m), which matches the D256 kernels' rounded products.
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (
        _read_k_g2r,
        _k_g2r_offset,
        _v_g2r_bases,
        _read_v_g2r,
        _store_r2s,
        _read_k_s2r,
        _read_v_s2r,
        _read_k_tile_g2r,
        _read_k_tile_s2r,
        _read_v_tile_s2r,
        _qk,
        _pv,
        _mask,
        _v_tail,
        _join,
        _mfma_hazard_wait,
        _maximum,
        _row_max,
        _row_max_local,
        _row_sum,
        _row_sum_local,
        _center_part,
        _exp_part,
        _pack_probability,
        _cross32_async,
        _rescale,
        _advance_max,
        _pin,
        _schedule,
        _stage_end,
        _wait,
        _reround_if,
        _first_max,
        _intrinsic,
        _next_max,
        _row_sum_tail,
        _stage_end_aligned,
        _align8,
        _interleave,
        _k_g2r_bases,
        _global_words,
    )
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 32 + (lane & 31)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    q_ptr = fx.get_iter(Q) + (
        (fx.Int64(q0) + fx.Int64(q_start)) * QROW + fx.Int64(head) * QHEAD
    )
    # Rows past `valid` read zero: the host keeps QROW >= DQ.
    gq = _buffer(fx.make_view(q_ptr, fx.make_layout(BM * QROW, 1)), valid * QROW * 2)
    gk, k_global_base = None, None
    if fx.const_expr(PAGE == 32):
        gk = _buffer(
            fx.make_view(
                fx.get_iter(K) + hkv * PAGE * DQ,
                fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1),
            ),
            (NP * HK - hkv) * PAGE * DQ * 2,
        )
    else:
        k_global_base = _pin_s64(
            fx.Int64(fx.ptrtoint(fx.get_iter(K))) + fx.Int64(hkv) * (PAGE * DQ * 2)
        )
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 32 + (lane & 31)) * (QROW * 2), lane, DQ)
    scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
    if fx.const_expr(PER_TOKEN):
        scale = (
            scale * rocdl.make_buffer_tensor(QS, max_size=False)[(q0 + row) * H + head]
        )
    else:
        scale = scale * QS[0]
    scale = fx.Float32(scale)
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last, last_page = tiles - 1, (kv_len - 1) // PAGE
    # Page ids of tiles 0-2; every later phase t requests pages(t+2).
    pages_0 = _read_tile_pages_g2r(table, fx.Int32(0), last, last_page, PAGE)
    pages_1 = _read_tile_pages_g2r(table, fx.Int32(1), last, last_page, PAGE)
    pages_2 = _read_tile_pages_g2r(table, fx.Int32(2), last, last_page, PAGE)
    pages_0 = _page_ready(pages_0[0]), _page_ready(pages_0[1])
    pages_1 = _page_ready(pages_1[0]), _page_ready(pages_1[1])
    pages_2 = _page_ready(pages_2[0]), _page_ready(pages_2[1])
    # BF16 P carries eight consecutive keys per b128, not FP8's sixteen.
    k_row = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (DV // 64) * PADDED_CHUNK + (lane & 31) * 16
    k_bytes = (DQ // 8) * PADDED_CHUNK
    half_page = (lane & 32) != 0
    # Loop-invariant lane addresses: LDS read/write (r/w) bases, K/V g2r lane offsets.
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    within = (tid & 63) * 16
    k_s2r = _pin_i32(shared + k_base)
    k_r2s = _pin_i32(
        shared + (tid >> 6) * PADDED_CHUNK + (within ^ ((within & 256) >> 2))
    )
    v_s2r = _pin_i32(shared + k_bytes + v_base)
    v_r2s = _pin_i32(shared + k_bytes + (tid >> 6) * PADDED_CHUNK + within)
    k_lane = _pin_i32((tid >> 6) * (PAGE * 16) + (tid & (min(PAGE, BN) - 1)) * 16)
    v_lane = _pin_i32(tid * 16)
    cross_address = _pin_i32((lane ^ 32) * 4)
    v_global_base = _pin_s64(
        fx.Int64(fx.ptrtoint(fx.get_iter(V))) + fx.Int64(hkv) * (PAGE * DV * 2)
    )

    def read_k_g2r(pages, tile):
        if fx.const_expr(PAGE == 32):
            address = _k_g2r_offset(
                k_lane, pages[0], pages[1], tile, half_page, HK, DQ, PAGE
            )
        else:
            address = _k_g2r_bases(k_global_base, pages[0], tile, HK, DQ, PAGE)
        return _read_k_g2r(gk, address, k_lane, DQ, PAGE)

    def read_v_g2r(pages, tile):
        first, second = _v_g2r_bases(v_global_base, pages[0], pages[1], tile, HK, PAGE)
        return _read_v_g2r(first, second, v_lane)

    def prologue():
        """K(0), V(0) -> LDS; QK S(0), m and S'(0); K(1) -> LDS (D192 loads it during QK(0))."""
        # Memory: g2r K(0), V(0); r2s K(0) and V(0) -> [0].
        k_0 = read_k_g2r(pages_0, fx.Int32(0))
        v_0 = read_v_g2r(pages_0, fx.Int32(0))
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_0, DQ // 64)
        _store_r2s(v_r2s, v_0, 2)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31.
        if fx.const_expr(DQ == 192):
            k_lo = _read_k_s2r(k_s2r, 0, DQ)
        else:
            k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV, True)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV, True)
        o0 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        o1 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        _schedule(DQ // 8, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63 (D192 also issues g2r K(1)).
        if fx.const_expr(DQ == 192):
            k_hi = _read_k_s2r(k_s2r, 1, DQ)
            k_1 = read_k_g2r(pages_1, _min(fx.Int32(1), last))
        else:
            k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV, True)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV, True)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        _mfma_hazard_wait()
        row_max = _first_max(s_0, scale)
        s_0 = _center_part(s_0, scale, row_max, 0, 32)
        # The exact rule of _rescale for S'(0); exact carries it through the phases.
        exact = fx.Int64(
            rocdl.ballot(fx.Int64.ir_type, (fx.absf(row_max) >= 2.0**22).ir_value())
        )
        s_0 = _reround_if(s_0, row_max, exact)
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: both groups read K(0); g2r K(1) (D128) and r2s it into the single K slot.
        if fx.const_expr(DQ == 128):
            tile_1 = _min(fx.Int32(1), last)
            bases_1 = None
            if fx.const_expr(PAGE != 32):
                bases_1 = _k_g2r_bases(k_global_base, pages_1[0], tile_1, HK, DQ, PAGE)
            k_1 = _read_k_tile_g2r(
                gk, bases_1, tid, pages_1[0], pages_1[1], tile_1, HK, DQ, PAGE
            )
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_1, DQ // 64)
        _wait(lgkmcnt=0)
        _stage_end()
        # The leading group waits one more stage for the trailing group's K(1)
        # writes; an LGKM wait alone cannot cover another wave.
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, *pages_1, *pages_2, exact

    @flyc.jit
    def phase(
        s_prev,
        row_max,
        row_sum,
        o0,
        o1,
        page_cur0,
        page_cur1,
        page_next0,
        page_next1,
        exact,
        t,
        CUR: fx.Constexpr[int],
        PREV: fx.Constexpr[int],
    ):
        """Phase t (table in _body_v128); page_cur*/page_next* are the page ids of tiles t and t+1.

        D128 unrolls two phases so CUR/PREV are constants; D192 runs one phase per trip and
        derives the V slots from t. exact carries the exact rule of _rescale.
        """
        s_prev, row_max, row_sum = (
            fx.Vector(s_prev),
            fx.Float32(row_max),
            fx.Float32(row_sum),
        )
        o0, o1, exact, t = fx.Vector(o0), fx.Vector(o1), fx.Int64(exact), fx.Int32(t)
        slot_cur = (t & 1) if DQ == 192 else CUR
        slot_prev = ((t - 1) & 1) if DQ == 192 else PREV
        stage_end = _stage_end_aligned if DQ == 128 else _stage_end
        # S0 memory: s2r K(t) keys 0-31; s_load pages(t+2); g2r V(t) (SGPR bases only change).
        k_lo = _read_k_s2r(k_s2r, 0, DQ)
        rocdl.sched_barrier(0)
        requests_next2 = _read_tile_pages_g2r(table, t + 2, last, last_page, PAGE)
        v_cur = read_v_g2r((page_cur0, page_cur1), t)
        rocdl.sched_barrier(0)
        page_next2_0, page_next2_1 = _page_ready(requests_next2[0]), _page_ready(
            requests_next2[1]
        )
        stage_end()
        # S1 compute: QK S(t) keys 0-31; P(t-1) values 0-23. D192 ring addresses are made here.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV, True)
        v_read, v_write = v_s2r, v_r2s
        if fx.const_expr(DQ == 192):
            v_read = _pin_i32(v_s2r + slot_prev * V_SLOT)
            v_write = _pin_i32(v_r2s + slot_cur * V_SLOT)
        p_prev = _pin(_exp_part(s_prev, 0, 24), 32)
        if fx.const_expr(DQ == 192):
            # A 16-cycle exp fits beside a 32-cycle MFMA; two per MFMA left each of them 4 cycles late.
            _interleave(DQ // 8, 1, exps=1, valus=1, salus=1)
        else:
            _schedule(DQ // 8, 2, 1, True)
        stage_end()
        # S2 memory: r2s V(t) -> [cur] once it lands; s2r K(t) keys 32-63.
        _wait(vmcnt=0)
        _store_r2s(v_write, v_cur, 2, CUR * V_SLOT if DQ == 128 else 0)
        k_hi = _read_k_s2r(k_s2r, 1, DQ)
        if fx.const_expr(PAGE != 32):
            # The SALU of K(t+1)'s 64-bit page bases runs while the LDS reads drain.
            k_next = _k_g2r_bases(
                k_global_base, page_next0, _min(t + 1, last), HK, DQ, PAGE
            )
        _wait(lgkmcnt=0)
        stage_end()
        # S3 compute: QK S(t) keys 32-63; P(t-1) values 24-31, lane row sum (D128: pair sums), BF16 P;
        # page32: K(t+1) address.
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV, True)
        p_prev = _exp_part(p_prev, 24, 32)
        if fx.const_expr(DQ == 128):
            sum_pairs = [p_prev[i] + p_prev[i + 1] for i in range(0, 32, 2)]
        else:
            lane_sum = _row_sum_local(p_prev)
        p_bf16 = _pack_probability(p_prev)
        if fx.const_expr(DQ == 192):
            # Every MFMA gets an exp or at most 6 VALU. The opaque v_perm packs are not VALU to the scheduler;
            # the last two MFMAs are left to them so they do not trail the stage.
            _interleave(8, 2, exps=1, valus=2)
            _interleave(DQ // 8 - 10, 2, valus=6)
            _interleave(2, 2)
        else:
            _schedule(4, 2, 2, True)
            _schedule(DQ // 8 - 4, 7, 2)
        if fx.const_expr(PAGE == 32):
            k_next = _k_g2r_offset(
                k_lane,
                page_next0,
                page_next1,
                _min(t + 1, last),
                half_page,
                HK,
                DQ,
                PAGE,
            )
        stage_end()
        # S4 memory: exchange row sums (D192); s2r V(t-1) dims 0-63; g2r K(t+1).
        if fx.const_expr(DQ == 192):
            other_sum = _cross32_async(lane_sum, cross_address)
        v_lo = _read_v_s2r(v_read, 0, PREV if DQ == 128 else 0)
        _wait(vmcnt=0)
        k_next = _read_k_g2r(gk, k_next, k_lane, DQ, PAGE)
        if fx.const_expr(DQ == 192):
            # The mask's skip branch is a taken jump; here it no longer delays the first MFMA of S5.
            s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        # D192's shuffle shares this LDS wait; it has no consumer before S5.
        _wait(lgkmcnt=0)
        stage_end()
        # S5 compute: PV O dims 0-63; l += row sum (D128: lane row sum); mask S(t) and its lane row max.
        if fx.const_expr(DQ == 128):
            s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        o0 = _pv(p_bf16, v_lo, o0, DV)
        if fx.const_expr(DQ == 192):
            row_sum = row_sum + (lane_sum + other_sum)
        lane_max = _row_max_local(s_cur)
        if fx.const_expr(DQ == 128):
            lane_sum = _row_sum_tail(sum_pairs)
        _schedule(DV // 8, 4 if DQ == 128 else 2, 3)
        stage_end()
        # S6 memory: exchange row maxima (D128: and row sums); s2r V(t-1) dims 64-127; r2s K(t+1) after
        # both groups read K(t).
        other_max = _cross32_async(lane_max, cross_address)
        if fx.const_expr(DQ == 128):
            other_sum = _cross32_async(lane_sum, cross_address)
        v_hi = _read_v_s2r(v_read, 1, PREV if DQ == 128 else 0)
        _wait(vmcnt=0)
        _store_r2s(k_r2s, k_next, DQ // 64)
        _wait(lgkmcnt=0)
        stage_end()
        # S7 compute: PV O dims 64-127; D128: l += row sum; new m and S'(t); lazy rescale of O and l.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        if fx.const_expr(DQ == 128):
            row_sum = row_sum + (lane_sum + other_sum)
        ballot, new_max = _next_max(
            _maximum(lane_max, other_max), scale, row_max, row_sum
        )
        s_cur = _center_part(s_cur, scale, new_max, 0, 32)
        _schedule(DV // 8, 4, 4)
        if fx.const_expr(DQ == 192):
            # The other group's next compute stage waits here; the cold-path skips (taken jumps) follow it.
            stage_end()
        o0, o1, row_sum, s_cur, exact = _rescale(
            o0, o1, row_sum, s_cur, row_max, new_max, ballot, exact
        )
        new_max = _advance_max(row_max, new_max)
        if fx.const_expr(DQ == 128):
            stage_end()
        return (
            s_cur,
            new_max,
            row_sum,
            o0,
            o1,
            page_next0,
            page_next1,
            page_next2_0,
            page_next2_1,
            exact,
        )

    def epilogue(state):
        """exp, row sum and PV of the last tile; V(last) already sits in slot last & 1."""
        s_last, row_max, row_sum, o0, o1 = state[:5]
        p_last = _exp_part(fx.Vector(s_last), 0, 32)
        row_sum = fx.Float32(row_sum) + _row_sum(p_last)
        p_bf16 = _pack_probability(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        # Memory: s2r V(last) dims 0-63; zero rows past kv_len (0 * NaN is NaN).
        v_lo = _read_v_tile_s2r(storage, v_base, last & 1, 0, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 0-63.
        o0 = _pv(p_bf16, v_lo, o0, DV)
        _stage_end()
        # Memory: s2r V(last) dims 64-127.
        v_hi = _read_v_tile_s2r(storage, v_base, last & 1, 1, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        # Compute: PV O dims 64-127.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait for waves 4-7 before the output shuffle reuses LDS.
            _stage_end()
        return row_max, row_sum, o0, o1

    # Pipeline: prologue stages K(0)/V(0), runs QK(0) and stages K(1); phases t = 1 .. tiles - 1;
    # epilogue finishes PV(last).
    state = prologue()
    if fx.const_expr(DQ == 128):
        # Two phases per trip keep the V ring slots compile-time constants.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            state = phase(*state, t, 1, 0)
            state = phase(*state, t + 1, 0, 1)
        if (tiles & 1) == 0:
            state = phase(*state, last, 1, 0)
    else:
        # D192 derives the slots from t; a single phase avoids doubling register pressure.
        for t in range(fx.Int32(1), tiles, fx.Int32(1)):
            state = phase(*state, t, 0, 0)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + (
        (fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV
    )
    obuf = rocdl.make_buffer_tensor(
        fx.make_view(optr, fx.make_layout(BM * H * DV, 1)),
        num_records_bytes=valid * H * DV * 2,
    )
    _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, H, DV)


@flyc.jit
def _body_wide(
    Q,
    K,
    V,
    O,
    LSE,
    QS,
    KS,
    VS,
    table,
    storage,
    q0,
    q_len,
    kv_len,
    head,
    qb,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NP: fx.Int32,
    DQ: fx.Constexpr[int],
    DV: fx.Constexpr[int],
    QROW: fx.Constexpr[int],
    QHEAD: fx.Constexpr[int],
    PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    STAGGER: fx.Constexpr[bool],
):
    """LSE or DV192: query rows 256 * qb .. + 255 of one head over KV tiles t, cooperative copies.

    LDS: one K slot; V uses a two-slot ring for D128/V128 (V(t) in slot t & 1) and one
    slot otherwise. Page ids run two tiles ahead and K one tile ahead (g2r in S0; r2s
    into the single slot in S4 for D128/V128, in S6 when D192/V192 QK streams K64
    chunks through S3). V(t) is loaded in S0; the ring publishes it in S2, the single
    slot in S0 of phase t + 1. Same stage roles and softmax as _body_v128, except the
    row sum/max reductions use lane shuffles inside compute stages.

    stage  phase t; ring: cur = t & 1, prev = (t - 1) & 1
    S0     s2r K(t) keys 0-31; single slot: r2s V(t-1); s_load pages(t+2); g2r V(t), K(t+1)
    S1     QK: S(t) keys 0-31          softmax: P(t-1) values 0-23
    S2     s2r K(t) keys 32-63; ring: V(t) lands, r2s V(t) -> [cur]
    S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, l += row sum, P to BF16
    S4     s2r V(t-1) dims 0-63; K(t+1) lands; D128/V128: r2s K(t+1)
    S5     PV: O dims 0-63 += P V      softmax: mask S(t), new m, S'(t) values 0..split
    S6     s2r V(t-1) dims 64-DV; D192/V192: r2s K(t+1)
    S7     PV: rest of O += P V        softmax: S'(t) values split..31; rescale O and l
    """
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (
        _read_k_tile_g2r,
        _read_v_tile_g2r,
        _store_tile_r2s,
        _read_k_tile_s2r,
        _read_v_tile_s2r,
        _qk,
        _pv,
        _mask,
        _v_tail,
        _join,
        _mfma_hazard_wait,
        _maximum,
        _row_max,
        _row_sum,
        _center_part,
        _exp_part,
        _pack_probability,
        _rescale,
        _pin,
        _schedule,
        _stage_end,
        _wait,
        _reround_if,
        _first_max,
        _intrinsic,
        _next_max,
        _k_g2r_bases,
        _global_words,
        _read_k_g2r,
    )
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 32 + (lane & 31)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    v_ring = DV == 128 and DQ == 128
    q_ptr = fx.get_iter(Q) + (
        (fx.Int64(q0) + fx.Int64(q_start)) * QROW + fx.Int64(head) * QHEAD
    )
    # Rows past `valid` read zero: the host keeps QROW >= DQ.
    gq = _buffer(fx.make_view(q_ptr, fx.make_layout(BM * QROW, 1)), valid * QROW * 2)
    gk, k_global_base = None, None
    if fx.const_expr(PAGE == 32):
        gk = _buffer(
            fx.make_view(
                fx.get_iter(K) + hkv * PAGE * DQ,
                fx.make_layout((NP * HK - hkv) * PAGE * DQ, 1),
            ),
            (NP * HK - hkv) * PAGE * DQ * 2,
        )
    else:
        k_global_base = _pin_s64(
            fx.Int64(fx.ptrtoint(fx.get_iter(K))) + fx.Int64(hkv) * (PAGE * DQ * 2)
        )
    # Q stays in registers for every KV tile.
    q = _read_q_g2r(gq, (wave * 32 + (lane & 31)) * (QROW * 2), lane, DQ)
    scale = fx.Float32(KS[0]) * fx.Float32(SCALE * LOG2E)
    if fx.const_expr(PER_TOKEN):
        scale = (
            scale * rocdl.make_buffer_tensor(QS, max_size=False)[(q0 + row) * H + head]
        )
    else:
        scale = scale * QS[0]
    scale = fx.Float32(scale)
    tiles = (kv_len + BN - 1) // BN
    if fx.const_expr(CAUSAL):
        end = (q_start + valid + kv_len - q_len + BN - 1) // BN
        tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last, last_page = tiles - 1, (kv_len - 1) // PAGE
    pages_0 = _read_tile_pages_g2r(table, fx.Int32(0), last, last_page, PAGE)
    pages_1 = _read_tile_pages_g2r(table, fx.Int32(1), last, last_page, PAGE)
    pages_2 = _read_tile_pages_g2r(table, fx.Int32(2), last, last_page, PAGE)
    pages_0 = _page_ready(pages_0[0]), _page_ready(pages_0[1])
    pages_1 = _page_ready(pages_1[0]), _page_ready(pages_1[1])
    pages_2 = _page_ready(pages_2[0]), _page_ready(pages_2[1])
    k_row = (lane & 3) | ((lane & 4) << 1) | ((lane & 8) >> 1) | (lane & 16)
    k_base = (lane >> 5) * PADDED_CHUNK + ((k_row * 16) ^ ((k_row & 16) << 2))
    v_base = (lane >> 5) * (DV // 64) * PADDED_CHUNK + (lane & 31) * 16
    k_bytes = (DQ // 8) * PADDED_CHUNK

    def read_k_g2r(pages, tile, bases=None):
        if fx.const_expr(PAGE != 32 and bases is None):
            bases = _k_g2r_bases(k_global_base, pages[0], tile, HK, DQ, PAGE)
        return _read_k_tile_g2r(gk, bases, tid, pages[0], pages[1], tile, HK, DQ, PAGE)

    def read_v_g2r(pages, tile):
        return _read_v_tile_g2r(
            V, tid, pages[0], pages[1], tile, hkv, HK, DV, PAGE, True
        )

    def store_k_r2s(words):
        _store_tile_r2s(storage, tid, words, 0, DQ // 64, True, True)

    def store_v_r2s(words, slot=0):
        _store_tile_r2s(
            storage,
            tid,
            words,
            k_bytes + slot * (DV // 8) * PADDED_CHUNK,
            DV // 64,
            bounded=True,
        )

    def prologue():
        """K(0), V(0) -> LDS; QK S(0), m and S'(0); K(1) -> LDS."""
        k_0 = read_k_g2r(pages_0, fx.Int32(0))
        v_0 = read_v_g2r(pages_0, fx.Int32(0))
        _wait(vmcnt=0)
        store_k_r2s(k_0)
        store_v_r2s(v_0)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: s2r K(0) keys 0-31 (D192/V192: first K64 chunk).
        k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV)
        o0 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        o1 = _pin(fx.Vector.filled(DV // 4, 0.0, fx.Float32))
        _schedule(DQ // 8, 3, 5)
        _stage_end()
        # Memory: s2r K(0) keys 32-63.
        k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV)
        _wait(lgkmcnt=0)
        _stage_end()
        # Compute: QK S(0) keys 32-63; softmax m and S'(0).
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV)
        s_0 = _mask(_join(s_lo, s_hi), fx.Int32(0), row, q_len, kv_len, CAUSAL)
        _mfma_hazard_wait()
        row_max = _first_max(s_0, scale)
        s_0 = _center_part(s_0, scale, row_max, 0, 32)
        # The exact rule of _rescale for S'(0); exact carries it through the phases.
        exact = fx.Int64(
            rocdl.ballot(fx.Int64.ir_type, (fx.absf(row_max) >= 2.0**22).ir_value())
        )
        s_0 = _reround_if(s_0, row_max, exact)
        row_sum = fx.Float32(0.0)
        _stage_end()
        if fx.const_expr(DQ == 192 or DV == 192):
            # Streaming QK reads K through the end of QK-high: both groups must finish first.
            _stage_end()
        # Memory: g2r K(1) and r2s it into the single K slot.
        k_1 = read_k_g2r(pages_1, _min(fx.Int32(1), last))
        _wait(vmcnt=0)
        store_k_r2s(k_1)
        _wait(lgkmcnt=0)
        _stage_end()
        # The leading group waits one more stage for the trailing group's K(1) writes.
        _stage_end()
        return s_0, row_max, row_sum, o0, o1, v_0, *pages_1, *pages_2, exact

    @flyc.jit
    def phase(
        s_prev,
        row_max,
        row_sum,
        o0,
        o1,
        v_pending,
        page_cur0,
        page_cur1,
        page_next0,
        page_next1,
        exact,
        t,
        CUR: fx.Constexpr[int],
        PREV: fx.Constexpr[int],
    ):
        """Phase t (table in _body_wide); v_pending is the staged V(t-1) of the single-slot path.

        exact carries the exact rule of _rescale.
        """
        s_prev, row_max, row_sum = (
            fx.Vector(s_prev),
            fx.Float32(row_max),
            fx.Float32(row_sum),
        )
        o0, o1, v_pending, t = (
            fx.Vector(o0),
            fx.Vector(o1),
            fx.Vector(v_pending),
            fx.Int32(t),
        )
        exact = fx.Int64(exact)
        # S0 memory: s2r K(t) keys 0-31; single slot publishes V(t-1); pages(t+2); g2r V(t), K(t+1).
        k_lo = _read_k_tile_s2r(storage, k_base, 0, DQ, DV)
        k_bases = None
        if fx.const_expr(PAGE != 32):
            # The SALU of K(t+1)'s 64-bit page bases issues among the K(t) LDS reads.
            k_bases = _k_g2r_bases(
                k_global_base, page_next0, _min(t + 1, last), HK, DQ, PAGE
            )
        rocdl.sched_barrier(0)
        if fx.const_expr(not v_ring):
            store_v_r2s(v_pending)
        requests_next2 = _read_tile_pages_g2r(table, t + 2, last, last_page, PAGE)
        v_cur = read_v_g2r((page_cur0, page_cur1), t)
        k_next = read_k_g2r((page_next0, page_next1), _min(t + 1, last), k_bases)
        rocdl.sched_barrier(0)
        page_next2_0, page_next2_1 = _page_ready(requests_next2[0]), _page_ready(
            requests_next2[1]
        )
        _stage_end()
        # S1 compute: QK S(t) keys 0-31; P(t-1) values 0-23.
        s_lo = _qk(q, k_lo, storage, k_base, 0, DQ, DV)
        p_prev = _pin(_exp_part(s_prev, 0, 24), 32)
        _schedule(DQ // 8, 2, 1, True)
        _stage_end()
        # S2 memory: s2r K(t) keys 32-63; the ring publishes V(t) -> [cur] once V lands.
        k_hi = _read_k_tile_s2r(storage, k_base, 1, DQ, DV)
        if fx.const_expr(v_ring):
            _wait(vmcnt=DQ // 64)
            store_v_r2s(v_cur, CUR)
        _wait(lgkmcnt=0)
        _stage_end()
        # S3 compute: QK S(t) keys 32-63; P(t-1) values 24-31, l += row sum, BF16 P.
        s_hi = _qk(q, k_hi, storage, k_base, 1, DQ, DV)
        p_prev = _exp_part(p_prev, 24, 32)
        row_sum = row_sum + _row_sum(p_prev)
        p_bf16 = _pack_probability(p_prev)
        _schedule(4, 2, 2, True)
        _schedule(DQ // 8 - 4, 8, 2)
        _stage_end()
        # S4 memory: s2r V(t-1) dims 0-63; D128/V128 publishes K(t+1) (both groups read K(t)).
        v_lo = _read_v_tile_s2r(storage, v_base, PREV if v_ring else 0, 0, DQ, DV)
        _wait(vmcnt=0)
        if fx.const_expr(DQ == 128 and DV == 128):
            store_k_r2s(k_next)
        _wait(lgkmcnt=0)
        _stage_end()
        # S5 compute: PV O dims 0-63; mask S(t), lazy new m and S'(t) values 0..split.
        s_cur = _mask(_join(s_lo, s_hi), t, row, q_len, kv_len, CAUSAL)
        o0 = _pv(p_bf16, v_lo, o0, DV)
        ballot, new_max = _next_max(_row_max(s_cur), scale, row_max, row_sum)
        split = 12 if STAGGER else 6
        s_cur = _center_part(s_cur, scale, new_max, 0, split)
        _schedule(3, 5, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 3)
        rocdl.sched_group_barrier(0x002, 3, 3)
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, 3)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 3)
        rocdl.sched_group_barrier(0x002, 10, 3)
        _schedule(DV // 8 - 6, 5, 3)
        _stage_end()
        # S6 memory: s2r V(t-1) dims 64-DV. D192/V192 publish K(t+1) only now: the partner
        # group's S3 streams K until the end of QK-high, two stages before the next K read.
        v_hi = _read_v_tile_s2r(storage, v_base, PREV if v_ring else 0, 1, DQ, DV)
        if fx.const_expr(DQ == 192 or DV == 192):
            store_k_r2s(k_next)
        _wait(lgkmcnt=0)
        _stage_end()
        # S7 compute: rest of PV; S'(t) values split..31; lazy rescale of O and l.
        o1 = _pv(p_bf16, v_hi, o1, DV)
        s_cur = _center_part(s_cur, scale, new_max, split, 32)
        _schedule(DV // 8, 4, 4)
        o0, o1, row_sum, s_cur, exact = _rescale(
            o0, o1, row_sum, s_cur, row_max, new_max, ballot, exact
        )
        _stage_end()
        v_pending = fx.Vector.filled(DV // 16, 0, fx.Int32) if v_ring else v_cur
        return (
            s_cur,
            new_max,
            row_sum,
            o0,
            o1,
            v_pending,
            page_next0,
            page_next1,
            page_next2_0,
            page_next2_1,
            exact,
        )

    def epilogue(state):
        """Single slot: r2s V(last). exp, row sum and PV of the last tile."""
        s_last, row_max, row_sum, o0, o1, v_pending = state[:6]
        if fx.const_expr(not v_ring):
            # Publish V(last) after both groups' S6 readers of V(last - 1).
            store_v_r2s(v_pending)
        p_last = _exp_part(fx.Vector(s_last), 0, 32)
        row_sum = fx.Float32(row_sum) + _row_sum(p_last)
        p_bf16 = _pack_probability(p_last)
        _wait(vmcnt=0, lgkmcnt=0)
        _stage_end()
        if fx.const_expr(not v_ring):
            # Wait for the trailing group's V(last) publication.
            _stage_end()
        # Memory: s2r V(last) dims 0-63; zero rows past kv_len.
        v_lo = _read_v_tile_s2r(storage, v_base, (last & 1) if v_ring else 0, 0, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_lo.store(_v_tail(v_lo.load(), kv_len - last * BN))
        _stage_end()
        o0 = _pv(p_bf16, v_lo, o0, DV)
        _stage_end()
        # Memory: s2r V(last) dims 64-DV.
        v_hi = _read_v_tile_s2r(storage, v_base, (last & 1) if v_ring else 0, 1, DQ, DV)
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        v_hi.store(_v_tail(v_hi.load(), kv_len - last * BN))
        _stage_end()
        o1 = _pv(p_bf16, v_hi, o1, DV)
        _stage_end()
        if fx.const_expr(not STAGGER):
            _stage_end()
        return row_max, row_sum, o0, o1

    state = prologue()
    if fx.const_expr(v_ring):
        # Two phases per trip keep the V ring slots compile-time constants.
        for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
            state = phase(*state, t, 1, 0)
            state = phase(*state, t + 1, 0, 1)
        if (tiles & 1) == 0:
            state = phase(*state, last, 1, 0)
    else:
        for t in range(fx.Int32(1), tiles, fx.Int32(1)):
            state = phase(*state, t, 0, 0)
    row_max, row_sum, o0, o1 = epilogue(state)

    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0)) * VS[0]
    out_tid = _pin_i32(fx.Int32(gpu.thread_id("x")))
    out_row = (out_tid >> 6) * 32 + (out_tid & 31)
    optr = fx.get_iter(O) + (
        (fx.Int64(q0) + fx.Int64(q_start)) * (H * DV) + fx.Int64(head) * DV
    )
    obuf = rocdl.make_buffer_tensor(
        fx.make_view(optr, fx.make_layout(BM * H * DV, 1)),
        num_records_bytes=valid * H * DV * 2,
    )
    _store_o_tile_r2g(o0, o1, inv, storage, obuf, out_tid, out_row, H, DV)
    if fx.const_expr(WITH_LSE):
        if ((out_tid & 63) < 32) & (out_row < valid):
            log_l = fx.Float32(
                llvm.call_intrinsic(
                    fx.Float32.ir_type, "llvm.log2.f32", [row_sum.ir_value()], [], []
                )
            )
            LSE[(q0 + q_start + out_row) * H + head] = (row_sum > 0.0).select(
                (fx.Float32(row_max) + log_l) * fx.Float32(math.log(2.0)),
                fx.Float32(float("-inf")),
            )


@flyc.jit
def _work(
    Q,
    K,
    V,
    O,
    LSE,
    CQ,
    KI,
    PAGES,
    LAST,
    QS,
    KS,
    VS,
    storage,
    head,
    batch,
    qb,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NP: fx.Int32,
    DQ: fx.Constexpr[int],
    DV: fx.Constexpr[int],
    QROW: fx.Constexpr[int],
    QHEAD: fx.Constexpr[int],
    PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
):
    # AST rewriting lifts branch bodies into local functions. Capture the
    # compute dependency before the branch so the native cache follows it.
    body = _body_wide
    if fx.const_expr(DV == 128 and not WITH_LSE):
        body = _body_v128
    q0 = _uniform(CQ[batch])
    q_len = _uniform(CQ[batch + 1]) - q0
    start = _uniform(KI[batch])
    pages = _uniform(KI[batch + 1]) - start
    kv_len = (pages - 1) * PAGE + _uniform(LAST[batch])
    table = fx.make_view(fx.get_iter(PAGES) + start, fx.make_layout(pages, 1))
    if qb * BM < q_len:
        group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
        if group != 0:
            body(
                Q,
                K,
                V,
                O,
                LSE,
                QS,
                KS,
                VS,
                table,
                storage,
                q0,
                q_len,
                kv_len,
                head,
                qb,
                H,
                HK,
                NP,
                DQ,
                DV,
                QROW,
                QHEAD,
                PAGE,
                CAUSAL,
                PER_TOKEN,
                WITH_LSE,
                SCALE,
                True,
            )
        else:
            body(
                Q,
                K,
                V,
                O,
                LSE,
                QS,
                KS,
                VS,
                table,
                storage,
                q0,
                q_len,
                kv_len,
                head,
                qb,
                H,
                HK,
                NP,
                DQ,
                DV,
                QROW,
                QHEAD,
                PAGE,
                CAUSAL,
                PER_TOKEN,
                WITH_LSE,
                SCALE,
                False,
            )


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _attention_kernel_942(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    CQ: fx.Tensor,
    KI: fx.Tensor,
    PAGES: fx.Tensor,
    LAST: fx.Tensor,
    QS: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NP: fx.Int32,
    B: fx.Int32,
    BLOCKS: fx.Int32,
    DQ: fx.Constexpr[int],
    DV: fx.Constexpr[int],
    QROW: fx.Constexpr[int],
    QHEAD: fx.Constexpr[int],
    PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool],
    CUS: fx.Constexpr[int],
):
    size = (
        (DQ + (2 if DV == 128 and (DQ == 128 or not WITH_LSE) else 1) * DV) // 8
    ) * PADDED_CHUNK
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, size, 16])
        .peek()
        .view(fx.make_layout(size, 1))
    )
    if fx.const_expr(PERSISTENT):
        # A fixed resident grid revisits the SAME FP8-derived body. Grid-stride
        # work assignment needs no shared counter, allocation, or reset kernel;
        # independent streams and captured graphs have no mutable shared state.
        # Query blocks run longest first (causal cost grows with qb) and every other
        # round of CUS tasks is reversed (snake), which evens out the per-CTA totals.
        tasks = H * B * BLOCKS
        worker, rnd = fx.Int32(gpu.block_id("x")), fx.Int32(0)
        while rnd < (tasks + CUS - 1) // CUS:
            task = rnd * CUS + ((rnd & 1) == 0).select(worker, CUS - 1 - worker)
            if task < tasks:
                pair = task // H
                head, batch, qb = task % H, pair % B, BLOCKS - 1 - pair // B
                _work(
                    Q,
                    K,
                    V,
                    O,
                    LSE,
                    CQ,
                    KI,
                    PAGES,
                    LAST,
                    QS,
                    KS,
                    VS,
                    storage,
                    head,
                    batch,
                    qb,
                    H,
                    HK,
                    NP,
                    DQ,
                    DV,
                    QROW,
                    QHEAD,
                    PAGE,
                    CAUSAL,
                    PER_TOKEN,
                    WITH_LSE,
                    SCALE,
                )
            rnd = rnd + 1
    else:
        # Blocks dispatch with z slowest: the longest causal query blocks start first.
        head, batch = fx.Int32(gpu.block_id("x")), fx.Int32(gpu.block_id("y"))
        qb = BLOCKS - 1 - fx.Int32(gpu.block_id("z"))
        _work(
            Q,
            K,
            V,
            O,
            LSE,
            CQ,
            KI,
            PAGES,
            LAST,
            QS,
            KS,
            VS,
            storage,
            head,
            batch,
            qb,
            H,
            HK,
            NP,
            DQ,
            DV,
            QROW,
            QHEAD,
            PAGE,
            CAUSAL,
            PER_TOKEN,
            WITH_LSE,
            SCALE,
        )


@flyc.jit
def _launch_attention(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    CQ: fx.Tensor,
    KI: fx.Tensor,
    PAGES: fx.Tensor,
    LAST: fx.Tensor,
    QS: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NP: fx.Int32,
    B: fx.Int32,
    BLOCKS: fx.Int32,
    DQ: fx.Constexpr[int],
    DV: fx.Constexpr[int],
    QROW: fx.Constexpr[int],
    QHEAD: fx.Constexpr[int],
    PAGE: fx.Constexpr[int],
    CAUSAL: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    WITH_LSE: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    PERSISTENT: fx.Constexpr[bool],
    CUS: fx.Constexpr[int],
    stream: fx.Stream,
):
    tasks = H * B * BLOCKS
    grid = ((tasks < CUS).select(tasks, CUS), 1, 1) if PERSISTENT else (H, B, BLOCKS)
    _attention_kernel_942(
        Q,
        K,
        V,
        O,
        LSE,
        CQ,
        KI,
        PAGES,
        LAST,
        QS,
        KS,
        VS,
        H,
        HK,
        NP,
        B,
        BLOCKS,
        DQ,
        DV,
        QROW,
        QHEAD,
        PAGE,
        CAUSAL,
        PER_TOKEN,
        WITH_LSE,
        SCALE,
        PERSISTENT,
        CUS,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "llvm.target_features": ir.Attribute.parse(
                '#llvm.target_features<["-packed-fp32-ops"]>'
            ),
        },
    ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)


class _PagedAttention:
    bf16_backend = "native-8wave-8stage"

    def __init__(self, heads, kv_heads, dq, dv, page, causal, mode, persistent):
        self.heads, self.kv_heads, self.dq, self.dv = heads, kv_heads, dq, dv
        self.page_size, self.causal, self.quant_query_mode = page, causal, mode
        self.memory_mode, self.persistent = "lds", persistent
        self._launch = _launch_attention
        self._compiled, self._row_strides = {}, {}

    def __call__(
        self,
        Q,
        K,
        V,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        q_descale,
        k_descale,
        v_descale,
        kv_last_page_lens,
        out=None,
        sink_ptr=None,
        stream=None,
        *,
        return_lse=False,
        lse=None,
        softmax_scale=None,
    ):
        """Run with contiguous SHUFFLE-5D KV and asynchronous device metadata.

        Q may be strided over tokens and heads (unit-stride Dq, token rows
        at least Dq apart), e.g. a slice of a fused QKV projection. Batch, query
        bound and KV pool size are runtime arguments; each Q stride pair
        compiles its own (one-token calls reuse a compiled row stride).
        Descales are finite positive FP32 and the scale product
        softmax_scale * log2(e) * q_descale * k_descale must stay a positive
        finite FP32. K/V within each KV length must be finite: rows of a causal
        tile share its V, so a non-finite value also reaches rows that mask its
        key. Metadata must be consistent/in-bounds; no GPU metadata is copied to
        the host. Active KV must be nonempty and causal KV >= Q. Caller output/LSE
        must not overlap inputs or each other. Empty Q returns without dispatch.
        Warm before graph capture.
        """
        if (
            not isinstance(causal, bool)
            or causal != self.causal
            or not isinstance(return_lse, bool)
        ):
            raise ValueError("causal must match the factory; return_lse must be bool")
        if sink_ptr is not None:
            raise NotImplementedError("gfx942 BF16 full MHA does not support sinks")
        if not isinstance(Q, torch.Tensor) or not Q.is_cuda:
            raise ValueError("Q/K/V must be tensors on the same gfx942 GPU")
        device = Q.device
        properties = torch.cuda.get_device_properties(device)
        if getattr(properties, "gcnArchName", "").split(":", 1)[0] != "gfx942":
            raise NotImplementedError("this BF16 backend requires gfx942")
        for name, tensor in (("Q", Q), ("K", K), ("V", V)):
            if not isinstance(tensor, torch.Tensor) or tensor.device != device:
                raise ValueError(f"{name} must be a tensor on the input GPU")
            if tensor.dtype != torch.bfloat16:
                raise NotImplementedError("gfx942 BF16 attention requires BF16 Q/K/V")
            if tensor.layout != torch.strided or not (
                tensor.is_contiguous() or name == "Q"
            ):
                raise ValueError(f"{name} must be contiguous")
        if Q.ndim != 3 or Q.shape[1:] != (self.heads, self.dq):
            raise ValueError("Q must be [tokens, query heads, Dq]")
        tokens = Q.shape[0]
        # A size-1 token axis may carry any stride: keep a valid one.
        q_row = (
            Q.stride(0)
            if tokens > 1 or Q.stride(0) >= self.dq
            else self.heads * self.dq
        )
        q_head = Q.stride(1) if self.heads > 1 else self.dq
        if Q.stride(2) != 1 or q_row < self.dq:
            raise ValueError(
                "Q needs a unit-stride Dq and token rows at least Dq apart"
            )
        q_span = (
            (tokens - 1) * q_row + (self.heads - 1) * q_head + self.dq if tokens else 0
        )
        # A query block's lane offsets and Q buffer bound reach BM rows. Dq/Dv 128/192 read V and
        # page64/page128 K from 64-bit page bases; page32 K uses 32-bit pool offsets.
        spans = [("Q", max(q_span, BM * q_row))]
        if self.page_size == 32:
            spans.append(("K", K.numel()))
        for name, span in spans:
            if span * 2 >= 2**31:
                raise NotImplementedError(
                    f"{name} byte span must fit signed int32 addressing"
                )
        if (
            K.ndim != 5
            or V.ndim != 5
            or K.shape != (V.shape[0], self.kv_heads, self.dq // 8, self.page_size, 8)
            or V.shape[1:] != (self.kv_heads, self.page_size // 8, self.dv, 8)
        ):
            raise ValueError("K/V must use the factory's BF16 SHUFFLE-5D layouts")
        for name, bound in (
            ("max_seqlen_q", max_seqlen_q),
            ("max_seqlen_k", max_seqlen_k),
        ):
            if (
                not isinstance(bound, int)
                or isinstance(bound, bool)
                or not 0 <= bound < 2**31
            ):
                raise ValueError(f"{name} must be a nonnegative signed int32 bound")
        if causal and max_seqlen_k < max_seqlen_q:
            raise ValueError("bottom-right causal attention requires KV >= Q")
        if tokens and max_seqlen_q == 0:
            raise ValueError("nonempty Q requires max_seqlen_q > 0")
        metadata = (
            ("cu_seqlens_q", cu_seqlens_q),
            ("kv_indptr", kv_indptr),
            ("kv_page_indices", kv_page_indices),
            ("kv_last_page_lens", kv_last_page_lens),
        )
        if cu_seqlens_k is not None:
            metadata += (("cu_seqlens_k", cu_seqlens_k),)
        for name, tensor in metadata:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 1
                or tensor.dtype != torch.int32
                or tensor.device != device
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous device int32 metadata")
        batch = cu_seqlens_q.numel() - 1
        if (
            batch < 0
            or tokens > 0
            and batch == 0
            or kv_indptr.shape != cu_seqlens_q.shape
            or kv_last_page_lens.numel() != batch
            or cu_seqlens_k is not None
            and cu_seqlens_k.shape != cu_seqlens_q.shape
        ):
            raise ValueError("inconsistent batch metadata shapes")
        if tokens and (
            max_seqlen_k == 0 or K.shape[0] == 0 or kv_page_indices.numel() == 0
        ):
            raise NotImplementedError("active sequences require nonempty KV")
        blocks = (max_seqlen_q + BM - 1) // BM
        if self.heads * batch * blocks >= 2**31:
            raise NotImplementedError(
                "heads * batch * query blocks must fit signed int32"
            )
        for name, tensor in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.device != device
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous FP32 on the input GPU")
        if k_descale.numel() != 1 or v_descale.numel() != 1:
            raise ValueError("K/V descales must each contain one value")
        if q_descale.numel() != 1 and (
            self.quant_query_mode != "per-token"
            or q_descale.numel() != tokens * self.heads
        ):
            raise ValueError("Q descale must be scalar or one value per token/head")
        scale = _softmax_scale(softmax_scale, self.dq**-0.5)
        out_shape, lse_shape = (tokens, self.heads, self.dv), (tokens, self.heads)
        if tokens * self.heads * self.dv * 2 >= 2**31:
            raise NotImplementedError(
                "output byte span must fit signed int32 addressing"
            )
        for name, tensor, shape, dtype in (
            ("out", out, out_shape, torch.bfloat16),
            ("lse", lse, lse_shape, torch.float32),
        ):
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != device
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} {shape} on the input GPU"
                )
        stream = torch.cuda.current_stream(device) if stream is None else stream
        if getattr(stream, "device", None) != device or not hasattr(
            stream, "cuda_stream"
        ):
            raise ValueError("stream must belong to the input GPU")
        with torch.cuda.device(device), torch.cuda.stream(stream):
            if out is None:
                out = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
            if return_lse and lse is None:
                lse = torch.empty(lse_shape, device=device, dtype=torch.float32)
            if tokens:
                # Dq/Dv 128/192 take K/V as [pages, page elements]: FlyDSL packs dynamic shapes as
                # int32, which a flat view of a pool past 2**31 elements would overflow.
                kv = (K.flatten(1), V.flatten(1))
                tensors = (
                    Q.as_strided((q_span,), (1,)),
                    *kv,
                    out.view(-1),
                    lse.view(-1) if lse is not None else k_descale.view(-1),
                    cu_seqlens_q,
                    kv_indptr,
                    kv_page_indices,
                    kv_last_page_lens,
                    q_descale.view(-1),
                    k_descale.view(-1),
                    v_descale.view(-1),
                )
                flags = (
                    self.page_size,
                    self.causal,
                    q_descale.numel() != 1,
                    lse is not None,
                    scale,
                    self.persistent,
                    properties.multi_processor_count,
                )
                known = self._row_strides.setdefault((device, q_head, flags), q_row)
                # A single token never reads past its row: any compiled row stride (all >= Dq) serves it.
                q_row = known if tokens == 1 else q_row
                consts = (
                    self.heads,
                    self.kv_heads,
                    self.dq,
                    self.dv,
                    q_row,
                    q_head,
                    *flags,
                )
                scalars = (
                    self.heads,
                    self.kv_heads,
                    K.shape[0],
                    batch,
                    blocks,
                    self.dq,
                    self.dv,
                    q_row,
                    q_head,
                    *flags,
                )
                args = (*tensors, *scalars, stream)
                # Shapes, strides and runtime scalars are kernel arguments; only these select code.
                key = (device, tuple(t.dtype for t in tensors), consts)
                compiled = self._compiled.get(key)
                if compiled is None:
                    # flyc.compile also launches this first call.
                    self._compiled[key] = flyc.compile(self._launch, *args)
                else:
                    compiled(*args)
        return (out, lse) if return_lse else out


@functools.lru_cache(maxsize=None, typed=True)
def PagedAttention(
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
    is_causal,
    quant_query_mode="per-token",
    key_layout="vectorized",
    window_left=-1,
    has_sink=False,
    *,
    memory_mode="lds",
    persistent=None,
):
    """FP8-derived BF16 body; persistent=None/True uses a fixed resident grid.

    persistent=False selects FP8's ordinary head/batch/query-tile grid. Both
    paths use the same eight-stage kernel and require no scheduler workspace.
    Dq/Dv128/192 use BM256. Only the paged D128/D192 path is vendored here.
    """
    if memory_mode != "lds" or key_layout != "vectorized":
        raise NotImplementedError("gfx942 BF16 supports LDS and vectorized K only")
    if persistent is not None and not isinstance(persistent, bool):
        raise ValueError("persistent must be bool or None")
    if not isinstance(is_causal, bool) or not isinstance(has_sink, bool):
        raise ValueError("is_causal and has_sink must be bools")
    if window_left != -1 or has_sink:
        raise NotImplementedError("gfx942 BF16 full MHA does not support SWA or sinks")
    if any(
        not isinstance(x, int) or isinstance(x, bool)
        for x in (num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size)
    ):
        raise ValueError("head counts, dimensions and page_size must be integers")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    supported_dims = head_dim_qk in (128, 192) and head_dim_v in (128, 192)
    if not supported_dims or page_size not in (32, 64, 128):
        raise NotImplementedError(
            "gfx942 BF16 supports Dq/Dv128/192 and pages32/64/128"
        )
    if quant_query_mode not in ("per-token", "per-tensor"):
        raise ValueError("query scale mode must be per-token or per-tensor")
    return _PagedAttention(
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        page_size,
        is_causal,
        quant_query_mode,
        persistent is not False,
    )
