# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small standalone CuTe DSL and PTX primitives shared by Kernel backends."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op

WARP_SIZE = 32
VEC_BF16 = 8
QUAD_BF16 = 4
NEGATIVE_ZERO_BF16_BITS = 0x8000
NEGATIVE_ZERO_BF16_PAIR = 0x80008000
# CUTLASS cute::TMA::CacheHintSm100::EVICT_FIRST policy descriptor.
L2_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    address = pointer.toint(loc=loc, ip=ip)
    if volatile:
        opcode = "ld.volatile.global.v4.u32"
    else:
        opcode = "ld.global.v4.u32"
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=volatile,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def load_global_u32x4_predicated(
    pointer: cute.Pointer,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            address.ir_value(loc=loc, ip=ip),
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            "@p ld.global.v4.u32 {$0, $1, $2, $3}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def load_global_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_u32_predicated(
    pointer: cute.Pointer,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                address.ir_value(loc=loc, ip=ip),
                Int32(predicate).ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.s32 p, $2, 0;\n\t"
                "@!p mov.u32 $0, 0;\n\t"
                "@p ld.global.u32 $0, [$1];\n\t"
                "}"
            ),
            "=r,l,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_u32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(2)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def load_global_u32x2_predicated(
    pointer: cute.Pointer,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [
            address.ir_value(loc=loc, ip=ip),
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $3, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@p ld.global.v2.u32 {$0, $1}, [$2];\n\t"
            "}"
        ),
        "=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(2)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def store_global_u32x4(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32_address(
    address: Int64,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32x2(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u16_bits(
    address: Int64,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .b16 bits;\n\t"
            "cvt.u16.u32 bits, $1;\n\t"
            "st.global.u16 [$0], bits;\n\t"
            "}"
        ),
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_u32x4(
    address: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), sentinel, sentinel, sentinel, sentinel],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_bf16_as_f32(
    address: Int64,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            (
                "{\n\t"
                ".reg .b16 bits;\n\t"
                "ld.global.b16 bits, [$1];\n\t"
                "cvt.f32.bf16 $0, bits;\n\t"
                "}"
            ),
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_bf16_as_f32_predicated(
    address: Int64,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                address.ir_value(loc=loc, ip=ip),
                Int32(predicate).ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n\t"
                ".reg .pred p;\n\t"
                ".reg .b16 bits;\n\t"
                "setp.ne.s32 p, $2, 0;\n\t"
                "@!p mov.b16 bits, 0;\n\t"
                "@p ld.global.b16 bits, [$1];\n\t"
                "cvt.f32.bf16 $0, bits;\n\t"
                "}"
            ),
            "=f,l,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            (
                "{\n\t"
                ".reg .b16 bits;\n\t"
                "cvt.rn.bf16.f32 bits, $1;\n\t"
                "cvt.u32.u16 $0, bits;\n\t"
                "}"
            ),
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shuffle_sync_idx_u32(
    value: Uint32,
    source_lane: Int32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                value.ir_value(loc=loc, ip=ip),
                source_lane.ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            # Preserve full-warp execution across later divergent consumers.
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_volatile_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([VEC_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, BFloat16)


@dsl_user_op
def packed_u32_to_bf16x2(packed: Uint32, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([2], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, 2, BFloat16)


@dsl_user_op
def packed_u32x2_to_bf16x4(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([QUAD_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, QUAD_BF16, BFloat16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([4], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def bf16x2_to_packed_u32(values, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.bitcast(
            T.i32(),
            values.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x4_to_packed_u32x2(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([2], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@cute.jit
def sanitize_negative_zero_u32x4(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for index in cutlass.range_constexpr(4):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def sanitize_negative_zero_u32(word: Uint32) -> Uint32:
    low = Uint16(word & Uint32(0xFFFF))
    high = Uint16(word >> Uint32(16))
    if low == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0xFFFF0000)
    if high == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def sanitize_negative_zero_u32x2(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
    for index in cutlass.range_constexpr(2):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def fragment_has_negative_zero(packed):
    dirty = False
    for index in cutlass.range_constexpr(4):
        word = packed[index]
        dirty = (
            dirty
            | (Uint16(word & Uint32(0xFFFF)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
            | (Uint16(word >> Uint32(16)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
        )
    return dirty


@dsl_user_op
def map_shared_to_peer(
    smem_pointer: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    address = smem_pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [address, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_cluster_f32(
    remote_address: Int32,
    value: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            remote_address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_u32x4(pointer: cute.Pointer, *, loc=None, ip=None):
    address = pointer.toint(loc=loc, ip=ip)
    # Prevent motion across the named-barrier pipeline protocol.
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def store_shared_u32x4(
    pointer: cute.Pointer,
    packed,
    *,
    loc=None,
    ip=None,
) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip), *words],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_u32x4_address(
    address: Int64,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=volatile,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def packed_negative_zero_bf16x8(*, loc=None, ip=None):
    word = Uint32(NEGATIVE_ZERO_BF16_PAIR).ir_value(loc=loc, ip=ip)
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [word, word, word, word],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    barrier_ptr: cute.Pointer,
    size_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    operands = [
        gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
        smem_ptr.toint(loc=loc, ip=ip).ir_value(),
        barrier_ptr.toint(loc=loc, ip=ip).ir_value(),
        size_bytes.ir_value(loc=loc, ip=ip),
        Int64(L2_EVICT_FIRST).ir_value(),
    ]
    llvm.inline_asm(
        None,
        operands,
        (
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
            ".L2::cache_hint [$1], [$0], $3, [$2], $4;"
        ),
        "l,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_shared_cta(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def remote_release_add1_u32(address: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "red.release.sys.global.add.u32 [$0], 1;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldmc_bf16x8(address: Int64, *, loc=None, ip=None):
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def stmc_bf16x2(
    address: Int64,
    packed: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), packed.ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.bf16x2 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x4(address: Int64, values, *, loc=None, ip=None) -> None:
    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x8(address: Int64, values, *, loc=None, ip=None) -> None:
    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
