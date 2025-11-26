# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T

import sglang.srt.sparse_attention.kernels.attention.mma_sm100_desc as sm100_desc


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | cutlass.Boolean = False,
) -> None:
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        tiled_mma.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    zero_init: bool | cutlass.Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
        assert (
            sA_swizzle is not None
        ), "sA_swizzle must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else None
    sB_layout = sB.layout
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    if cutlass.const_expr(not is_ts):
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (
                    sm100_desc.Major.K
                    if cutlass.const_expr(
                        op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                    )
                    else sm100_desc.Major.MN
                ),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (
                sm100_desc.Major.K
                if cutlass.const_expr(
                    op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else sm100_desc.Major.MN
            ),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)

    if cutlass.const_expr(not is_ts):
        smem_desc_start_a_lo = cutlass.Int32(
            smem_desc_base_a_lo
        ) | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = cutlass.Int32(
        smem_desc_base_b_lo
    ) | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(not is_ts):
            smem_desc_a_lo = smem_desc_start_a_lo + (
                (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            )
        smem_desc_b_lo = smem_desc_start_b_lo + (
            (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        )
        # with cute.arch.elect_one():
        #     cute.printf("smem_desc_a_lo = {}, smem_desc_b_lo = {}", smem_desc_a_lo, smem_desc_b_lo)
        #     cute.printf("smem_desc_a_lo_correct = {}, smem_desc_b_lo_correct = {}", smem_desc_a_lo_correct, smem_desc_b_lo_correct)
        with cute.arch.elect_one():
            if cutlass.const_expr(not is_ts):
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        smem_desc_a_lo.ir_value(),
                        smem_desc_b_lo.ir_value(),
                        cutlass.Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
                    ".reg .b32 idesc;\n\t"
                    f"mov.b32 idesc, {hex(idesc)};\n\t"
                    f"mov.b64 smem_desc_a, {{$1, {hex(smem_desc_a_hi)}}};\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            else:
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        tCrA[None, None, k].iterator.toint().ir_value(),
                        smem_desc_b_lo.ir_value(),
                        cutlass.Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_b;\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], smem_desc_b, {hex(idesc)}, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )


@cute.jit
def gemm_ptx_loop(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    zero_init: bool | cutlass.Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
        assert (
            sA_swizzle is not None
        ), "sA_swizzle must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    if cutlass.const_expr(not is_ts):
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (
                    sm100_desc.Major.K
                    if cutlass.const_expr(
                        op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                    )
                    else sm100_desc.Major.MN
                ),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (
                sm100_desc.Major.K
                if cutlass.const_expr(
                    op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else sm100_desc.Major.MN
            ),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)

    if cutlass.const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1]
        for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        for k in cutlass.range_constexpr(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1]
        for k in cutlass.range_constexpr(1, cute.size(tCrB.shape[2]))
    ]

    if cutlass.const_expr(not is_ts):
        smem_desc_start_a_lo = cutlass.Int32(
            smem_desc_base_a_lo
            | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = cutlass.Int32(
        smem_desc_base_b_lo
        | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = (
        "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    )
    if cutlass.const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                cutlass.Int32(
                    cute.arch.make_warp_uniform(smem_desc_start_a_lo)
                ).ir_value(),
                cutlass.Int32(
                    cute.arch.make_warp_uniform(smem_desc_start_b_lo)
                ).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 smem_desc_a_lo, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                cutlass.Int32(smem_desc_start_b_lo).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_a, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[cutlass.Int32] = None,
    zero_init: bool | cutlass.Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
        assert (
            sA_swizzle is not None
        ), "sA_swizzle must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    if cutlass.const_expr(not is_ts):
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (
                    sm100_desc.Major.K
                    if cutlass.const_expr(
                        op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                    )
                    else sm100_desc.Major.MN
                ),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (
                sm100_desc.Major.K
                if cutlass.const_expr(
                    op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else sm100_desc.Major.MN
            ),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if cutlass.const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if cutlass.const_expr(not is_ts):
        smem_desc_start_a_lo = cutlass.Int32(
            smem_desc_base_a_lo
            | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = cutlass.Int32(
        smem_desc_base_b_lo
        | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = (
        "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    )
    if cutlass.const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                cutlass.Int32(smem_desc_start_a_lo).ir_value(),
                cutlass.Int32(smem_desc_start_b_lo).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            "mov.b32 smem_desc_a_lo, $0;\n\t"
            "mov.b32 smem_desc_b_lo, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        input_args = [
            cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            cutlass.Int32(smem_desc_start_b_lo).ir_value(),
            cutlass.Int32(not zero_init).ir_value(),
        ]
        if cutlass.const_expr(mbar_ptr is not None):
            assert (
                mbar_phase is not None
            ), "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(cutlass.Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$3], $4, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     cutlass.Int32(smem_desc_start_b_lo).ir_value(),
            #     cutlass.Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    (
                        cute.size(tCrA.shape[2])
                        if cutlass.const_expr(mbar_ptr is None)
                        else cute.size(tCrA.shape[2]) // 4 * 3
                    ),
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(
                        cute.size(tCrA.shape[2]) // 4 * 3, cute.size(tCrA.shape[2])
                    )
                )
                if cutlass.const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            # "r,r,r",
            "r,r,r" if cutlass.const_expr(mbar_ptr is None) else "r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA_base_addr_for_desc: cutlass.Int32,
    sA_addr_offset_for_desc: cutlass.Constexpr[int],
    sA_stage: cutlass.Int32,
    sB_base_addr_for_desc: cutlass.Int32,
    sB_addr_offset_for_desc: cutlass.Constexpr[int],
    sB_stage: cutlass.Int32,
    sA_layout: Optional[cute.Layout],
    sB_layout: Optional[cute.Layout],
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    zero_init: bool | cutlass.Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if cutlass.const_expr(not is_ts):
        assert (
            sA_layout is not None
        ), "sA_layout must be provided when a_src is not TMEM"
        assert (
            sA_swizzle is not None
        ), "sA_swizzle must be provided when a_src is not TMEM"
    idesc: int = cutlass.const_expr(sm100_desc.mma_op_to_idesc(op))
    if cutlass.const_expr(not is_ts):
        smem_desc_base_a: int = cutlass.const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (
                    sm100_desc.Major.K
                    if cutlass.const_expr(
                        op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                    )
                    else sm100_desc.Major.MN
                ),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = cutlass.const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = cutlass.const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (
                sm100_desc.Major.K
                if cutlass.const_expr(
                    op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else sm100_desc.Major.MN
            ),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = cutlass.const_expr(smem_desc_b_hi)
    mask = [cutlass.Int32(0)] * 4

    if cutlass.const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 8) >> 4
            for k in range(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in range(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * op.b_dtype.width // 8) >> 4
        for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if cutlass.const_expr(not is_ts):
        # smem_desc_start_a_lo = cutlass.Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
        smem_desc_start_a_lo = cutlass.const_expr(smem_desc_base_a_lo)
    else:
        smem_desc_start_a_lo = None
    # smem_desc_start_b_lo = cutlass.Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    smem_desc_start_b_lo = cutlass.const_expr(smem_desc_base_b_lo)
    pred_str = (
        "p" if isinstance(zero_init, cutlass.Boolean) else "0" if zero_init else "1"
    )
    if cutlass.const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                # cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                cutlass.Int32(sA_base_addr_for_desc).ir_value(),
                cutlass.Int32(sA_stage).ir_value(),
                # cutlass.Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                cutlass.Int32(sB_base_addr_for_desc).ir_value(),
                cutlass.Int32(sB_stage).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            # "mov.b32 smem_desc_a_lo, $0;\n\t"
            # f"add.u32 smem_desc_a_lo, $0, {hex(smem_desc_start_a_lo)};\n\t"
            f"mad.lo.u32 smem_desc_a_lo, $1, {hex(sA_addr_offset_for_desc)}, $0;\n\t"
            # "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mad.lo.u32 smem_desc_b_lo, $3, {hex(sB_addr_offset_for_desc)}, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $4, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                cutlass.Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                cutlass.Int32(smem_desc_start_b_lo).ir_value(),
                cutlass.Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_a, $1;\n\t"
            f"mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
