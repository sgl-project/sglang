# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""DPP helpers used by paged-attention kernels."""

import flydsl.expr as fx


def update_dpp_i32(
    old,
    src,
    dpp_ctrl: int,
    row_mask: int = 0xF,
    bank_mask: int = 0xF,
    bound_ctrl: bool = False,
    **kw,
):
    """Wrapper for ``llvm.amdgcn.update.dpp.i32``.

    DPP controls are immediate operands. Common CDNA values:
    280/264 for row xor-8, 276/260 for row xor-4, 78 for xor-2,
    and 177 for xor-1 within a 16-lane row.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    return _llvm.call_intrinsic(
        fx.Int32.ir_type,
        "llvm.amdgcn.update.dpp.i32",
        [
            fx.Int32(old).ir_value(),
            fx.Int32(src).ir_value(),
            fx.Int32(dpp_ctrl).ir_value(),
            fx.Int32(row_mask).ir_value(),
            fx.Int32(bank_mask).ir_value(),
            fx.Boolean(bound_ctrl).ir_value(),
        ],
        [],
        [],
        **kw,
    )


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    src_i32 = fx.Float32(src).bitcast(fx.Int32)
    if offset == 8:
        out_i32 = update_dpp_i32(src_i32, src_i32, 280, 0xF, 0xC, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 264, 0xF, 0x3, False, **kw)
    elif offset == 4:
        out_i32 = update_dpp_i32(src_i32, src_i32, 276, 0xF, 0xA, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 260, 0xF, 0x5, False, **kw)
    elif offset == 2:
        out_i32 = update_dpp_i32(src_i32, src_i32, 78, 0xF, 0xF, False, **kw)
    elif offset == 1:
        out_i32 = update_dpp_i32(src_i32, src_i32, 177, 0xF, 0xF, False, **kw)
    else:
        raise ValueError(
            f"dpp_xor_f32 only supports 16-lane offsets 1, 2, 4, 8; got {offset}"
        )
    return fx.Int32(out_i32).bitcast(fx.Float32).ir_value()
