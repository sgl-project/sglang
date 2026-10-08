# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Vendored FlyDSL vector wrappers for the Kimi-K3 FlyDSL kernels.

Re-exports the raw dialect, unwraps DSL values, and fills dynamic-index
sentinels. FlyDSL removed ``flydsl.expr.vector`` before ROCm/FlyDSL#880 and
AITER no longer ships a copy, so SGLang keeps this one. Remove it once the
callers use the raw dialect or the typed vector API.
Source: FlyDSL ``python/flydsl/expr/vector.py``.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector

# Re-export the raw dialect so vector.broadcast / vector.shuffle work directly
from flydsl._mlir.dialects.vector import *
from flydsl.expr.meta import dsl_loc_tracing

# Vector and related types, which flydsl.expr.vector also re-exported
from flydsl.expr.typing import (  # noqa: F401
    ReductionOp,
    Vector,
    as_ir_value,
    empty_like,
    full,
    full_like,
    ones_like,
    zeros_like,
)

# ═══════════════════════════════════════════════════════════════════════
# Dialect helper wrappers (legacy, will be deprecated)
# Prefer using Vector methods or _mlir.dialects.vector directly.
# ═══════════════════════════════════════════════════════════════════════


def _as_index_ir_value(value):
    if isinstance(value, int):
        from flydsl.expr import arith as _arith_ext

        return _arith_ext.constant(value, index=True)
    v = as_ir_value(value)
    # vector/memref indices must be index-typed; cast integer offsets
    # (e.g. fx.Int64/i32) so callers can pass explicit-width offsets.
    if isinstance(v.type, ir.IntegerType):
        from flydsl._mlir.dialects import arith as _std_arith

        v = _std_arith.IndexCastOp(ir.IndexType.get(), v).result
    return v


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    """Build a vector from scalars, unwrapping DSL values."""
    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [as_ir_value(v) for v in elems]
        return _vector.from_elements(*args, **kwargs)

    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def store(value, memref, indices, **kwargs):
    """Store a vector, unwrapping DSL values and converting indices."""
    return _vector.store(
        as_ir_value(value),
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Thin wrappers for common op classes that otherwise require `.result` access.
# -----------------------------------------------------------------------------


@dsl_loc_tracing
def extract(vector, static_position=None, dynamic_position=None):
    """Extract vector elements, filling missing dynamic-index sentinels."""
    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_as_index_ir_value(i) for i in dynamic_position]

    n_static = len(static_position)
    n_dynamic = len(dynamic_position)
    if n_dynamic > 0 and n_static < n_dynamic:
        kDynamic = ir.ShapedType.get_dynamic_size()
        static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

    return _vector.ExtractOp(
        as_ir_value(vector),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result


@dsl_loc_tracing
def load_op(result_type, memref, indices):
    """Load a vector, unwrapping DSL values and converting indices."""
    return _vector.LoadOp(
        result_type,
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
    ).result


@dsl_loc_tracing
def bitcast(result_type, source):
    """Bitcast a vector, unwrapping its DSL value."""
    return _vector.BitCastOp(
        result_type,
        as_ir_value(source),
    ).result
