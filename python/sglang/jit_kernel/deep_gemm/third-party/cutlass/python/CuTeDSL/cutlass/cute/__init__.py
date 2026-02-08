# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

# Use the auto-generated enum AddressSpace
from cutlass._mlir.dialects.cute import AddressSpace

# Explicitly import types that might be directly used by other modules.
# This is a fix for using Sphinx to generate documentation
# Because Sphinx processes each module in isolation, it won't be able to rely
# on re-exported symbols via wildcard imports (from .typing import *) in the
# same way that Python does at runtime.
from .typing import (
    Shape,
    Stride,
    IntTuple,
    Coord,
    Tile,
    XTuple,
    Tiler,
    Layout,
    Pointer,
    Tensor,
)

# Import everything else
from .typing import *

from .core import (
    assume,
    is_integer,
    is_int_tuple,
    is_static,
    size,
    has_underscore,
    slice_,
    make_ptr,
    make_layout,
    recast_layout,
    make_fragment_like,
    depth,
    rank,
    flatten_to_tuple,
    flatten,
    unflatten,
    product,
    product_like,
    shape,
    size_in_bytes,
    make_identity_layout,
    make_ordered_layout,
    make_composed_layout,
    make_layout_tv,
    make_swizzle,
    recast_ptr,
    make_tensor,
    make_identity_tensor,
    make_fragment,
    recast_tensor,
    get,
    select,
    front,
    is_major,
    leading_dim,
    find,
    find_if,
    coalesce,
    group_modes,
    cosize,
    dice,
    product_each,
    prepend,
    append,
    prepend_ones,
    append_ones,
    ceil_div,
    slice_and_offset,
    crd2idx,
    domain_offset,
    elem_less,
    transform_leaf,
    filter_zeros,
    filter,
    tile_to_shape,
    shape_div,
    composition,
    complement,
    right_inverse,
    left_inverse,
    max_common_layout,
    max_common_vector,
    logical_product,
    zipped_product,
    tiled_product,
    flat_product,
    raked_product,
    blocked_product,
    flat_divide,
    logical_divide,
    zipped_divide,
    tiled_divide,
    local_partition,
    local_tile,
    printf,
    print_tensor,
    # tiled mma/tiled copy
    make_mma_atom,
    make_tiled_mma,
    make_copy_atom,
    make_tiled_copy_tv,
    make_tiled_copy,
    make_tiled_copy_S,
    make_tiled_copy_D,
    make_tiled_copy_A,
    make_tiled_copy_B,
    make_tiled_copy_C,
    make_tiled_copy_C_atom,
    basic_copy,
    basic_copy_if,
    autovec_copy,
    copy,
    copy_atom_call,
    gemm,
    # Wrapper classes
    ComposedLayout,
    Swizzle,
    E,
    Atom,
    MmaAtom,
    CopyAtom,
    TiledCopy,
    TiledMma,
    TensorSSA,
    ReductionOp,
    full,
    full_like,
    empty_like,
    ones_like,
    zeros_like,
    where,
    any_,
    all_,
    # User defined struct
    struct,
    pretty_str,
    make_layout_image_mask,
    repeat_like,
    round_up,
    is_congruent,
    is_weakly_congruent,
    ScaledBasis,
    get_divisibility,
    Ratio,
)

from . import arch
from . import nvgpu
from . import testing
from . import runtime

# Export all math ops without "math."
from .math import *

# Used as internal symbol
from .. import cutlass_dsl as _dsl

# Aliases
jit = _dsl.CuTeDSL.jit
kernel = _dsl.CuTeDSL.kernel
register_jit_arg_adapter = _dsl.JitArgAdapterRegistry.register_jit_arg_adapter
compile = _dsl.compile

# Explicitly export all symbols for documentation generation
__all__ = [
    # Core types
    "AddressSpace",
    "Tensor",
    "Layout",
    "ComposedLayout",
    "Swizzle",
    "E",
    "Atom",
    "MmaAtom",
    "CopyAtom",
    "TiledCopy",
    "TiledMma",
    "TensorSSA",
    # Basic utility functions
    "assume",
    "is_integer",
    "is_int_tuple",
    "is_static",
    "size",
    "has_underscore",
    "slice_",
    "depth",
    "rank",
    "shape",
    "printf",
    "print_tensor",
    "pretty_str",
    # Layout functions
    "make_layout",
    "recast_layout",
    "make_identity_layout",
    "make_ordered_layout",
    "make_composed_layout",
    "make_layout_tv",
    "make_layout_image_mask",
    # Tensor functions
    "make_ptr",
    "make_tensor",
    "make_identity_tensor",
    "make_fragment",
    "make_fragment_like",
    "recast_ptr",
    "recast_tensor",
    # Tensor manipulation
    "get",
    "select",
    "front",
    "is_major",
    "leading_dim",
    "find",
    "find_if",
    "coalesce",
    "group_modes",
    "cosize",
    "size_in_bytes",
    # Tuple operations
    "flatten_to_tuple",
    "flatten",
    "product",
    "product_like",
    "product_each",
    "prepend",
    "append",
    "prepend_ones",
    "append_ones",
    # Math operations
    "ceil_div",
    "round_up",
    # Layout operations
    "slice_and_offset",
    "crd2idx",
    "domain_offset",
    "elem_less",
    "filter_zeros",
    "filter",
    "tile_to_shape",
    "shape_div",
    "dice",
    # Layout algebra
    "composition",
    "complement",
    "right_inverse",
    "left_inverse",
    "max_common_layout",
    "max_common_vector",
    "is_congruent",
    "is_weakly_congruent",
    # Product operations
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    "raked_product",
    "blocked_product",
    # Division operations
    "flat_divide",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "local_partition",
    "local_tile",
    # MMA and Copy operations
    "make_mma_atom",
    "make_tiled_mma",
    "make_copy_atom",
    "make_tiled_copy_tv",
    "make_tiled_copy",
    "make_tiled_copy_C_atom",
    "basic_copy",
    "basic_copy_if",
    "autovec_copy",
    "copy",
    "copy_atom_call",
    "gemm",
    # Tensor creation
    "full",
    "full_like",
    "empty_like",
    "ones_like",
    "zeros_like",
    "where",
    "any_",
    "all_",
    "repeat_like",
    "ScaledBasis",
    # User defined struct
    "struct",
    # Modules
    "arch",
    "nvgpu",
    "testing",
    "runtime",
    # Decorators and code generation
    "jit",
    "kernel",
    "register_jit_arg_adapter",
    "compile",
]
