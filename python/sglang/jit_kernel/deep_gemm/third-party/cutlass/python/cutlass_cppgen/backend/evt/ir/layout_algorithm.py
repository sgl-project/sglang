#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Layout algebras
"""

from pycute import Layout, composition, make_layout, flatten, product


def _infer_split(old_shape, new_shape):
    old_shape = _tuple_to_list(old_shape)
    new_shape = _tuple_to_list(new_shape)
    if len(old_shape) == 0 and len(new_shape) == 0:
        return []
    if len(old_shape) == 0:
        if product(tuple(new_shape)) != 1:
            raise ValueError("Invalid reshape size")
        else:
            return new_shape
    if len(new_shape) == 0:
        if product(tuple(old_shape)) != 1:
            raise ValueError("Invalid reshape size")
        else:
            return old_shape
    # This is done recursively by only process the last dimension at each time
    old_dim = old_shape[-1]
    new_dim = new_shape[-1]
    # Exact match
    if old_dim == new_dim:
        return _infer_split(old_shape[:-1], new_shape[:-1]) + [new_dim,]
    # Needs split
    if old_dim > new_dim and old_dim % new_dim == 0:
        residual = old_dim // new_dim
        return _infer_split(old_shape[:-1] + [residual,], new_shape[:-1]) + [new_dim,]
    # Needs merge
    if old_dim < new_dim and new_dim % old_dim == 0:
        residual = new_dim // old_dim
        return _infer_split(old_shape[:-1], new_shape[:-1] + [residual,]) + [old_dim,]

    raise NotImplementedError(f"Unsupported split: {old_shape} -> {new_shape}")

def _infer_merge(flatten_shape, shape):
    flatten_shape = _tuple_to_list(flatten_shape)
    shape = _tuple_to_list(shape)
    idx_flat = 0
    merged_shape = []
    for dim in shape:
        # Exact match
        if dim == flatten_shape[idx_flat]:
            merged_shape.append(dim)
            idx_flat += 1
        # Need group
        elif dim > flatten_shape[idx_flat] and dim % flatten_shape[idx_flat] == 0:
            residual = dim
            group = []
            while(residual > 1):
                group.append(flatten_shape[idx_flat])
                residual = residual // flatten_shape[idx_flat]
                idx_flat += 1
            merged_shape.append(group)
        else:
            raise NotImplementedError(f"Unsupported merge: {flatten_shape} -> {shape}")

    return merged_shape

def _list_to_tuple(nested_list):
    if isinstance(nested_list, list) or isinstance(nested_list, tuple):
        return tuple(_list_to_tuple(item) for item in nested_list)
    return nested_list

def _tuple_to_list(nested_tuple):
    if isinstance(nested_tuple, list) or isinstance(nested_tuple, tuple):
        return list(_tuple_to_list(item) for item in nested_tuple)
    return nested_tuple

def _reverse_tuple(nested_tuple: tuple):
    if isinstance(nested_tuple, tuple):
        return tuple([_reverse_tuple(item) for item in nested_tuple][::-1])
    return nested_tuple

def _get_first_lhs_nonzero_stride(stride_list, idx):
    for i in reversed(range(idx)):
        if stride_list[i] != 0:
            return i
    else:
        return None

def _get_first_rhs_nonzero_stride(stride_list, idx):
    for i in range(idx+1, len(stride_list)):
        if stride_list[i] != 0:
            return i
        else:
            return None

def reshape(layout, new_shape):
    """
    General reshape of input layout.
    It takes two steps:
    1. split the dimensions of the old layout
    2. merge the splitted dimensions according to the new shape
    """
    #
    # Step 1: Split the dimensions of the old layout
    #
    # 1.1 Flat old and new shape
    old_flatten_shape = list(flatten(layout.shape))
    new_flatten_shape = list(flatten(new_shape))

    # 1.2 Infer the flatten splitted shape
    splitted_flatten_shape = _infer_split(old_flatten_shape, new_flatten_shape)

    # 1.3 Unflat the splitted shape based on the old shape
    splited_shape = _infer_merge(splitted_flatten_shape, old_flatten_shape)

    # 1.4 Infer the type of each split
    # If the split type is in row-major (R), the dimension list is reversed because
    # the cute::composition only support column-major split
    split_type = []  # the type of each split (ColumnMajor or RowMajor)
    permuted_splitted_shape = []
    old_flatten_stride = list(flatten(layout.stride))
    for idx, dim in enumerate(splited_shape):
        if not isinstance(dim, list):
            permuted_splitted_shape.append(dim)
            split_type.append("C")
        else:
            lhs_stride = _get_first_lhs_nonzero_stride(old_flatten_stride, idx)
            rhs_stride = _get_first_rhs_nonzero_stride(old_flatten_stride, idx)
            # Special case for single tuple
            # Use column-major by default
            if lhs_stride is None and rhs_stride is None:
                permuted_splitted_shape.append(dim)
                split_type.append("C")
            else:
                if lhs_stride is not None and rhs_stride is not None:
                    # We consider shape[idx]:stride[idx]
                    # Case 1: stride[idx - 1] <= stride[idx] <= stride[idx + 1]: column major
                    if lhs_stride <= old_flatten_stride[idx] and old_flatten_stride[idx] <= rhs_stride:
                        permuted_splitted_shape.append(dim)
                        split_type.append("C")
                    # Case 2: stride[idx - 1] > stride[idx] > stride[idx + 1]: row major
                    elif lhs_stride > old_flatten_stride[idx] and old_flatten_stride[idx] > rhs_stride:
                        permuted_splitted_shape.append([d for d in reversed(dim)])
                        split_type.append("R")
                    # Case 3: stride[idx - 1] <= stride[idx] > stride[idx + 1]: concave
                    elif lhs_stride <= old_flatten_stride[idx] and old_flatten_stride[idx] > rhs_stride:
                        if lhs_stride >= rhs_stride:
                            permuted_splitted_shape.append(dim)
                            split_type.append("C")
                        else:
                            permuted_splitted_shape.append([d for d in reversed(dim)])
                            split_type.append("R")
                    # Case 4: stride[idx - 1] > stride[idx] <= stride[idx + 1]: concave
                    elif lhs_stride > old_flatten_stride[idx] and old_flatten_stride[idx] <= rhs_stride:
                        if lhs_stride >= rhs_stride:
                            permuted_splitted_shape.append(dim)
                            split_type.append("C")
                        else:
                            permuted_splitted_shape.append([d for d in reversed(dim)])
                            split_type.append("R")
                    else:
                        raise NotImplementedError()
                elif lhs_stride is None:
                    # Case 1: dim's stride < dim+1's stride, expand in column major
                    if old_flatten_stride[idx] > rhs_stride:
                        permuted_splitted_shape.append([d for d in reversed(dim)])
                        split_type.append("R")
                    else:
                        permuted_splitted_shape.append(dim)
                        split_type.append("C")
                else:
                    # Case 1: dim's stride > dim-1's stride
                    if old_flatten_stride[idx] < lhs_stride:
                        permuted_splitted_shape.append([d for d in reversed(dim)])
                        split_type.append("R")
                    else:
                        permuted_splitted_shape.append(dim)
                        split_type.append("C")

    # 1.4 Generate the splitted layout
    permuted_splitted_layout = composition(layout, Layout(_list_to_tuple(permuted_splitted_shape)))

    # 1.5 Reverse the permutation in 1.4 before merge
    splitted_shape = []
    splitted_stride = []
    for shape_dim, stride_dim, type in zip(
            permuted_splitted_layout.shape,
            permuted_splitted_layout.stride,
            split_type):
        if type == "C":
            splitted_shape.append(shape_dim)
            splitted_stride.append(stride_dim)
        else:
            splitted_shape.append(tuple([d for d in reversed(shape_dim)]))
            splitted_stride.append(tuple([d for d in reversed(stride_dim)]))
    splitted_layout = Layout(tuple(splitted_shape), tuple(splitted_stride))


    #
    # Step 2: Merge the splitted dimensions according to the new shape
    #
    # 2.1 Merge layout
    merged_layout = composition(splitted_layout, Layout(new_shape))

    # 2.2 Cleaning up
    output_layout = composition(merged_layout, Layout(new_shape))
    return output_layout


def permutation(layout, permutation):
    """
    Permute the layout
    """
    new_shape = tuple([layout.shape[idx] for idx in permutation])
    new_stride = tuple([layout.stride[idx] for idx in permutation])
    return Layout(new_shape, new_stride)


def _broadcast(layout, new_shape):
    if len(layout) == 1 and isinstance(new_shape, int):
        old_dim = layout.shape
        old_stride = layout.stride
        new_dim = new_shape
        if old_dim == new_dim:
            return Layout(old_dim, old_stride)
        elif old_dim == 1:
            return Layout(new_dim, 0)
        else:
            raise NotImplementedError(f"Invalid Broadcast: {old_dim} -> {new_dim}")

    # Align the dimensions
    old_shape = layout.shape
    if isinstance(old_shape, int):
        old_shape = (old_shape,)
        sub_layouts = [layout,]
    else:
        sub_layouts = [sub_layout for sub_layout in layout]
    rhs_broadcast_layouts = [Layout(1, 0)] * (len(new_shape) - len(old_shape))
    # Get the broadcasted layout
    broadcast_layouts = []
    try:
        layout = make_layout(*sub_layouts, *rhs_broadcast_layouts)
        broadcast_layouts = []
        for idx, sub_layout in enumerate(layout):
            broadcast_layouts.append(_broadcast(sub_layout, new_shape[idx]))
    except NotImplementedError:
        layout = make_layout(*rhs_broadcast_layouts, *sub_layouts)
        for idx, sub_layout in enumerate(layout):
            broadcast_layouts.append(_broadcast(sub_layout, new_shape[idx]))
    return make_layout(*broadcast_layouts)


def broadcast(layout, new_shape):
    """
    Broadcast the new layout based on the input shape
    The broadcasted shape equals to the new shape
    The stride of broadcasted dimensions are 0
    """
    return _broadcast(layout, new_shape)


def debroadcast(layout, dims):
    """
    Squeeze the 0-stride
    """
    for dim in dims:
        if layout.stride[dim] != 0:
            raise ValueError(f"Dim{dim} cannot be debroadcasted as it has stride {layout.stride[dim]}")
    new_shape = tuple([s for idx, s in enumerate(layout.shape) if idx not in dims])
    new_stride = tuple([s for idx, s in enumerate(layout.stride) if idx not in dims])
    return Layout(new_shape, new_stride)


def canonicalization_(shapes, strides):
    if isinstance(shapes, tuple):
        c_shapes = []
        c_strides = []
        for shape, stride in zip(shapes, strides):
            c_shape, c_stride = canonicalization_(shape, stride)
            c_shapes.append(c_shape)
            c_strides.append(c_stride)
        return tuple(c_shapes), tuple(c_strides)
    else:
        if shapes == 1:
            return 1, 0
        else:
            return shapes, strides

def canonicalization(layout):
    """
    Canonicalize the input layout
    1. set the stride of shape "1" to 0
    """
    new_shape, new_stride = canonicalization_(layout.shape, layout.stride)
    return Layout(new_shape, new_stride)
