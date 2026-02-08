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
Layout manipulation nodes and implementations

The layout Nodes change the layout of intermediate nodes in epilogue visitor graph
"""

from copy import deepcopy

from cutlass_library import LayoutType
from pycute import product, flatten

import cutlass_cppgen
from cutlass_cppgen.backend.evt.ir.layout_algorithm import _list_to_tuple, _tuple_to_list
from cutlass_cppgen.backend.evt.ir.node import NodeBase
from cutlass_cppgen.backend.evt.ir.tensor import Tensor


class PermutationImpl:
    """
    Detailed implementation and helper functions for permutation
    """
    def __init__(self, node) -> None:
        assert "indices" in node.kwargs.keys()
        self.indices = list(node.kwargs["indices"])
        self.inverse_indices = self.get_inverse_indices(self.indices)

    def get_inverse_impl(self):
        inverse_impl = deepcopy(self)
        inverse_impl.indices = self.inverse_indices
        inverse_impl.inverse_indices = self.indices
        return inverse_impl

    def update(self, shape):
        num_dim = len(shape)
        indices = self.indices
        num_old_dim = len(indices)
        # Add offset
        for i, idx in enumerate(indices):
            indices[i] = idx + num_dim - num_old_dim
        # Add broadcast dims
        for i in range(num_dim - num_old_dim):
            indices = [i,] + indices

        self.indices = indices
        self.inverse_indices = self.get_inverse_indices(self.indices)

    def get_inverse_indices(self, indices):
        """
        Get the indices for inverse permutation
        """
        num_dim = len(indices)
        inverse_indices = [0] * num_dim
        for i in range(num_dim):
            inverse_indices[indices[i]] = i
        return inverse_indices

    def shape_propagation(self, input_node_meta):
        input_shape = input_node_meta.tensor.shape
        output_shape = tuple([input_shape[idx] for idx in self.indices])
        return output_shape

    def broadcast(self, shape, node_meta: NodeBase):
        """
        Broadcast the inputs based on current shape
        """
        self.update(shape)
        inverse_shape = tuple([shape[idx] for idx in self.inverse_indices])
        node_meta.tensor.broadcast(inverse_shape)

    def apply_to_user(self, usr_meta: NodeBase):
        """
        Propagate the permutation to the users of the current nodes
        """
        usr_meta.tensor.permute(self.inverse_indices)
        if hasattr(usr_meta, "store_tensor"):
            if usr_meta.store_tensor is not None:
                usr_meta.store_tensor.permute(self.inverse_indices)

    def apply_to_input(self, input_meta: NodeBase):
        """
        Propagate the permutation to inputs of the current nodes
        """
        input_meta.tensor.permute(self.indices)
        if hasattr(input_meta, "store_tensor"):
            if input_meta.store_tensor is not None:
                input_meta.store_tensor.permute(self.indices)


class ReshapeImpl:
    """
    Detailed implementation and helper functions for reshape
    """
    def __init__(self, node) -> None:
        self.node = node
        assert "new_shape" in node.kwargs.keys()
        self.output_shape = _list_to_tuple(node.kwargs["new_shape"])

    def get_inverse_impl(self):
        inverse_impl = deepcopy(self)
        inverse_impl.output_shape = self.input_shape
        inverse_impl.input_shape = self.output_shape
        return inverse_impl

    def shape_propagation(self, input_node_meta):
        self.input_shape = input_node_meta.tensor.shape
        return _list_to_tuple(self.output_shape)

    def broadcast(self, shape, node_meta: NodeBase):
        """
        Broadcast the inputs based on current shape.
        """
        # Step 1: infer split
        flatten_split_shape = self.infer_split(flatten(self.input_shape), flatten(self.output_shape))
        split_input_shape = self.infer_merge(flatten_split_shape, self.input_shape)
        split_output_shape = self.infer_merge(flatten_split_shape, self.output_shape)

        # broadcast shape -> split_output_shape -> flatten_split_shape
        if len(shape) - len(split_output_shape) > 0:
            for _ in range(len(shape) - len(split_output_shape)):
                split_output_shape = [1,] + split_output_shape
                flatten_split_shape = [1,] + flatten_split_shape
                split_input_shape = [1,] + split_input_shape
        broadcast_factor = []
        for dim, old_dim in zip(shape, split_output_shape):
            if not isinstance(dim, list):
                dim = [dim,]
            if not isinstance(old_dim, list):
                old_dim = [old_dim,]
            if product(tuple(dim)) == product(tuple(old_dim)):
                broadcast_factor += [1] * len(old_dim)
            elif product(tuple(old_dim)) == 1:
                assert len(dim) == 1
                broadcast_factor.append(dim[0])
            else:
                raise NotImplementedError(f"Invalid Broadcast: {old_dim} -> {dim}")

        # flatten_split_shape -> split_input_shape
        factor_idx = 0
        broadcast_split_input_shape = []
        for dim in split_input_shape:
            if isinstance(dim, list):
                new_dim = []
                for d in dim:
                    new_dim.append(d * broadcast_factor[factor_idx])
                    factor_idx += 1
                broadcast_split_input_shape.append(new_dim)
            else:
                broadcast_split_input_shape.append(dim * broadcast_factor[factor_idx])
                factor_idx += 1
        broadcast_split_input_shape = _list_to_tuple(broadcast_split_input_shape)
        node_meta.tensor.reshape(_list_to_tuple(split_input_shape))
        node_meta.tensor.broadcast(broadcast_split_input_shape)
        # Last reshape op to clean up
        broadcast_input_shape = tuple([product(dim) for dim in broadcast_split_input_shape])
        node_meta.tensor.reshape(broadcast_input_shape)
        # Update the input shape and output shape
        self.input_shape = _list_to_tuple(node_meta.tensor.shape)
        self.output_shape = _list_to_tuple(shape)

    def apply_to_user(self, user_meta: NodeBase):
        """
        Propagate the reshape to user nodes
        """
        user_meta.tensor.reshape(tuple(self.input_shape))
        if hasattr(user_meta, "store_tensor"):
            if user_meta.store_tensor is not None:
                user_meta.store_tensor.reshape(tuple(self.input_shape))

    def apply_to_input(self, input_meta: NodeBase):
        """
        Propagate the reshape to input nodes
        """
        input_meta.tensor.reshape(tuple(self.output_shape))
        if hasattr(input_meta, "store_tensor"):
            if input_meta.store_tensor is not None:
                input_meta.store_tensor.reshape(tuple(self.output_shape))

    #
    # Helper functions
    #

    def infer_split(self, input_shape, output_shape):
        """
        Infer the flatten splitted shape that can be merged to both input_shape and output_shape
        """
        input_shape = _tuple_to_list(input_shape)
        output_shape = _tuple_to_list(output_shape)
        if len(input_shape) == 0 and len(output_shape) == 0:
            return []
        if len(input_shape) == 0:
            if product(tuple(output_shape)) != 1:
                raise ValueError("Invalid reshape size")
            else:
                return output_shape
        if len(output_shape) == 0:
            if product(tuple(input_shape)) != 1:
                raise ValueError("Invalid reshape size")
            else:
                return input_shape
        # This is done recursively by only process the last dimension at each time
        old_dim = input_shape[-1]
        new_dim = output_shape[-1]
        # Exact match
        if old_dim == new_dim:
            return self.infer_split(input_shape[:-1], output_shape[:-1]) + [new_dim,]
        # Needs split
        if old_dim > new_dim and old_dim % new_dim == 0:
            residual = old_dim // new_dim
            return self.infer_split(input_shape[:-1] + [residual,], output_shape[:-1]) + [new_dim,]
        # Needs merge
        if old_dim < new_dim and new_dim % old_dim == 0:
            residual = new_dim // old_dim
            return self.infer_split(input_shape[:-1], output_shape[:-1] + [residual,]) + [old_dim,]

        raise NotImplementedError(f"Unsupported split: {input_shape} -> {output_shape}")

    def infer_merge(self, flatten_shape, shape):
        flatten_shape = _tuple_to_list(flatten_shape)
        shape = _tuple_to_list(shape)
        idx_flat = len(flatten_shape) - 1
        merged_shape = []
        for dim in reversed(shape):
            # Exact match
            if dim == flatten_shape[idx_flat]:
                merged_shape.append(dim)
                idx_flat -= 1
            # need group
            elif dim > flatten_shape[idx_flat] and dim % flatten_shape[idx_flat] == 0:
                residual = dim
                group = []
                while(residual > 1):
                    group.append(flatten_shape[idx_flat])
                    residual = residual // flatten_shape[idx_flat]
                    idx_flat -= 1
                merged_shape.append(group[::-1])
            else:
                raise NotImplementedError(f"Unsupported merge: {flatten_shape} -> {shape}")

        return merged_shape[::-1]


class LayoutNode(NodeBase):
    """
    Layout manipulation nodes
    """
    fn_to_impl = {
        "permute": PermutationImpl,
        "reshape": ReshapeImpl
    }
    def __init__(self, name: str, fn, kwargs: dict) -> None:
        super().__init__(name)
        self.op = "layout"
        self.fn = fn
        self.kwargs = kwargs
        self.underlying_impl = self.fn_to_impl[self.fn.__name__](self)

    def get_inverse_node(self):
        inverse_node = deepcopy(self)
        inverse_node.underlying_impl = self.underlying_impl.get_inverse_impl()
        return inverse_node

    def shape_propagation(self, input_node_metas):
        if self._tensor is not None:
            return
        assert len(input_node_metas) == 1, "Layout node can only have one input node"

        output_shape = self.underlying_impl.shape_propagation(input_node_metas[0])

        self._tensor = Tensor(
            element=self.element_output,
            shape=output_shape, layout_tag=LayoutType.RowMajor
        )

        return super().shape_propagation(input_node_metas)

    def type_propagation(self, input_node_metas: 'list[NodeBase]'):
        """
        The store nodes has element_output = element_input
        """
        assert len(input_node_metas) == 1, "Layout node can only have one input node"
        self.element_output = input_node_metas[0].element_output

    def broadcast_propagation(self, input_node_metas: 'list[NodeBase]'):
        """
        Propagate the broadcast in the reversed topological order
        """
        if self.tensor is None:
            raise RuntimeError(f"The tensor of node {self.name} is unknown.")
        shape = self.tensor.shape

        for child in input_node_metas:
            self.underlying_impl.broadcast(shape, child)

    def apply_to_user(self, usr_meta: NodeBase):
        """
        Propagate the permutation to user nodes
        """
        self.underlying_impl.apply_to_user(usr_meta)

    def apply_to_input(self, input_meta: NodeBase):
        """
        Propagate the permutation to input nodes
        """
        self.underlying_impl.apply_to_input(input_meta)
