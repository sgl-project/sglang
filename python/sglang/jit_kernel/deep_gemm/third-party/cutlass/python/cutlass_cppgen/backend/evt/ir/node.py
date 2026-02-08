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
Base & visitor classes of DAGIR Nodes
"""

import ctypes
from re import sub

from cutlass_library import LayoutType

from cutlass_cppgen.backend.evt.ir.layout_algorithm import _list_to_tuple, _reverse_tuple
from cutlass_cppgen.backend.evt.ir.tensor import Tensor


class TupleEmitter:
    """
    Emit the cute tuple to C++ code
    """
    def __init__(self, stride_dtype):
        self.stride_dtype = stride_dtype

    def emit(self, py_tuple):
        if isinstance(py_tuple, int):
            if py_tuple in [0, 1]:
                return f"cute::Int<{py_tuple}>"
            else:
                return f"{self.stride_dtype}"
        elif isinstance(py_tuple, tuple):
            decl = "cute::Stride<"
            for item in py_tuple:
                decl += self.emit(item) + ", "
            return decl[:-2] + ">"
        else:
            raise ValueError(f"TupleEmitter.emit only accepts tuple or int, got {type(py_tuple).__name__}")


class ImplBase:
    """
    Base class for Node Implementation
    """
    def __init__(self, node) -> None:
        self.node = node
        self.name = node.name
        self.tensor = node.tensor
        self._type_decl = None
        self.tuple_emitter = TupleEmitter("int64_t")

    @property
    def stride_dtype(self):
        return self.tuple_emitter.stride_dtype

    @stride_dtype.setter
    def stride_dtype(self, stride_dtype):
        self.tuple_emitter.stride_dtype = stride_dtype

    @staticmethod
    def match(node, problem_size: tuple):
        """
        Match function used in get_underlying_impl
        """
        raise NotImplementedError(f"The `match` function is not defined.")

    @property
    def argument_type(self):
        """
        Default class for Argument Type
        """
        class _Argument(ctypes.Structure):
            _fields_ = []

            def __init__(self, *args, **kwargs) -> None:
                pass

        return _Argument

    @property
    def name_camel(self) -> str:
        """
        Return the CamelCase name.
        """
        return sub(r"(_|-)+", " ", self.name).title().replace(" ", "")

    @property
    def stride_mnl(self):
        """
        Typename StrideMNL
        """
        stride = _list_to_tuple([self.stride[-2], self.stride[-1]] + list(_reverse_tuple(tuple(self.stride[:-2]))))
        return self.tuple_emitter.emit(stride)

    def get_non_constant_stride(self, py_tuple):
        if isinstance(py_tuple, int):
            if py_tuple not in [0, 1]:
                return py_tuple
            else:
                return None
        non_constant_stride = []
        for item in py_tuple:
            item_out = self.get_non_constant_stride(item)
            if item_out:
                non_constant_stride.append(item_out)
        return tuple(non_constant_stride)

    def get_stride_mnl(self):
        """
        Get the non-zero stride mnl. This is used in argument construction
        """
        stride = _list_to_tuple([self.stride[-2], self.stride[-1]] + list(_reverse_tuple(tuple(self.stride[:-2]))))
        return stride

    def get_smem_size(self, *args, **kwargs):
        """
        Get the shared memory size and alignment of current node
        """
        return (0, 1)


class NoOpImpl(ImplBase):
    """
    The NoOpImpl does nothing but forward its input to users
    """
    def __init__(self, node) -> None:
        super().__init__(node)

    @staticmethod
    def match(node, problem_size: tuple):
        if node.op == "store":
            # Store that is not output is a No OP
            return not node.is_output


class NodeBase:
    """
    Base class of DAG Node
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.underlying_impl = None

        self._tensor = None

        # Whether the node is disabled for emit
        self.disabled = False

    @property
    def name_camel(self) -> str:
        """
        Return the CamelCase name.
        """
        return self.underlying_impl.name_camel

    @property
    def tensor(self) -> Tensor:
        """
        Return the output tensor (concept: cutlass_cppgen.backend.evt.ir.tensor)
        """
        return self._tensor

    @tensor.setter
    def tensor(self, kwargs):
        """
        Setting the tensor
        """
        self._tensor = Tensor(**kwargs)

    #
    # Helper functions for type/shape propagation
    #

    def shape_propagation(self, input_node_metas):
        """
        Infer shape from input nodes
        General Broadcasting Rules from NumPy
        When operating on two arrays, we compare their shapes element-wise.
        It starts with the trailing (i.e. rightmost) dimension and works its
        way left. Two dimensions are compatible when
        1. they are equal
        2. one of them is 1
        """
        if self._tensor is not None:
            return

        shape = None
        for src in input_node_metas:
            src_shape = src.tensor.shape
            if shape is None:
                shape = src_shape
            else:
                len_difference = len(shape) - len(src_shape)
                if len_difference > 0:
                    for _ in range(len_difference):
                        src_shape = [1, ] + list(src_shape)
                elif len_difference < 0:
                    for _ in range(-len_difference):
                        shape = [1, ] + list(shape)
                broadcasted_shape = []
                # Infer broadcast shape
                for shape_dim, src_dim in zip(reversed(shape), reversed(src_shape)):
                    if shape_dim == 1:
                        broadcasted_shape = [src_dim, ] + list(broadcasted_shape)
                    elif src_dim == 1:
                        broadcasted_shape = [shape_dim, ] + list(broadcasted_shape)
                    elif shape_dim == src_dim:
                        broadcasted_shape = [shape_dim, ] + list(broadcasted_shape)
                    else:
                        error_msg = "Dimension mismatch between "
                        for src_ in input_node_metas:
                            error_msg += f"{src_.name}{src_.tensor.shape}, "
                        error_msg = error_msg[:-2] + "."
                        raise RuntimeError(error_msg)
                shape = tuple(broadcasted_shape)

        self._tensor = Tensor(element=self.element_output, shape=shape, layout_tag=LayoutType.RowMajor)

    def type_propagation(self, *args, **kwargs):
        """
        Each node is associated with two data types: `element` and `element_output`.
        The `element_output` is the type of return array of the node. The `element`
        has specific meaning for different node types.
        * Load Node: data type of tensor in gmem
        * Compute Node: element compute
        * Store Node: data type of tensor in gmem
        This function must be overloaded in the derived classes
        """
        raise NotImplementedError(f"Function `type_propagation` is not overloaded in {self.__class__.__name__}")

    def broadcast_propagation(self, input_node_metas: 'list[NodeBase]'):
        """
        Propagate the broadcast in the reversed topological order.
        For example:
            C[l, m, n] = A[m, 1] + B[l, m, n]
        After the broadcast propagation, it will be come
            C[l, m, n] = A[l, m, n] + B[l, m, n]
        and each tensor will have a proper stride accessing the underlying tensor
        """
        if self.tensor is None:
            raise RuntimeError(f"The tensor of node {self.name} is unknown.")
        for child in input_node_metas:
            child.tensor.broadcast(self.tensor.shape)

    def get_underlying_impl(self, problem_size: tuple):
        """
        Get the underlying implementation of the current node.
        """
        if self.tensor is None:
            raise RuntimeError(f"The Layout of node {self.name} is unknown. Please call PassShapeTypePropagation first.")

        for impl in self.possible_impls:
            if impl.match(self, problem_size):
                self.underlying_impl = impl(self)
                break

        if self.underlying_impl is None:
            raise NotImplementedError(f"No matching op for node {self.name} with stride {self.tensor.stride}.")

#
# Visitor Nodes & Impls
#

class TopoVisitorImpl(ImplBase):
    """
    Impl for topological visitor
    """
    def __init__(self, node) -> None:
        super().__init__(node.output_node)
        self.name = node.name
        self.element_output = node.output_node.element_output

class TopoVisitorNode(NodeBase):
    def __init__(self, name: str, subgraph, output_node) -> None:
        super().__init__(name)
        self.subgraph = subgraph
        self.output_node = output_node
        self.op = "dag"
        self.underlying_impl = TopoVisitorImpl(self)
