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
Store node and implementations
"""

import ctypes

from cutlass_library import DataType

from cutlass_cppgen.backend.c_types import tuple_factory
from cutlass_cppgen.backend.epilogue import dtype2ctype, to_ctype_value
from cutlass_cppgen.backend.evt.ir.node import NodeBase, ImplBase, NoOpImpl
from cutlass_cppgen.backend.evt.ir.tensor import Tensor
from cutlass_cppgen.backend.library import FloatRoundStyle, FunctionalOp


class StoreImplBase(ImplBase):
    """
    Base class for store node implementation
    """
    reserved_names = ["D"]
    def __init__(self, node) -> None:
        super().__init__(node)
        self.element = node.element
        self.element_output = node.element_output
        self.stride = node.store_tensor.stride


class StoreDImpl(StoreImplBase):
    """
    Store D implementation
    """

    @property
    def argument_type_d(self):
        stride_mnl = self.get_stride_mnl()
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_D", ctypes.c_void_p),
                ("stride_D", tuple_type)
            ]
            def __init__(self, ptr: int) -> None:
                self.ptr_D = ptr
                self.stride_D = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if node.name == "D" and node.store_tensor.shape == problem_size:
            return True
        return False


class AuxStoreImpl(StoreImplBase):
    def __init__(self, node) -> None:
        super().__init__(node)
        self.round_style = FloatRoundStyle.ToNearest

    @property
    def argument_type(self):
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_aux", ctypes.c_void_p),
                ("dAux", tuple_type)
            ]
            def __init__(self, kwargs) -> None:
                ptr = kwargs[name]
                self.ptr_aux = ptr
                self.dAux = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if not node.is_output:
            return False
        if node.name in StoreImplBase.reserved_names:
            return False

        strideMN = node.store_tensor.stride[-2:]
        if (strideMN[0] == 1 and strideMN[1] != 0 or
            strideMN[0] != 0 and strideMN[1] == 1 ):
            return True
        else:
            return False


class ReductionImplBase(StoreImplBase):
    def __init__(self, node) -> None:
        super().__init__(node)
        self.element = node.store_tensor.element
        self.element_compute = node.element_compute
        self.reg_reduce_fn = self.node.reg_reduce_fn
        self.gmem_reduce_fn = self.node.gmem_reduce_fn
        self.round_style = node.round_style
        self.stride_dtype = "int"

    def get_reduce_identity(self):
        """
        Return the reduction identity of the current reduce_fn
        """
        maxes = {
            DataType.f32: (2 ** 31) - 1,
            DataType.f16: (2 ** 15),
            DataType.s32: (2 ** 31) - 1,
            DataType.s8: (2 ** 7) - 1
        }
        mins = {
            DataType.f32: -maxes[DataType.f32],
            DataType.f16: -maxes[DataType.f16],
            DataType.s32: -maxes[DataType.s32],
            DataType.s8: -maxes[DataType.s8]
        }
        if self.reg_reduce_fn == FunctionalOp.Maximum:
            if self.element_compute not in mins:
                raise Exception(f"No min entry for data type {self.element_compute}")
            return to_ctype_value(mins[self.element_compute], self.element_compute)
        elif self.reg_reduce_fn == FunctionalOp.Multiplies:
            return to_ctype_value(1., self.element_compute)
        elif self.reg_reduce_fn == FunctionalOp.Minimum:
            if self.element_compute not in maxes:
                raise Exception(f"No max entry for data type {self.element_compute}")
            return to_ctype_value(maxes[self.element_compute], self.element_compute)
        else:
            return to_ctype_value(0., self.element_compute)

    @property
    def argument_type(self):
        self.get_reduce_identity()
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        element_compute = self.element_compute
        reduce_identity = self.get_reduce_identity()
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr", ctypes.c_void_p),
                ("reduce_identity", dtype2ctype[element_compute]),
                ("dMNL", tuple_type)
            ]
            def __init__(self, kwargs) -> None:
                ptr = kwargs[name]
                self.ptr = ptr
                self.reduce_identity = reduce_identity
                self.dMNL = tuple_type(stride_mnl)

        return _Argument


class ColumnReductionImpl(ReductionImplBase):

    @staticmethod
    def match(node, problem_size: tuple):
        if not node.is_output:
            return False
        if node.name in StoreImplBase.reserved_names:
            return False

        strideMN = node.store_tensor.stride[-2:]
        if strideMN == (1, 0):
            return True
        else:
            return False


class RowReductionImpl(ReductionImplBase):

    @staticmethod
    def match(node, problem_size: tuple):
        if not node.is_output:
            return False
        if node.name in StoreImplBase.reserved_names:
            return False

        strideMN = node.store_tensor.stride[-2:]
        if strideMN == (0, 1):
            return True
        else:
            return False


class ScalarReductionImpl(ReductionImplBase):

    @staticmethod
    def match(node, problem_size: tuple):
        if not node.is_output:
            return False
        if node.name in StoreImplBase.reserved_names:
            return False

        strideMN = node.store_tensor.stride[-2:]
        if strideMN == (0, 0):
            return True
        else:
            return False


class StoreNode(NodeBase):
    """
    Store node
    """
    possible_impls = [
        AuxStoreImpl, RowReductionImpl,
        ColumnReductionImpl, ScalarReductionImpl,
        NoOpImpl, StoreDImpl
    ]
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.op = "store"
        self.is_output = False
        self._store_tensor = None

    @property
    def store_tensor(self) -> Tensor:
        """
        Return the output tensor (concept: cutlass_cppgen.backend.evt.ir.tensor)
        """
        return self._store_tensor

    @store_tensor.setter
    def store_tensor(self, kwargs):
        """
        Setting the tensor
        """
        self._store_tensor = Tensor(**kwargs)

    def type_propagation(self, input_node_metas: 'list[NodeBase]'):
        """
        The store nodes has element_output = element_input
        """
        if self.is_output:
            if self.store_tensor is None:
                raise RuntimeError(f"The store tensor of node {self.name} is unknown.")
            self.element = self.store_tensor.element
        assert len(input_node_metas) == 1, "Store node can only have one input node"
        self.element_output = input_node_metas[0].element_output

    def broadcast_propagation(self, input_node_metas: 'list[NodeBase]'):
        super().broadcast_propagation(input_node_metas)
        if self.is_output:
            self._store_tensor.broadcast(self.tensor.shape)
