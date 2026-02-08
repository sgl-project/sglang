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
Load nodes and implementations
"""

import ctypes

from cutlass_cppgen.backend.c_types import tuple_factory
from cutlass_cppgen.backend.epilogue import dtype2ctype, to_ctype_value
from cutlass_cppgen.backend.evt.ir.node import NodeBase, ImplBase


class LoadImplBase(ImplBase):
    """
    Base class for load node implementations
    """
    reserved_names = ["accum", "C"]
    def __init__(self, node) -> None:
        super().__init__(node)
        self.element = node.element
        self.element_output = node.element_output
        self.stride = node.tensor.stride


class AccumulatorImpl(LoadImplBase):
    """
    Accumulator node implementation
    """

    @staticmethod
    def match(node, problem_size: tuple):
        return node.name == "accum" and node.tensor.shape == problem_size


class LoadSrcImpl(LoadImplBase):
    """
    Load C implementation
    """
    @property
    def name_camel(self) -> str:
        return "TensorC"

    @property
    def argument_type_c(self):
        stride_mnl = self.get_stride_mnl()
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_C", ctypes.c_void_p),
                ("stride_C", tuple_type)
            ]
            def __init__(self, ptr) -> None:
                self.ptr_C = ptr
                self.stride_C = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        return node.name == "C" and node.tensor.shape == problem_size


class AuxLoadImpl(LoadImplBase):
    """
    Load arbitrary tensor
    """
    @property
    def argument_type(self):
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        element_type = self.element
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_aux", ctypes.c_void_p),
                ("null_default", dtype2ctype[element_type]),
                ("dAux", tuple_type)
            ]
            def __init__(self, kwargs) -> None:
                ptr = kwargs[name]
                self.ptr_aux = ptr
                self.null_default = to_ctype_value(0, element_type)
                self.dAux = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if node.name in LoadImplBase.reserved_names:
            return False
        strideMN = node.tensor.stride[-2:]
        if (strideMN[0] == 1 and strideMN[1] != 0 or
            strideMN[0] != 0 and strideMN[1] == 1 ):
            return True
        else:
            return False


class RowBroadcastImpl(LoadImplBase):
    """
    Broadcast a row vector
    """
    def __init__(self, node) -> None:
        super().__init__(node)
        self.stride_dtype = "int"

    @property
    def argument_type(self):
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        element_type = self.element
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_row", ctypes.c_void_p),
                ("null_default", dtype2ctype[element_type]),
                ("dRow", tuple_type)
            ]
            def __init__(self, kwargs) -> None:
                ptr = kwargs[name]
                self.ptr_row = ptr
                self.null_default = to_ctype_value(0, element_type)
                self.dRow = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if node.name in LoadImplBase.reserved_names:
            return False

        strideMN = node.tensor.stride[-2:]
        if strideMN == (0, 1):
            return True
        else:
            return False


class ColumnBroadcastImpl(LoadImplBase):
    """
    Broadcast a column vector
    """
    def __init__(self, node) -> None:
        super().__init__(node)
        self.stride_dtype = "int"

    @property
    def argument_type(self):
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        element_type = self.element
        class _Argument(ctypes.Structure):
            _fields_ = [
                ("ptr_col", ctypes.c_void_p),
                ("null_default", dtype2ctype[element_type]),
                ("dCol", tuple_type)
            ]
            def __init__(self, kwargs) -> None:
                ptr = kwargs[name]
                self.ptr_col = int(ptr)
                self.null_default = to_ctype_value(0, element_type)
                self.dCol = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if node.name in LoadImplBase.reserved_names:
            return False

        strideMN = node.tensor.stride[-2:]
        if strideMN == (1, 0):
            return True
        else:
            return False


class ScalarBroadcastImpl(LoadImplBase):
    """
    Broadcast a scalar
    """
    def __init__(self, node) -> None:
        super().__init__(node)
        self.stride_dtype = "int"

    @property
    def argument_type(self):
        stride_mnl = self.get_stride_mnl()
        name = self.name
        tuple_type = tuple_factory(stride_mnl, self.stride_dtype)
        element_type = self.element

        if self.tensor.is_constant:
            value = self.tensor.value
            class _Argument(ctypes.Structure):
                _fields_ = [
                    ("scalars", dtype2ctype[element_type]),
                    ("scalar_ptrs", ctypes.c_void_p),
                    ("dScalar", tuple_type)
                ]
                def __init__(self, kwargs) -> None:
                    self.scalars = to_ctype_value(value, element_type)
                    self.scalar_ptrs = 0
                    self.dScalar = tuple_type(stride_mnl)

        else:
            class _Argument(ctypes.Structure):
                _fields_ = [
                    ("scalars", dtype2ctype[element_type]),
                    ("scalar_ptrs", ctypes.c_void_p),
                    ("dScalar", tuple_type)
                ]
                def __init__(self, kwargs) -> None:
                    scalar_or_ptr = kwargs[name]
                    if isinstance(scalar_or_ptr, float):
                        self.scalars = to_ctype_value(scalar_or_ptr, element_type)
                        self.scalar_ptrs = 0
                    else:
                        self.scalar_ptrs = int(scalar_or_ptr)

                    self.dScalar = tuple_type(stride_mnl)

        return _Argument

    @staticmethod
    def match(node, problem_size: tuple):
        if node.name in LoadImplBase.reserved_names:
            return False

        strideMN = node.tensor.stride[-2:]
        if strideMN == (0, 0):
            return True
        else:
            return False


class LoadNode(NodeBase):
    """
    Load Node
    """
    cnt = 0
    possible_impls = [
        AccumulatorImpl, LoadSrcImpl, AuxLoadImpl,
        RowBroadcastImpl, ColumnBroadcastImpl,
        ScalarBroadcastImpl
    ]
    def __init__(self, name: str) -> None:
        if name is None:
            name = f"load{LoadNode.cnt}"
            LoadNode.cnt += 1
        super().__init__(name)
        self.op = "load"

    def type_propagation(self, *args, **kwargs):
        """
        Load node loads tensor under type `tensor.element` and returns an array of type `tensor.element`.
        """
        if self.tensor is None:
            raise RuntimeError(f"The tensor of node {self.name} is unknown.")

        self.element = self.tensor.element
        self.element_output = self.tensor.element
