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

from cutlass_library import DataTypeSize, DataTypeTag

from cutlass_cppgen.backend.evt.ir import (
    # Load Node
    AccumulatorImpl,
    AuxLoadImpl,
    ColumnBroadcastImpl,
    LoadNode,
    LoadSrcImpl,
    RowBroadcastImpl,
    ScalarBroadcastImpl,
    # Compute Node
    ComputeImpl,
    # Store Node
    AuxStoreImpl,
    ColumnReductionImpl,
    RowReductionImpl,
    ScalarReductionImpl
)

from cutlass_cppgen.backend.library import (
    FloatRoundStyleTag,
    FunctionalOp,
    op_tag,
)


class Sm80AccumulatorImpl(AccumulatorImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""\nusing {self.name_camel} = cutlass::epilogue::threadblock::VisitorAccFetch;\n"""
        return self._type_decl


class Sm80AuxLoadImpl(AuxLoadImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, {DataTypeTag[self.element]}, {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80LoadSrcImpl(Sm80AuxLoadImpl):
    pass


class Sm80ScalarBroadcastImpl(ScalarBroadcastImpl):
    def __init__(self, node: LoadNode) -> None:
        super().__init__(node)
        self.broadcast_count = 1
        self.reduction_fn = FunctionalOp.Multiplies

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorScalarBroadcast<
    {DataTypeTag[self.element]}, {self.stride_mnl}, {self.broadcast_count}, {op_tag(self.reduction_fn)}
>;
"""
        return self._type_decl


class Sm80RowBroadcastImpl(RowBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, {DataTypeTag[self.element]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80ColumnBroadcastImpl(ColumnBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, {DataTypeTag[self.element]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80ComputeImpl(ComputeImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorCompute<
    {op_tag(self.fn)}, {DataTypeTag[self.element_output]}, {DataTypeTag[self.element_compute]},
    {FloatRoundStyleTag[self.round_style]}
>;
"""
        return self._type_decl


class Sm80AuxStoreImpl(AuxStoreImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, {DataTypeTag[self.element]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80StoreDImpl(Sm80AuxStoreImpl):
    pass


class Sm80ColumnReductionImpl(ColumnReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorColReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},
    OutputTileThreadMap, {DataTypeTag[self.element]},
    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80RowReductionImpl(RowReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorRowReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},
    OutputTileThreadMap, {DataTypeTag[self.element]},
    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm80ScalarReductionImpl(ScalarReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::threadblock::VisitorScalarReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},
    OutputTileThreadMap, {DataTypeTag[self.element]},
    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl
