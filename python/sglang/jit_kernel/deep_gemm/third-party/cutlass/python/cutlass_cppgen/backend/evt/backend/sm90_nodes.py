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

from pycute import product

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
    ComputeNode,
    # Store Node
    AuxStoreImpl,
    ColumnReductionImpl,
    RowReductionImpl,
    ScalarReductionImpl,
    StoreNode,
    StoreDImpl,
)
from cutlass_cppgen.backend.library import (
    FloatRoundStyleTag,
    FunctionalOp,
    op_tag,
)


class Sm90AccumulatorImpl(AccumulatorImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""\nusing {self.name_camel} = cutlass::epilogue::fusion::Sm90AccFetch;\n"""
        return self._type_decl


class Sm90LoadSrcImpl(LoadSrcImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using ElementC = {DataTypeTag[self.element]};
using StrideC = {self.stride_mnl};
using {self.name_camel} = cutlass::epilogue::fusion::Sm90SrcFetch<{DataTypeTag[self.element]}>;
"""
        return self._type_decl


class Sm90AuxLoadImpl(AuxLoadImpl):

    @property
    def descriptor(self) -> str:
        """
        Descriptor for Aux Load
        """
        return f"{self.name_camel}Descriptor"

    def decl_descriptor(self) -> str:
        """
        Declare the descriptor type
        """
        return f"\nusing {self.descriptor} = cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, {self.stride_mnl}, {DataTypeTag[self.element]}>;\n"

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = self.decl_descriptor()
        self._type_decl += f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90AuxLoad<
    {self.descriptor}::Stages, typename {self.descriptor}::EpilogueTile, {DataTypeTag[self.element]},
    {self.stride_mnl}, typename {self.descriptor}::SmemLayoutAtom, typename {self.descriptor}::CopyOpS2R
>;
"""
        return self._type_decl

    def get_smem_size(self, cta_tile_mnk, epilogue_tile_mn, stages_c, stages_d, epi_tiles):
        """
        Get the shared memory size based on epilogue_tile_mn, stages_c, and stages_d
        """
        return (DataTypeSize[self.element] * stages_c * product(epilogue_tile_mn) // 8, 128)


class Sm90ScalarBroadcastImpl(ScalarBroadcastImpl):
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
using {self.name_camel} = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
    {DataTypeTag[self.element]}, {self.stride_mnl}, {self.broadcast_count}, {op_tag(self.reduction_fn)}
>;
"""
        return self._type_decl


class Sm90RowBroadcastImpl(RowBroadcastImpl):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90RowBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]}, {DataTypeTag[self.element_output]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm90ColumnBroadcastImpl(ColumnBroadcastImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]}, {DataTypeTag[self.element_output]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm90ComputeImpl(ComputeImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90Compute<
    {op_tag(self.fn)}, {DataTypeTag[self.element_output]}, {DataTypeTag[self.element_compute]},
    {FloatRoundStyleTag[self.round_style]}
>;
"""
        return self._type_decl


class Sm90AuxStoreImpl(AuxStoreImpl):

    @property
    def descriptor(self) -> str:
        """
        Descriptor for Aux Load
        """
        return f"{self.name_camel}Descriptor"

    def decl_descriptor(self) -> str:
        """
        Declare the descriptor type
        """
        return f"""
using {self.descriptor} = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, {self.stride_mnl}, {DataTypeTag[self.element]}
>;
"""
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = self.decl_descriptor()
        self._type_decl += f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90AuxStore<
    {self.descriptor}::Stages, typename {self.descriptor}::EpilogueTile, {DataTypeTag[self.element]},
    {FloatRoundStyleTag[self.round_style]}, {self.stride_mnl}, typename {self.descriptor}::SmemLayoutAtom,
    typename {self.descriptor}::CopyOpR2S
>;
"""
        return self._type_decl

    def get_smem_size(self, cta_tile_mnk, epilogue_tile_mn, stages_c, stages_d, epi_tiles):
        """
        Get the shared memory size based on epilogue_tile_mn, stages_c, and stages_d
        """
        return (DataTypeSize[self.element] * stages_d * product(epilogue_tile_mn) // 8, 128)


class Sm90StoreDImpl(StoreDImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        return f"""
using ElementD = {DataTypeTag[self.element]};
using StrideD = {self.stride_mnl};
"""


class Sm90ColumnReductionImpl(ColumnReductionImpl):

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90ColReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)}, 0,
    typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]},
    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm90RowReductionImpl(RowReductionImpl):


    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90RowReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)}, 0 /* Stages */,
    typename EpilogueDescriptor::TileShape, {DataTypeTag[self.element]},
    {DataTypeTag[self.element_compute]}, {FloatRoundStyleTag[self.round_style]},
    {self.stride_mnl}
>;
"""
        return self._type_decl


class Sm90ScalarReductionImpl(ScalarReductionImpl):


    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.name_camel} = cutlass::epilogue::fusion::Sm90ScalarReduction<
    {op_tag(self.reg_reduce_fn)}, {op_tag(self.gmem_reduce_fn)},
    {DataTypeTag[self.element]}, {DataTypeTag[self.element_compute]},
    {FloatRoundStyleTag[self.round_style]}, {self.stride_mnl}
>;
"""
        return self._type_decl
