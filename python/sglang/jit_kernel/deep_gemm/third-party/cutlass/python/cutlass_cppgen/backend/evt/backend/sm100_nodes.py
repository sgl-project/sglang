#################################################################################################
#
# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cutlass_cppgen.backend.evt.ir import AuxLoadImpl, AuxStoreImpl
import cutlass_cppgen.backend.evt.backend.sm90_nodes as sm90_nodes

from cutlass_cppgen.backend.library import FloatRoundStyleTag


Sm100AccumulatorImpl = sm90_nodes.Sm90AccumulatorImpl
Sm100LoadSrcImpl = sm90_nodes.Sm90LoadSrcImpl
Sm100ScalarBroadcastImpl = sm90_nodes.Sm90ScalarBroadcastImpl
Sm100RowBroadcastImpl = sm90_nodes.Sm90RowBroadcastImpl
Sm100ColumnBroadcastImpl = sm90_nodes.Sm90ColumnBroadcastImpl
Sm100ComputeImpl = sm90_nodes.Sm90ComputeImpl
Sm100StoreDImpl = sm90_nodes.Sm90StoreDImpl
Sm100ColumnReductionImpl = sm90_nodes.Sm90ColumnReductionImpl
Sm100RowReductionImpl = sm90_nodes.Sm90RowReductionImpl
Sm100ScalarReductionImpl = sm90_nodes.Sm90ScalarReductionImpl


class Sm100AuxLoadImpl(AuxLoadImpl):

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
        return f"\nusing {self.descriptor} = cutlass::epilogue::collective::detail::Sm100AuxLoadDescriptor<EpilogueDescriptor, {self.stride_mnl}, {DataTypeTag[self.element]}>;\n"

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


class Sm100AuxStoreImpl(AuxStoreImpl):

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
using {self.descriptor} = cutlass::epilogue::collective::detail::Sm100AuxStoreDescriptor<
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
