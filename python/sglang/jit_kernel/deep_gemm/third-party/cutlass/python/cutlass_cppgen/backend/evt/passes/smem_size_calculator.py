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
Compute the shared memory size in bytes
"""

from math import gcd

import cutlass_library
from pycute import flatten, shape_div, product

import cutlass_cppgen
from cutlass_cppgen.backend.evt.ir import TopoVisitorNode, DAGIR
from cutlass_cppgen.backend.library import DataType, DataTypeSize


class GetSmemSize:
    """
    Get the size in byte of shared memory used by the kernel
    """
    def __init__(self, dag_ir: DAGIR) -> None:
        self.dag_ir = dag_ir
        self.cc = self.dag_ir.cc

    #
    # Sm90 epilogue specific
    #

    def sm90_epilogue_tile(self, tile_description):
        # Get the epilogue tile size
        schedule = tile_description.epilogue_schedule
        if schedule == cutlass_library.EpilogueScheduleType.TmaWarpSpecialized:
            element_d = self.dag_ir.get_node_meta("D").element
            nperf = 64 if (DataTypeSize[element_d] == 8 and tile_description.threadblock_shape[1] % 64 == 0) else 32
            epi_tile_m = min(64, tile_description.threadblock_shape[0])
            epi_tile_n = gcd(min(nperf, tile_description.threadblock_shape[1]), tile_description.threadblock_shape[1])
            epilogue_tile_mn = (epi_tile_m, epi_tile_n)
        elif schedule == cutlass_library.EpilogueScheduleType.TmaWarpSpecializedCooperative:
            epi_tile_m = min(128, tile_description.threadblock_shape[0])
            epi_tile_n = gcd(min(32, tile_description.threadblock_shape[1]), tile_description.threadblock_shape[1])
            epilogue_tile_mn = (epi_tile_m, epi_tile_n)
        else:
            raise NotImplementedError(f"Unsupported schedule: {schedule}")

        # Get the pipeline stages
        stages_d = 2
        epi_tiles = product(shape_div(tuple(tile_description.threadblock_shape)[:2], epilogue_tile_mn))
        if self.dag_ir.has_node("C"):
            element_c = self.dag_ir.get_node_meta("C").element
        else:
            element_c = None

        element_d = self.dag_ir.get_node_meta("D").element
        if element_c == element_d:
            reuse_smem_c = True
        else:
            reuse_smem_c = False
        stages_c = max(epi_tiles, stages_d + 1) if reuse_smem_c else epi_tiles

        # Record the epilogue tile
        self.cta_tile_mnk = tuple(tile_description.threadblock_shape)
        self.epilogue_tile_mn = epilogue_tile_mn
        self.epi_tiles = epi_tiles
        self.stages_c = stages_c
        self.stages_d = stages_d
        self.reuse_smem_c = reuse_smem_c
        self.element_c = element_c
        self.element_d = element_d
        self.is_source_supported = element_c is not None

    def sm90_or_sm100_epilogue_smem_size(self, tile_description):
        # Get the Fusion Storage
        nodes = self.dag_ir.nodes_topological_order()
        self.smem_types = {}
        for node in nodes:
            meta = self.dag_ir.get_node_meta(node)
            if not meta.disabled:
                self.smem_types[node] = meta.underlying_impl.get_smem_size(
                    self.cta_tile_mnk, self.epilogue_tile_mn,
                    self.stages_c, self.stages_d, self.epi_tiles)
            if node == "D":
                continue
            if isinstance(meta, TopoVisitorNode):
                self.get_dag_smem_type(node)
            else:
                self.get_evt_smem_type(node)

        thread_smem_size = self.smem_types[self.dag_ir.get_all_inputs("D")[0]][0]
        # Get the Tensor Storage
        tensors = []
        if self.is_source_supported:
            smem_C = DataTypeSize[self.element_c] * product(self.epilogue_tile_mn) * self.stages_c // 8
            tensors.append((smem_C, 128))
        else:
            tensors.append((0, 1))
        if self.reuse_smem_c:
            tensors.append((0, 128))
        else:
            smem_D = DataTypeSize[self.element_d] * product(self.epilogue_tile_mn) * self.stages_d // 8
            tensors.append((smem_D, 128))
        tensors.append((thread_smem_size, 128))

        tensor_smem_size = self.get_struct_size(tensors)
        # Get pipeline storage size
        # sizeof(uint64_t * stages_c * 2), alignment of uint64_t
        # 2 is for FullBarrier and EmptyBarrier
        pipeline_smem_size = (8 * self.stages_c * 2, 8)

        # get SharedStorage size
        smem_size = self.get_struct_size([tensor_smem_size, pipeline_smem_size])
        return smem_size[0]

    def sm90_epilogue_smem_size(self, tile_description):
        """
        Compute the shared memory size of sm90 collective epilogue
        """
        self.sm90_epilogue_tile(tile_description)
        return self.sm90_or_sm100_epilogue_smem_size(tile_description)

    #
    # Sm100 epilogue specific
    #

    def sm100_epilogue_tile(self, tile_description):
        cta_tile = (tile_description.blackwell_threadblock_shape[0], tile_description.blackwell_threadblock_shape[1])
        mma_tile = cta_tile

        if tile_description.is_2sm:
            cta_tile = (cta_tile[0] // 2, cta_tile[1])

        if tile_description.is_2sm and mma_tile[0] == 128:
            tmem_warps = (2, 2)
        else:
            tmem_warps = (4, 1)

        if self.dag_ir.has_node("C"):
            element_c = self.dag_ir.get_node_meta("C").element
            element_c_size = DataTypeSize[element_c]
        else:
            element_c = None
            element_c_size = 0

        element_d = self.dag_ir.get_node_meta("D").element

        DisableSource = element_c is None or not self.dag_ir.has_node("C") or self.dag_ir.get_node_meta("C").element == DataType.void

        CtaM = cta_tile[0]
        CtaN = cta_tile[1]
        WarpM = tmem_warps[0]
        WarpN = tmem_warps[1]
        MaxBits = max(element_c_size, DataTypeSize[element_d])
        DpFull = 32
        M = min(CtaM, DpFull * WarpM)

        if DisableSource:
            # Epilogues w/o residual load are less sensitive to smem allocation
            # Target a fixed amount of compute per epilogue iteration
            if MaxBits == 4:
                # Make epilogue tile larger to reduce the epilogue iterations.
                # 64 is the experimental value. It will minimize epilogue iterations but keep the number of A/B buffers the same.
                ComputeElts = 8192
                Nperf = ComputeElts // M
            else:
                ComputeElts = 4096
                Nperf = ComputeElts // M
        else:
            # Epilogues w/ residual load are more sensitive to smem allocation
            # Target optimal smem distribution between epilogue+mainloop based on datatype+tilesize
            if MaxBits == 32:
                Nperf = 16 if CtaM > 64 and CtaN <= 128 else 32
            elif MaxBits == 16:
                Nperf = 32 if CtaN <= 128 else 64
            else:
                Nperf = 64

        def is_m_major(layout):
            return flatten(layout.stride[0]) == 1

        if DisableSource or is_m_major(self.dag_ir.get_node_meta("C").tensor.layout):
            N_min_C = 8 * WarpN
        elif element_c_size == 6:
            N_min_C = 128 * WarpN
        else:
            N_min_C = (128 // element_c_size) * WarpN

        if is_m_major(self.dag_ir.get_node_meta("D").tensor.layout):
            N_min_D = 8 * WarpN
        elif DataTypeSize[element_d] == 6:
            N_min_D = 128 * WarpN
        else:
            N_min_D = (128 // DataTypeSize[element_d]) * WarpN

        N = min(CtaN, max(Nperf, N_min_C, N_min_D))

        tile_m = M
        tile_n_size = N // WarpN * WarpN

        epilogue_tile_mn = (tile_m, tile_n_size)
        epi_tiles = product(shape_div(tuple(tile_description.threadblock_shape)[:2], epilogue_tile_mn))

        stages_d = min(epi_tiles, 2)
        reuse_smem_c = (element_c_size > 8)

        if reuse_smem_c:
            stages_c = max(min(epi_tiles, 4), stages_d + 1)
        else:
            stages_c = min(epi_tiles, 4)

        # Record the epilogue tile
        self.cta_tile_mnk = tuple(tile_description.threadblock_shape)
        self.epilogue_tile_mn = epilogue_tile_mn
        self.epi_tiles = epi_tiles
        self.stages_c = stages_c
        self.stages_d = stages_d
        self.reuse_smem_c = reuse_smem_c
        self.element_c = element_c
        self.element_d = element_d
        self.is_source_supported = not DisableSource

    def sm100_epilogue_smem_size(self, tile_description):
        """
        Compute the shared memory size of sm100 collective epilogue
        """
        self.sm100_epilogue_tile(tile_description)
        return self.sm90_or_sm100_epilogue_smem_size(tile_description)

    def __call__(self, tile_description):
        return getattr(self, f"sm{self.cc}_epilogue_smem_size")(tile_description)

    #
    # Helper functions
    #

    @staticmethod
    def get_visitor_size(members: list, ebo: bool):
        """
        Get the size of struct in bytes
        """
        offset = 0
        max_alignment = 1
        if len(members) > 0:
            # Get alignment
            for _, alignment in members:
                max_alignment = max(max_alignment, alignment)

            for type_size, _ in members:
                if type_size != 0:
                    offset = ((offset + max_alignment - 1) // max_alignment) * max_alignment
                if type_size == 0 and not ebo:
                    offset += 1
                else:
                    offset += type_size
            offset = ((offset + max_alignment - 1) // max_alignment) * max_alignment
            return (offset, max_alignment)
        else:
            # Struct size is at least 1
            return (1, 1)

    def get_struct_size(self, members: list):
        """
        Get the size of struct in bytes
        """
        return self.get_visitor_size(members, False)

    def get_evt_smem_type(self, node):
        # Sort the input nodes by edge weight
        input_types = [self.smem_types[child] for child in self.dag_ir.get_all_inputs(node)]
        input_types.append(self.smem_types[node])
        if len(input_types) > 1:
            ebo = len(input_types) > 4
            self.smem_types[node] = self.get_visitor_size(input_types, ebo)

    def get_dag_smem_type(self, node):
        meta = self.dag_ir.get_node_meta(node)
        subgraph = meta.subgraph
        subgraph_nodes = subgraph.nodes_topological_order()
        # Visit the unvisited nodes in subgraph
        for n in subgraph_nodes:
            m = subgraph.get_node_meta(n)
            if m.disabled:
                continue
            else:
                self.smem_types[n] = m.underlying_impl.get_smem_size(
                    self.cta_tile_mnk, self.epilogue_tile_mn,
                    self.stages_c, self.stages_d, self.epi_tiles)
        input_types = [self.smem_types[child] for child in subgraph_nodes[:-1]]
        if len(input_types) > 0:
            ebo = len(input_types) > 4
            self.smem_types[node] = self.get_visitor_size(input_types, ebo)
