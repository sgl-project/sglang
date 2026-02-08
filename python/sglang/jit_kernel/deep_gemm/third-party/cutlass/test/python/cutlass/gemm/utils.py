#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cutlass_library import SubstituteTemplate

import cutlass_cppgen
from cutlass_library import (
    DataTypeNames,
    EpilogueScheduleSuffixes,
    KernelScheduleSuffixes,
    LayoutType,
    OpcodeClassNames,
    ShortDataTypeNames,
    ShortLayoutTypeNames
)
from cutlass_cppgen.backend import library

from gemm_testbed import test_all_gemm


class Layout:
    """
    Utility class to map transpose and non-transpose terminology to row- and column-major terminology
    """

    T = LayoutType.RowMajor
    N = LayoutType.ColumnMajor


class LayoutCombination:
    """
    Utility class defining all combinations of row- and column-major layouts for operands to a GEMMs
    """

    NNN = (Layout.N, Layout.N, Layout.N)
    NNT = (Layout.N, Layout.N, Layout.T)
    NTN = (Layout.N, Layout.T, Layout.N)
    NTT = (Layout.N, Layout.T, Layout.T)
    TNN = (Layout.T, Layout.N, Layout.N)
    TNT = (Layout.T, Layout.N, Layout.T)
    TTN = (Layout.T, Layout.T, Layout.N)
    TTT = (Layout.T, Layout.T, Layout.T)


def get_name(
    layouts,
    alignments,
    element_output,
    element_accumulator,
    element_epilogue,
    cluster_shape,
    threadblock_shape,
    stages,
    element_a,
    element_b,
    element_c,
    arch,
    opclass,
    kernel_schedule=None,
    epilogue_schedule=None,
    suffix="",
):
    """
    Generates a procedural name for a test case.

    :param layouts: indexable container of layouts of A, B, and C operands
    :param alignments: indexable container of alignments of A, B, and C operands
    :param element_output: data type of the output element
    :param element_accumulator: data type used in accumulation
    :param element_epilogue: data type used in computing the epilogue
    :param cluster_shape: indexable container of dimensions of threadblock cluster to be launched
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param element_a: data type of operand A
    :param element_b: data type of operand B
    :param element_c: data type of operand C
    :param arch: compute capability of kernel being generated
    :type arch: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass_cppgen.OpcodeClass
    :param kernel_schedule: kernel_schedule type
    :type kernel_schedule: cutlass_cppgen.KernelScheduleType
    :param epilogue_schedule: epilogue_schedule type
    :type epilogue_schedule: cutlass_cppgen.EpilogueScheduleType
    :param suffix: additional string to add to the suffix of the name
    :type suffix: str

    :return: str
    """
    name_format = "test_SM${arch}_Device_Gemm_${eA}${lA}_${eB}${lB}_${eC}${lC}_${opclass}_${acc}_${tbM}x${tbN}x${tbK}_${cM}x${cN}x${cK}_${stages}_align${aA}-${aB}-${aC}${k}${e}${suffix}"
    return SubstituteTemplate(
        name_format,
        {
            "arch": str(arch),
            "eA": DataTypeNames[element_a],
            "eB": DataTypeNames[element_b],
            "eC": DataTypeNames[element_c],
            "lA": ShortLayoutTypeNames[layouts[0]],
            "lB": ShortLayoutTypeNames[layouts[1]],
            "lC": ShortLayoutTypeNames[layouts[2]],
            "opclass": OpcodeClassNames[opclass],
            "acc": DataTypeNames[element_accumulator],
            "cM": str(cluster_shape[0]),
            "cN": str(cluster_shape[1]),
            "cK": str(cluster_shape[2]),
            "tbM": str(threadblock_shape[0]),
            "tbN": str(threadblock_shape[1]),
            "tbK": str(threadblock_shape[2]),
            "stages": str(stages) if stages is not None else "auto",
            "aA": str(alignments[0]),
            "aB": str(alignments[1]),
            "aC": str(alignments[2]),
            "k": "" if kernel_schedule is None else KernelScheduleSuffixes[kernel_schedule],
            "e": "" if epilogue_schedule is None else EpilogueScheduleSuffixes[epilogue_schedule],
            "suffix": "" if suffix is None else suffix,
        },
    )


def add_test_gemm(
    cls=None,
    cc=None,
    element=None,
    layouts=None,
    alignments=None,
    element_output=None,
    element_accumulator=None,
    cluster_shape=None,
    threadblock_shape=None,
    warp_count=None,
    stages=None,
    opclass=None,
    swizzle=None,
    kernel_schedule=None,
    epilogue_schedule=None,
    compilation_modes=['nvcc', 'nvrtc'],
    element_A=None,
    element_B=None,
    element_C=None):
    """
    Create test-running functions with the given specification and set it as a method of ``cls``.

    :param cls: class to which the generated method will be added
    :type cls: type
    :param cc: compute capability to compile for
    :type cc: int
    :param element: data type of A and B operands
    :type element: cutlass_cppgen.DataType.f16
    :param layouts: layouts of A, B, and C operands
    :type layouts: list or tuple
    :param alignments: alingments of A, B, and C operands
    :type alignments: list or tuple
    :param element_output: data type of the output element
    :type element_output: cutlass_cppgen.DataType
    :param element_accumulator: data type used in accumulation
    :type element_accumulator: cutlass_cppgen.DataType
    :param cluster_shape: dimensions of clusters
    :type cluster_shape: list or tuple
    :param threadblock_shape: dimensions of threadblock tiles
    :type threadblock_shape: list or tuple
    :param warp_count: warps to be launched per threadblock dimension
    :type warp_count: list or tuple
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass_cppgen.OpcodeClass
    :param swizzle: threadblock swizzling functor
    :param kernel_schedule: kernel schedule to use
    :type kernel_schedule: cutlass_cppgen.KernelScheduleType
    :param epilogue_schedule: epilogue schedule to use
    :type epilogue_schedule: cutlass_cppgen.EpilogueScheduleType
    :param compilation_modes: list of compilers to used in testing the kernel (options: 'nvrtc', 'nvcc')
    :type compilation_modes: list,
    :param element_A: data type of operand A. If set, overrides ``element``
    :type element_A: cutlass_cppgen.DataType
    :param element_B: data type of operand B. If set, overrides ``element``
    :type element_B: cutlass_cppgen.DataType
    :param element_C: data type of operand C. If set, overrides ``element``
    :type element_C: cutlass_cppgen.DataType
    """

    if element_A is None:
        element_A = element
    if element_B is None:
        element_B = element
    if element_C is None:
        element_C = element
    if element_output is None:
        element_output = element
    if element_accumulator is None:
        element_accumulator = element

    for compilation_mode in compilation_modes:
        def run(self):
            """
            Dynamically-generated function that constructs a GEMM operation and verifies it against
            multiple test cases.
            """

            layout_A, layout_B, layout_C = layouts
            alignment_A, alignment_B, alignment_C = alignments

            plan = cutlass_cppgen.op.Gemm(element_A=element_A, element_B=element_B,
                                element_C=element_C, element_D=element_output,
                                layout_A=layout_A, layout_B=layout_B, layout_C=layout_C,
                                element_accumulator=element_accumulator,
                                kernel_cc=cc)

            plan.opclass = opclass
            if swizzle is not None:
                plan.swizzling_functor = swizzle

            td = plan.tile_descriptions()[0]

            if warp_count is not None:
                td.warp_count = warp_count
            td.threadblock_shape = threadblock_shape
            td.stages = stages
            td.cluster_shape = cluster_shape
            op = plan.construct(tile_description=td, alignment_A=alignment_A, alignment_B=alignment_B, alignment_C=alignment_C)
            self.assertTrue(test_all_gemm(op, 'universal', compilation_mode=compilation_mode))

        element_epilogue = element_accumulator
        name = get_name(
            layouts=layouts, alignments=alignments, element_output=element_output, element_accumulator=element_accumulator,
            element_epilogue=element_epilogue, cluster_shape=cluster_shape, threadblock_shape=threadblock_shape,
            stages=stages, element_a=element_A, element_b=element_B, element_c=element_C, arch=cc, opclass=opclass,
            kernel_schedule=kernel_schedule, epilogue_schedule=epilogue_schedule, suffix=f'_{compilation_mode}')

        setattr(cls, name, run)
