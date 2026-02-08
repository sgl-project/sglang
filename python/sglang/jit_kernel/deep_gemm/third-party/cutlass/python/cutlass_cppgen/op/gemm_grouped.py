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
    Ease-of-use interface for constructing, compiling, and running GEMMs.

    The ``GroupedGemm`` interface is meant to allow one to easily instantiate, compile, and run
    grouped GEMM operations in CUTLASS via Python, without specifying many configuration parameters.
    Under the hood, the interface will select sensible default parameters for the many template
    parameters for CUTLASS grouped GEMMs.

    Note: optimal performance is not to be expected from this interface. To achieve optimal
    performance, one should specify and tune each configuration parameter.

    The simplest example of using this interface is the following:

    .. highlight:: python
    .. code-block:: python

        # As, Bs, Cs, and Ds are torch/numpy/cupy tensor objects
        plan = cutlass_cppgen.op.GroupedGemm(element=cutlass_cppgen.DataType.f16, layout=cutlass_cppgen.LayoutType.RowMajor)
        plan.run([A0, A1], [B0, B1], [C0, C1], [D0, D1])
"""
from __future__ import annotations
from typing import Optional
from cutlass_library import DataTypeSize

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
from cutlass_cppgen.backend.gemm_operation import (
    GemmGroupedArguments,
    GemmOperationGrouped,
)
from cutlass_cppgen.backend.library import (
    SchedulerMode,
    TensorDescription,
    TileDescription,
)
from cutlass_cppgen.op.gemm import Gemm
from cutlass_cppgen.shape import GemmCoord
from cutlass_cppgen.utils import check, datatypes


class GroupedGemm(Gemm):
    """
    Constructs a ``GroupedGemm`` object.

    The data types and layouts of operands A, B, and C, along with the data type of output D
    and that used for accumulation, are bound to the ``GroupedGemm`` object throughout its lifetime --
    these are not to be changed after a ``GroupedGemm`` has been constructed.

    The constructor has optional parameters for flexibly setting these parameters. Please see the constructor
    for ``Gemm`` for examples of these.

    :param cc: compute capability of device to generate kernels for
    :type cc: int
    :param A: tensor representing data type and layout of operands A
    :param B: tensor representing data type and layout of operands B
    :param C: tensor representing data type and layout of operands C
    :param D: tensor representing data type and layout of operands D
    :param alpha: scalar paramter alpha from GEMM computation that scales the product of operands A and B
    :param beta: scalar parameter beta from GEMM operation that scales operand C
    :param element_accumulator: data type to be used in accumulation of the product of operands A and B
    :type element_accumulator: cutlass_cppgen.DataType
    :param element: generic data type to be used for operands A, B, C, D, as well as the accumulation data type
    :type element: cutlass_cppgen.DataType
    :param layout: generic layout type to be used for operands A, B, C, and D
    :type layout: cutlass_cppgen.LayoutType
    :param element_A: data type to be used for operand A
    :type element_A: cutlass_cppgen.DataType
    :param element_B: data type to be used for operand B
    :type element_B: cutlass_cppgen.DataType
    :param element_C: data type to be used for operand C
    :type element_C: cutlass_cppgen.DataType
    :param element_D: data type to be used for operand D
    :type element_D: cutlass_cppgen.DataType
    :type layout_A: layout of operand A
    :param layout_A: cutlass_cppgen.LayoutType
    :type layout_B: layout of operand B
    :param layout_B: cutlass_cppgen.LayoutType
    :type layout_C: layout of operand C
    :param layout_C: cutlass_cppgen.LayoutType
    :type layout_D: layout of operand D
    :param layout_D: cutlass_cppgen.LayoutType
    """

    def __init__(
        self, A=None, B=None, C=None, D=None,
        alpha=1.0, beta=0.0, element_accumulator=None,
        element=None, layout=None,
        element_A=None, element_B=None, element_C=None, element_D=None,
        layout_A=None, layout_B=None, layout_C=None,
        cc: int = None,
    ):
        super().__init__(
            A=A, B=B, C=C, D=D,
            alpha=alpha, beta=beta,
            element_accumulator=element_accumulator,
            element=element, layout=layout,
            element_A=element_A, element_B=element_B,
            element_C=element_C, element_D=element_D,
            layout_A=layout_A, layout_B=layout_B, layout_C=layout_C,
            cc=cc
        )

        # Grouped GEMM specializations for SM90 are currently unavailable. Revert to using SM80
        if self.current_cc in [90, 100, 101, 103]:
            self._reset_options(80)
            self._reset_operations(reset_epilogue=False)

        self.name = "grouped_gemm"

    @Gemm.swizzling_functor.setter
    def swizzling_functor(self, swizzling_functor):
        """
        Sets the swizzling functor to the type specified by `swizzling_functor`
        """
        raise Exception('Grouped GEMM does not currently support different swizzling functors')

    def construct(self, tile_description: TileDescription = None,
                  alignment_A: int = None,
                  alignment_B: int = None,
                  alignment_C: int = None) -> GemmOperationGrouped:
        """
        Constructs a ``cutlass_cppgen.backend.GemmOperationGrouped`` based on the input parameters and current
        kernel specification of the ``Gemm`` object.

        :param tile_description: tile description specifying shapes and operand types to use in the kernel
        :type tile_description: cutlass_cppgen.backend.TileDescription
        :param alignment_A: alignment of operand A
        :type alignment_A: int
        :param alignment_B: alignment of operand B
        :type alignment_B: int
        :param alignment_C: alignment of operand C
        :type alignment_C: int

        :return: operation that was constructed
        :rtype: cutlass_cppgen.backend.GemmOperationGrouped
        """
        alignment_A = check.alignment_or_default(alignment_A, max(self.possible_operations.alignments("A")))
        alignment_B = check.alignment_or_default(alignment_B, max(self.possible_operations.alignments("B")))
        alignment_C = check.alignment_or_default(alignment_C, max(self.possible_operations.alignments("C")))

        self.epilogue_functor = self._reset_epilogue_functor_alignment(alignment_C, self.epilogue_functor)

        tensor_A = TensorDescription(self._element_a, self._layout_b, alignment_A)
        tensor_B = TensorDescription(self._element_b, self._layout_b, alignment_B)
        tensor_C = TensorDescription(self._element_c, self._layout_c, alignment_C)

        if tile_description is None:
            op = self.possible_operations.operations(alignment_A, alignment_B, alignment_C, self._math_operation)[0]
            tile_description = datatypes.td_from_profiler_op(op)
        else:
            valid, err_str = self._valid_tile_description(tile_description)
            if not valid:
                raise Exception(f"Invalid tile description. {err_str}")
            self.tile_description = tile_description

        operation = GemmOperationGrouped(
            arch=self.current_cc,
            tile_description=tile_description,
            A=tensor_A, B=tensor_B, C=tensor_C,
            epilogue_functor=self.epilogue_functor,
            swizzling_functor=self._swizzling_functor,
            precompute_mode=SchedulerMode.Device)

        return operation

    def run(self, A, B, C, D,
            alpha=None, beta=None, sync: bool = True,
            print_module: bool = False,
            stream: Optional[cuda.CUstream] = None) -> GemmGroupedArguments:
        """
        Runs the kernel currently specified.

        By default, this call returns only once the kernel has completed. To launch the kernel
        and immediately return, set ``sync=False``. In this case, it is the responsibility of the
        caller to syncrhonize the results of the kernel before attempting to access outputs
        by calling ``sync()`` on the arguments returned from this call.

        :param A: list of tensors representing data type and layout of operand A
        :type A: list
        :param B: list of tensors representing data type and layout of operand B
        :type B: list
        :param C: list of tensors representing data type and layout of operand C
        :type C: list
        :param D: list of tensors representing data type and layout of operand D
        :type D: list
        :param alpha: scalar paramter alpha from GEMM computation that scales the product of operands A and B
        :param beta: scalar parameter beta from GEMM operation that scales operand C
        :param sync: whether the call should wait for the kernel to complete before returning
        :type sync: bool
        :param print_module: whether to print the emitted C++ code
        :type print_module: bool
        :param stream: cuda stream, defaults to cuda.cuda.CUstream(0)
        :type stream: :class:`cuda.cuda.CUstream`

        :return: arguments passed in to the kernel
        :rtype: cutlass_cppgen.backend.GemmGroupedArguments
        """
        if not stream:
            stream = cuda.CUstream(0)

        super().run_setup()

        if len(A) != len(B) or len(A) != len(C) or len(A) != len(D):
            raise Exception("Lengths of A, B, C, and D lists must be equal")

        problem_sizes = []
        As, Bs, Cs, Ds = ([None] * len(A) for _ in range(4))
        for i in range(len(A)):
            As[i] = self._verify_tensor(A[i], self.A, self._element_a, self._layout_a, "A")
            Bs[i] = self._verify_tensor(B[i], self.B, self._element_b, self._layout_b, "B")
            Cs[i] = self._verify_tensor(C[i], self.C, self._element_c, self._layout_c, "C")
            Ds[i] = self._verify_tensor(D[i], self.D, self._element_d, self._layout_d, "D")
            problem_sizes.append(GemmCoord(A[i].shape[0], B[i].shape[1], A[i].shape[1]))

        alpha = self._verify_scalar(alpha, self.alpha, self._element_c, "alpha")
        beta = self._verify_scalar(beta, self.beta, self._element_c, "beta")

        alignment_a = min((self.possible_operations.find_alignment(A.shape, self._layout_a, operand="A") for A in As))
        alignment_b = min((self.possible_operations.find_alignment(B.shape, self._layout_b, operand="B") for B in Bs))
        alignment_c = min((self.possible_operations.find_alignment(C.shape, self._layout_c, operand="C") for C in Cs))
        self.compile(self.tile_description, alignment_A=alignment_a, alignment_B=alignment_b,
                     alignment_C=alignment_c, print_module=print_module)

        arguments = GemmGroupedArguments(
            operation=self.operation,
            problem_sizes=problem_sizes,
            A=As, B=Bs, C=Cs, D=Ds,
            output_op=self.operation.epilogue_type(alpha, beta),
            stream=stream
        )

        self.operation.run(arguments)

        if sync:
            arguments.sync()

        return arguments
