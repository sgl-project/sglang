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
    Ease-of-use interface for constructing, compiling, and running CONVs

    The ``Conv2d`` interface is meant to allow one to easily instantiate, compile, and run
    CONV2D operations in CUTLASS via Python, without specifying many configuration parameters.
    Under the hood, the interface will select sensible default parameters for the many template
    parameters for CUTLASS CONVs.

    Note: optimal performance is not to be expected from this interface. To achieve optimal
    performance, one should specify and tune each configuration parameter.

    The simplest example of using this interface is the following:

    .. highlight:: python
    .. code-block:: python

        # A, B, C, and D are torch/numpy/cupy tensor objects
        plan = cutlass_cppgen.op.Conv(A, B, C, D)
        plan.run(stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    One can also use the interface by specifying data types of operands at construction
    and using different tensor objects with these data types at runtime:

    .. highlight:: python
    .. code-block:: python

        # The following is shorthand for:
        #        cutlass_cppgen.op.Conv2d(kind="fprop",
        #                          element_A=torch.float32, element_B=torch.float32,
        #                          element_C=torch.float32, element_D=torch.float32,
        #                          element_accumulator=torch.float32)
        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=torch.float32)

        A0 = torch.rand((128, 256), dtype=torch.float32, device='cuda')
        B0 = torch.rand((256, 64), dtype=torch.float32, device='cuda')
        C0 = torch.zeros((128, 64), dtype=torch.float32, device='cuda')
        D0 = torch.zeros((128, 64), dtype=torch.float32, device.'cuda')
        plan.run(A0, B0, C0, D0, stride=(1, 1), padding=(0, 0), dilation=(1, 1))

        A = torch.rand((32, 128), dtype=torch.float32, device='cuda')
        B = torch.rand((128, 256), dtype=torch.float32, device='cuda')
        C = torch.zeros((32, 256), dtype=torch.float32, device='cuda')
        D = torch.zeros((32, 256), dtype=torch.float32, device.'cuda')
        plan.run(A1, B1, C1, D1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    The interface additionally enables one to decouple the compilation of the underlying CUTLASS
    kernel from its execution:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=np.float32)

        # Do other work...

        plan.run(A0, B0, C0, D0, stride=(1, 1), padding=(0, 0), dilation=(1, 1))

        # Do other work...

        plan.run(A1, B1, C1, D1, stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    Elementwise activation functions are easily fused to the GEMM via the interface:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=np.float32)
        plan.activation = cutlass_cppgen.epilogue.relu

    Operations can also be run asynchronously:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Conv2d(kind="fprop", element=np.float32)
        args = plan.run()

        # Do other work...

        args.sync()
"""

from __future__ import annotations
from typing import Optional
from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
cudart =  lazy_import("cuda.cudart")
from cutlass_library import (
    ConvKind,
    ConvMode,
    DataTypeSize,
    IteratorAlgorithm,
    OperationKind,
    SplitKMode,
    StrideSupport,
)

import cutlass_cppgen
from cutlass_cppgen import epilogue
from cutlass_cppgen.backend import compiler
from cutlass_cppgen.backend.conv2d_operation import Conv2dArguments, Conv2dOperation
from cutlass_cppgen.backend.reduction_operation import ReductionOperation, ReductionArguments
from cutlass_cppgen.backend.library import TensorDescription, TileDescription
from cutlass_cppgen.op.op import OperationBase
from cutlass_cppgen.shape import Conv2DProblemSize, MatrixCoord
from cutlass_cppgen.utils import check, datatypes


class Conv2d(OperationBase):
    """
    Constructs a ``Conv2d`` object.

    The convolution kind (fprop, wgrad, degrad), the data types of operands A, B, and C,
    along with the data type of output D and that used for accumulation, are bound to the ``Conv``
    object throughout its lifetime -- these are not to be changed after a ``Conv2d`` has been constructed.

    The constructor has optional parameters for flexibly setting these parameters. The following
    constructors are equivalent:

    .. highlight:: python
    .. code-block:: python

        # Use F32 for A, B, C, D, and accumulation in fprop

        # Use the generic ``element`` parameter to concisely set all data types for operands to the same values.
        Conv2d(kind="fprop", element=cutlass_cppgen.DataType.f32)

        # Explicitly specify the data types to use for A, B, C, and D.
        Conv2d(kind="fprop", element_A=cutlass_cppgen.DataType.f32, element_B=cutlass_cppgen.DataType.f32,
            element_C=cutlass_cppgen.DataType.f32, element_D=cutlass_cppgen.DataType.f32)

        # Set the data types and elements from existing tensors. Note that one can use different tensors when
        # executing GEMM via the ``run()`` method than passed in here (though those passed in to ``run()`` must
        # have the same data type as those passed in here).
        # A, B, C, and D are torch.Tensor objects of type torch.float32 under the channel-last layout
        Conv2d(kind="fprop", A=A, B=B, C=C, D=D)

        # Explicitly specify the data type for only some of A, B, C, and D. Unspecified data types will inherit
        # those passed in via the generic ``element``
        Conv2d(kind="fprop", element_A=cutlass_cppgen.DataType.f32, element_accumulator=cutlass_cppgen.DataType.f32,
            element=cutlass_cppgen.DataType.f32)

    The order of precedence for the setting of the data type for a given operand/output is as follows:
        1) If the tensor type is specified (e.g., ``A``), use the data type inferred from this tensor
        2) Otherwise, if the data type (e.g., ``element_A``) is specified, use those
        3) Otherwise, use the generic values (e.g., ``element``)

    :param kind: the convolution kind (i.e. fprop, wgrad, and dgrad)
    :type kind: str
    :param A: tensor representing data type of operand A
    :param B: tensor representing data type of operand B
    :param C: tensor representing data type of operand C
    :param D: tensor representing data type of operand D
    :param alpha: scalar paramter alpha from GEMM computation that scales the product of operands A and B
    :param beta: scalar parameter beta from GEMM operation that scales operand C
    :param element: generic data type to be used for operands A, B, C, D, as well as the accumulation data type
    :type element: cutlass_cppgen.DataType
    :param element_A: data type to be used for operand A
    :type element_A: cutlass_cppgen.DataType
    :param element_B: data type to be used for operand B
    :type element_B: cutlass_cppgen.DataType
    :param element_C: data type to be used for operand C
    :type element_C: cutlass_cppgen.DataType
    :param element_D: data type to be used for operand D
    :type element_D: cutlass_cppgen.DataType
    :param element_accumulator: data type to be used in accumulation of the product of operands A and B
    :type element_accumulator: cutlass_cppgen.DataType
    :param cc: compute capability of device for which kernels should be compiled. For example, if running on H100, this should be set to 90
    :type cc: int
    :param kernel_cc: compute capability of kernels to generate. For example, if running on SM90, but desiring to use a CUTLASS 2.x-style Ampere kernel, this should be set to 80
    :type kernel_cc: int
    """
    def __init__(
        self, kind="fprop",
        A=None, B=None, C=None, D=None, alpha=1.0, beta=0.0,
        element=None,
        element_A=None, element_B=None, element_C=None, element_D=None,
        element_accumulator=None,
        cc: int = None, kernel_cc: int = None
    ):
        super().__init__(cc=cc, kernel_cc=kernel_cc, operation_kind=OperationKind.Conv2d)
        # Verify the kernel cc
        if self.current_cc in [90, 100, 101, 103]:
            # The Conv2d kernel on Hopper (SM90) is currently unsupported
            # Revert to use SM80-tagged kernels
            cutlass_cppgen.logger.warning("Reverting to using SM80-tagged kernel. Opclass may change.")
            self.specified_kernel_cc = 80
            self._reset_options(80)

        # The arch is used in testing
        self.arch = self.current_cc
        self.name = "conv2d" + kind

        # The convolution kind. (concept: cutlass_library.library.ConvKind)
        self.conv_kind = datatypes.getattr_enum(ConvKind, kind)

        # The element types (concept: cutlass library types) of A, B, C, and D
        elements = []
        layouts = []

        # Complete the data types based on user-provided arguments
        for elt, tens, name in zip([element_A, element_B, element_C, element_D],
                                   [A, B, C, D],
                                   ["A", "B", "C", "D"]):
            if elt is not None and tens is not None:
                raise Exception(f'Must not specify both element_{name} and tensor {name}')
            if elt is None and tens is None and element is None:
                raise Exception(f'Must specify one of element_{name}, tensor {name}, or generic element.')

            elt_to_set = None
            lay_to_set = None

            if tens is not None:
                elt_to_set, _ = datatypes.get_datatype_and_layout(tens)
            else:
                elt_to_set = elt if elt is not None else element

            assert elt_to_set is not None

            # Currently we only support layout TensorNHWC
            lay_to_set = cutlass_cppgen.LayoutType.TensorNHWC
            elements.append(datatypes.library_type(elt_to_set))
            layouts.append(lay_to_set)

        self._element_a, self._element_b, self._element_c, self._element_d = elements
        self._layout_a, self._layout_b, self._layout_c, self._layout_d = layouts

        self.A, self.B, self.C, self.D, self.alpha, self.beta = A, B, C, D, alpha, beta

        if element_accumulator is None:
            self._element_accumulator = self._element_c
        else:
            self._element_accumulator = datatypes.library_type(element_accumulator)

        # Default inputs if none is supplied in run()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.alpha = alpha
        self.beta = beta

        # We only specify the stride of the swizzling functor here
        # The actual swizzling functor is determined in run based on conv_kind and stride
        self._swizzling_stride = 1

        # Arguments that will be set to default value in _reset_operations
        # The default tile_description and op_class are fetched from manifest of cutlass library
        self._tile_description = None
        self.op_class = None
        # The default identity epilogue will be created
        self.epilogue_functor = None

        self._reset_operations()

        # Arguments that will be determined online based on arguments of "run"
        # based on stride, input/output channels, alignment, and conv_kind
        self._iterator_algorithm = None
        self._stride_support = None

    def _reset_operations(self, reset_epilogue: bool = True):
        # Set the default op class
        datatype_comb = (self._element_a, self._element_b, self._element_accumulator)
        layout_comb = (self._layout_a, self._layout_b)

        self.possible_op_classes = self.options.supporting_opclasses(
            self._element_a, self._element_b, self._element_accumulator,
            self._layout_a, self._layout_b, self._math_operation
        )

        if cutlass_cppgen.OpcodeClass.TensorOp in self.possible_op_classes:
            self.opclass = cutlass_cppgen.OpcodeClass.TensorOp
        elif cutlass_cppgen.OpcodeClass.Simt in self.possible_op_classes:
            self.opclass = cutlass_cppgen.OpcodeClass.Simt
        else:
            if self._math_operation is not None:
                math_op_str = f' and math operation {self._math_operation}'
            else:
                math_op_str = ''

            raise Exception(f'No kernel configuration found for supported data type and layout '
                            f'combination {datatype_comb}x{layout_comb}{math_op_str}')

        if reset_epilogue:
            self._reset_epilogue_functor_activation(epilogue.identity)

        self.alignment_pref_A = min(
            128 // DataTypeSize[self._element_a], max(self.possible_operations.alignments("A")))
        self.alignment_pref_B = min(
            128 // DataTypeSize[self._element_b], max(self.possible_operations.alignments("B")))
        self.alignment_pref_C = min(
            128 // DataTypeSize[self._element_c], max(self.possible_operations.alignments("C")))

    #
    # Tile description Related
    #

    @property
    def tile_description(self) -> TileDescription:
        """
        Returns the tile description
        """
        return self._tile_description

    @tile_description.setter
    def tile_description(
        self, td=None):
        """
        Set the tile description

        :param td: tile description
        :type td: cutlass_cppgen.backend.TileDescription, or a dict with keys
                  {
                      "threadblock_shape": [int, int, int],
                      "warp_count": [int, int, int],
                      "stages": int,
                      "instruction_shape": [int, int, int] (optional),
                      "cluster_shape": [int, int, int] (optional)
                  }
        """
        if td is None:
            return
        if isinstance(td, dict):
            if self._tile_description is None:
                op = self.possible_operations.default_operation(self._math_operation)
                self._tile_description = datatypes.td_from_profiler_op(op)
            if "cluster_shape" in td.keys():
                if td["cluster_shape"] != [1, 1, 1]:
                    cutlass_cppgen.logger.warning("Conv2d currently only support 'cluster_shape'=[1, 1, 1]'.")
                    td["cluster_shape"] = [1, 1, 1]
            td = self._tile_description.clone_and_update(td)

        valid, msg = self._valid_tile_description(td)
        if valid:
            self._tile_description = td
        else:
            raise Exception(msg)

    def _valid_tile_description(self, td: TileDescription) -> tuple:
        """
        Checks whether the provided tile description is valid for the given compute capability. At present,
        this checks the following:

        - Does the tile description use a number of stages supported by the compute capability in question?
        - Does the tile size requested fit within shared memory?
        - Are cluster dimensions outside the valid range requested for a given architecture (e.g.,
          more non-unit cluster dimensions for pre-SM90 architectures)?
        - Is the kernel schedule being used supported on the architecture in question?

        :param td: tile description to validate
        :type td: cutlass_cppgen.backend.TileDescription
        :return: tuple in which the first element is a bool indicating that the tile description is valid
                 and the second element is a string providing an optional error message.
        :rtype: tuple
        """
        valid, msg = check.valid_stage_count(self.cc, self.current_cc, td)
        if not valid:
            return (valid, msg)

        valid, msg = check.valid_cluster_shape(self.current_cc, td.cluster_shape)
        if not valid:
            return (valid, msg)

        return valid, msg

    def tile_descriptions(self) -> list:
        """
        Returns a list of valid tile descriptions for the operations

        :returns: list of valid tile descriptions for the operations
        :rtype: list
        """
        descriptions = []
        description_str = []
        for op in self.possible_operations.all_operations:
            td = datatypes.td_from_profiler_op(op)

            if self._math_operation is not None:
                if td.math_instruction.math_operation != self._math_operation:
                    continue

            if str(td) not in description_str:
                description_str.append(str(td))
                descriptions.append(td)
        return descriptions

    #
    # Swizzling functor Related
    #

    @property
    def swizzling_stride(self):
        """
        Returns the stride of swizzling currently being used by the Conv2d

        :return: swizzing stride
        """
        return self._swizzling_stride

    @swizzling_stride.setter
    def swizzling_stride(self, stride: int):
        """
        Sets the swizzling functor to the type specified by `swizzling_functor`
        """
        if not isinstance(stride, int):
            raise Exception(f"Expect integer (1, 2, 4, 8), got {stride}")
        self._swizzling_stride = stride

    def _propose_swizzling_functor(self, stride):
        """
        Automatically propose the swizzling functor based on the stride
        """
        if self.conv_kind == ConvKind.Dgrad:
            if stride[0] != 1 or stride[1] != 1:
                return getattr(cutlass_cppgen.swizzle, f"StridedDgradIdentitySwizzle{self._swizzling_stride}")

        return getattr(cutlass_cppgen.swizzle, f"IdentitySwizzle{self._swizzling_stride}")

    #
    # Iterator Algorithm Related
    #

    @property
    def iterator_algorithm(self) -> IteratorAlgorithm:
        """
        Returns the iterator algorithm
        """
        return self._iterator_algorithm

    @iterator_algorithm.setter
    def iterator_algorithm(self, alg: str):
        """
        Sets the iterator algorithm

        :param alg: The iterator algorithm
        :type td: string, options: "analytic", "optimized", "few_channels", and "fixed_channels"
        """
        iterator_alg = datatypes.getattr_enum(IteratorAlgorithm, alg)

        # Check if the iterator algorithm is valid
        if iterator_alg in [IteratorAlgorithm.FewChannels, IteratorAlgorithm.FixedChannels] and self.conv_kind != ConvKind.Fprop:
            raise Exception(f"{self.conv_kind} does not support iterator algorithm {alg}.")

        self._iterator_algorithm = iterator_alg

    def _propose_iterator_algorithm(self, problem_size, alignment_a, alignment_b) -> IteratorAlgorithm:
        """
        Propose a valid iterator algorithm based on problem size and alignment
        """
        if self.conv_kind == ConvKind.Fprop:
            # Check whether the fixed channel is applicable
            if problem_size.C == alignment_a:
                return IteratorAlgorithm.FixedChannels
            elif (problem_size.C % alignment_a == 0 and
                  problem_size.R <= 32 and problem_size.S <= 32):
                return IteratorAlgorithm.Optimized
            else:
                return IteratorAlgorithm.Analytic
        elif self.conv_kind == ConvKind.Dgrad:
            if (problem_size.K % alignment_a == 0 and
                problem_size.R <= 32 and problem_size.S <= 32 and
                problem_size.C % alignment_b == 0):
                return IteratorAlgorithm.Optimized
            else:
                return IteratorAlgorithm.Analytic
        elif self.conv_kind == ConvKind.Wgrad:
            if (problem_size.K % alignment_a == 0 and
                problem_size.C % alignment_b == 0):
                return IteratorAlgorithm.Optimized
            else:
                return IteratorAlgorithm.Analytic

    def _validate_iterator_algorithm(self, iterator_algorithm, problem_size, alignment_a, alignment_b) -> bool:
        """
        Validate whether the user provide iterator algorithm works for the given problem size
        """
        if self.conv_kind == ConvKind.Fprop:
            if iterator_algorithm == IteratorAlgorithm.FixedChannels:
                return problem_size.C == alignment_a
            elif iterator_algorithm == IteratorAlgorithm.Optimized:
                return (problem_size.C % alignment_a == 0 and
                  problem_size.R <= 32 and problem_size.S <= 32)
            elif iterator_algorithm == IteratorAlgorithm.FewChannels:
                return problem_size.C % alignment_a == 0
        elif self.conv_kind == ConvKind.Dgrad:
            if iterator_algorithm == IteratorAlgorithm.Optimized:
                return (problem_size.K % alignment_a == 0 and
                        problem_size.R <= 32 and problem_size.S <= 32 and
                        problem_size.C % alignment_b == 0)
        elif self.conv_kind == ConvKind.Wgrad:
            if iterator_algorithm == IteratorAlgorithm.Optimized:
                return (problem_size.K % alignment_a == 0 and
                problem_size.C % alignment_b == 0)

        return True

    #
    # Stride Support Related
    #

    def _propose_stride_support(self, stride):
        if self.conv_kind == ConvKind.Dgrad:
            if stride[0] == 1 and stride[1] == 1:
                return StrideSupport.Unity

        return StrideSupport.Strided

    #
    # Construct and Compilation
    #

    def construct(
        self, tile_description: TileDescription = None,
        alignment_A: int = None, alignment_B: int = None, alignment_C: int = None,
        iterator_algorithm: IteratorAlgorithm = None,
        stride_support = None, swizzling_functor: cutlass_cppgen.swizzle = None,
        epilogue_functor=None) -> cutlass_cppgen.backend.Conv2dOperation:
        """
        Constructs a ``cutlass_cppgen.backend.Conv2dOperation`` based on the input parameters and current
        kernel specification of the ``Conv2d`` object.

        :param tile_description: tile description specifying shapes and operand types to use in the kernel
        :type tile_description: cutlass_cppgen.backend.TileDescription
        :param alignment_A: alignment of operand A
        :type alignment_A: int
        :param alignment_B: alignment of operand B
        :type alignment_B: int
        :param alignment_C: alignment of operand C
        :type alignment_C: int
        :param iterator_algorithm: the iterator algorithm used
        :type iterator_algorithm: cutlass_library.library.IteratorAlgorithm
        :param stride_support: the stride support of dgrad
        :type stride_support: cutlass_library.library.StrideSupport
        :param swizzling_functor: the swizzling functor
        :type swizzling_functor: cutlass_cppgen.swizzle
        :param epilogue_functor: the epilogue functor

        :return: operation that was constructed
        :rtype: cutlass_cppgen.backend.Conv2dOperation
        """
        # Get alignment
        alignment_A = check.alignment_or_default(alignment_A, self.alignment_pref_A)
        alignment_B = check.alignment_or_default(alignment_B, self.alignment_pref_B)
        alignment_C = check.alignment_or_default(alignment_C, self.alignment_pref_C)

        tensor_A = TensorDescription(self._element_a, self._layout_b, alignment_A)
        tensor_B = TensorDescription(self._element_b, self._layout_b, alignment_B)
        tensor_C = TensorDescription(self._element_c, self._layout_c, alignment_C)

        if tile_description is None:
            if self.tile_description is not None:
                tile_description = self.tile_description
            else:
                op = self.possible_operations.operations(alignment_A, alignment_B, alignment_C, self._math_operation)[0]
                tile_description = datatypes.td_from_profiler_op(op)
        else:
            valid, err_str = self._valid_tile_description(tile_description)
            if not valid:
                raise Exception(f"Invalid tile description. {err_str}")
            self.tile_description = tile_description

        if iterator_algorithm is None:
            # If the iterator algorithm is already set
            if self.iterator_algorithm is not None:
                iterator_algorithm = self.iterator_algorithm
            else:
                # Otherwise, we conservatively use the analytic iterator for correctness
                iterator_algorithm = IteratorAlgorithm.Analytic

        if stride_support is None:
            # If the stride support is already set
            if self._stride_support is not None:
                stride_support = self._stride_support
            else:
                # Otherwise, we assume strided
                stride_support = StrideSupport.Strided

        if swizzling_functor is None:
            # If the swizzling functor is already set
            swizzling_functor = self._propose_swizzling_functor(stride=(2, 2))

        if epilogue_functor is None:
            if self.epilogue_functor is not None:
                epilogue_functor = self.epilogue_functor
            else:
                epilogue_functor = self._create_epilogue_functor_activation(self._activation)

        # Reset the alignment of the epilogue functor
        epilogue_functor = self._reset_epilogue_functor_alignment(alignment_C, epilogue_functor)

        operation = Conv2dOperation(
            conv_kind=self.conv_kind,
            iterator_algorithm=iterator_algorithm,
            arch=self.current_cc,
            tile_description=tile_description,
            A=tensor_A, B=tensor_B, C=tensor_C,
            stride_support=stride_support,
            epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor,
        )

        return operation

    def compile(self, tile_description: TileDescription = None,
                alignment_A: int = None, alignment_B: int = None, alignment_C: int = None,
                iterator_algorithm: IteratorAlgorithm = None,
                stride_support = None, swizzling_functor: cutlass_cppgen.swizzle = None,
                epilogue_functor = None, print_module: bool = False) -> cutlass_cppgen.backend.Conv2dOperation:
        """
        Emits and compiles the kernel currently specified. If ``tile_description`` and any
        of the ``alignment`` parameters are set, the kernel will be chosen using this
        tile description and alignments. Otherwise, a default tile description and alignment
        will be used.

        ::param tile_description: tile description specifying shapes and operand types to use in the kernel
        :type tile_description: cutlass_cppgen.backend.TileDescription
        :param alignment_A: alignment of operand A
        :type alignment_A: int
        :param alignment_B: alignment of operand B
        :type alignment_B: int
        :param alignment_C: alignment of operand C
        :type alignment_C: int
        :param iterator_algorithm: the iterator algorithm used
        :type iterator_algorithm: cutlass_library.library.IteratorAlgorithm
        :param stride_support: the stride support of dgrad
        :type stride_support: cutlass_library.library.StrideSupport
        :param swizzling_functor: the swizzling functor
        :type swizzling_functor: cutlass_cppgen.swizzle
        :param epilogue_functor: the epilogue functor

        :return: operation that was compiled
        :rtype: cutlass_cppgen.backend.Conv2dOperation
        """

        self.operation = self.construct(
            tile_description, alignment_A, alignment_B, alignment_C,
            iterator_algorithm, stride_support, swizzling_functor, epilogue_functor)

        if print_module:
            print(self.operation.rt_module.emit())

        compiler.add_module([self.operation,])
        return self.operation

    #
    # Run Related
    #

    def _verify_type_and_layout(self, tensor, ref_type, ref_layout, name):
        """
        Verifies that ``tensor`` has data type ``ref_type`` and layout ``ref_layout``. An exception
        is raised if it does not.

        :param tensor: object representing a tensor passed in to verify, or ``None`` if no tensor was passed in
        :type tensor: numpy/cupy/torch array/tensor object
        :param ref_dtype: data type for the tensor that this object was initialized to
        :param name: identifier of the tensor to verify. Used in raising exceptions
        :type name: str
        """
        dtype, _ = datatypes.get_datatype_and_layout(tensor)
        if dtype != ref_type:
            raise Exception(f'Tensor {name} with type and layout {dtype} '
                            f'does not match the expected type of {ref_type}.')

    def _get_and_verify_conv_problem_size(self, A, B, C, stride, padding, dilation):
        if self.conv_kind == ConvKind.Fprop:
            input = A
            weight = B
            output = C
            output_tensor = "C"
        elif self.conv_kind == ConvKind.Dgrad:
            output = A
            weight = B
            input = C
            output_tensor = "A"
        elif self.conv_kind == ConvKind.Wgrad:
            output = A
            input = B
            weight = C
            output_tensor = "A"
        else:
            raise Exception(f"Convolution kind {self.conv_kind} is not supported")

        N_, H_, W_, C_ = datatypes.get_tensor_shape(input, op="CONV")
        K_, R_, S_, _ = datatypes.get_tensor_shape(weight, op="CONV")
        _, P_, Q_, _ = datatypes.get_tensor_shape(output, op="CONV")

        problem_size = Conv2DProblemSize(
            N_, H_, W_, C_,
            K_, R_, S_, C_,
            padding[0], padding[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            ConvMode.CrossCorrelation,
            1, 1
        )

        if P_ != problem_size.P or Q_ != problem_size.Q:
            raise Exception(
                f"Tensor {output_tensor} size should be ({N_}, {problem_size.P}, {problem_size.Q}, {K_}), got ({N_}, {P_}, {Q_}, {K_})")

        return problem_size

    def run(self, A=None, B=None, C=None, D=None,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1),
            alpha=None, beta=None,
            split_k=("serial", 1), sync: bool = True,
            print_module: bool = False,
            stream: Optional[cuda.CUstream] = None) -> Conv2dArguments:
        """
        Runs the kernel currently specified. If it has not already been, the kernel is emitted and
        compiled. Tensors holding operands and outputs of the kernel are sourced either from the
        ``A``, ``B``, ``C``, ``D``, ``alpha``, and ``beta``
        parameters provided in the call, or from those
        passed in on the construction of this object -- one of the two must be specified.

        By default, this call returns only once the kernel has completed. To launch the kernel
        and immediately return, set ``sync=False``. In this case, it is the responsibility of the
        caller to syncrhonize the results of the kernel before attempting to access outputs
        by calling ``sync()`` on the arguments returned from this call.

        :param A: tensor representing data type and layout of operand A
        :param B: tensor representing data type and layout of operand B
        :param C: tensor representing data type and layout of operand C
        :param D: tensor representing data type and layout of operand D
        :param stride: (stride_h, stride_w) describing the convolution stride. Default: (1, 1)
        :param padding: (pad_h, pad_w) describing the convolution padding. Default: (0, 0)
        :param dilation: (dilation_h, dilation_w) describing the dilation of convolution. Default: (1, 1)
        :param alpha: scalar paramter alpha from GEMM computation that scales the product of operands A and B
        :param beta: scalar parameter beta from GEMM operation that scales operand C
        :param split_k: a tuple (split_k_mode, split_k_slices)
        :param sync: whether the call should wait for the kernel to complete before returning
        :type sync: bool
        :param print_module: whether to print the emitted C++ code
        :type print_module: bool
        :param stream: cuda stream, defaults to cuda.cuda.CUstream(0)
        :type stream: :class:`cuda.cuda.CUstream`

        :return: arguments passed in to the kernel
        :rtype: cutlass_cppgen.backend.Conv2dArguments
        """
        if not stream:
            stream = cuda.CUstream(0)
        super().run_setup()

        A = self._verify_tensor(A, self.A, self._element_a, self._layout_a, "A")
        B = self._verify_tensor(B, self.B, self._element_b, self._layout_b, "B")
        C = self._verify_tensor(C, self.C, self._element_c, self._layout_c, "C")
        D = self._verify_tensor(D, self.D, self._element_d, self._layout_d, "D")
        alpha = self._verify_scalar(alpha, self.alpha, self._element_c, "alpha")
        beta = self._verify_scalar(beta, self.beta, self._element_c, "beta")

        # handle the case when there is no C
        if C is None:
            if beta != 0:
                raise Exception(f"With beta {beta} != 0, C has to be provided.")
            else:
                C = D

        # Construct problem size based on input
        # It also verifies whether the A, B, C, D, stride, padding, and dilation are matching
        problem_size = self._get_and_verify_conv_problem_size(A, B, C, stride, padding, dilation)

        # Propose stride support based on input
        stride_support = self._propose_stride_support(stride)

        # Propose swizzling functor
        swizzling_functor = self._propose_swizzling_functor(stride)

        shape_a = datatypes.get_tensor_shape(A, op="CONV")
        shape_b = datatypes.get_tensor_shape(B, op="CONV")
        shape_c = datatypes.get_tensor_shape(C, op="CONV")

        # Get the alignment
        alignment_a = self.possible_operations.find_alignment(shape_a, self._layout_a, operand="A")
        alignment_b = self.possible_operations.find_alignment(shape_b, self._layout_b, operand="B")
        alignment_c = self.possible_operations.find_alignment(shape_c, self._layout_c, operand="C")

        alignment_a = check.update_alignment(alignment_a, self.alignment_pref_A)
        alignment_b = check.update_alignment(alignment_b, self.alignment_pref_B)
        alignment_c = check.update_alignment(alignment_c, self.alignment_pref_C)

        # Propose iterator algorithm based on input
        if self._iterator_algorithm is None:
            # Propose a default iterator algorithm based on the problem size
            iterator_algorithm = self._propose_iterator_algorithm(problem_size, alignment_a, alignment_b)
        else:
            if (self._validate_iterator_algorithm(self._iterator_algorithm, problem_size, alignment_a, alignment_b)):
                iterator_algorithm = self._iterator_algorithm
            else:
                raise Exception(f"Iterator algorithm {self._iterator_algorithm} is invalid for current problem.")

        epilogue_args = [alpha, beta]

        if hasattr(self, "_activation_args"):
            if isinstance(self._activation_args, list):
                epilogue_args += self._activation_args
            else:
                epilogue_args.append(self._activation_args)

        if split_k[0] == "parallel" and split_k[1] > 1:
            epilogue_functor = self._create_epilogue_functor_activation(epilogue.identity)
        else:
            epilogue_functor = self.epilogue_functor

        # The alignment is determined by the iterator function (I believe)
        self.compile(tile_description=self.tile_description, alignment_A=alignment_a, alignment_B=alignment_b,
                     alignment_C=alignment_c, iterator_algorithm=iterator_algorithm, stride_support=stride_support,
                     swizzling_functor=swizzling_functor, epilogue_functor=epilogue_functor, print_module=print_module)

        # Create reduction operation for parallel split-k
        if split_k[0] == "parallel" and split_k[1] > 1:
            epilogue_functor_reduction = self._reset_epilogue_functor_alignment(alignment_c, self.epilogue_functor)
            self.reduction_operation = ReductionOperation(
                shape=MatrixCoord(4, 32 * alignment_c), C=self.operation.C,
                element_accumulator=self._element_accumulator,
                element_compute=self._element_accumulator,
                epilogue_functor=epilogue_functor_reduction,
                count=alignment_c
            )
            if print_module:
                print(self.reduction_operation.rt_module.emit())
            compiler.add_module([self.reduction_operation,])

        arguments = Conv2dArguments(
            operation=self.operation, problem_size=problem_size,
            A=A, B=B, C=C, D=D,
            output_op=self.operation.epilogue_type(*epilogue_args),
            split_k_mode=datatypes.getattr_enum(SplitKMode, split_k[0]),
            split_k_slices=split_k[1],
            stream=stream
        )

        self.operation.run(arguments)

        if split_k[0] == "parallel" and split_k[1] > 1:
            implicit_gemm_size = arguments.problem_size.implicit_gemm_size(self.conv_kind)
            reduction_arguments = ReductionArguments(
                self.reduction_operation,
                problem_size=[implicit_gemm_size.m, implicit_gemm_size.n],
                partitions=split_k[1],
                workspace=arguments.ptr_D,
                destination=D,
                source=C,
                output_op=self.reduction_operation.epilogue_type(*epilogue_args),
                stream=stream
            )
            self.reduction_operation.run(reduction_arguments)

        if sync:
            if split_k[0] == "parallel" and split_k[1] > 1:
                reduction_arguments.sync()

                # Free memory allocated by args because we are not
                # calling `arguments.sync()` in this case (which will free memory)
                arguments.free()
            else:
                arguments.sync()

        return arguments

    #
    # Helper functions
    #
    @staticmethod
    def output_size(input_size, weight_size, padding, stride, dilation):
        problem_size = Conv2DProblemSize(
            *input_size,
            *weight_size,
            padding[0], padding[1],
            stride[0], stride[1],
            dilation[0], dilation[1],
            ConvMode.CrossCorrelation,
            1, 1
        )
        return (problem_size.N, problem_size.P, problem_size.Q, problem_size.K)


#
# Easy to use interfaces for fprop, wgrad, and dgrad
#

class Conv2dFprop(Conv2d):
    def __init__(
        self,
        input=None, weight=None, C=None, output=None, alpha=1, beta=0,
        element=None,
        element_input=None, element_weight=None, element_C=None, element_output=None,
        element_accumulator=None,
        cc: int = None, kernel_cc: int = None):
        A, B, D = input, weight, output
        element_A, element_B, element_D = element_input, element_weight, element_output
        super().__init__(
            "fprop", A, B, C, D, alpha, beta, element,
            element_A, element_B, element_C, element_D,
            element_accumulator, cc, kernel_cc)

    def run(
        self, input=None, weight=None, C=None, output=None, alpha=None, beta=None,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), split_k=("serial", 1),
        sync: bool = True, print_module: bool = False,
        stream: Optional[cuda.CUstream] = None) -> Conv2dArguments:

        if not stream:
            stream = cuda.CUstream(0)

        A, B, D = input, weight, output
        return super().run(
            A, B, C, D, alpha, beta, stride, padding, dilation, split_k, sync, print_module, stream)


class Conv2dDgrad(Conv2d):
    def __init__(
        self,
        grad_output=None, weight=None, C=None, grad_input=None, alpha=1, beta=0,
        element=None,
        element_grad_output=None, element_weight=None, element_C=None, element_grad_input=None,
        element_accumulator=None,
        cc: int = None, kernel_cc: int = None):
        A, B, D = grad_output, weight, grad_input
        element_A, element_B, element_D = element_grad_output, element_weight, element_grad_input
        super().__init__(
            "dgrad", A, B, C, D, alpha, beta, element,
            element_A, element_B, element_C, element_D,
            element_accumulator, cc, kernel_cc)

    def run(self, grad_output=None, weight=None, C=None, grad_input=None, alpha=None, beta=None,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), split_k=("serial", 1),
        sync: bool = True, print_module: bool = False,
        stream: Optional[cuda.CUstream] = None) -> Conv2dArguments:
        #
        if not stream:
            stream = cuda.CUstream(0)

        A, B, D = grad_output, weight, grad_input
        return super().run(
            A, B, C, D, alpha, beta, stride, padding, dilation, split_k, sync, print_module, stream)


class Conv2dWgrad(Conv2d):
    def __init__(
        self,
        grad_output=None, input=None, C=None, grad_weight=None, alpha=1, beta=0,
        element=None,
        element_grad_output=None, element_input=None, element_C=None, element_grad_weight=None,
        element_accumulator=None,
        cc: int = None, kernel_cc: int = None):
        A, B, D = grad_output, input, grad_weight
        element_A, element_B, element_D = element_grad_output, element_input, element_grad_weight
        super().__init__(
            "wgrad", A, B, C, D, alpha, beta, element,
            element_A, element_B, element_C, element_D,
            element_accumulator, cc, kernel_cc)

    def run(self, grad_output=None, input=None, C=None, grad_weight=None, alpha=None, beta=None,
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), split_k=("serial", 1),
        sync: bool = True, print_module: bool = False,
        stream: Optional[cuda.CUstream] = None) -> Conv2dArguments:
        if not stream:
            stream = cuda.CUstream(0)

        A, B, D = grad_output, input, grad_weight
        return super().run(
            A, B, C, D, alpha, beta, stride, padding, dilation, split_k, sync, print_module, stream)
