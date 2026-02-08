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

    The ``Gemm`` interface is meant to allow one to easily instantiate, compile, and run
    GEMM operations in CUTLASS via Python, without specifying many configuration parameters.
    Under the hood, the interface will select sensible default parameters for the many template
    parameters for CUTLASS GEMMs.

    Note: optimal performance is not to be expected from this interface. To achieve optimal
    performance, one should specify and tune each configuration parameter.

    The simplest example of using this interface is the following:

    .. highlight:: python
    .. code-block:: python

        # A, B, C, and D are torch/numpy/cupy tensor objects
        plan = cutlass_cppgen.op.Gemm(A, B, C, D)
        plan.run()


    One can also use the interface by specifying data types of operands at construction
    and using different tensor objects with these data types at runtime:

    .. highlight:: python
    .. code-block:: python

        # The following is shorthand for:
        #        cutlass_cppgen.op.Gemm(element_A=torch.float32, element_B=torch.float32,
        #                        element_C=torch.float32, element_D=torch.float32,
        #                        element_accumulator=torch.float32,
        #                        layout=cutlass_cppgen.LayoutType.RowMajor)
        plan = cutlass_cppgen.op.Gemm(element=torch.float32, layout=cutlass_cppgen.LayoutType.RowMajor)

        A0 = torch.rand((128, 256), device='cuda')
        B0 = torch.rand((256, 64), device='cuda')
        C0 = torch.zeros((128, 64), device='cuda')
        D0 = torch.zeros((128, 64), device.'cuda')
        plan.run(A0, B0, C0, D0)

        A = torch.rand((32, 128), device='cuda')
        B = torch.rand((128, 256), device='cuda')
        C = torch.zeros((32, 256), device='cuda')
        D = torch.zeros((32, 256), device.'cuda')
        plan.run(A1, B1, C1, D1)

    The interface additionally enables one to decouple the compilation of the underlying CUTLASS
    kernel from its execution:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Gemm(element=np.float32, layout=cutlass_cppgen.LayoutType.RowMajor)
        plan.compile()

        # Do other work...

        plan.run(A0, B0, C0, D0)

        # Do other work...

        plan.run(A1, B1, C1, D1)

    Elementwise activation functions are easily fused to the GEMM via the interface:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Gemm(element=np.float32, layout=cutlass_cppgen.LayoutType.RowMajor)
        plan.activation = cutlass_cppgen.epilogue.relu

    Operations can also be run asynchronously:

    .. highlight:: python
    .. code-block:: python

        plan = cutlass_cppgen.op.Gemm(element=np.float32, layout=cutlass_cppgen.LayoutType.RowMajor)
        args = plan.run()

        # Do other work...

        args.sync()
"""
from __future__ import annotations
from typing import Optional
from math import prod

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
from cutlass_library import (
    DataType,
    DataTypeSize,
    GemmUniversalMode,
    KernelScheduleSuffixes,
)

import cutlass_cppgen
from cutlass_cppgen import epilogue, swizzle
from cutlass_cppgen.backend import compiler
from cutlass_cppgen.backend.evt import EpilogueFunctorVisitor
from cutlass_cppgen.backend.gemm_operation import GemmArguments, GemmOperationUniversal
from cutlass_cppgen.backend.library import TensorDescription, TileDescription
from cutlass_cppgen.op.op import OperationBase
from cutlass_cppgen.shape import GemmCoord
from cutlass_cppgen.utils import check, datatypes


class Gemm(OperationBase):
    """
    Constructs a ``Gemm`` object.

    The data types and layouts of operands A, B, and C, along with the data type of output D
    and that used for accumulation, are bound to the ``Gemm`` object throughout its lifetime --
    these are not to be changed after a ``Gemm`` has been constructed.

    The constructor has optional parameters for flexibly setting these parameters. The following
    constructors are equivalent:

    .. highlight:: python
    .. code-block:: python

        # Use F32 for A, B, C, D, and accumulation. All operands are row major.

        # Use the generic ``element`` and ``layout`` parameters to concisely set all data types and layouts
        # for operands to the same values.
        Gemm(element=cutlass_cppgen.DataType.f32, layout=cutlass_cppgen.LayoutType.RowMajor)

        # Explicitly specify the data types to use for A, B, C, and D. Use the generic ``layout``.
        Gemm(element_A=cutlass_cppgen.DataType.f32, element_B=cutlass_cppgen.DataType.f32, element_C=cutlass_cppgen.DataType.f32,
            element_D=cutlass_cppgen.DataType.f32, layout=cutlass_cppgen.LayoutType.RowMajor)

        # Set the data types and elements from existing tensors. Note that one can use different tensors when
        # executing GEMM via the ``run()`` method than passed in here (though those passed in to ``run()`` must
        # have the same data type and layout as those passed in here).
        # A, B, C, and D are row-major torch.Tensor objects of type torch.float32
        Gemm(A=A, B=B, C=C, D=D)

        # Use the generic ``element`` and explicitly specify the layouts to use for A, B, and C (layout of D is
        # the same as that for D, at present)
        Gemm(element=cutlass_cppgen.DataType.f32, layout_A=cutlass_cppgen.LayoutType.RowMajor,
            layout_B=cutlass_cppgen.LayoutType.RowMajor, layout_C=cutlass_cppgen.LayoutType.RowMajor)

        # Explicitly specify the data type and layout for only some of A, B, C, and D. Unspecified data types
        # and layouts will inherit those passed in via the generic ``element`` and ``layout``
        Gemm(element_A=cutlass_cppgen.DataType.f32, layout_B=cutlass_cppgen.LayoutType.RowMajor,
            element=cutlass_cppgen.DataType.f32, layout=cutlass_cppgen.LayoutType.RowMajor)

    The order of precedence for the setting of the data type and layout for a given operand/output is as follows:
        1) If the tensor type is specified (e.g., ``A``), use the data type and layout inferred from this tensor
        2) Otherwise, if the data type/layout (e.g., ``element_A``, ``layout_A``) is specified, use those
        3) Otherwise, use the generic values (e.g., ``element``, ``layout``)

    :param cc: compute capability of device for which kernels should be compiled. For example, if running on H100, this should be set to 90
    :type cc: int
    :param kernel_cc: compute capability of kernels to generate. For example, if running on SM90, but desiring to use a CUTLASS 2.x-style Ampere kernel, this should be set to 80
    :type kernel_cc: int
    :param A: tensor representing data type and layout of operand A
    :param B: tensor representing data type and layout of operand B
    :param C: tensor representing data type and layout of operand C
    :param D: tensor representing data type and layout of operand D
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
    :param layout_A: layout of operand A
    :type layout_A: cutlass_cppgen.LayoutType
    :param layout_B: layout of operand B
    :type layout_B: cutlass_cppgen.LayoutType
    :param layout_C: layout of operand C
    :type layout_C: cutlass_cppgen.LayoutType
    :param layout_D: layout of operand D
    :type layout_D: cutlass_cppgen.LayoutType
    """

    def __init__(
        self, A=None, B=None, C=None, D=None,
        alpha=1.0, beta=0.0, element_accumulator=None,
        element=None, layout=None,
        element_A=None, element_B=None, element_C=None, element_D=None,
        layout_A=None, layout_B=None, layout_C=None,
        cc: int = None, kernel_cc: int = None
    ):
        super().__init__(cc=cc, kernel_cc=kernel_cc)
        self.name = "gemm"
        self.compiled = False

        elements = []
        layouts = []

        # Check that at least one of the following is set for each tensor (illustrated assuming tensor A):
        # ``A``, ``element_A``, ``element`` and ``A``, ``layout_A``, ``layout``
        for elt, lay, tens, name in zip([element_A, element_B, element_C, element_D],
                                        [layout_A, layout_B, layout_C, layout_C],
                                        [A, B, C, D],
                                        ["A", "B", "C", "D"]):
            if elt is not None and tens is not None:
                raise Exception(f'Must not specify both element_{name} and tensor {name}')
            if lay is not None and tens is not None:
                raise Exception(f'Must not specify both layout_{name} and tensor {name}')
            if elt is None and tens is None and element is None:
                raise Exception(f'Must specify one of element_{name}, tensor {name}, or generic element.')
            if lay is None and tens is None and layout is None:
                raise Exception(f'Must specify one of layout_{name}, tensor {name}, or generic layout.')

            elt_to_set = None
            lay_to_set = None
            if tens is not None:
                elt_to_set, lay_to_set = datatypes.get_datatype_and_layout(tens)
            else:
                elt_to_set = elt if elt is not None else element
                lay_to_set = lay if lay is not None else layout

            elements.append(datatypes.library_type(elt_to_set))
            layouts.append(lay_to_set)

        self._element_a, self._element_b, self._element_c, self._element_d = elements
        self._layout_a, self._layout_b, self._layout_c, self._layout_d = layouts

        if element_accumulator is None:
            self._element_accumulator = self._element_c
        else:
            self._element_accumulator = datatypes.library_type(element_accumulator)

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.alpha = alpha
        self.beta = beta

        self.epilogue_functor = None
        self.op_class = None
        self._tile_description = None

        self._reset_operations()

        self._swizzling_functor = cutlass_cppgen.swizzle.IdentitySwizzle1

    def _reset_operations(self, reset_epilogue: bool = True):
        # Set the default op class
        datatype_comb = (self._element_a, self._element_b, self._element_accumulator)
        layout_comb = (self._layout_a, self._layout_b)

        self.possible_op_classes = self.options.supporting_opclasses(
            self._element_a, self._element_b, self._element_accumulator,
            self._layout_a, self._layout_b, self._math_operation)

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
            self._reset_epilogue_functor_activation(cutlass_cppgen.epilogue.identity)

    @property
    def swizzling_functor(self):
        """
        Returns the type of the swizzling functor currently being used by the GEMM

        :return: swizzing functor type
        """
        return self._swizzling_functor

    @swizzling_functor.setter
    def swizzling_functor(self, swizzling_functor):
        """
        Sets the swizzling functor to the type specified by `swizzling_functor`
        """
        if swizzling_functor == cutlass_cppgen.swizzle.ThreadblockSwizzleStreamK:
            if self.op_class == cutlass_cppgen.OpcodeClass.Simt:
                raise Exception('ThreadblockSwizzleStreamK is currently only supported with opcode class TensorOp')

            if self.current_cc in [90, 100, 101, 103]:
                raise Exception('ThreadblockSwizzleStreamK is currently unsupported on SM90+')
        self._swizzling_functor = swizzling_functor

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
        valid, msg = check.valid_stage_count(self.cc, self.current_cc, td, self._element_c, self._element_d)
        if not valid:
            return (valid, msg)

        valid, msg = check.valid_cluster_shape(self.current_cc, td.cluster_shape)
        if not valid:
            return (valid, msg)

        valid, msg = check.valid_schedule(self.current_cc, td.kernel_schedule, td.epilogue_schedule, td.tile_scheduler)

        if self.cc in [100, 101, 103] and td.kernel_schedule is not None and td.is_2sm and td.cluster_shape[0] % 2 != 0:
            valid = False
            msg = "Cluster shape must be divisible by 2 for 2SM kernels on SM100, SM101, and SM103"

        return valid, msg

    def tile_descriptions(self) -> list:
        """
        Returns a list of valid tile descriptions for the operations

        :returns: list of valid tile descriptions for the operations
        :rtype: list
        """
        tds = [datatypes.td_from_profiler_op(op) for op in self.possible_operations.all_operations]
        if self._math_operation is not None:
            tds = [td for td in tds if td.math_instruction.math_operation == self._math_operation]
        return tds

    def construct(
        self, tile_description: TileDescription = None,
        alignment_A: int = None, alignment_B: int = None, alignment_C: int = None) -> GemmOperationUniversal:
        """
        Constructs a ``cutlass_cppgen.backend.GemmUniversalOperation`` based on the input parameters and current
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
        :rtype: cutlass_cppgen.backend.GemmOperationUniversal
        """
        alignment_pref_A = min(128 // DataTypeSize[self._element_a], max(self.possible_operations.alignments("A")))
        alignment_pref_B = min(128 // DataTypeSize[self._element_b], max(self.possible_operations.alignments("B")))
        alignment_A = check.alignment_or_default(alignment_A, alignment_pref_A)
        alignment_B = check.alignment_or_default(alignment_B, alignment_pref_B)

        tensor_A = TensorDescription(self._element_a, self._layout_a, alignment_A)
        tensor_B = TensorDescription(self._element_b, self._layout_b, alignment_B)

        if alignment_C is None:
            alignment_C = max(self.possible_operations.alignments("C"))
            if self._element_c != DataType.void:
                alignment_C = min(128 // DataTypeSize[self._element_c], alignment_C)

        if tile_description is None:
            if self._tile_description is None:
                op = self.possible_operations.operations(alignment_A, alignment_B, alignment_C, self._math_operation)[0]
                tile_description = datatypes.td_from_profiler_op(op)

                # The selected op may have lower alignment than that determined above, so we must
                # reset alignment here.
                alignment_C = op.C.alignment
            else:
                tile_description = self._tile_description
        else:
            valid, err_str = self._valid_tile_description(tile_description)
            if not valid:
                raise Exception(f"Invalid tile description. {err_str}")
            self._tile_description = tile_description

        tensor_C = TensorDescription(self._element_c, self._layout_c, alignment_C)
        self.epilogue_functor = self._reset_epilogue_functor_alignment(alignment_C, self.epilogue_functor)

        operation = GemmOperationUniversal(
            arch=self.current_cc,
            tile_description=tile_description,
            A=tensor_A, B=tensor_B, C=tensor_C,
            epilogue_functor=self.epilogue_functor,
            swizzling_functor=self._swizzling_functor,
        )

        return operation

    def compile(self, tile_description: TileDescription = None,
                alignment_A: int = None, alignment_B: int = None, alignment_C: int = None,
                print_module: bool = False) -> cutlass_cppgen.backend.GemmOperationUniversal:
        """
        Emits and compiles the kernel currently specified. If ``tile_description`` and any
        of the ``alignment`` parameters are set, the kernel will be chosen using this
        tile description and alignments. Otherwise, a default tile description and alignment
        will be used.

        :param tile_description: tile description specifying shapes and operand types to use in the kernel
        :type tile_description: cutlass_cppgen.backend.TileDescription
        :param alignment_A: alignment of operand A
        :type alignment_A: int
        :param alignment_B: alignment of operand B
        :type alignment_B: int
        :param alignment_C: alignment of operand C
        :type alignment_C: int
        :param print_module: whether to print the emitted C++ code
        :type print_module: bool

        :return: operation that was compiled
        :rtype: cutlass_cppgen.backend.GemmOperationUniversal
        """
        self.operation = self.construct(tile_description, alignment_A, alignment_B, alignment_C)

        if print_module:
            print(self.operation.rt_module.emit())

        compiler.add_module([self.operation,])
        return self.operation

    def _verify_rank(self, tensor):
        """
        Verifies that ``tensor`` has rank greater than 1

        :param tensor: object representing a tensor passed in to verify, or ``None`` if no tensor was passed in
        :type tensor: numpy/cupy/torch array/tensor object
        """
        if len(tensor.shape) < 2:
            raise Exception(f"Tensors must be of rank greater than 1. Received tensor of shape: {tensor.shape}")

    def _get_batch_count(self, A, B, C, D) -> int:
        """
        Returns the batch count specified by the tensors A, B, C, and D and verifies that these
        tensors match in batch size. Presence of a batch dimension is detected by one of the
        tensors being rank 3. If a batch dimension is present, it must be present in one of
        operands A, B, or C (but need not be in all), and must be present in D.

        :param A: tensor A
        :type A: numpy/cupy/torch array/tensor object
        :param B: tensor B
        :type B: numpy/cupy/torch array/tensor object
        :param C: tensor C
        :type C: numpy/cupy/torch array/tensor object
        :param D: tensor D
        :type D: numpy/cupy/torch array/tensor object

        :return: tuple of batch count dimensions
        :rtype: tuple
        """
        A_batch = prod(A.shape[:-2]) if len(A.shape) > 2 else 1
        B_batch = prod(B.shape[:-2]) if len(B.shape) > 2 else 1

        if 1 not in [A_batch, B_batch]:
            if A_batch != B_batch:
                raise Exception(f"Get invalid batch counts: A={A_batch}, B={B_batch}")
        return max(A_batch, B_batch)

    def _get_batch_stride(self, tensor) -> int:
        """
        Returns the batch stride of ``tensor``. If ``tensor`` is only rank-2, batch stride is 0.

        :param tensor: tensor object to process
        :type tensor: numpy/cupy/torch array/tensor object

        :return: stride between each matrix in the batch
        :rtype: int
        """
        if tensor is not None and len(tensor.shape) > 2:
            return tensor.shape[-2] * tensor.shape[-1]
        else:
            return 0

    def _get_problem_args(self, A, B, C, D) -> tuple:
        """
        Returns the problem size and GEMM universal mode to use for the
        given operands.

        :param A: tensor A
        :type A: numpy/cupy/torch array/tensor object
        :param B: tensor B
        :type B: numpy/cupy/torch array/tensor object
        :param C: tensor C
        :type C: numpy/cupy/torch array/tensor object
        :param D: tensor D
        :type D: numpy/cupy/torch array/tensor object

        :return: tuple containing the problem size (cutlass_cppgen.shape.GemmCoord), the GEMM mode (cutlass_cppgen.GemmUniversalMode), and the batch count (int)
        :rtype: tuple
        """
        M, K = A.shape[-2:]
        N = B.shape[-1]
        mode = GemmUniversalMode.Gemm

        batch_count = self._get_batch_count(A, B, C, D)
        returned_batch_count = batch_count

        # If we are running a batched GEMM in which there is a nonzero batch stride
        # only for A, then we can fold the batched dimension of A into the M dimension
        # (i.e., (b, m, k) x (k, n) -> (m*b, k) x (k, n)). This works only if both A
        # and C are row major. A similar operation can be performed if only B has a nonzero
        # batch dimension
        if batch_count > 1:
            A_row = self._layout_a == cutlass_cppgen.LayoutType.RowMajor
            B_row = self._layout_b == cutlass_cppgen.LayoutType.RowMajor
            C_row = self._layout_c == cutlass_cppgen.LayoutType.RowMajor

            # Consider a Tensor to be batched if its rank is > 2 and
            # the product of the modes beyond rank 2 equals our pre-determined batch size.
            batched = lambda x : x is None or (len(x.shape) > 2 and prod(x.shape[:-2]) == batch_count)

            if batched(A) and not batched(B) and (C is None or batched(C)) and A_row and C_row:
                M *= batch_count
                returned_batch_count = 1
            elif not batched(A) and batched(B) and (C is None or batched(C)) and not B_row and not C_row:
                N *= batch_count
                returned_batch_count = 1
            else:
                mode = GemmUniversalMode.Batched

        return GemmCoord(M, N, K), mode, returned_batch_count

    def _verify_type_and_layout(self, tensor, ref_type, ref_layout, name):
        """
        Verifies that ``tensor`` has data type ``ref_type`` and layout ``ref_layout``. An exception
        is raised if it does not.

        :param tensor: object representing a tensor passed in to verify, or ``None`` if no tensor was passed in
        :type tensor: numpy/cupy/torch array/tensor object
        :param ref_dtype: data type for the tensor that this object was initialized to
        :param ref_layout: layout for the tensor that this object was initialized to
        :param name: identifier of the tensor to verify. Used in raising exceptions
        :type name: str
        """
        dtype, layout = datatypes.get_datatype_and_layout(tensor)
        if dtype != ref_type or layout != ref_layout:
            try:
                # Attempt to transpose the tensor to fit the desired layout
                tensor = tensor.transpose(-1, -2)
            except:
                raise Exception(f'Tensor {name} with type and layout ({dtype}, {layout}) '
                                f'does not match the expected type and '
                                f'layout of ({ref_type}, {ref_layout}) and transpose failed.')

    def run(self, A=None, B=None, C=None, D=None,
            alpha=None, beta=None, sync: bool = True, print_module: bool = False, visitor_args: dict = None,
            stream: Optional[cuda.CUstream] = None) -> GemmArguments:
        """
        Runs the kernel currently specified. If it has not already been, the kernel is emitted and
        compiled. Tensors holding operands and outputs of the kernel are sourced either from the
        ``A``, ``B``, ``C``, ``D``, ``alpha``, and ``beta``
        parameters provided in this call, or from those
        passed in on the construction of this object -- one of the two must be specified.

        By default, this call returns only once the kernel has completed. To launch the kernel
        and immediately return, set ``sync=False``. In this case, it is the responsibility of the
        caller to syncrhonize the results of the kernel before attempting to access outputs
        by calling ``sync()`` on the arguments returned from this call.

        :param A: tensor representing data type and layout of operand A
        :param B: tensor representing data type and layout of operand B
        :param C: tensor representing data type and layout of operand C
        :param D: tensor representing data type and layout of operand D
        :param alpha: scalar paramter alpha from GEMM computation that scales the product of operands A and B
        :param beta: scalar parameter beta from GEMM operation that scales operand C
        :param sync: whether the call should wait for the kernel to complete before returning
        :type sync: bool
        :param print_module: whether to print the emitted C++ code
        :type print_module: bool
        :param stream: cuda stream, defaults to cuda.cuda.CUstream(0)
        :type stream: :class:`cuda.cuda.CUstream`

        :return: arguments passed in to the kernel
        :rtype: cutlass_cppgen.backend.GemmArguments
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

        is_void_c = self._element_c == DataType.void

        self._verify_rank(A)
        self._verify_rank(B)
        if not is_void_c:
            self._verify_rank(C)
        self._verify_rank(D)

        alignment_a = self.possible_operations.find_alignment(A.shape, self._layout_a, operand="A")
        alignment_b = self.possible_operations.find_alignment(B.shape, self._layout_b, operand="B")

        # Set C alignment based on D.shape so as to correctly get an alignment with void-C
        # kernels, for which `C` is None.
        alignment_c = self.possible_operations.find_alignment(D.shape, self._layout_c, operand="C")
        self.compile(self._tile_description, alignment_A=alignment_a, alignment_B=alignment_b,
                     alignment_C=alignment_c, print_module=print_module)

        problem_size, mode, batch_count = self._get_problem_args(A, B, C, D)

        if mode == GemmUniversalMode.Gemm or batch_count == 1:
            kwargs = {'split_k_slices': 1}
        else:
            kwargs = {
                'batch': batch_count,
                'batch_strides': {
                    'A': self._get_batch_stride(A),
                    'B': self._get_batch_stride(B),
                    'C': self._get_batch_stride(C),
                    'D': self._get_batch_stride(D)
                }
            }

        kwargs['stream'] = stream

        if isinstance(self.epilogue_functor, EpilogueFunctorVisitor):
            output_op = self.operation.epilogue_type(visitor_args)
        else:
            output_op = self.operation.epilogue_type(alpha, beta)

        arguments = GemmArguments(
            operation=self.operation, problem_size=problem_size,
            A=A, B=B, C=C, D=D,
            output_op=output_op,
            gemm_mode=mode,
            **kwargs
        )

        self.operation.run(arguments)

        if sync:
            arguments.sync()

        return arguments
