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
Classes containing valid operations for a given compute capability and data types.
"""

from itertools import combinations_with_replacement
import logging

import cutlass_library
from cutlass_library.library import ConvKind, IteratorAlgorithm, StrideSupport, GroupMode

import cutlass_cppgen
from cutlass_cppgen.utils.check import valid_stage_count
from cutlass_cppgen.utils.datatypes import td_from_profiler_td, td_from_profiler_op


_generator_ccs = [50, 60, 61, 70, 75, 80, 90, 100]


class KernelsForDataType:
    """
    Container class for keeping track of kernels that correspond to a particular combination
    of data types for operands A, B, and accumulator
    """

    def __init__(self, datatype_comb: tuple, layout_comb: tuple):
        self.datatype_comb = datatype_comb
        self.layout_comb = layout_comb
        self.math_operations = set()

        # Dictionary mapping from alignment (int) to a list of kernels that fit the alignment
        # constraint for the data type combination
        self.kernels_by_alignment = {}

    def add(self, operation):
        """
        Add an operation to the list of supported kernels
        """
        alignment_key = f"{operation.A.alignment} {operation.B.alignment} {operation.C.alignment}"
        if alignment_key not in self.kernels_by_alignment:
            self.kernels_by_alignment[alignment_key] = []
        self.kernels_by_alignment[alignment_key].append(operation)
        self.math_operations.add(operation.tile_description.math_instruction.math_operation)

    def alignments(self, operand: str):
        """
        Returns an unsorted list of alignments supported by this data type combination

        :param operand: identifier of operand in question (e.g., A, B, C)
        :type operand: str

        :return: unsorted list of alignments supported by this data type combination
        :rtype: list
        """
        operand_idx = self._operand_idx(operand)
        return [int(key.split(" ")[operand_idx]) for key in self.kernels_by_alignment.keys()]

    @property
    def all_operations(self):
        """
        Returns a list of all operations supported by this data type combination

        :return: list of all operations supported by this data type combination
        :rtype: list
        """
        ops = []
        for _, alignment_ops in self.kernels_by_alignment.items():
            ops.extend(alignment_ops)
        return ops

    def default_operation(self, math_operation: cutlass_cppgen.MathOperation):
        key = sorted(list(self.kernels_by_alignment.keys()))[0]
        kernels = self.kernels_by_alignment[key]
        if math_operation is not None:
            kernels = [x for x in kernels if x.tile_description.math_instruction.math_operation == math_operation]
        return kernels[0]

    def operations(self, alignment_A: int, alignment_B: int, alignment_C: int, math_operation: cutlass_cppgen.MathOperation):
        """
        Returns operations satisfying the alignment constraints

        :param alignment_A: alignment constraint of operations to return
        :type alignment_A: int
        :param alignment_B: alignment constraint of operations to return
        :type alignment_B: int
        :param alignment_C: alignment constraint of operations to return
        :type alignment_C: int
        :param math_operation: math operation to consider
        :type math_operation: cutlass_cppgen.MathOperation

        :return: list of operations
        :rtype: list
        """
        key = f"{alignment_A} {alignment_B} {alignment_C}"

        if key not in self.kernels_by_alignment:
            og_key = key
            # Reconcile A, B, and C alignments by trying to align to the minimum
            min_alignment = min(alignment_A, alignment_B, alignment_C)
            key = f"{min_alignment} {min_alignment} {min_alignment}"
            if key not in self.kernels_by_alignment:
                # Finally, go through all available alignment combinations and find
                # one for which all values are less than those passed in.
                key = None
                alignments = sorted([tuple(int(x) for x in k.split(" ")) for k in self.kernels_by_alignment.keys()], reverse=True)
                for align_A, align_B, align_C in alignments:
                    if alignment_A % align_A == 0 and alignment_B % align_B == 0 and alignment_C % align_C == 0:
                        key = f"{align_A} {align_B} {align_C}"
                        break

                if key is None:
                    raise Exception(
                        f"No operations of alignment {og_key} found for data type and layout "
                        f"combination {self.datatype_comb} {self.layout_comb}. Compatible alignments "
                        f"are {self.kernels_by_alignment.keys()}"
                    )

        ops = self.kernels_by_alignment[key]
        if math_operation is not None:
            ops = [op for op in ops if op.tile_description.math_instruction.math_operation == math_operation]
        return ops

    def _operand_idx(self, key: str) -> int:
        operand_list = ["A", "B", "C"]
        if key not in operand_list:
            raise Exception(f"Unexpected operand {operand}")

        return operand_list.index(key)

    def find_alignment(self, shape: tuple, layout: cutlass_cppgen.LayoutType, operand=str) -> int:
        """
        Returns the most preferable alignment for a given shape and layout

        :param shape: extent of each dimension of the tensor
        :type shape: tuple
        :param layout: layout of the tensor
        :type layout: cutlass_cppgen.LayoutType
        :param operand: descriptor of the operand in question
        :type operand: str

        :return: maximum alignment supported by the data type combination and tensor size
        :rtype: int
        """
        operand_idx = self._operand_idx(operand)

        # Determine the leading dimension of the shape
        if layout == cutlass_cppgen.LayoutType.ColumnMajor:
            ld = shape[-2]
        elif layout == cutlass_cppgen.LayoutType.RowMajor:
            ld = shape[-1]
        elif layout == cutlass_cppgen.LayoutType.TensorNHWC:
            ld = shape[-1]
        else:
            raise Exception(f"Unexpected or unsupported layout {layout}")

        for alignments in sorted(list(self.kernels_by_alignment.keys()), reverse=True):
            alignment = int(alignments.split(" ")[operand_idx])
            if ld % alignment == 0:
                return alignment

        # Default to alignment of 1 if no others match
        return 1

    def sort(self):
        """
        Sorts each list of kernels in `kernels_by_alignment` in descending order of threadblock shape
        """
        key = lambda op: (
            op.tile_description.threadblock_shape[0]
            * op.tile_description.threadblock_shape[1]
            * op.tile_description.threadblock_shape[2]
        )
        for alignment in self.kernels_by_alignment.keys():
            self.kernels_by_alignment[alignment].sort(key=key, reverse=True)

    def supports_math_operation(self, math_operation: cutlass_cppgen.MathOperation) -> bool:
        """
        Returns whether `math_operation` is supported by at least one operation.

        :param math_operation: math operation to consider
        :type math_operation: cutlass_cppgen.MathOperation

        :return: whether math_operation is supported by at least one operation
        :rtype: bool
        """
        return math_operation is None or math_operation in self.math_operations


class ArchOptions:
    """
    Structure for keeping track of kernels available on a given compute capability

    :param target_cc: compute capability of the device on which kernels will be run
    :type target_cc: int
    :param kernel_cc: compute capability of the kernels to generate
    :type kernel_cc: int
    :param operation_kind: type of operation to register
    :type operation_kind: cutlass_library.OperationKind
    :param gemm_kinds: types of GEMM operations that can be included
    :type gemm_kinds: list
    :param allowed_math_operations: types of primitive math operations allowed
    :type allowed_math_operations: list
    """

    def __init__(
        self,
        target_cc: int,
        kernel_cc: int,
        operation_kind: cutlass_library.OperationKind,
        gemm_kinds: list,
        allowed_math_operations: list = [
            cutlass_library.MathOperation.multiply_add,
            cutlass_library.MathOperation.multiply_add_saturate,
            cutlass_library.MathOperation.multiply_add_mixed_input_upcast,
            cutlass_library.MathOperation.multiply_add_fast_f32
        ]
    ):
        self.cc = kernel_cc

        # Dictionary with following structure:
        #  Key: OpcodeClass
        #  Value: Dictionary with the following structure:
        #     Key: tuple of ((DataType, DataType, DataType), (LayoutType, LayoutType, LayoutType),
        #          representing ((element_a, element_b, element_accumulator), (layout_a, layout_b))
        #     Value: KernelsForDataType
        self.operations_by_opclass = {}
        self.op_class = None
        self.allowed_math_operations = allowed_math_operations

        if target_cc == 100 and kernel_cc == 90 or target_cc == 90 and kernel_cc == 100:
            return

        # Identify the method within CUTLASS generator script that generates kernel
        # descriptions for the target CC
        generate_function_name = "GenerateSM" + str(kernel_cc)
        if not hasattr(cutlass_library.generator, generate_function_name):
            cutlass_cppgen.logger.warning(f"No generator found for architecture {kernel_cc}")
            return
        generate_function = getattr(cutlass_library.generator, generate_function_name)

        # Initialize a default manifest and populate it with valid kernel descriptions
        # for the target CC
        args = [
            "--kernels=all",
            f"--log-level={logging.getLevelName(cutlass_cppgen.logger.level)}"
        ]
        manifest_args = cutlass_library.generator.define_parser().parse_args(args)
        manifest = cutlass_library.manifest.Manifest(manifest_args)
        generate_function(manifest, cutlass_cppgen._nvcc_version)

        if operation_kind not in manifest.operations:
            # No kernels generated for this architecture, this could be because the CUDA
            # toolkit is insufficient to support operations in this CC
            cutlass_cppgen.logger.warning(f"No operations of type {operation_kind} found for CC {kernel_cc}")
            return

        # Only one CC should be returned, given the setup above of calling only the generation scripts
        # for a given CC
        if len(manifest.operations[operation_kind].keys()) != 1 or kernel_cc not in manifest.operations[operation_kind]:
            raise Exception(f"Error finding kernels for SM{kernel_cc}. Check that your CUDA toolkit version "
                             "is sufficient for the architecture in question.")

        # Iterate through the available operations for this operation kind and
        # find available opclasses and data types
        for name, op_list in manifest.operations[operation_kind][kernel_cc].items():
            for op in op_list:

                if operation_kind == cutlass_library.OperationKind.Gemm:
                    if op.gemm_kind not in gemm_kinds:
                        continue

                mi = op.tile_description.math_instruction
                if mi.math_operation not in self.allowed_math_operations:
                    continue

                # Prune operations that don't fit in shared memory
                td = td_from_profiler_op(op)
                if not valid_stage_count(target_cc, kernel_cc, td, verbose=False)[0]:
                    continue

                if mi.opcode_class not in self.operations_by_opclass:
                    self.operations_by_opclass[mi.opcode_class] = {}

                datatype_comb = (mi.element_a, mi.element_b, mi.element_accumulator)
                layout_comb = (op.A.layout, op.B.layout)

                # Register TF32 kernels as F32 to enable F32 -> TF32 conversion + TF32 Tensor Core operations
                if datatype_comb == (cutlass_library.DataType.tf32, cutlass_library.DataType.tf32, cutlass_library.DataType.f32):
                    # TF32 kernels only supported on SM80 and beyond
                    if self.cc < 80:
                        continue
                    elif self.cc == 90 or self.cc == 100:
                        if (op.A.element != cutlass_library.DataType.f32
                            or op.B.element != cutlass_library.DataType.f32
                            or op.C.element != cutlass_library.DataType.f32):
                            continue

                    datatype_comb = (cutlass_library.DataType.f32, cutlass_library.DataType.f32, cutlass_library.DataType.f32)

                opclass_dict = self.operations_by_opclass[mi.opcode_class]
                key = (datatype_comb, layout_comb)
                if key not in opclass_dict:
                    opclass_dict[key] = KernelsForDataType(datatype_comb, layout_comb)
                opclass_dict[key].add(op)

        # Set the default opclass to TensorOp, if available. Otherwise default to SIMT
        if cutlass_library.OpcodeClass.TensorOp in self.operations_by_opclass:
            self.op_class = cutlass_library.OpcodeClass.TensorOp
        else:
            self.op_class = cutlass_library.OpcodeClass.Simt

        # The profiler's generator may generate only a limited set of combinations of operands for SIMT kernels.
        # Here, we generate additional versions via a generic TileDescription.
        if cutlass_library.OpcodeClass.Simt not in self.operations_by_opclass:
            self.operations_by_opclass[cutlass_library.OpcodeClass.Simt] = {}

        if operation_kind == cutlass_library.OperationKind.Gemm:
            types = [
                (cutlass_library.DataType.s8, cutlass_library.DataType.s8, cutlass_library.DataType.s8),
                (cutlass_library.DataType.s8, cutlass_library.DataType.s8, cutlass_library.DataType.s32),
                (cutlass_library.DataType.f16, cutlass_library.DataType.f16, cutlass_library.DataType.f16),
                (cutlass_library.DataType.f16, cutlass_library.DataType.f16, cutlass_library.DataType.f32),
                (cutlass_library.DataType.f32, cutlass_library.DataType.f32, cutlass_library.DataType.f32),
                (cutlass_library.DataType.f64, cutlass_library.DataType.f64, cutlass_library.DataType.f64),
            ]

            # Add FP8 A/B/C
            fp8_types = [cutlass_library.DataType.e4m3, cutlass_library.DataType.e5m2]
            for type_comb in combinations_with_replacement(fp8_types, 3):
                types.append(type_comb)

            # Add FP8 A/B with FP32 C
            for type_comb in combinations_with_replacement(fp8_types, 2):
                types.append(type_comb + (cutlass_cppgen.DataType.f32,))

            layouts = [
                (cutlass_library.LayoutType.RowMajor, cutlass_library.LayoutType.RowMajor),
                (cutlass_library.LayoutType.RowMajor, cutlass_library.LayoutType.ColumnMajor),
                (cutlass_library.LayoutType.ColumnMajor, cutlass_library.LayoutType.RowMajor),
                (cutlass_library.LayoutType.ColumnMajor, cutlass_library.LayoutType.ColumnMajor),
            ]
        elif operation_kind == cutlass_library.OperationKind.Conv2d:
            types = [
                (cutlass_library.DataType.f16, cutlass_library.DataType.f16, cutlass_library.DataType.f16),
                (cutlass_library.DataType.f16, cutlass_library.DataType.f16, cutlass_library.DataType.f32),
                (cutlass_library.DataType.f32, cutlass_library.DataType.f32, cutlass_library.DataType.f32),
                (cutlass_library.DataType.f64, cutlass_library.DataType.f64, cutlass_library.DataType.f64),
            ]

            layouts = [
                (cutlass_library.LayoutType.TensorNHWC, cutlass_library.LayoutType.TensorNHWC),
            ]
        else:
            raise NotImplementedError(f"Operation kind {operation_kind} is currently unsupported.")

        alignment = 1
        epilogue_functor = cutlass_library.EpilogueFunctor.LinearCombination
        swizzling_functor = cutlass_library.SwizzlingFunctor.Identity8
        for type_comb in types:
            for layout_comb in layouts:
                comb = (type_comb, layout_comb)
                if comb in self.operations_by_opclass[cutlass_library.OpcodeClass.Simt]:
                    continue

                A = cutlass_library.TensorDescription(type_comb[0], layout_comb[0], alignment)
                B = cutlass_library.TensorDescription(type_comb[1], layout_comb[1], alignment)
                C = cutlass_library.TensorDescription(type_comb[2], cutlass_library.LayoutType.ColumnMajor, alignment)
                math_inst = cutlass_library.MathInstruction(
                    [1, 1, 1],
                    type_comb[0],
                    type_comb[1],
                    type_comb[2],
                    cutlass_library.OpcodeClass.Simt,
                    cutlass_library.MathOperation.multiply_add
                )

                td = cutlass_library.TileDescription(
                    [128, 128, 8], 2, [4, 2, 1], math_inst, 50, 1024)

                # Prune operations that don't fit in shared memory
                if not valid_stage_count(target_cc, kernel_cc, td_from_profiler_td(td), verbose=False)[0]:
                    continue

                new_kernels = KernelsForDataType(type_comb, layout_comb)

                if operation_kind == cutlass_library.OperationKind.Gemm:
                    new_operation = cutlass_library.manifest.GemmOperation(
                        cutlass_library.GemmKind.Universal, td.minimum_compute_capability,
                        td, A, B, C, type_comb[2], epilogue_functor, swizzling_functor)
                    new_kernels.add(new_operation)
                elif operation_kind == cutlass_library.OperationKind.Conv2d:
                    for conv_kind in [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad]:
                        new_operation = cutlass_library.manifest.Conv2dOperation(
                            conv_kind, IteratorAlgorithm.Analytic, td.minimum_compute_capability, td,
                            A, B, C, type_comb[2], StrideSupport.Strided, epilogue_functor, swizzling_functor,
                            group_mode=GroupMode.SingleGroup
                        )
                        new_kernels.add(new_operation)

                self.operations_by_opclass[cutlass_library.OpcodeClass.Simt][comb] = new_kernels

        # Sort all operations
        for oc in self.operations_by_opclass.keys():
            for comb in self.operations_by_opclass[oc].keys():
                self.operations_by_opclass[oc][comb].sort()

    def opclass_supports_combination(
        self, op_class: cutlass_library.OpcodeClass, datatype_comb: tuple, layout_comb: tuple, math_operation: cutlass_library.MathOperation
    ) -> bool:
        """
        Returns whether the provided operation class supports the provided data type and layout combination

        :param op_class: operation class to consider
        :type op_class: cutlass_library.OpcodeClass
        :param datatype_comb: tuple of data types for (element_A, element_B, element_accumulator)
        :type datatype_comb: tuple[cutlass_library.DataType]
        :param layout_comb: tuple of data types for (layout_A, layout_B)
        :type layout_comb: tuple[cutlass_library.LayoutType]
        :param math_operation: math operation to consider or None if any can be considered
        :type math_operation: cutlass_cppgen.MathOperation

        :return: set of operation classes that support the provided data type and layout combination
        :rtype: set
        """
        if op_class not in self.operations_by_opclass:
            raise Exception(f"Unexpected or unsupported operation class {op_class}")

        if operations := self.operations_by_opclass[op_class].get((datatype_comb, layout_comb)):
            if math_operation is not None:
                return operations.supports_math_operation(math_operation)
            else:
                return True

        return False


    def supporting_opclasses(
        self,
        element_a: cutlass_library.DataType,
        element_b: cutlass_library.DataType,
        element_accumulator: cutlass_library.DataType,
        layout_a: cutlass_library.LayoutType,
        layout_b: cutlass_library.LayoutType,
        math_operation: cutlass_library.MathOperation,
    ) -> set:
        """
        Returns a set of operation classes that support the provided data type combination

        :param element_a: data type of operand A
        :type element_a: cutlass_library.DataType
        :param element_b: data type of operand B
        :type element_b: cutlass_library.DataType
        :param element_accumulator: data type of accumulator
        :type element_accumulator: cutlass_library.DataType
        :param layout_a: layout of operand A
        :type layout_a: cutlass_library.LayoutType
        :param layout_b: layout of operand B
        :type layout_b: cutlass_library.LayoutType
        :param math_operation: math operation to consider
        :type math_operation: cutlass_cppgen.MathOperation

        :return: set of operation classes that support the provided data type combination
        :rtype: set
        """
        supporting_op_classes = set()
        datatype_comb = (element_a, element_b, element_accumulator)
        layout_comb = (layout_a, layout_b)

        for op_class in self.operations_by_opclass.keys():
            if self.opclass_supports_combination(op_class, datatype_comb, layout_comb, math_operation):
                supporting_op_classes.add(op_class)
        return supporting_op_classes

    def operations(
        self,
        op_class: cutlass_library.OpcodeClass,
        element_a: cutlass_library.DataType,
        element_b: cutlass_library.DataType,
        element_accumulator: cutlass_library.DataType,
        layout_a: cutlass_library.LayoutType,
        layout_b: cutlass_library.LayoutType,
        math_operation: cutlass_library.MathOperation,
    ) -> KernelsForDataType:
        """
        Returns whether the provided operation class supports the provided data type combination

        :param op_class: operation class to consider
        :type op_class: cutlass_library.OpcodeClass
        :param element_a: data type of operand A
        :type element_a: cutlass_library.DataType
        :param element_b: data type of operand B
        :type element_b: cutlass_library.DataType
        :param element_accumulator: data type of accumulator
        :type element_accumulator: cutlass_library.DataType
        :param layout_a: layout of operand A
        :type layout_a: cutlass_library.LayoutType
        :param layout_b: layout of operand B
        :type layout_b: cutlass_library.LayoutType
        :param math_operation: math operation to consider
        :type math_operation: cutlass_cppgen.MathOperation

        :return: container of kernels by alignment supported by the provided combination of parameters
        :rtype: KernelsForDataType
        """
        datatype_comb = (element_a, element_b, element_accumulator)
        layout_comb = (layout_a, layout_b)
        if not self.opclass_supports_combination(op_class, datatype_comb, layout_comb, math_operation):
            raise Exception(
                f"Data type layout combination {datatype_comb}, {layout_comb} "
                f"is not supported by opcode class {op_class} on CC {self.cc}."
            )
        return self.operations_by_opclass[op_class][(datatype_comb, layout_comb)]


class OptionRegistry:
    """
    Container of all architecture-specific options

    :param target_cc: compute capability of the device on which operations will be run
    :type target_cc: int
    """

    def __init__(self, target_cc: int):
        self.registry = {}

        if target_cc > 100 and (target_cc not in [101, 103, 120, 121]):
            raise Exception(f"Unsupported compute capability {target_cc}. The CUTLASS Python interface only supports compute capabilities up to the Blackwell architecture.")

        gemm_kinds = [cutlass_library.GemmKind.Universal, cutlass_library.GemmKind.Universal3x]
        operation_kinds = [cutlass_library.OperationKind.Gemm, cutlass_library.OperationKind.Conv2d]
        # Construct options for each CC
        for kernel_cc in _generator_ccs:
            self.registry[kernel_cc] = {}
            for opkind in operation_kinds:
                self.registry[kernel_cc][opkind] = ArchOptions(target_cc, kernel_cc, opkind, gemm_kinds)

    def options_for_cc(self, cc: int, op_kind=cutlass_library.OperationKind.Gemm) -> ArchOptions:
        return self.registry.get(cc, None)[op_kind]
