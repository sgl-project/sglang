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
Base operation used for defining high-level CUTLASS operations (e.g., GEMM, Conv2d)
"""

from bisect import bisect_left

from cutlass_library import (
    DataType,
    DataTypeSize,
    MathOperation,
    OperationKind,
    SharedMemPerCC
)

import cutlass_cppgen
from cutlass_cppgen import get_option_registry
from cutlass_cppgen.backend.evt import EpilogueFunctorVisitor
from cutlass_cppgen.backend.evt.passes.util import cc_map
from cutlass_cppgen.backend.utils.device import device_cc
from cutlass_cppgen.epilogue import get_activations, get_activation_epilogue, identity
from cutlass_cppgen.library_defaults import KernelsForDataType, _generator_ccs
from cutlass_cppgen.swizzle import get_swizzling_functors
from cutlass_cppgen.utils import datatypes, check


class OperationBase:
    """
    Base operation used for defining high-level CUTLASS operations (e.g., GEMM, Conv2d)
    """

    def __init__(self, cc: int = None, kernel_cc: int = None, operation_kind = OperationKind.Gemm):
        """
        :param cc: compute capability of device for which kernels should be compiled. For example, if running on H100, this should be set to 90
        :type cc: int
        :param kernel_cc: compute capability of kernels to generate. For example, if running on SM90, but desiring to use a CUTLASS 2.x-style Ampere kernel, this should be set to 80
        :type kernel_cc: int
        :param operation_kind: class of operation that will be performed (e.g., GEMM, Conv)
        :type operation_kind: cutlass_library.OperationKind
        """
        self.operation_kind = operation_kind
        self.cc = cc if cc is not None else device_cc()
        self.specified_kernel_cc = kernel_cc is not None
        self.current_cc = kernel_cc if kernel_cc is not None else self._find_closest_cc(self.cc)
        self.tile_description = None
        self._math_operation = None

        self.options = get_option_registry().options_for_cc(self.current_cc, operation_kind)

        if self.options is None:
            raise Exception(f"Invalid or unsupported compute capability: {self.current_cc}")

        # Default activation function: identity
        self._activation = identity

    def _find_closest_cc(self, cc: int) -> int:
        """
        Returns the closest CC in _generator_ccs less than or equal to `cc`

        :param cc: compute capability to query
        :type cc: int

        :returns: closest CC in _generator_ccs less than or equal to `cc`
        :rtype: int
        """
        if cc in _generator_ccs:
            return cc

        # Find closest CC lower than this CC
        idx = bisect_left(_generator_ccs, cc)
        if idx == 0:
            raise Exception(f'No valid CC to fall back to for {cc}')
        return _generator_ccs[idx-1]

    def activations(self) -> list:
        """
        Returns possible activation functions that can be used

        :return: list of activation functions that can be used
        :rtype: list
        """
        return get_activations()

    def swizzling_functors(self) -> list:
        """
        Returns possible swizzling functions that can be used

        :return: list of swizzling functions that can be used
        :rtype: list
        """
        return get_swizzling_functors()

    def _reset_options(self, cc: int):
        """
        Resets the kernel options based on cc

        :param cc: compute capability to reset to
        :type cc: int
        """
        if cc != self.current_cc:
            if cc not in _generator_ccs:
                raise Exception(f'Invalid CC for CUTLASS kernels: {cc}.')
            self.current_cc = cc
            self.options = get_option_registry().options_for_cc(self.current_cc, self.operation_kind)

    def _verify_scalar(self, scalar, ref_scalar, ref_dtype, name):
        """
        Verifies the following properties:
            1) Either ``scalar`` or ``ref_scakar`` must be set (i.e., not ``None``)
            2) If ``scalar`` is not ``None``, its datatype must match matches the current version
               set by the plan (i.e., those in ``ref_dtype``)

        If either of these properties does not hold, an exception is raised. If these properties hold and
        ``scalar`` is not ``None``, ``scalar`` is returned. Otherwise, ``ref_scalar`` is returned.

        :param scalar: object representing a tensor passed in to verify, or ``None`` if no tensor was passed in
        :type scalar: numpy/cupy/torch scalar
        :param ref_scalar: object representing a tensor passed in on construction of this object, or ``None`` if no tensor was passed in
        :type ref_scalar: numpy/cupy/torch scalar
        :param ref_dtype: data type for the scalar that this object was initialized to
        :param name: identifier of the scalar to verify. Used in raising exceptions
        :type name: str

        :return: valid scalar to use
        :rtype: numpy/cupy/torch scalar
        """
        if scalar is None:
            if ref_scalar is None:
                raise Exception(f"Scalar {name} must be set.")
            return ref_scalar
        if hasattr(scalar, "dtype"):
            dtype = datatypes.library_type(scalar.dtype)
            if dtype != ref_dtype:
                raise Exception(
                    f"Tensor {name} with type {dtype} does not match expected type {ref_dtype}."
                )
        return scalar

    def _verify_tensor(self, tensor, ref_tensor, ref_dtype, ref_layout, name):
        """
        Verifies the following properties:
            If ref_dtype is not void:
                1) Either ``tensor`` or ``ref_tensor`` must be set (i.e., not ``None``)
                2) If ``tensor`` is not ``None``, its datatype and layout must match matches the current versions
                set by the plan (i.e., those in ``ref_dtype`` and ``ref_layout``)
            If ref_dtype is void:
                Neither ``tensor`` nor ``ref_tensor`` are set

        If either of these properties does not hold, an exception is raised. If these properties hold and
        ``tensor`` is not ``None``, ``tensor`` is returned. Otherwise, ``ref_tensor`` is returned.

        :param tensor: object representing a tensor passed in to verify, or ``None`` if no tensor was passed in
        :type tensor: numpy/cupy/torch array/tensor object
        :param ref_tensor: object representing a tensor passed in on construction of this object, or ``None`` if no tensor was passed in
        :type ref_tensor: numpy/cupy/torch array/tensor object
        :param ref_dtype: data type for the tensor that this object was initialized to
        :param ref_layout: layout for the tensor that this object was initialized to
        :param name: identifier of the tensor to verify. Used in raising exceptions
        :type name: str

        :return: valid tensor object to use
        :rtype: numpy/cupy/torch array/tensor object
        """
        if ref_dtype == DataType.void:
            if tensor is not None or ref_tensor is not None:
                raise Exception("Operands with element DataType.void must not be provided a tensor")
            return None

        if tensor is None:
            if ref_tensor is None:
                raise Exception(f"Tensor {name} must be set.")
            return ref_tensor

        self._verify_type_and_layout(tensor, ref_dtype, ref_layout, name)
        return tensor

    @property
    def opclass(self) -> cutlass_cppgen.OpcodeClass:
        """
        Returns the opcode class currently in use

        :return: opcode class currently in use
        :rtype: cutlass_cppgen.OpcodeClass
        """
        return self.op_class

    @opclass.setter
    def opclass(self, oc: cutlass_cppgen.OpcodeClass):
        if isinstance(oc, str):
            oc = datatypes.getattr_enum(cutlass_cppgen.OpcodeClass, oc)
        if oc in self.possible_op_classes:
            self.op_class = oc
        else:
            raise Exception(
                f'Unsupported operation class {oc} for CC {self.cc} and data type combination '
                f'({self._element_a}, {self._element_b}, {self._element_accumulator}) and '
                f'layout combination ({self._layout_a}, {self._layout_b}).')

        # Changing the op class also changes the possible operations available. Reset these.
        self.possible_operations = self.options.operations(
            self.op_class, self._element_a, self._element_b,
            self._element_accumulator, self._layout_a, self._layout_b, self._math_operation)

        # Changing the op class changes the elements per access in the epilogue. Reset this.
        if self.epilogue_functor is not None:
            self.epilogue_functor = self._reset_epilogue_functor_alignment(self._elements_per_access(), self.epilogue_functor)

    @property
    def math_operation(self) -> cutlass_cppgen.MathOperation:
        """
        Returns the math operation currently in use

        :return: math operation currently in use
        :rtype: cutlass_cppgen.MathOperation
        """
        return self._math_operation

    @math_operation.setter
    def math_operation(self, mo: cutlass_cppgen.MathOperation):
        if isinstance(mo, str):
            mo = datatypes.getattr_enum(cutlass_cppgen.MathOperation, mo)

        if not self.specified_kernel_cc:
            if self.current_cc in [90, 100, 101, 103]:
                # CUTLASS 3.0 kernels do not use different math operations. If one is specified, we
                # revert to using a CUTLASS 2.x kernel by using SM80-tagged kernels.
                cutlass_cppgen.logger.warning("Reverting to using SM80-tagged kernel. Opclass may change.")
                self._reset_options(80)
                self._reset_operations(reset_epilogue=False)
        elif self.current_cc in [90, 100, 101, 103]:
            raise Exception("CUTLASS 3.0 kernels do not use different math operations. "
                "To use 2.x kernels with a specific math operation, do not set the `kernel_cc`"
                "parameter when constructing the plan.")

        self._math_operation = mo
        self._reset_operations()

    def _elements_per_access(self):
        if self.op_class == cutlass_cppgen.OpcodeClass.Simt:
            return 1
        elif self._element_c != DataType.void:
            return 128 // DataTypeSize[self._element_c]
        else:
            return 128 // max(self.possible_operations.alignments("C"))

    def _create_epilogue_functor_activation(self, activation):
        """
        Returns the epilogue functor with given activation function
        """
        if self.epilogue_functor is None:
            elements_per_access = self._elements_per_access()
        else:
            elements_per_access = self.epilogue_functor.epilogue_vector_length

        if not self.specified_kernel_cc:
            if self.current_cc in [90, 100, 101, 103] and activation != identity:
                # CUTLASS 3.0 kernels in Python currently only support identity activation. If one requests a non-identity activation,
                # revert to using a CUTLASS 2.x kernel by using SM80-tagged kernels.
                cutlass_cppgen.logger.warning("Reverting to using SM80-tagged kernel. Opclass may change.")
                if self._element_c != self._element_d:
                    raise Exception("CUTLASS 2.x kernels require element C to be the same as element D")
                self._reset_options(80)
                self._reset_operations(reset_epilogue=False)
            elif (self.cc in [90, 100, 101, 103] and self.current_cc not in [90, 100, 101, 103] and activation == identity and self._math_operation is None):
                # SM80 fallback kernels are currently used. Since an identity activation is requested,
                # we can switch back to using SM90 kernels.
                self._reset_options(self.cc)
                self._reset_operations(reset_epilogue=False)
        else:
            if self.current_cc in [90, 100, 101, 103] and activation != identity:
                raise Exception("Epilogues with elementwise fusion are not currently supported "
                                "in the Python interface for 3.x kernels. To use 2.x kernels "
                                "with fused elementwise epilogues, do not set the `kernel_cc` "
                                "parameter when constructing the plan.")

        return get_activation_epilogue(
            activation,
            self._element_d,
            elements_per_access,
            self._element_accumulator,
            self._element_accumulator,
        )

    def _reset_epilogue_functor_activation(self, activation):
        """
        Set the epilogue functor based on the provided activation function
        """
        self.epilogue_functor = self._create_epilogue_functor_activation(activation)

    def _reset_epilogue_functor_alignment(self, alignment, epilogue_functor):
        """
        Reset the alignment of the current epilogue functor based on alignment C
        """
        if isinstance(epilogue_functor, EpilogueFunctorVisitor):
            return epilogue_functor

        if epilogue_functor is None or not hasattr(epilogue_functor, 'activation_functor'):
            # Identity epilogue does not have 'activation_functor'
            activation = identity
        else:
            activation = epilogue_functor.activation_functor

        epilogue_functor = get_activation_epilogue(
            activation,
            self._element_d,
            alignment,
            self._element_accumulator,
            self._element_accumulator,
        )
        return epilogue_functor

    @property
    def activation(self):
        """
        Returns the type of the current activation function used
        """
        if hasattr(self.epilogue_functor, "activation_functor"):
            return self.epilogue_functor.activation_functor
        else:
            return identity

    @activation.setter
    def activation(self, act):
        """
        Sets the type of the activation function to use
        Activation can come with a set of arguments

        :param act: type of activation function to use
        :type act: str or tuple. e.g. "relu", ("leaky_relu", 0.01)

        """
        if isinstance(act, tuple):
            if isinstance(act[0], str):
                act_fn = getattr(cutlass_cppgen.backend.epilogue, act[0])
            else:
                act_fn = act[0]
            self._reset_epilogue_functor_activation(act_fn)
            self._activation_args = act[1]
            self._activation = act[0]
        else:
            if isinstance(act, str):
                act = getattr(cutlass_cppgen.backend.epilogue, act)
            self._reset_epilogue_functor_activation(act)
            self._activation = act

    @property
    def epilogue_visitor(self):
        """
        Return the epilogue functor
        """
        return self.epilogue_functor

    @epilogue_visitor.setter
    def epilogue_visitor(self, visitor):
        """
        Create the epilogue visitor
        """
        self.epilogue_functor = EpilogueFunctorVisitor(cc_map[self.cc], visitor)

        # The epilogue_functor may consume too much shared memory
        # Reset the possible operations
        if self.cc not in [90, 100, 101, 103]:
            # The shared memory is only a concern for sm90+ epilogue
            # In sm80, the epilogue and mainloop share the shared memory
            return

        datatype_comb = self.possible_operations.datatype_comb
        layout_comb = self.possible_operations.layout_comb
        new_possible_operations = KernelsForDataType(datatype_comb, layout_comb)
        for operation in self.possible_operations.all_operations:
            td = datatypes.td_from_profiler_op(operation)
            # Filter invalid epilogue schedules
            if cc_map[self.cc] == 90 and td.epilogue_schedule not in [
                cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecialized,
                cutlass_cppgen.EpilogueScheduleType.TmaWarpSpecializedCooperative]:
                continue
            epilogue_smem_bytes = self.epilogue_functor.get_smem_size(td)

            # Verify the maximum number of mainloop stages
            mainloop_smem_per_stage = check.calculate_smem_usage_per_stage(td, OperationKind.Gemm)
            smem_capacity_bytes = SharedMemPerCC[self.cc] << 10
            mainloop_stages = (smem_capacity_bytes - epilogue_smem_bytes) // mainloop_smem_per_stage
            if mainloop_stages < 2:
                # Mainloop stages must >= 2
                continue

            new_possible_operations.add(operation)
        if len(new_possible_operations.all_operations) == 0:
            raise RuntimeError(
                "The epilogue consumes too much shared memory. "
                "No valid tile description is found in the generator.")
        self.possible_operations = new_possible_operations


    def run_setup(self):
        """
        Steps that must be taken before caling `plan.run()`
        """
        # Initialize the memory pool if, if not already done
        cutlass_cppgen.get_memory_pool()
