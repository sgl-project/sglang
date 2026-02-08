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
from __future__ import annotations

import ctypes
from typing import Union

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
from cutlass_library import SubstituteTemplate
import numpy as np

from cutlass_library import (
    ConvKindNames,
    ConvKindTag,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    IteratorAlgorithmNames,
    IteratorAlgorithmTag,
    LayoutTag,
    LayoutType,
    MathOperation,
    MathOperationTag,
    OpcodeClass,
    OpcodeClassNames,
    OpcodeClassTag,
    OperationKind,
    ShortDataTypeNames,
    ShortLayoutTypeNames,
    SplitKMode,
    StrideSupport,
    StrideSupportTag,
    SwizzlingFunctor,
    SwizzlingFunctorTag,
    get_complex_from_real,
)

from cutlass_cppgen.backend.arguments import ArgumentBase
from cutlass_cppgen.backend.c_types import dim3_, get_conv2d_arguments
from cutlass_cppgen.backend.library import (
    EmissionType,
    TensorDescription,
    TileDescription,
)
from cutlass_cppgen.backend.memory_manager import device_mem_alloc
from cutlass_cppgen.backend.operation import ExecutableOperation, LaunchConfiguration
from cutlass_cppgen.backend.utils.device import to_device_ptr
from cutlass_cppgen.shape import GemmCoord


class Conv2dArguments(ArgumentBase):
    """
    Argument wrapper for Conv2d. It encodes problem information and
    user-provide tensors into the kernel's argument.

    :param operation: the Conv2d operation to take the argument
    :type operation: :class:`cutlass_cppgen.backend.Conv2dOperation`
    :param problem_size: the Conv2d problem size
    :type problem_size: :class:`cutlass_cppgen.shape.Conv2dProblemSize`
    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray
    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray
    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray
    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray
    :param split_k_mode: conv2d split K mode, defaults to cutlass_library.library.SplitKMode.Serial
    :type split_k_mode: cutlass_library.library.SplitKMode, optional
    :param output_op: output operator, optional
    :type output_op: :class:`cutlass_cppgen.backend.LinearCombinationFunctorArguments`
    :param stream: cuda stream, defaults to cuda.cuda.CUstream(0)
    :type stream: :class:`cuda.cuda.CUstream`
    """

    def __init__(self, operation, problem_size, A, B, C, D,
        split_k_mode=SplitKMode.Serial, **kwargs, ) -> None:
        self.operation = operation
        self.conv_kind = operation.conv_kind
        self.layout_A = operation.A.layout
        self.layout_B = operation.B.layout
        self.layout_C = operation.C.layout

        self.element_A = operation.A.element
        self.element_B = operation.B.element
        self.element_C = operation.C.element

        if self.layout_C == LayoutType.TensorNC32HW32:
            raise Exception("Layout type TensorNC32HW32 is not currently supported")

        super().__init__(A, B, C, D, **kwargs)

        if "split_k_slices" in kwargs.keys() and kwargs["split_k_slices"] > 1:
            self.split_k_mode = split_k_mode
            self.split_k_slices = kwargs["split_k_slices"]
        else:
            self.split_k_mode = SplitKMode.Serial
            self.split_k_slices = 1

        if "output_op" in kwargs.keys() and self.split_k_mode != SplitKMode.Parallel:
            self.output_op = kwargs["output_op"]
        else:
            self.output_op = self.operation.epilogue_type(1.0, 0.0)

        self.problem_size = problem_size
        self.problem_size.split_k_slices = self.split_k_slices

        self.initialize()

    def get_arguments(self):
        tc_numel = -1
        if hasattr(self, "tensor_c_numel"):
            tc_numel = self.tensor_c_numel

        self.c_arguments = self.operation.argument_type(
            int(self.conv_kind),
            self.problem_size.ctype,
            int(to_device_ptr(self.ptr_A)),
            int(to_device_ptr(self.ptr_B)),
            int(to_device_ptr(self.ptr_C)),
            int(to_device_ptr(self.ptr_D)),
            tc_numel,
            self.output_op,
            int(self.split_k_mode)
        )

    def initialize(self):
        self.launch_config = self.operation.rt_module.plan(self)

        self.get_arguments()

        # Allocate and initialize device workspace
        device_workspace_size = self.operation.rt_module.get_workspace_size(self.c_arguments)
        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        self.semaphore = 0
        if workspace_ptr is not None and self.split_k_mode == SplitKMode.Parallel:
            self.ptr_D = workspace_ptr
            # Reset arguments now that ptr_D has been updated
            self.get_arguments()
        elif workspace_ptr is not None and self.split_k_mode == SplitKMode.Serial:
            self.semaphore = workspace_ptr

        params_ = self.operation.rt_module.get_args(
            self.c_arguments, ctypes.c_void_p(int(self.semaphore)))
        self.host_workspace = bytearray(params_.contents)
        self.device_workspace = None

    def sync(self):
        """
        Synchronize the arguments. If the input tensor is in host,
        copy it from device to host.
        """
        return super().sync()


class Conv2dRT(ExecutableOperation):
    """
    Conv2dRT manages the CUTLASS runtime components
    """

    KernelTemplate = r"""
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  ${operation_name}${operation_suffix}::SharedStorage *shared_storage =
      reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);

  ${operation_name}${operation_suffix} op;

  op(params, *shared_storage);
}
    """

    HostTemplate = r"""
extern "C" {
  // Get the size of params in bytes
  int ${operation_name}_get_param_size(){
    return sizeof(${operation_name}${operation_suffix}::Params);
  }

  // Get the size of dynamic shared memory in bytes
  int ${operation_name}_shared_memory_size() {
    return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
  }

  using ElementA = typename ${operation_name}_base::ElementA;
  using ElementB = typename ${operation_name}_base::ElementB;
  using ElementC = typename ${operation_name}_base::ElementC;
  using LayoutA = typename ${operation_name}_base::LayoutA;
  using LayoutB = typename ${operation_name}_base::LayoutB;
  using LayoutC = typename ${operation_name}_base::LayoutC;
  using EpilogueOutputOp = typename ${operation_name}_base::EpilogueOutputOp;

  struct ${operation_name}_TemporaryArgs {
    int conv_kind;
    cutlass::conv::Conv2dProblemSize problem_size;
    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementC* ptr_C;
    ElementC* ptr_D;
    int tensor_c_numel;
    typename EpilogueOutputOp::Params epilogue_params;
    int split_k_mode;
  };

  typename ${operation_name}${operation_suffix}::Arguments
  construct_arguments(${operation_name}_TemporaryArgs args) {
    cutlass::conv::Operator conv_operator = static_cast<cutlass::conv::Operator>(args.conv_kind);
    auto tc_A = cutlass::conv::implicit_gemm_tensor_a_extent(conv_operator, args.problem_size);
    auto tc_B = cutlass::conv::implicit_gemm_tensor_b_extent(conv_operator, args.problem_size);
    auto tc_C = cutlass::conv::implicit_gemm_tensor_c_extent(conv_operator, args.problem_size);
    auto tc_D = cutlass::conv::implicit_gemm_tensor_c_extent(conv_operator, args.problem_size);

    auto size_C = tc_C.at(0) * tc_C.at(1) * tc_C.at(2) * tc_C.at(3);
    if (args.tensor_c_numel >= 0 && args.tensor_c_numel == tc_C.at(3) && args.tensor_c_numel < size_C) {
      // C is interpreted as bias
      tc_C = {0, 0, 0, 0};
    }

    cutlass::TensorRef<ElementA, LayoutA> tref_A(args.ptr_A, LayoutA::packed(tc_A));
    cutlass::TensorRef<ElementB, LayoutA> tref_B(args.ptr_B, LayoutB::packed(tc_B));
    cutlass::TensorRef<ElementC, LayoutA> tref_C(args.ptr_C, LayoutC::packed(tc_C));
    cutlass::TensorRef<ElementC, LayoutA> tref_D(args.ptr_D, LayoutC::packed(tc_D));

    return {
      args.problem_size,
      tref_A,
      tref_B,
      tref_C,
      tref_D,
      args.epilogue_params,
      static_cast<cutlass::conv::SplitKMode>(args.split_k_mode)
    };
  }

  // Get the params as byte array
  char* ${operation_name}_get_params(${operation_name}_TemporaryArgs args, int *semaphore=nullptr) {
    auto arguments = construct_arguments(args);
    typename ${operation_name}${operation_suffix}::Params* params;
    params = new ${operation_name}${operation_suffix}::Params(arguments, semaphore);

    char *bytes = ((char*)(params));
    char *output = new char[sizeof(${operation_name}${operation_suffix}::Params)];
    for (unsigned int i = 0; i < sizeof(${operation_name}${operation_suffix}::Params); i ++)
      output[i] = bytes[i];

    return output;
  }

  dim3 ${operation_name}_get_grid_shape(
    int conv_kind,
    cutlass::conv::Conv2dProblemSize problem_size,
    cutlass::gemm::GemmCoord tile_size,
    int split_k_slices
  ) {

    using Swizzle = typename ${operation_name}_base::ThreadblockSwizzle;
    auto tiled_shape = Swizzle::get_tiled_shape(
      static_cast<cutlass::conv::Operator>(conv_kind),
      problem_size,
      tile_size,
      split_k_slices);

    return Swizzle::get_grid_shape(tiled_shape);
  }

  size_t ${operation_name}_get_workspace_size(${operation_name}_TemporaryArgs args) {
    auto arguments = construct_arguments(args);

    // Temporarily define device::-level Conv2d so that we can call get_workspace_size
    using DeviceConv = cutlass::conv::device::ImplicitGemmConvolution<${operation_name}_base>;
    return DeviceConv::get_workspace_size(arguments);
  }
}

    """

    def __init__(self, operation: "Conv2dOperation"):
        super().__init__(operation)
        self.extra_funcs = {
            "get_grid_shape": dim3_,
            "get_workspace_size": ctypes.c_uint64
        }
        self.argument_type, self.epilogue_type = get_conv2d_arguments(operation.epilogue_functor)
        self.argtype = [ctypes.POINTER(self.argument_type), ctypes.c_void_p]
        self.conv_kind = operation.conv_kind

        self.operation: Conv2dOperation = operation

        self.emitter = EmitConv2dInstance("_type")

        self.threads = operation.tile_description.num_threads

        self.swizzle_functor = operation.swizzling_functor

    def emit(self):
        return self.emitter.emit(self.operation)

    def plan(self, arguments: Conv2dArguments):
        tile_size = GemmCoord(
            self.operation.tile_description.threadblock_shape[0],
            self.operation.tile_description.threadblock_shape[1],
            self.operation.tile_description.threadblock_shape[2],
        )

        grid = self.get_grid_shape(
            int(self.conv_kind),
            arguments.problem_size.ctype,
            tile_size.ctype,
            arguments.split_k_slices
        )

        return LaunchConfiguration(
            [grid.x, grid.y, grid.z], [self.threads, 1, 1],
            self.shared_memory_capacity)

    def initialize(self):
        err, = cuda.cuFuncSetAttribute(
            self.kernel,
            attrib=cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value=self.shared_memory_capacity)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error: {err}")


class Conv2dOperation:
    """
    CUTLASS Conv2d operation description.

    :param conv_kind: convolution operator
    :type conv_kind: :class:`cutlass_library.library.ConvKind`

    :param iterator_algorithm: Selects among several implementation
    variants trading off performance with simplicity
    :type iterator_algorithm: :class:`cutlass_library.library.IteratorAlgorithm`

    :param arch: GPU compute capability (sm_xx)
    :type arch: int

    :param tile_description: tile description
    :type tile_description: :class:`cutlass_cppgen.backend.TileDescription`

    :param A: tensor A description
    :type A: :class:`cutlass_cppgen.backend.TensorDescription`

    :param B: tensor B description
    :type B: :class:`cutlass_cppgen.backend.TensorDescription`

    :param C: tensor C description
    :type C: :class:`cutlass_cppgen.backend.TensorDescription`

    :param D: tensor D description
    :type D: :class:`cutlass_cppgen.backend.TensorDescription`

    :param element_epilogue: element type for computation in epilogue \
    :type element_epilogue: cutlass_library.library.DataType

    :param stride_support: distinguish among partial specializations that \
    accelerate certain problems where convolution stride is unit \
    :type stride_support: :class:`cutlass_library.library.StrideSupport`

    :param epilogue_functor: convolution epilogue functor
    :type epilogue_functor: :class:`EpilogueFunctor`

    :param swizzling_functor: threadblock swizzling functor
    """
    def __init__(
        self,
        conv_kind,
        iterator_algorithm,
        arch: int,
        tile_description: TileDescription,
        A: TensorDescription,
        B: TensorDescription,
        C: TensorDescription,
        stride_support,
        epilogue_functor,
        swizzling_functor=SwizzlingFunctor.Identity1,
        emission_type=EmissionType.Kernel,
        **kwargs
    ):
        self.operation_kind: OperationKind = OperationKind.Conv2d
        self.arch: int = arch
        self.tile_description: TileDescription = tile_description
        self.conv_kind = conv_kind
        self.A: TensorDescription = A
        self.B: TensorDescription = B
        self.C: TensorDescription = C
        self.epilogue_functor = epilogue_functor
        self.iterator_algorithm = iterator_algorithm
        self.stride_support = stride_support
        self.swizzling_functor = swizzling_functor

        self.emission_type = emission_type

        self.rt_module: Conv2dRT = Conv2dRT(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    def run(self, arguments: Conv2dArguments) -> cuda.CUresult:
        """
        Launch the cuda kernel with input arguments

        :param arguments: conv2d arguments
        :type arguments: :class:`cutlass_cppgen.backend.Conv2dArguments`
        """

        # launch the kernel
        err = self.rt_module.run(
            arguments.host_workspace,
            arguments.device_workspace,
            arguments.launch_config,
            arguments.stream
        )

        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA Error {err}")

        return err

    #
    # Get function name
    #

    def procedural_name(self):
        """The full procedural name indicates architecture, extended name, tile size, and layout."""
        return self.configuration_name()

    def configuration_name(self):
        """The full procedural name indicates architecture, extended name, tile size, and layout."""

        opcode_class_name = OpcodeClassNames[
            self.tile_description.math_instruction.opcode_class
        ]

        threadblock = "%dx%d_%dx%d" % (
            self.tile_description.threadblock_shape[0],
            self.tile_description.threadblock_shape[1],
            self.tile_description.threadblock_shape[2],
            self.tile_description.stages,
        )

        if self.stride_support == StrideSupport.Unity:
            configuration_name = "cutlass_sm${arch}_${opcode_class}_${extended_name}_${threadblock}_${layout}_unity_stride_align${alignment}"
        else:
            configuration_name = "cutlass_sm${arch}_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}"

        return SubstituteTemplate(
            configuration_name,
            {
                "arch": str(self.arch),
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": "%d" % self.A.alignment
            },
        )

    def extended_name(self):
        """Append data types if they differ from compute type."""
        if self.C.element != self.tile_description.math_instruction.element_accumulator and \
                self.A.element != self.tile_description.math_instruction.element_accumulator:
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
                self.A.element != self.tile_description.math_instruction.element_accumulator:
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = SubstituteTemplate(extended_name, {
            "element_a": DataTypeNames[self.A.element],
            "element_c": DataTypeNames[self.C.element],
            "core_name": self.core_name(),
        })

        return extended_name

    def layout_name(self):
        return "%s" % (ShortLayoutTypeNames[self.A.layout])

    def core_name(self):
        """The basic operation kind is prefixed with a letter indicating the accumulation type."""

        intermediate_type = ""

        if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp:
            inst_shape = "%dx%dx%d" % tuple(
                self.tile_description.math_instruction.instruction_shape)
            if self.tile_description.math_instruction.element_a != self.A.element and \
                    self.tile_description.math_instruction.element_a != self.accumulator_type():
                intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
        else:
            inst_shape = ""

        return "%s%s%s%s_%s" % (
            ShortDataTypeNames[self.accumulator_type()],
            inst_shape,
            intermediate_type,
            ConvKindNames[self.conv_kind],
            IteratorAlgorithmNames[self.iterator_algorithm]
        )

    def is_complex(self):
        complex_operators = [
            MathOperation.multiply_add_complex,
            MathOperation.multiply_add_complex_gaussian,
        ]
        return self.tile_description.math_instruction.math_operation in complex_operators

    def accumulator_type(self):
        accum = self.tile_description.math_instruction.element_accumulator

        if self.is_complex():
            return get_complex_from_real(accum)

        return accum

    def device_op(self):
        """
        Returns a new Conv2dOperation object that is constructed with emission type
        ``EmissionType.Device``.

        :return: operation ready for device-level code emission
        :rtype: Conv2dOperation
        """
        return Conv2dOperation(
            self.conv_kind, self.iterator_algorithm, self.arch, self.tile_description,
            self.A, self.B, self.C, self.stride_support, self.epilogue_functor, self.swizzling_functor,
            emission_type=EmissionType.Device)


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################


class EmitConv2dInstance:
    def __init__(self, operation_suffix=""):
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/conv/kernel/default_conv2d_fprop.h",
            "cutlass/conv/kernel/default_conv2d_dgrad.h",
            "cutlass/conv/kernel/default_conv2d_wgrad.h",
            "cutlass/conv/device/implicit_gemm_convolution.h"
        ]
        self.template = """
// Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
using ${operation_name}_base =
typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}<
  ${element_a},
  ${layout_a},
  ${element_b},
  ${layout_b},
  ${element_c},
  ${layout_c},
  ${element_accumulator},
  ${opcode_class},
  ${arch},
  cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
  cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
  cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
  ${epilogue_functor},
  ${swizzling_functor},
  ${stages},
  ${math_operator},
  ${iterator_algorithm},
  ${stride_support},
  ${align_a},
  ${align_b}
>::Kernel;

struct ${operation_name}${operation_suffix}:
  public ${operation_name}_base { };

"""

        self.template_device = """
// Conv2d operation ${operation_name}

using Conv2d${conv_kind_name}Kernel = typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}<
  ${element_a},
  ${layout_a},
  ${element_b},
  ${layout_b},
  ${element_c},
  ${layout_c},
  ${element_accumulator},
  ${opcode_class},
  ${arch},
  cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
  cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
  cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
  ${epilogue_functor},
  ${swizzling_functor},
  ${stages},
  ${math_operator},
  ${iterator_algorithm},
  ${stride_support},
  ${align_a},
  ${align_b}
>::Kernel;

using DeviceKernel =
    typename cutlass::conv::device::ImplicitGemmConvolution<Conv2d${conv_kind_name}Kernel>;
"""

    def emit(self, operation):
        warp_shape = [int(operation.tile_description.threadblock_shape[idx] /
                          operation.tile_description.warp_count[idx]) for idx in range(3)]

        epilogue_vector_length = int(min(
            operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

        values = {
            "operation_name": operation.procedural_name(),
            "operation_suffix": self.operation_suffix,
            "conv_kind": ConvKindTag[operation.conv_kind],
            "conv_kind_name": ConvKindNames[operation.conv_kind].capitalize(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(operation.tile_description.math_instruction.instruction_shape[0]),
            "instruction_shape_n": str(operation.tile_description.math_instruction.instruction_shape[1]),
            "instruction_shape_k": str(operation.tile_description.math_instruction.instruction_shape[2]),
            "epilogue_vector_length": str(epilogue_vector_length),
            "epilogue_functor": operation.epilogue_functor.emit(),
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "iterator_algorithm": IteratorAlgorithmTag[operation.iterator_algorithm],
            "iterator_algorithm_name": IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
            "stride_support": StrideSupportTag[operation.stride_support],
            "math_operator": "cutlass::arch::OpMultiplyAddComplex" if operation.is_complex() else MathOperationTag[operation.tile_description.math_instruction.math_operation],
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
        }

        if operation.emission_type == EmissionType.Kernel:
            conv2d_template = self.template
        else:
            conv2d_template = self.template_device

        return SubstituteTemplate(conv2d_template, values)
