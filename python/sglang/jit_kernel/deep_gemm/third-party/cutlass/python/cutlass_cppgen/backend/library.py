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

"""
Common data types and string names/tags for them
"""

import enum

from cutlass_library import (
    ComplexTransform,
    DataType,
    DataTypeSize,
    EpilogueScheduleType,
    KernelScheduleSuffixes,
    KernelScheduleType,
    MathOperation,
    OpcodeClass,
    TileSchedulerType
)


# The following block implements enum.auto() for Python 3.5 variants that don't include it such
# as the default 3.5.2 on Ubuntu 16.04.
#
# https://codereview.stackexchange.com/questions/177309/reimplementing-pythons-enum-auto-for-compatibility

try:
    from enum import auto as enum_auto
except ImportError:
    __cutlass_library_auto_enum = 0

    def enum_auto() -> int:
        global __cutlass_library_auto_enum
        i = __cutlass_library_auto_enum
        __cutlass_library_auto_enum += 1
        return i


class DataTypeSizeBytes:
    """
    Static class to mimic the `DataTypeSize` dictionary, but with checks for whether the
    data type key is less than a full byte or a non-integer number of bytes.
    """

    @staticmethod
    def __class_getitem__(datatype):
        """
        Returns the number of bytes in size the data type is. Raises an exception if the data type
        is either less than a full byte or a non-integer number of bytes in size.

        :param datatype: data type to query

        :return: number of bytes the data type occupies
        :rtype: int
        """
        bits = DataTypeSize[datatype]
        if bits < 8:
            raise Exception(
                f"Data type {datatype} is less than one byte in size."
            )
        elif bits % 8 != 0:
            raise Exception(
                f"Data type datatype is not an integer number of bytes."
            )
        return bits // 8


class SchedulerMode(enum.Enum):
    Device = enum_auto()
    Host = enum_auto()


SchedulerModeTag = {
    SchedulerMode.Device: "cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly",
    SchedulerMode.Host: "cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute",
}


ShortSchedulerModeNames = {SchedulerMode.Device: "Device", SchedulerMode.Host: "Host"}


class FunctionalOp(enum.Enum):
    AtomicAdd = enum_auto()
    AtomicMaximum = enum_auto()
    Divides = enum_auto()
    Maximum = enum_auto()
    Minimum = enum_auto()
    Minus = enum_auto()
    Multiplies = enum_auto()
    MultiplyAdd = enum_auto()
    Plus = enum_auto()
    Exp = enum_auto()


FunctionalOpTag = {
    FunctionalOp.AtomicAdd: "cutlass::atomic_add",
    FunctionalOp.AtomicMaximum: "cutlass::atomic_maximum",
    FunctionalOp.Divides: "cutlass::divides",
    FunctionalOp.Maximum: "cutlass::maximum",
    FunctionalOp.Minimum: "cutlass::minimum",
    FunctionalOp.Minus: "cutlass::minus",
    FunctionalOp.Multiplies: "cutlass::multiplies",
    FunctionalOp.MultiplyAdd: "cutlass::multiply_add",
    FunctionalOp.Plus: "cutlass::plus",
    FunctionalOp.Exp: "cutlass::fast_exp_op",
}


class ActivationOp(enum.Enum):
    DGelu = enum_auto()
    Gelu = enum_auto()
    GeluTaylor = enum_auto()
    HardSwish = enum_auto()
    Identity = enum_auto()
    LeakyReLU = enum_auto()
    ReLU = enum_auto()
    Sigmoid = enum_auto()
    SiLU = enum_auto()
    Tanh = enum_auto()


ActivationOpTag = {
    ActivationOp.DGelu: "cutlass::epilogue::thread::dGELU",
    ActivationOp.Gelu: "cutlass::epilogue::thread::GELU",
    ActivationOp.GeluTaylor: "cutlass::epilogue::thread::GELU_taylor",
    ActivationOp.HardSwish: "cutlass::epilogue::thread::HardSwish",
    ActivationOp.Identity: "cutlass::epilogue::thread::Identity",
    ActivationOp.LeakyReLU: "cutlass::epilogue::thread::LeakyReLU",
    ActivationOp.ReLU: "cutlass::epilogue::thread::ReLu",
    ActivationOp.Sigmoid: "cutlass::epilogue::thread::Sigmoid",
    ActivationOp.SiLU: "cutlass::epilogue::thread::SiLu",
    ActivationOp.Tanh: "cutlass::epilogue::thread::Tanh",
}


def op_tag(op) -> str:
    """
    Dispatches `op` to the appropriate *Tag dictionary depending on whether
    `op` is an ActivationOp or FunctionalOp. This is useful for cases in which
    either type can be used.

    :param op: operation to emit a tag for
    :type op: ActivationOp | FunctionalOp

    :return: tag corresponding to op
    :rtype: str
    """
    if isinstance(op, ActivationOp):
        return ActivationOpTag[op]
    elif isinstance(op, FunctionalOp):
        return FunctionalOpTag[op]
    else:
        raise Exception(f"Unexpected op type {op}. Must be one of ActivationOp or FunctionalOp.")


class FloatRoundStyle(enum.Enum):
    ToNearest = enum_auto()
    ToNearestSatfinite = enum_auto()
    Indeterminate = enum_auto()
    TowardZero = enum_auto()
    TowardInfinity = enum_auto()
    TowardNegInfinity = enum_auto()
    HalfUlpTruncDntz = enum_auto()
    HalfUlpTruncate = enum_auto()


FloatRoundStyleTag = {
    FloatRoundStyle.ToNearest: "cutlass::FloatRoundStyle::round_to_nearest",
    FloatRoundStyle.ToNearestSatfinite: "cutlass::FloatRoundStyle::round_to_nearest_satfinite",
    FloatRoundStyle.Indeterminate: "cutlass::FloatRoundStyle::round_indeterminate",
    FloatRoundStyle.TowardZero: "cutlass::FloatRoundStyle::round_toward_zero",
    FloatRoundStyle.TowardInfinity: "cutlass::FloatRoundStyle::round_toward_infinity",
    FloatRoundStyle.TowardNegInfinity: "cutlass::FloatRoundStyle::round_toward_neg_infinity",
    FloatRoundStyle.HalfUlpTruncDntz: "cutlass::FloatRoundStyle::round_half_ulp_trunc_dntz",
    FloatRoundStyle.HalfUlpTruncate: "cutlass::FloatRoundStyle::round_half_ulp_truncate",
}


class MathInstruction:
    """
    Description of a the lowest-level matrix-multiply-accumulate operation to be used in a kernel
    """

    def __init__(
        self,
        instruction_shape,
        element_a,
        element_b,
        element_accumulator,
        opcode_class=OpcodeClass.Simt,
        math_operation=MathOperation.multiply_add,
    ):
        """
        :param instruction_shape: size of the [M, N, K] dimensions of the instruction
        :type instruction_shape: list or tuple
        :param element_a: data type of operand A
        :param element_b: data type of operand B
        :param element_accumulator: data type used in accumulation
        :param opcode_class: higher-level class of the instruction (e.g., SIMT or Tensor Core)
        :type opcode_class: cutlass_library.library.OpcodeClass
        :param math_operation: the type of low-level operation to be performed (e.g., multiply accumulate)
        :type math_operation: MathOperation
        """
        self.instruction_shape = instruction_shape
        self.element_a = element_a
        self.element_b = element_b
        self.element_accumulator = element_accumulator
        self.opcode_class = opcode_class
        self.math_operation = math_operation


def to_blackwell_threadblock_shape(tile_description, cluster_shape, kernel_schedule):
    blackwell_threadblock_shape = tile_description.threadblock_shape
    is_2sm = False if kernel_schedule is None else ("2sm" in KernelScheduleSuffixes[kernel_schedule])
    if cluster_shape[0] > 0:
        blackwell_threadblock_shape = [
            tile_description.threadblock_shape[0] // cluster_shape[0],
            tile_description.threadblock_shape[1] // cluster_shape[1],
            tile_description.threadblock_shape[2] // cluster_shape[2]
        ]
        if is_2sm:
            blackwell_threadblock_shape[0] *= 2
    else:
        blackwell_threadblock_shape = tile_description.math_instruction.instruction_shape
    return blackwell_threadblock_shape, is_2sm


class TileDescription:
    """
    Description of a tile of computation to be performed in the kernel, encompassing threadblock, cluster, and warp shapes,
    stage count, and math instruction specification
    """

    def __init__(
        self,
        threadblock_shape,
        stages,
        warp_count,
        math_instruction,
        cluster_shape=[1, 1, 1],
        kernel_schedule: KernelScheduleType = None,
        epilogue_schedule: EpilogueScheduleType = None,
        tile_scheduler: TileSchedulerType = None
    ):
        """
        :param threadblock_shape: shape of a threadblock tyle
        :type threadblock_shape: list or tuple
        :param stages: number of pipline stages in the operation. For SM90 kernels, this can be set to `None` and the maximum
                       number of stages that can be supported for an operation on a given architecture will be computed at a later time
        :type stages: int or None
        :param warp_count: number of warps in each [M, N, K] dimension of a threadblock tile
        :type warp_count: list, tuple, or None
        :param math_instruction: specification of the instruction type and shape to be performed and the types of its operands
        :type math_instruction: MathInstruction
        :param cluster_shape: number of threadblocks in the [X, Y, Z] dimensions of a threadblock cluster
        :param kernel_schedule: type of kernel schedule to use (only available for SM90+)
        :type kernel_schedule: cutlass_library.KernelScheduleType
        :param epilogue_schedule: type of epilogue schedule to use (only available for SM90+)
        :type epilogue_schedule: cutlass_library.EpilogueScheduleType
        :param tile_scheduler: type of tile scheduler to use (only available for SM90+)
        :type tile_scheduler: cutlass_library.TileSchedulerType
        """
        if ((kernel_schedule is None and epilogue_schedule is not None) or
            (kernel_schedule is not None and epilogue_schedule is None)):
            raise Exception("Kernel and epilogue schedule must either both be Auto or neither be Auto.")

        self.threadblock_shape = threadblock_shape
        self.cluster_shape = cluster_shape
        self.kernel_schedule = kernel_schedule
        self.epilogue_schedule = epilogue_schedule
        self.tile_scheduler = tile_scheduler
        self.stages = stages

        self.math_instruction = math_instruction
        self.instruction_shape = math_instruction.instruction_shape

        # Number of warps along x, y, z directions
        self.warp_count = warp_count

        self.blackwell_threadblock_shape, self.is_2sm = to_blackwell_threadblock_shape(self, self.cluster_shape, self.kernel_schedule)

    def clone_and_update(self, td: dict):
        attrs = {
            "cluster_shape": None,
            "threadblock_shape": None,
            "warp_count": None,
            "stages": None,
            "instruction_shape": None,
            "kernel_schedule": None,
            "epilogue_schedule": None,
            "tile_scheduler": None
        }
        for key in attrs.keys():
            if key in td.keys():
                attrs[key] = td[key]
            else:
                attrs[key] = getattr(self, key)

        attrs["math_instruction"] = MathInstruction(
            attrs["instruction_shape"],
            self.math_instruction.element_a,
            self.math_instruction.element_b,
            self.math_instruction.element_accumulator,
            self.math_instruction.opcode_class,
            self.math_instruction.math_operation
        )

        # Remove the instruction shape
        del attrs["instruction_shape"]

        return TileDescription(**attrs)

    @property
    def num_threads(self):
        """
        Returns the number of threads in the threadblock

        :return: number of threads in the threadblock
        :rtype: int or None (if warp count is None)
        """
        if self.warp_count is not None:
            threads = 32
            for cnt in self.warp_count:
                threads *= cnt
            return threads
        return None

    def procedural_name(self):
        """
        Returns a name identifying the tile description

        :return: name identifying the tile description
        :rtype: int
        """
        emit_stages = 0 if self.stages is None else self.stages
        name = "%dx%dx%d_%dx%d_%dx%d" % (
            self.cluster_shape[0],
            self.cluster_shape[1],
            self.cluster_shape[2],
            self.threadblock_shape[0],
            self.threadblock_shape[1],
            self.threadblock_shape[2],
            emit_stages
        )

        return name

    def procedural_name_2x(self):
        """
        Returns a name identifying the tile description

        :return: name identifying the tile description
        :rtype: int
        """
        return "%dx%d_%dx%d" % (self.threadblock_shape[0], self.threadblock_shape[1], self.threadblock_shape[2], self.stages)

    def __str__(self):
        """
        Returns a string with containing each of the tile description's values

        :return: contents of tile description
        :rtype: str
        """
        if self.kernel_schedule is not None:
            kschedule = self.kernel_schedule
        else:
            kschedule = KernelScheduleType.ScheduleAuto

        if self.epilogue_schedule is not None:
            eschedule = self.epilogue_schedule
        else:
            eschedule = EpilogueScheduleType.ScheduleAuto

        if self.tile_scheduler is not None:
            tschedule = self.tile_scheduler.name
        else:
            tschedule = "None"
        return f"""
{{
  ClusterShape: {self.cluster_shape}
  ThreadblockShape: {self.threadblock_shape}
  WarpCount: {self.warp_count}
  Stages: {self.stages if self.stages is not None else 'Auto'}
  InstructionShape: {self.math_instruction.instruction_shape}
  Kernel schedule: {kschedule.name}
  Epilogue schedule: {kschedule.name}
  TileScheduler: {tschedule}
}}"""


class TensorDescription:
    def __init__(self, element, layout, alignment=1, complex_transform=ComplexTransform.none):
        self.element = element
        self.layout = layout
        if element != DataType.void:
            self.alignment = min(128 // DataTypeSize[self.element], alignment)
        else:
            self.alignment = alignment
        self.complex_transform = complex_transform


def CalculateSmemUsagePerStage(operation):
    """
    Returns the amount of shared memory in bytes consumed in a single stage of a kernel.

    :param op: operation for which the maximum stages should be computed. If stages are
               set via the `op.tile_description.stages` parameter, this setting is ignored
               in the present calculation
    :type op: cutlass_cppgen.backend.Operation

    :return: number of bytes of shared memory consumed by a single stage
    :rtype: int
    """
    m, n, k = operation.tile_description.threadblock_shape

    if operation.operation_kind == OperationKind.Gemm:
        stage_barrier_bytes = 32
        return (
            (DataTypeSize[operation.A.element] * m * k // 8)
            + (DataTypeSize[operation.B.element] * k * n // 8)
            + stage_barrier_bytes
        )
    else:
        raise Exception("Unsupported operation kind {}.".format(operation.operation_kind))


def CalculateSmemUsage(operation):
    """
    Returns the amount of shared memory in bytes consumed by a kernel.

    :param op: operation for which the maximum stages should be computed. If stages are
               set via the `op.tile_description.stages` parameter, this setting is ignored
               in the present calculation
    :type op: cutlass_cppgen.backend.Operation

    :return: int
    """
    return operation.tile_description.stages * CalculateSmemUsagePerStage(operation)


class ApiVersion(enum.Enum):
    """
    Differentiate between CUTLASS 2.x and 3.x API versions
    """

    v2x = enum_auto()
    v3x = enum_auto()


def api_version(arch, opclass, dtype):
    """
    Returns whether the architecture, opcode class, and datatype in question require using CUTLASS 2.x
    or 3.x for code emission.

    :param arch: compute capability of device on which to run
    :type arch: int
    :param opclass: class of the operation being performed
    :type opclass: cutlass_library.OpcodeClass
    :param dtype: data type to be used in operation (assumes that ElementA and ElementB are the same)
    :type dtype: cutlass_library.DataType

    :return: API version to be used in code emission
    :rtype: ApiVersion
    """
    if (arch in [90, 100, 101, 103] and
        opclass == OpcodeClass.TensorOp and
        (dtype != DataType.f64)):
        return ApiVersion.v3x
    else:
        return ApiVersion.v2x


class EmissionType(enum.Enum):
    """
    Tags for whether to emit a kernel- or device-level operation
    """

    Kernel = enum_auto()
    Device = enum_auto()
