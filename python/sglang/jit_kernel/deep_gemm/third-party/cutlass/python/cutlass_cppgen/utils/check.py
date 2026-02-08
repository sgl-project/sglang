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
Utility functions for checking constraints on kernels and calculating kernel attributes
"""

import ctypes

from cutlass_library import DataTypeSize, KernelScheduleSuffixes, OperationKind, SharedMemPerCC

import cutlass_cppgen
from cutlass_cppgen.backend.library import TileDescription


def calculate_smem_usage_per_stage(td: TileDescription, operation_kind: OperationKind) -> int:
    """
    Returns the amount of shared memory in bytes consumed in a single stage of a kernel.

    :param td: tile description to compute shared memory of
    :type td: TileDescription
    :param operation_kind: identifier for the type of operation being performed
    :type operation_kind: cutlass_library.OperationKind

    :return: number of bytes of shared memory consumed by a single stage
    :rtype: int
    """
    m, n, k = td.blackwell_threadblock_shape
    if td.is_2sm:
        m //= 2

    if operation_kind == OperationKind.Gemm:
        stage_barrier_bytes = 32
        return (
            (DataTypeSize[td.math_instruction.element_a] * m * k // 8)
            + (DataTypeSize[td.math_instruction.element_b] * k * n // 8)
            + stage_barrier_bytes
        )
    else:
        raise Exception(f"No available shared memory calculation for operation kind {operation.operation_kind}")


def calculate_smem_usage(operation) -> int:
    """
    Returns the amount of shared memory in bytes consumed by a kernel.

    :return: number of bytes of shared memory consumed by the operation
    :return: int
    """
    _per_stage = calculate_smem_usage_per_stage(operation.tile_description, operation.operation_kind)
    return _per_stage * operation.tile_description.stages


def valid_stage_count(
    cc: int,
    kernel_cc: int,
    td: TileDescription,
    element_C: cutlass_cppgen.DataType = None,
    element_D: cutlass_cppgen.DataType = None,
    verbose: bool = True) -> tuple:
    """
    Checks whether a device with `cc` supports the number of stages within `tile_description`, both
    based on raw limits on the number of stages and based on shared memory capacity

    :param cc: compute capability of device in question
    :type cc: int
    :param kernel_cc: compute capability that the kernel targets (corresponding to the arch::SMxy tag in CUTLASS)
    :type kernel_cc: int
    :param td: tile description to check
    :type td: TileDescription
    :param element_C: data type of operand C
    :type element_C: cutlass_cppgen.DataType
    :param element_D: data type of operand D
    :type element_D: cutlass_cppgen.DataType
    :param verbose: whether to log warnings
    :type verbose: bool

    :return: tuple with the first element indicating whether the provided tile description is
             valid for the provided device and the second element being an error message
    :rtype: tuple
    """
    if kernel_cc in [90, 100, 101, 103]:
        if (td.stages is None or td.stages == 0):
            # Stage count of None or 0 for SM90 indicates that the CollectiveBuilder automatically
            # determines the stage count to use. Thus, all settings are valid in these scenarios.
            return (True, "")
        elif verbose:
            cutlass_cppgen.logger.warning(
                "Setting an explicit stage count for SM90 kernels currently may "
                "result in compilation errors if the combination of tile shape, "
                "stage count, and shared memory requirement of the epilogue exceeds "
                "the available shared memory per SM.")

    if td.stages <= 0:
        return (False, f"Stage counts must be positive integers. Tile description has stage count of {td.stages}.")

    if cc < 80 and td.stages != 2:
        return (False, f"Tile description has stage count of {td.stages}, "
                       f"but only 2 stages are supported on SM{cc}.")

    # The calculation below does not consider shared memory used by the epilogue and, thus,
    # only catches cases in which the mainloop exceeds the device's shared memory capacity.
    # This is not a concern for CUTLASS 2.x kernels, for which the shared memory of the
    # mainloop and epilogue is shared.
    smem_per_stage = calculate_smem_usage_per_stage(td, OperationKind.Gemm)
    smem_usage_mainloop = (smem_per_stage * td.stages)
    smem_arch = SharedMemPerCC[cc] << 10
    if smem_usage_mainloop > smem_arch:
        return ( False,
            "Configuration uses too much shared memory. Consider reducing stage count or tile shape.\n"
            f"Details:\n"
            f"Mainloop uses {smem_per_stage} bytes of shared memory per stage, and "
            f"{td.stages} stages for a total of {smem_usage_mainloop} bytes.\n"
            f"The maxmium amount of shared memory that can be used per block on CC {cc} is {smem_arch}.")

    return (True, "")


def valid_cluster_shape(cc: int, cluster_shape: list) -> tuple:
    """
    Checks whether a device with `cc` supports a thread block cluster of shape `cluster_shape`.

    :param cc: compute capability of device in question
    :type cc: int
    :param cluster_shape: dimensions of thread block cluster shape to check
    :type cluster_shape: list

    :return: tuple with the first element indicating whether the provided cluster shape is
             valid for the provided device and the second element being an error message
    :rtype: tuple
    """

    if cc < 90 or cc in [120, 121]:
        if cluster_shape != [1, 1, 1]:
            return (False,
                    f"Cluster shape for pre-SM90 architectures and SM 120 and 121 must be [1, 1, 1]. Received cluster shape of "
                    f"{cluster_shape} for SM{cc}.")
        else:
            return (True, "")

    if len(cluster_shape) != 3:
        return (False,
                f"Cluster shapes must be rank-3. Received {cluster_shape} (rank {len(cluster_shape)}")

    if cluster_shape[2] != 1:
        return (False,
                "CUTLASS kernels currently require the third dimension of cluster shape to be 1. "
                f"Received cluster shape of {cluster_shape}.")

    return (True, "")


def valid_schedule(
    cc: int,
    kernel_schedule: cutlass_cppgen.KernelScheduleType,
    epilogue_schedule: cutlass_cppgen.EpilogueScheduleType,
    tile_scheduler: cutlass_cppgen.TileSchedulerType) -> tuple:
    """
    Checks that the kernel and epilogue schedules passed in are a valid combination for
    a device of compute capability ``cc``.

    :param cc: compute capability of device in question
    :type cc: int
    :param kernel_schedule: kernel schedule type
    :type kernel_schedule: cutlass_cppgen.KernelScheduleType
    :param epilogue_schedule: epilogue schedule type
    :type epilogue_schedule: cutlass_cppgen.EpilogueScheduleType
    :param tile_scheduler: tile scheduler type
    :type tile_scheduler: cutlass_cppgen.TileSchedulerType

    :return: tuple with the first element indicating whether the provided schedules are
             valid for the provided device and the second element being an error message
    :rtype: tuple
    """
    kernel_auto = (kernel_schedule == cutlass_cppgen.KernelScheduleType.ScheduleAuto)
    epilogue_auto = (epilogue_schedule == cutlass_cppgen.EpilogueScheduleType.ScheduleAuto)
    tile_scheduler_default = (tile_scheduler == cutlass_cppgen.TileSchedulerType.Default)
    if (cc < 90 or cc in [120, 121]) and not (kernel_auto and epilogue_auto and tile_scheduler_default):
        return (False, "Non-default schedules are only supported on SM90 and beyond (excluding SM120 and SM121)")

    if cc == 90 and ((kernel_auto and not epilogue_auto) or (not kernel_auto and epilogue_auto)):
        return (False, "Kernel and epilogue schedules must either both be auto or neither be auto")

    if not tile_scheduler_default:
        cooperative_kernels = [cutlass_cppgen.KernelScheduleType.TmaWarpSpecializedCooperative, 
                               cutlass_cppgen.KernelScheduleType.CpAsyncWarpSpecializedCooperative]
        if cc == 90 and (tile_scheduler == cutlass_cppgen.TileSchedulerType.StreamK) and (kernel_schedule not in cooperative_kernels):
            return (False, "Stream-K tile scheduler is currently only supported with the cooperative kernel schedule")
    return (True, "")


def alignment_or_default(alignment_provided: int, default_alignment: int) -> int:
    """
    Returns `alignment_provided` if it is set, otherwise `default_alignment` and checks
    that `alignment_provided` does not exceed `default_alignment`.

    :param alignment_provided: alignment preference specified. Can be None.
    :type alignment_provided: int
    :param default_alignment: alignment to use if `alignment_provided` is None
    :type default_alignment: int

    :return: alignment to use
    :rtype: int
    """
    if alignment_provided is not None:
        if alignment_provided > default_alignment:
            raise Exception(f"Alignment {alignment_provided} exceeds the maximum supported of {default_alignment}.")
        return alignment_provided

    return default_alignment


def update_alignment(alignment_provided:int, default_alignment: int) -> int:
    """
    Returns `alignment_provided` if it is set, otherwise `default_alignment` and checks
    that `alignment_provided` does not exceed `default_alignment`.

    :param alignment_provided: alignment preference specified. Can be None.
    :type alignment_provided: int
    :param default_alignment: alignment to use if `alignment_provided` is None
    :type default_alignment: int

    :return: alignment to use
    :rtype: int
    """
    if alignment_provided is not None:
        if alignment_provided > default_alignment:
            if alignment_provided % default_alignment == 0:
                return default_alignment
            raise Exception(f"Alignment {alignment_provided} exceeds the maximum supported of {default_alignment}.")
        return alignment_provided

    return default_alignment
