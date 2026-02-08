# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from enum import Enum
from typing_extensions import deprecated
import warnings


@deprecated("Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead")
class SmemCapacity(Enum):
    SM80_SMEM_CAPACITY_BYTES = (164 - 1) * 1024
    SM86_SMEM_CAPACITY_BYTES = (100 - 1) * 1024
    SM89_SMEM_CAPACITY_BYTES = (100 - 1) * 1024


warnings.warn(
    "SMEM_CAPACITY is deprecated: Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead",
    DeprecationWarning,
    stacklevel=2,
)
# Dictionary to map compute capability to SMEM capacity
SMEM_CAPACITY = {
    "sm80": SmemCapacity.SM80_SMEM_CAPACITY_BYTES.value,
    "sm86": SmemCapacity.SM86_SMEM_CAPACITY_BYTES.value,
    "sm89": SmemCapacity.SM89_SMEM_CAPACITY_BYTES.value,
}
