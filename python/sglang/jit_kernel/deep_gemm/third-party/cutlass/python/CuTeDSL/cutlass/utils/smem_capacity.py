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


SMEM_CAPACITY_MAP = {
    "sm_120": (100 - 1) * 1024,
    "sm_100": (228 - 1) * 1024,
    "sm_90": (228 - 1) * 1024,
    "sm_80": (164 - 1) * 1024,
    "sm_86": (100 - 1) * 1024,
    "sm_89": (100 - 1) * 1024,
}


def get_smem_capacity_in_bytes(compute_capability: str) -> int:
    if compute_capability not in SMEM_CAPACITY_MAP:
        raise ValueError(f"Unsupported compute capability: {compute_capability}")
    return SMEM_CAPACITY_MAP[compute_capability]
