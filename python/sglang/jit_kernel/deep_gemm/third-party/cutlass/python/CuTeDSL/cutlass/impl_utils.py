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


def check_value_in(
    value, possible_values: list, value_description: str, prefix=""
) -> None:
    if value not in possible_values:
        err_msg = prefix
        if err_msg != "":
            err_msg += ": "
        err_msg += f"invalid {value_description}, got {value}, must be one of {possible_values}"
        raise ValueError(err_msg)


def check_type_in(ty, possible_types: list, type_description: str, prefix="") -> None:
    if not isinstance(ty, type):
        ty = type(ty)
    if ty not in possible_types:
        err_msg = prefix
        if err_msg != "":
            err_msg += ": "
        err_msg += f"invalid type for {type_description}, got {ty}, must be one of {possible_types}"
        raise TypeError(err_msg)
