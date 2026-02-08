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

import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu import tcgen05


class LayoutEnum(Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"

    def mma_major_mode(self):
        return (
            tcgen05.OperandMajorMode.K
            if self == LayoutEnum.ROW_MAJOR
            else tcgen05.OperandMajorMode.MN
        )

    def sm90_mma_major_mode(self):
        return (
            warpgroup.OperandMajorMode.K
            if self == LayoutEnum.ROW_MAJOR
            else warpgroup.OperandMajorMode.MN
        )

    def is_n_major_c(self):
        return self == LayoutEnum.ROW_MAJOR

    def is_m_major_c(self):
        return self == LayoutEnum.COL_MAJOR

    @staticmethod
    def from_tensor(tensor: cute.Tensor) -> "LayoutEnum":
        ret = None
        if tensor.leading_dim == 1:
            ret = LayoutEnum.ROW_MAJOR
        elif tensor.leading_dim == 0:
            ret = LayoutEnum.COL_MAJOR
        else:
            raise ValueError(f"Invalid leading dimension: {tensor.leading_dim}")

        return ret


__all__ = ["LayoutEnum"]
