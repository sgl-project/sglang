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
Utilities for expressing shapes
"""

from cutlass_library import (
    ConvMode,
    ConvKind,
    LayoutType
)
from cutlass_cppgen.backend.c_types import (
    Conv2DProblemSize_,
    GemmCoord_,
    GemmCoordBatched_
)


class MatrixCoord:
    def __init__(self, row, col):
        self._row = row
        self._col = col

    @property
    def row(self):
        return self._row

    @property
    def column(self):
        return self._col

    def leading_dimension(self, layout: LayoutType) -> int:
        """
        Returns the leading dimension for a matrix with layout ``layout`` and shape provided by the MatrixCoord.

        :param layout: layout of matrix
        :type layout: cutlass_library.LayoutType

        :returns: leading dimension
        :rtype: int
        """
        if layout == LayoutType.RowMajor:
            return self._col
        elif layout == LayoutType.ColumnMajor:
            return self._row
        else:
            raise Exception(f'Unsupported layout for leading dimension calculation: {layout}')


class GemmCoord:
    def __init__(self, m: int, n: int, k: int):
        self._m = m
        self._n = n
        self._k = k

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    @property
    def k(self) -> int:
        return self._k

    @property
    def mk(self) -> MatrixCoord:
        return MatrixCoord(self._m, self._k)

    @property
    def mn(self) -> MatrixCoord:
        return MatrixCoord(self._m, self._n)

    @property
    def kn(self) -> MatrixCoord:
        return MatrixCoord(self._k, self._n)

    @property
    def ctype(self) -> GemmCoord_:
        return GemmCoord_(self._m, self._n, self._k)

    def batched_ctype(self, batch_count: int) -> GemmCoordBatched_:
        return GemmCoordBatched_(self._m, self._n, self._k, batch_count)


class Conv2DProblemSize:
    def __init__(
        self, n: int, h: int, w: int, c: int,
        k: int, r: int, s: int, c_: int,
        pad_h: int, pad_w: int, stride_h: int, stride_w: int,
        dilation_h: int, dilation_w: int, mode: ConvMode=ConvMode.CrossCorrelation,
        split_k_slices: int=1, groups: int=1):

        self.N = n
        self.H = h
        self.W = w
        self.C = c
        self.K = k
        self.R = r
        self.S = s
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.mode = int(mode)
        self.split_k_slices = split_k_slices
        self.groups = groups
        self.P = ((h + pad_h * 2 - r * dilation_h) // stride_h) + 1
        self.Q = ((w + pad_w * 2 - s * dilation_w) // stride_w) + 1

    @property
    def ctype(self) -> Conv2DProblemSize_:
        return Conv2DProblemSize_(self)

    def implicit_gemm_size(self, kind: ConvKind):
        if kind == ConvKind.Fprop:
            return GemmCoord(
                self.N * self.P * self.Q,
                self.K,
                self.R * self.S * self.C // self.groups
            )
        elif kind == ConvKind.Dgrad:
            return GemmCoord(
                self.N * self.H * self.W,
                self.C,
                self.R * self.S * self.K
            )
        elif kind == ConvKind.Wgrad:
            return GemmCoord(
                self.K,
                self.R * self.S * self.C,
                self.N * self.P * self.Q
            )

    @staticmethod
    def from_sizes(input_size, weight_size):
        K, R, S, _ = weight_size
        pad_h = R // 2
        pad_w = S // 2
        stride_h = 1
        stride_w = 1
        dilation_h = 1
        dilation_w = 1
        return Conv2DProblemSize(
            *input_size,
            *weight_size,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w
        )
