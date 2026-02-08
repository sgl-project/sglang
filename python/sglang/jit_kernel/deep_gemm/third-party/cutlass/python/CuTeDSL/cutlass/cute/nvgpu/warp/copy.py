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

from dataclasses import dataclass
from typing import Type

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir import ir

from ..common import OpError
from ...core import CopyOp, Trait, _pack_shape
from ...typing import Numeric


@dataclass(frozen=True)
class BaseOp(CopyOp):
    transpose: bool = False
    num_matrices: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.transpose, bool):
            raise OpError(
                self,
                "expects the 'transpose' Op parameter to be a bool instance",
            )

    def __str__(self) -> str:
        res = (
            f"{self.__class__.__name__[:-2]} Copy Operation"
            + f"\n  number of matrices = {self.num_matrices}"
        )
        if self.transpose:
            res += f"\n  transposed"
        return res


@dataclass(frozen=True)
class LdMatrix8x8x16bOp(BaseOp):
    """
    8x8 ``ldmatrix`` Operation.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>`__.
    This operation corresponds to the ``.m8n8`` qualifier.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_matrices not in [1, 2, 4]:
            raise OpError(
                self,
                "expects the 'num_matrices' Op parameter to be one of [1,2,4]",
            )

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "LdMatrix8x8x16bTrait":
        mode = _pack_shape((8, 8), loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.CopyAtomLdsmType.get(
            copy_internal_type.mlir_type,
            mode.type.attribute,
            _cute_nvgpu_ir.LdsmSzPattern.u16,
            self.num_matrices,
            ir.UnitAttr.get() if self.transpose else None,
        )
        return LdMatrix8x8x16bTrait(_cute_ir.atom(ty, loc=loc, ip=ip))


class LdMatrix8x8x16bTrait(Trait):
    pass


@dataclass(frozen=True)
class LdMatrix16x16x8bOp(BaseOp):
    """
    16x16 8-bit ``ldmatrix`` Operation.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>`__.
    This operation corresponds to the ``.m16n16`` and the ``.b16`` qualifiers.
    """

    def __init__(self, num_matrices: int) -> None:
        super().__init__(transpose=True, num_matrices=num_matrices)
        self._verify()

    def _verify(self):
        assert self.transpose, "transpose must be True"
        if self.num_matrices not in [1, 2]:
            raise OpError(
                self,
                "expects the 'num_matrices' Op parameter to be one of [1,2]",
            )

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "LdMatrix16x16x8bTrait":
        mode = _pack_shape((16, 16), loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.CopyAtomLdsmType.get(
            copy_internal_type.mlir_type,
            mode.type.attribute,
            _cute_nvgpu_ir.LdsmSzPattern.u8,
            self.num_matrices,
            ir.UnitAttr.get(),
        )
        return LdMatrix16x16x8bTrait(_cute_ir.atom(ty, loc=loc, ip=ip))


class LdMatrix16x16x8bTrait(Trait):
    pass


@dataclass(frozen=True)
class StMatrix8x8x16bOp(BaseOp):
    """
    8x8 ``stmatrix`` Operation.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix>`__.
    This operation corresponds to the ``m8n8`` qualifier.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_matrices not in [1, 2, 4]:
            raise OpError(
                self,
                "expects the 'num_matrices' Op parameter to be one of [1,2,4]",
            )

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "StMatrix8x8x16bTrait":
        mode = _pack_shape((8, 8), loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.CopyAtomStsmType.get(
            copy_internal_type.mlir_type,
            mode.type.attribute,
            self.num_matrices,
            ir.UnitAttr.get() if self.transpose else None,
        )
        return StMatrix8x8x16bTrait(_cute_ir.atom(ty, loc=loc, ip=ip))


class StMatrix8x8x16bTrait(Trait):
    pass


@dataclass(frozen=True)
class StMatrix16x8x8bOp(BaseOp):
    """
    16x8 ``stmatrix`` Operation.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix>`__.
    This operation corresponds to the ``m16n8`` qualifier.
    """

    def __init__(self, num_matrices: int) -> None:
        super().__init__(transpose=True, num_matrices=num_matrices)
        self._verify()

    def _verify(self):
        if self.num_matrices not in [1, 2, 4]:
            assert self.transpose, "transpose must be True"
            raise OpError(
                self,
                "expects the 'num_matrices' Op parameter to be one of [1,2,4]",
            )

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "StMatrix16x8x8bTrait":
        mode = _pack_shape((16, 8), loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.CopyAtomStsmType.get(
            copy_internal_type.mlir_type,
            mode.type.attribute,
            self.num_matrices,
            ir.UnitAttr.get(),
        )
        return StMatrix16x8x8bTrait(_cute_ir.atom(ty, loc=loc, ip=ip))


class StMatrix16x8x8bTrait(Trait):
    pass
