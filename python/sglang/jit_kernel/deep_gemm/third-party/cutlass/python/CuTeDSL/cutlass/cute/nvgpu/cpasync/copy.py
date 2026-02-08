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

import enum
from dataclasses import dataclass
from typing import Optional, Type

from cutlass.cutlass_dsl import CuTeDSL, t

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir import ir

from ...core import CopyOp, Trait, ReductionOp
from ...typing import Int16, Pointer, Integer, Numeric
from ..common import OpError
from ..tcgen05.mma import CtaGroup


####################################################################################################
#
# Aynchronous copies
#
####################################################################################################


class LoadCacheMode(enum.Enum):
    """
    An enumeration for the possible cache modes of a non-bulk ``cp.async`` instruction.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators>`__.
    """

    ALWAYS = _cute_nvgpu_ir.LoadCacheMode.always
    GLOBAL = _cute_nvgpu_ir.LoadCacheMode.global_
    STREAMING = _cute_nvgpu_ir.LoadCacheMode.streaming
    LAST_USE = _cute_nvgpu_ir.LoadCacheMode.last_use
    NONE = _cute_nvgpu_ir.LoadCacheMode.none

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}>"

    def _to_ir(self) -> _cute_nvgpu_ir.LoadCacheMode:
        return self.value


@dataclass(frozen=True)
class CopyG2SOp(CopyOp):
    """
    Non-bulk asynchronous GMEM to SMEM Copy Operation.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy>`__.
    """

    cache_mode: LoadCacheMode = LoadCacheMode.ALWAYS

    def __str__(self) -> str:
        res = "cp.async GMEM -> SMEM copy Operation"
        if self.cache_mode != LoadCacheMode.ALWAYS:
            res += f"\n  with cache mode = {self.cache_mode}"
        return res

    def _make_trait(
        self,
        copy_internal_type: Type[t.Numeric],
        *,
        loc=None,
        ip=None,
        **kwargs,
    ) -> "CopyG2STrait":
        num_bits_per_copy = kwargs.get("num_bits_per_copy", None)
        # Verify that the user provided enum values
        if not isinstance(self.cache_mode, LoadCacheMode):
            raise OpError(
                self,
                "expects the 'cache_mode' Op parameter to be a LoadCacheMode instance",
            )
        if not isinstance(num_bits_per_copy, int) or (num_bits_per_copy <= 0):
            raise ValueError(
                "expects a 'num_bits_per_copy' kw argument of type int that is positive "
                f"when creating a copy Atom for {self.__class__.__name__}"
            )
        # Verify that the user provided enum values
        if not isinstance(self.cache_mode, LoadCacheMode):
            raise OpError(
                self,
                "expects the 'cache_mode' Op parameter to be a LoadCacheMode instance",
            )
        ty = _cute_nvgpu_ir.CopyAtomSIMTAsyncCopyType.get(
            copy_internal_type.mlir_type, self.cache_mode._to_ir(), num_bits_per_copy
        )
        return CopyG2STrait(_cute_ir.atom(ty, loc=loc, ip=ip))


class CopyG2STrait(Trait):
    pass


####################################################################################################
#
# Bulk tensor copies a.k.a TMA copies
#
####################################################################################################

TMA_MBAR_PTR_FIELD_NAME = "tma_bar"
TMA_MASK_FIELD_NAME = "mcast_mask"
TMA_DESC_PTR_FIELD_NAME = "tma_descriptor_ptr"

#
# TMA GMEM -> SMEM copies
#


@dataclass(frozen=True)
class CopyBulkTensorTileG2SOp(CopyOp):
    """
    Bulk tensor asynchrnous GMEM to SMEM Copy Operation using the TMA unit.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`__.
    This Operation uses TMA in the ``.tile`` mode.
    """

    cta_group: CtaGroup = CtaGroup.ONE

    admissible_archs = [
        "sm_90",
        "sm_90a",
        "sm_100a",
        "sm_100f",
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.cta_group, CtaGroup):
            raise OpError(
                self, "expects the 'cta_group' parameter to be a CtaGroup instance"
            )
        # Arch verification
        arch = CuTeDSL._get_dsl().envar.arch
        if arch not in self.admissible_archs:
            raise OpError(
                self,
                f"expects arch to be one of {self.admissible_archs}, but got {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )
        if (self.cta_group == CtaGroup.TWO) and arch[:5] == "sm_90":
            raise OpError(
                self,
                f"CTA group of 2 is tcgen05-specific and is not and is not compatible with {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )

    def __str__(self) -> str:
        res = "cp.async GMEM -> SMEM bulk tensor copy Operation"
        if self.cta_group == CtaGroup.TWO:
            res += f"\n  CTA group = 2"
        return res

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "CopyBulkTensorTileG2SNonExecTrait":
        raise NotImplementedError(
            "Use cpasync.make_tiled_tma_atom to obtain a copy Atom for TMA"
        )

    def _to_ir(self) -> _cute_nvgpu_ir.TiledTmaLoadEnum:
        if self.cta_group == CtaGroup.ONE:
            return _cute_nvgpu_ir.TiledTmaLoadEnum.sm_90
        elif self.cta_group == CtaGroup.TWO:
            return _cute_nvgpu_ir.TiledTmaLoadEnum.sm_100_2sm
        else:
            assert False, "unrecognized self.cta_group"


class CopyBulkTensorTileG2SNonExecTrait(Trait):
    # We allow kw args to be dropped so that the user can write common code for non-multicast
    # and multicast loads.
    def unpack(
        self,
        *,
        loc=None,
        ip=None,
        tma_bar_ptr: Optional[Pointer] = None,
        tma_desc_ptr: Optional[Pointer] = None,
        **kwargs,
    ):
        """
        Custom implementation of unpack for non-executable TMAs.

        The non-multicast TMA load requires a `tma_bar_ptr` keyword argument to be provided when
        using `cute.copy`. Any other kw arguments will be ignored instead of triggering an error.
        """
        if not isinstance(tma_bar_ptr, Pointer):
            raise ValueError(
                "expects a pointer to an mbarrier to be provided via the tma_bar_ptr kw argument"
            )
        exec_value = _cute_nvgpu_ir.atom_make_exec_tma(self.value, loc=loc, ip=ip)
        attr_str = f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_MBAR_PTR_FIELD_NAME}>"
        attr = ir.Attribute.parse(attr_str)
        exec_value = _cute_nvgpu_ir.atom_set_value(
            exec_value, attr, tma_bar_ptr.value, loc=loc, ip=ip
        )
        if isinstance(tma_desc_ptr, Pointer):
            attr_str = f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_DESC_PTR_FIELD_NAME}>"
            attr = ir.Attribute.parse(attr_str)
            exec_value = _cute_nvgpu_ir.atom_set_value(
                exec_value, attr, tma_desc_ptr.value, loc=loc, ip=ip
            )
        return exec_value


#
# TMA GMEM -> SMEM multicast copies
#


@dataclass(frozen=True)
class CopyBulkTensorTileG2SMulticastOp(CopyOp):
    """
    Bulk tensor asynchrnous multicast GMEM to SMEM Copy Operation using the TMA unit.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`__.
    This Operation uses TMA in the ``.tile`` mode.
    """

    cta_group: CtaGroup = CtaGroup.ONE

    admissible_archs = [
        "sm_90",
        "sm_90a",
        "sm_100a",
        "sm_100f",
    ]

    def __post_init__(self):
        if not isinstance(self.cta_group, CtaGroup):
            raise OpError(
                self, "expects the 'cta_group' parameter to be a CtaGroup instance"
            )
        # Arch verification
        arch = CuTeDSL._get_dsl().envar.arch
        if arch not in self.admissible_archs:
            raise OpError(
                self,
                f"expects arch to be one of {self.admissible_archs}, but got {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )
        if (self.cta_group == CtaGroup.TWO) and arch[:5] == "sm_90":
            raise OpError(
                self,
                f"CTA group of 2 is tcgen05-specific and is not and is not compatible with {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )

    def __str__(self) -> str:
        res = "cp.async GMEM -> SMEM bulk tensor multicast copy Operation"
        if self.cta_group == CtaGroup.TWO:
            res += f"\n  CTA group = 2"
        return res

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "CopyBulkTensorTileG2SMulticastNonExecTrait":
        raise NotImplementedError(
            "Use cpasync.make_tiled_tma_atom to obtain a copy Atom for TMA"
        )

    def _to_ir(self) -> _cute_nvgpu_ir.TiledTmaLoadEnum:
        if self.cta_group == CtaGroup.ONE:
            return _cute_nvgpu_ir.TiledTmaLoadEnum.sm_90_multicast
        elif self.cta_group == CtaGroup.TWO:
            return _cute_nvgpu_ir.TiledTmaLoadEnum.sm_100_2sm_multicast
        else:
            assert False, "unrecognized self.cta_group"


class CopyBulkTensorTileG2SMulticastNonExecTrait(Trait):
    def unpack(
        self,
        *,
        loc=None,
        ip=None,
        tma_bar_ptr: Optional[Pointer] = None,
        mcast_mask=None,
        tma_desc_ptr=None,
    ):
        """
        Custom implementation of unpack for non-executable TMAs.

        The multicast TMA load requires a `tma_bar_ptr`  and a `mcast_mask` keyword arguments to be
        provided when using `cute.copy`.
        """
        if not isinstance(tma_bar_ptr, Pointer):
            raise ValueError(
                "expects a pointer to an mbarrier to be provided via the tma_bar_ptr kw argument"
            )
        if not isinstance(mcast_mask, Integer):
            raise ValueError(
                "expects a multicast mask to be provided via the mcast_mask kw argument"
            )
        exec_value = _cute_nvgpu_ir.atom_make_exec_tma(self.value, loc=loc, ip=ip)
        attr_str = f"#cute_nvgpu.atom_copy_field_tmaload<tma_bar>"
        attr = ir.Attribute.parse(attr_str)
        exec_value = _cute_nvgpu_ir.atom_set_value(
            exec_value, attr, tma_bar_ptr.value, loc=loc, ip=ip
        )
        attr_str = f"#cute_nvgpu.atom_copy_field_tmaload<mcast_mask>"
        attr = ir.Attribute.parse(attr_str)
        exec_value = _cute_nvgpu_ir.atom_set_value(
            exec_value, attr, Int16(mcast_mask).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
        if isinstance(tma_desc_ptr, Pointer):
            attr_str = f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_DESC_PTR_FIELD_NAME}>"
            attr = ir.Attribute.parse(attr_str)
            exec_value = _cute_nvgpu_ir.atom_set_value(
                exec_value, attr, tma_desc_ptr.value, loc=loc, ip=ip
            )
        return exec_value


#
# TMA SMEM -> GMEM copies
#


@dataclass(frozen=True)
class CopyBulkTensorTileS2GOp(CopyOp):
    """
    Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`__.
    This Operation uses TMA in the ``.tile`` mode.
    """

    admissible_archs = [
        "sm_90",
        "sm_90a",
        "sm_100a",
        "sm_100f",
    ]

    def __post_init__(self):
        # Arch verification
        arch = CuTeDSL._get_dsl().envar.arch
        if arch not in self.admissible_archs:
            raise OpError(
                self,
                f"expects arch to be one of {self.admissible_archs}, but got {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )

    def __str__(self) -> str:
        return "cp.async SMEM -> GMEM bulk tensor copy Operation"

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "CopyBulkTensorTileS2GTrait":
        raise NotImplementedError(
            "Use cpasync.make_tiled_tma_atom to obtain a copy Atom for TMA"
        )


class CopyBulkTensorTileS2GTrait(Trait):
    def unpack(self, *, loc=None, ip=None, tma_desc_ptr: Optional[Pointer] = None):
        """
        Custom implementation of unpack for non-executable TMAs.
        """
        exec_value = _cute_nvgpu_ir.atom_make_exec_tma(self.value, loc=loc, ip=ip)
        if isinstance(tma_desc_ptr, Pointer):
            attr_str = (
                f"#cute_nvgpu.atom_copy_field_tmastore<{TMA_DESC_PTR_FIELD_NAME}>"
            )
            attr = ir.Attribute.parse(attr_str)
            exec_value = _cute_nvgpu_ir.atom_set_value(
                exec_value, attr, tma_desc_ptr.value, loc=loc, ip=ip
            )
        return exec_value

@dataclass(frozen=True)
class CopyReduceBulkTensorTileS2GOp(CopyOp):
    """
    Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk>`__.
    This Operation uses TMA in the ``.tile`` mode.
    """

    reduction_kind: ReductionOp = ReductionOp.ADD

    admissible_archs = [
        "sm_90",
        "sm_90a",
        "sm_100a",
        "sm_100f",
    ]

    def __post__init__(self):
        # Arch verification
        arch = CuTeDSL.__get_dsl().envar.arch
        if arch not in self.admissible_archs:
            raise OpError(
                self,
                f"expects arch to be one of {self.admissible_archs}, but got {arch}",
                suggestion="Ensure env CUTE_DSL_ARCH matches your GPU architecture",
            )

    def __str__(self) -> str:
        return "cp.async SMEM -> GMEM bulk tensor reduction Operation"

    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ) -> "CopyReduceBulkTensorTileS2GTrait":
        raise NotImplementedError(
            "Use cpasync.make_tiled_tma_atom to obtain a copy Atom for TMA"
        )

    def _to_ir(self) -> _cute_nvgpu_ir.ReductionKind:
        if self.reduction_kind == ReductionOp.ADD:
            return _cute_nvgpu_ir.ReductionKind.ADD
        elif self.reduction_kind == ReductionOp.MIN:
            return _cute_nvgpu_ir.ReductionKind.MIN
        elif self.reduction_kind == ReductionOp.MAX:
            return _cute_nvgpu_ir.ReductionKind.MAX
        elif self.reduction_kind == ReductionOp.INC:
            return _cute_nvgpu_ir.ReductionKind.INC
        elif self.reduction_kind == ReductionOp.DEC:
            return _cute_nvgpu_ir.ReductionKind.DEC
        elif self.reduction_kind == ReductionOp.AND:
            return _cute_nvgpu_ir.ReductionKind.AND
        elif self.reduction_kind == ReductionOp.OR:
            return _cute_nvgpu_ir.ReductionKind.OR
        elif self.reduction_kind == ReductionOp.XOR:
            return _cute_nvgpu_ir.ReductionKind.XOR
        else:
            assert False, "unrecognized self.reduction_kind"


class CopyReduceBulkTensorTileS2GTrait(Trait):
    def unpack(self, *, loc=None, ip=None, tma_desc_ptr: Optional[Pointer] = None):
        """
        Custom implementation of unpack for non-executable TMAs.
        """
        exec_value = _cute_nvgpu_ir.atom_make_exec_tma(self.value, loc=loc, ip=ip)
        if isinstance(tma_desc_ptr, Pointer):
            attr_str = (
                f"#cute_nvgpu.atom_copy_field_tmareduce<{TMA_DESC_PTR_FIELD_NAME}>"
            )
            attr = ir.Attribute.parse(attr_str)
            exec_value = _cute_nvgpu_ir.atom_set_value(
                exec_value, attr, tma_desc_ptr.value, loc=loc, ip=ip
            )
        return exec_value

__all__ = [
    "LoadCacheMode",
    "CopyG2SOp",
    "CopyBulkTensorTileG2SOp",
    "CopyBulkTensorTileG2SMulticastOp",
    "CopyBulkTensorTileS2GOp",
    "CopyReduceBulkTensorTileS2GOp",
]
