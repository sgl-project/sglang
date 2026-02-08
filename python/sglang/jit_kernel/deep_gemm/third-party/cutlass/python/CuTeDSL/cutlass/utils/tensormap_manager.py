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
from enum import Enum, auto
from typing import Tuple

from cutlass.cutlass_dsl import const_expr

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir

import cutlass.cute as cute


class TensorMapUpdateMode(Enum):
    """
    Enum class defining tensor map update modes.

    Modes:
    GMEM: Update tensormap in global memory
    SMEM: Load tensormap from global memory to shared memory,
    update it in shared memory, then store back to global memory
    """

    GMEM = auto()  # Update tensormap in global memory
    SMEM = auto()  # Update tensormap in shared memory


@dataclass(frozen=True)
class TensorMapManager:
    """
    Manages TensorMap operations including initialization and updates.
    Provides utilities to convert tensormap pointer to across different memory spaces.
    """

    tensormap_update_mode: TensorMapUpdateMode
    bytes_per_tensormap: int

    # convert given cute.Pointer or cutlass.Int64 to a cute.Pointer to tensormap.
    # address_space: the address space of the resulting tensormap pointer. It could be generic or gmem
    def get_tensormap_ptr(
        self,
        ptr: cute.Pointer,
        address_space=_cute_ir.AddressSpace.gmem,
    ) -> cute.Pointer:
        if address_space not in [
            _cute_ir.AddressSpace.gmem,
            _cute_ir.AddressSpace.generic,
        ]:
            raise ValueError(f"Invalid address space: {address_space} for tensormap")

        gmem_ptr_i64 = ptr.toint().ir_value()
        gmem_ptr_i64_align_ty = _cute_ir.ConstrainedIntType.get(
            self.bytes_per_tensormap, gmem_ptr_i64.type.width
        )
        gmem_ptr_i64_align = _cute_ir.assume(gmem_ptr_i64_align_ty, gmem_ptr_i64)
        gmem_ptr_ty = _cute_ir.PtrType.get(
            _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
            address_space,
            self.bytes_per_tensormap,
        )
        return _cute_ir.inttoptr(gmem_ptr_ty, gmem_ptr_i64_align)

    # init tensormap pointed by dst_ptr with the one inside copy_atom.
    # dst_ptr should be pointing to a global memory location or a smem location
    # warp_id specifies which warp to perform the initialization
    @cute.jit
    def init_tensormap_from_atom(
        self, copy_atom: cute.CopyAtom, dst_ptr: cute.Pointer, warp_id: int
    ) -> None:
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == warp_id:
            with cute.arch.elect_one():
                cute.nvgpu.cpasync.copy_tensormap(copy_atom, dst_ptr)
        cute.arch.sync_warp()
        return

    # Perform a fence operation to ensure previous `init_tensormap_from_atom` calls have been completed
    def fence_tensormap_initialization(
        self,
    ) -> None:
        if self.tensormap_update_mode == TensorMapUpdateMode.GMEM:
            cute.arch.fence_acq_rel_cta()
        return

    # Perform a fence operation to ensure previous `update_tensormap` calls have been completed
    def fence_tensormap_update(
        self,
        tensormap_ptr: cute.Pointer,
    ) -> None:
        cute.nvgpu.cpasync.fence_tma_desc_acquire(tensormap_ptr)
        return

    @cute.jit
    def update_tensormap(
        self,
        tensor_gmem: Tuple[cute.Tensor, ...],
        tma_copy_atom: Tuple[cute.CopyAtom, ...],
        tensormap_gmem_ptr: Tuple[cute.Pointer, ...],
        warp_id: int,
        tensormap_smem_ptr: Tuple[cute.Pointer, ...],
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # updates before touching tensormap in global memory
        if warp_idx == warp_id:
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for copy_atom, tensor, smem_ptr in zip(
                    tma_copy_atom, tensor_gmem, tensormap_smem_ptr
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(
                        copy_atom, tensor, smem_ptr
                    )
            # wait until it's safe to update tensormap in global memory
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            # updates to tensormap in global memory
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for gmem_ptr, smem_ptr in zip(tensormap_gmem_ptr, tensormap_smem_ptr):
                    cute.nvgpu.cpasync.cp_fence_tma_desc_release(gmem_ptr, smem_ptr)
            else:
                for copy_atom, tensor, gmem_ptr in zip(
                    tma_copy_atom, tensor_gmem, tensormap_gmem_ptr
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(
                        copy_atom, tensor, gmem_ptr
                    )
                cute.arch.sync_warp()
                cute.nvgpu.cpasync.fence_tma_desc_release()
