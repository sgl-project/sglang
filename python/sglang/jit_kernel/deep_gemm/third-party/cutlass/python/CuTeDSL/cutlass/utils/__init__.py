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

from .static_persistent_tile_scheduler import (
    WorkTileInfo,
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)

from .hardware_info import (
    HardwareInfo,
)

from .blackwell_helpers import (
    compute_epilogue_tile_shape,
    get_smem_store_op,
    get_tmem_load_op,
    get_num_tmem_alloc_cols,
    make_smem_layout_a,
    make_smem_layout_b,
    make_smem_layout_epi,
    make_trivial_tiled_mma,
    make_blockscaled_trivial_tiled_mma,
)

from .hopper_helpers import (
    sm90_get_smem_store_op,
)

from .blockscaled_layout import (
    BlockScaledBasicChunk,
    tile_atom_to_shape_SF,
    make_smem_layout_sfa,
    make_smem_layout_sfb,
    make_tmem_layout_sfa,
    make_tmem_layout_sfb,
)

from .grouped_gemm_tile_scheduler_helper import (
    GroupSearchResult,
    GroupedGemmGroupSearchState,
    GroupedGemmTileSchedulerHelper,
    create_initial_search_state,
)

from .tensormap_manager import (
    TensorMapUpdateMode,
    TensorMapManager,
)

from .smem_allocator import SmemAllocator

from .layout import LayoutEnum

from .smem_capacity import (
    get_smem_capacity_in_bytes,
)

from .distributed_helpers import (
    spin_lock_wait,
    spin_lock_multimem_arrive,
    multimem_ld_reduce_8xf16,
    multimem_ld_reduce_4xf32,
    multimem_ld_reduce_8xbf16,
    multimem_ld_reduce_16xe4m3,
    multimem_ld_reduce_16xe5m2,
    multimem_st_4xb32,
    sm_wise_inter_gpu_multimem_barrier,
)

__all__ = [
    "get_smem_capacity_in_bytes",
    "SmemAllocator",
    "LayoutEnum",
    "WorkTileInfo",
    "PersistentTileSchedulerParams",
    "StaticPersistentTileScheduler",
    "TensorMapUpdateMode",
    "TensorMapManager",
    "GroupSearchResult",
    "GroupedGemmGroupSearchState",
    "create_initial_search_state",
    "GroupedGemmTileSchedulerHelper",
    "HardwareInfo",
]
