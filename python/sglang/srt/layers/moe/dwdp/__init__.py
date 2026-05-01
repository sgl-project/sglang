# Copyright 2024-2025 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Distributed Weight Data Parallelism (DWDP) for Sparse MoE Models.

DWDP distributes MoE expert weights across GPUs within a node, using async
P2P prefetch via CUDA IPC to pull remote expert weights before they are needed.
Each GPU holds a partition of experts locally and prefetches the remaining
experts from peers.
"""

from sglang.srt.layers.moe.dwdp.dwdp_manager import (
    DwdpExpertLayout,
    DwdpManager,
    DwdpWeightView,
    enable_dwdp,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)

__all__ = [
    "DwdpExpertLayout",
    "DwdpManager",
    "DwdpWeightView",
    "enable_dwdp",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]
