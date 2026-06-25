"""DWDP (Distributed Weight Data Parallelism) for MoE prefill.

DWDP eliminates workload imbalance and all-to-all communication during MoE
prefill by keeping tokens on-rank and asynchronously prefetching peer expert
weights via NVLink.  Uses CUDA VMM to present a composite virtual address space
where every MoE kernel sees a single contiguous [num_experts, ...] tensor.
"""

from sglang.srt.layers.moe.dwdp.dwdp_manager import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)

__all__ = [
    "DwdpManager",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]
