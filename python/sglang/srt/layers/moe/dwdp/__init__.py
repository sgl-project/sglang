"""DWDP (Distributed Weight Data Parallelism): MoE prefill with tokens kept on-rank and peer expert weights prefetched via NVLink into a composite VMM address space."""

from sglang.srt.runtime_context import (
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from sglang.srt.utils import is_npu

# Import only the selected backend. The CUDA manager owns CUDA VMM resources;
# the NPU manager owns ACL VMM views and native stream-ordered peer copies.
if is_npu():
    from sglang.srt.hardware_backend.npu.dwdp.manager import (
        NpuDwdpManager as DwdpManager,
    )
else:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager

__all__ = [
    "DwdpManager",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]
