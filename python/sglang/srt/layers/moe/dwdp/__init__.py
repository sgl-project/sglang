"""DWDP: keep tokens local and prefetch distributed expert weights."""

from sglang.srt.runtime_context import (
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from sglang.srt.utils import is_npu

# Import only the selected backend. The CUDA manager owns CUDA VMM resources;
# the NPU manager uses ordinary ND compute buffers and native ACL stream copies.
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
