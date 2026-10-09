"""DWDP: keep tokens on-rank and prefetch peer expert weights on demand."""

from sglang.srt.runtime_context import get_global_dwdp_manager, set_global_dwdp_manager


def __getattr__(name):
    if name == "DwdpManager":
        from sglang.srt.utils import is_npu

        if is_npu():
            from sglang.srt.hardware_backend.npu.dwdp.manager import NPUDwdpManager

            return NPUDwdpManager
        from .dwdp_manager import DwdpManager

        return DwdpManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DwdpManager", "get_global_dwdp_manager", "set_global_dwdp_manager"]
