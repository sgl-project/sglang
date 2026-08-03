"""DWDP: rank-local MoE tokens with prefetched peer expert weights."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager

__all__ = [
    "DwdpManager",
    "create_dwdp_manager",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]


def create_dwdp_manager(server_args):
    """Select a DWDP backend without importing vendor modules eagerly."""

    from sglang.srt.utils.common import is_hip

    backend = getattr(server_args, "dwdp_weight_backend", "auto")
    if is_hip():
        if backend == "vmm":
            try:
                from sglang.srt.layers.moe.dwdp.rocm_vmm_manager import (
                    RocmVmmDwdpManager,
                )

                if RocmVmmDwdpManager.is_available():
                    return RocmVmmDwdpManager(server_args)
                raise RuntimeError(
                    "ROCm DWDP VMM backend was requested but HIP VMM "
                    "capability is unavailable"
                )
            except ImportError:
                raise

        if backend in ("auto", "ipc"):
            from sglang.srt.layers.moe.dwdp.rocm_ipc import RocmIpcDwdpManager

            return RocmIpcDwdpManager(server_args)
        raise ValueError(f"Unsupported ROCm DWDP weight backend: {backend}")

    if backend == "ipc":
        raise ValueError("DWDP IPC/multi-B backend is only supported on ROCm")
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager

    return DwdpManager(server_args)


def __getattr__(name: str):
    # Keep submodules such as hip_vmm importable without eagerly loading the
    # CUDA-specific DWDP manager and transport dependency chain.
    if name == "DwdpManager":
        from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager

        value = DwdpManager
    elif name in ("get_global_dwdp_manager", "set_global_dwdp_manager"):
        from sglang.srt.runtime_context import (
            get_global_dwdp_manager,
            set_global_dwdp_manager,
        )

        value = {
            "get_global_dwdp_manager": get_global_dwdp_manager,
            "set_global_dwdp_manager": set_global_dwdp_manager,
        }[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
