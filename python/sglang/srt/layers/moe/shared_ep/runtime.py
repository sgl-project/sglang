"""Platform runtime hooks for SharedEP state and GPU epochs.

The framework layer deliberately does not import a platform VMM implementation.
CUDA keeps its existing implementation behind a lazy hook, while ROCm provides
its VMM/epoch implementation through ``shared_ep.rocm_runtime`` (or explicit
registration by an out-of-tree backend).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from sglang.srt.utils import is_hip


class SharedEpRuntimeCapability(str, Enum):
    RANK_MAJOR_VMM = "rank_major_vmm"
    SYSTEM_SCOPE_GPU_EPOCH = "system_scope_gpu_epoch"


_REQUIRED_CAPABILITIES = frozenset(
    {
        SharedEpRuntimeCapability.RANK_MAJOR_VMM,
        SharedEpRuntimeCapability.SYSTEM_SCOPE_GPU_EPOCH,
    }
)


@dataclass(frozen=True)
class SharedEpRuntimeHooks:
    """Framework-facing capabilities supplied by a platform runtime."""

    name: str
    platform: str
    create_state: Callable[..., Any]
    capabilities: frozenset[SharedEpRuntimeCapability]

    def validate(self) -> None:
        if self.platform not in ("cuda", "rocm"):
            raise ValueError(f"Unsupported SharedEP runtime platform {self.platform!r}")
        missing = _REQUIRED_CAPABILITIES.difference(self.capabilities)
        if missing:
            missing_names = sorted(capability.value for capability in missing)
            raise RuntimeError(
                f"SharedEP runtime {self.name!r} is missing capabilities "
                f"{missing_names}"
            )


_RUNTIME_HOOKS: dict[str, SharedEpRuntimeHooks] = {}


def register_shared_ep_runtime(
    hooks: SharedEpRuntimeHooks,
    *,
    replace: bool = False,
) -> None:
    """Register one platform implementation without importing the other."""

    hooks.validate()
    existing = _RUNTIME_HOOKS.get(hooks.platform)
    if existing is not None and not replace:
        raise ValueError(
            f"SharedEP runtime for {hooks.platform!r} is already registered "
            f"as {existing.name!r}"
        )
    _RUNTIME_HOOKS[hooks.platform] = hooks


def _create_cuda_state(**kwargs):
    # Keep CUDA VMM/PTX modules out of ROCm import paths.
    from sglang.srt.layers.moe.shared_ep.state import create_shared_ep_state

    return create_shared_ep_state(**kwargs)


_CUDA_RUNTIME = SharedEpRuntimeHooks(
    name="cuda_vmm",
    platform="cuda",
    create_state=_create_cuda_state,
    capabilities=_REQUIRED_CAPABILITIES,
)


def _load_rocm_runtime() -> None:
    module_name = "sglang.srt.layers.moe.shared_ep.rocm_runtime"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return

    if "rocm" in _RUNTIME_HOOKS:
        return
    get_hooks = getattr(module, "get_shared_ep_runtime_hooks", None)
    if get_hooks is not None:
        register_shared_ep_runtime(get_hooks())


def get_shared_ep_runtime_hooks(
    platform: str | None = None,
) -> SharedEpRuntimeHooks:
    """Resolve hooks for the active platform without cross-platform imports."""

    platform = platform or ("rocm" if is_hip() else "cuda")
    if platform == "cuda":
        hooks = _RUNTIME_HOOKS.get(platform, _CUDA_RUNTIME)
    elif platform == "rocm":
        _load_rocm_runtime()
        hooks = _RUNTIME_HOOKS.get(platform)
        if hooks is None:
            raise RuntimeError(
                "ROCm SharedEP requires a registered rank-major VMM and "
                "system-scope GPU epoch runtime"
            )
    else:
        raise ValueError(f"Unsupported SharedEP platform {platform!r}")

    hooks.validate()
    return hooks
