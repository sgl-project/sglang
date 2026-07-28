"""ROCm runtime registration for SharedEP's HIP VMM and native epochs."""

from __future__ import annotations

from sglang.srt.layers.moe.shared_ep.runtime import (
    SharedEpRuntimeCapability,
    SharedEpRuntimeHooks,
)


def _create_rocm_state(**kwargs):
    from sglang.srt.layers.moe.shared_ep.state import create_shared_ep_state

    return create_shared_ep_state(**kwargs)


_ROCM_RUNTIME = SharedEpRuntimeHooks(
    name="hip_vmm",
    platform="rocm",
    create_state=_create_rocm_state,
    capabilities=frozenset(
        {
            SharedEpRuntimeCapability.RANK_MAJOR_VMM,
            SharedEpRuntimeCapability.SYSTEM_SCOPE_GPU_EPOCH,
        }
    ),
)


def get_shared_ep_runtime_hooks() -> SharedEpRuntimeHooks:
    return _ROCM_RUNTIME
