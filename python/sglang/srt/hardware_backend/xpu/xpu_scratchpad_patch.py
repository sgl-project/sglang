# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""XPU graph-safe scratchpad buffer management for sgl_kernel.

Ensures that sgl_kernel internal scratchpads (MoE and GDN) are graph-safe,
memory-efficient, and compatible with both current and future sgl-kernel-xpu releases:

1. Graph Capture mode: All captured graphs and all layers within graphs share
   one dedicated, static graph workspace (reused across layers and batch sizes
   via narrow slice). Retired buffers during graph-capture expansion are
   retained to ensure physical memory addresses are never recycled.
2. Eager mode: Retains sgl-kernel's native eager workspace behavior without
   altering it or imposing arbitrary thresholds.
3. Forward/Backward compatibility:
   - If sgl-kernel-xpu defines `_XPU_GRAPH_SAFE = True`, the patch automatically
     bypasses itself, delegating to the kernel's native graph-safe implementation.
   - If future sgl-kernel-xpu refactors remove `_get_moe_ws` / `_get_gdn_workspace`,
     hasattr checks ensure zero breaking changes.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_GRAPH_SCRATCHPAD_STATE: Dict[str, Any] = {
    "moe_cache": {},
    "moe_views": {},
    "moe_retired": [],
    "gdn_cache": {},
    "gdn_retired": [],
}

_ACTIVE_GRAPH_FINALIZERS: Dict[int, weakref.finalize] = {}


def _on_owner_finalized(owner_id: int) -> None:
    _ACTIVE_GRAPH_FINALIZERS.pop(owner_id, None)
    if len(_ACTIVE_GRAPH_FINALIZERS) == 0:
        reset_sgl_kernel_graph_scratchpads()


def register_graph_scratchpad_owner(owner: object) -> None:
    """Register an active graph runner / backend instance as a scratchpad owner."""
    if owner is not None:
        oid = id(owner)
        if oid not in _ACTIVE_GRAPH_FINALIZERS:
            fin = weakref.finalize(owner, _on_owner_finalized, oid)
            fin.atexit = False
            _ACTIVE_GRAPH_FINALIZERS[oid] = fin


def unregister_graph_scratchpad_owner(owner: object) -> None:
    """Unregister a graph scratchpad owner; clear caches only when all owners exit."""
    if owner is not None:
        fin = _ACTIVE_GRAPH_FINALIZERS.pop(id(owner), None)
        if fin is not None:
            fin.detach()
        if len(_ACTIVE_GRAPH_FINALIZERS) == 0:
            reset_sgl_kernel_graph_scratchpads()


def reset_sgl_kernel_graph_scratchpads() -> None:
    """Clear cached and retired graph scratchpads after ALL XPU graphs are reset."""
    for container in _GRAPH_SCRATCHPAD_STATE.values():
        container.clear()


def _normalize_xpu_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        device = "xpu"
    dev = torch.device(device)
    if dev.type == "xpu" and dev.index is None:
        dev = torch.device("xpu", torch.xpu.current_device())
    return dev


def _is_xpu_capturing(dev: torch.device) -> bool:
    return (
        dev.type == "xpu"
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.xpu, "is_current_stream_capturing")
        and torch.xpu.is_current_stream_capturing()
    )


def _in_fake_mode() -> bool:
    try:
        from torch._guards import detect_fake_mode

        return detect_fake_mode() is not None
    except Exception:
        return False


def _is_fake(t: torch.Tensor) -> bool:
    if hasattr(t, "fake_mode") and t.fake_mode is not None:
        return True
    try:
        from torch._subclasses.fake_tensor import is_fake

        return is_fake(t)
    except Exception:
        return False


def _is_compiling() -> bool:
    try:
        import torch.compiler

        return torch.compiler.is_compiling()
    except Exception:
        return False


_orig_moe_ws: Optional[Callable] = None
_orig_gdn_ws: Optional[Callable] = None


def _get_shared_graph_moe_ws(
    name: str, shape: tuple, dtype: torch.dtype, dev: torch.device
) -> torch.Tensor:
    numel = 1
    for d in shape:
        numel *= d
    shape_tuple = tuple(shape)
    if numel == 0 or _in_fake_mode():
        return torch.empty(shape_tuple, dtype=dtype, device=dev)

    key = (name, dtype, dev)
    graph_moe_ws_cache = _GRAPH_SCRATCHPAD_STATE["moe_cache"]
    graph_moe_ws_view_cache = _GRAPH_SCRATCHPAD_STATE["moe_views"]
    graph_moe_retired_buffers = _GRAPH_SCRATCHPAD_STATE["moe_retired"]

    cur = graph_moe_ws_cache.get(key)
    if cur is None or cur.numel() < numel:
        old_cur = cur
        cur = torch.empty(numel, dtype=dtype, device=dev)
        if not _is_fake(cur):
            graph_moe_ws_cache[key] = cur
            if old_cur is not None:
                graph_moe_retired_buffers.append(old_cur)

    cached_view = graph_moe_ws_view_cache.get(key)
    if (
        cached_view is None
        or cached_view[0] is not cur
        or cached_view[1] != shape_tuple
    ):
        view = cur.narrow(0, 0, numel).view(shape_tuple)
        if not _is_fake(cur):
            graph_moe_ws_view_cache[key] = (cur, shape_tuple, view)
        return view
    return cached_view[2]


def _get_shared_graph_gdn_workspace(nbytes: int, dev: torch.device) -> torch.Tensor:
    if nbytes <= 0 or _in_fake_mode():
        return torch.empty(max(nbytes, 0), dtype=torch.uint8, device=dev)

    graph_gdn_ws_cache = _GRAPH_SCRATCHPAD_STATE["gdn_cache"]
    graph_gdn_retired_buffers = _GRAPH_SCRATCHPAD_STATE["gdn_retired"]

    cur = graph_gdn_ws_cache.get(dev)
    if cur is None or cur.numel() < nbytes:
        old_cur = cur
        cur = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        if not _is_fake(cur):
            graph_gdn_ws_cache[dev] = cur
            if old_cur is not None:
                graph_gdn_retired_buffers.append(old_cur)
    return cur.narrow(0, 0, nbytes)


# Custom ops for zero-break torch.compile integration.
# Note: torch.library.infer_schema maps `list[int]` -> `SymInt[]` and `int` -> `SymInt`
# (using `torch.SymInt` directly in type annotations is rejected by infer_schema).
try:

    @torch.library.custom_op("sgl_xpu::get_moe_ws_op", mutates_args=())
    def _get_moe_ws_op(
        name: str,
        shape: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        dev = _normalize_xpu_device(device)
        shape_tuple = tuple(int(d) for d in shape)
        if _is_xpu_capturing(dev):
            return _get_shared_graph_moe_ws(name, shape_tuple, dtype, dev)
        if _orig_moe_ws is not None:
            return _orig_moe_ws(name, shape_tuple, dtype, device)
        return torch.empty(shape_tuple, dtype=dtype, device=device)

    @_get_moe_ws_op.register_fake
    def _(
        name: str,
        shape: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(tuple(shape), dtype=dtype, device=device)

    _HAS_MOE_WS_CUSTOM_OP = True
except Exception:
    logger.warning("Failed to register sgl_xpu::get_moe_ws_op", exc_info=True)
    _HAS_MOE_WS_CUSTOM_OP = False

try:

    @torch.library.custom_op("sgl_xpu::get_gdn_ws_op", mutates_args=())
    def _get_gdn_ws_op(nbytes: int, device: torch.device) -> torch.Tensor:
        dev = _normalize_xpu_device(device)
        size = int(nbytes)
        if _is_xpu_capturing(dev):
            return _get_shared_graph_gdn_workspace(size, dev)
        if _orig_gdn_ws is not None:
            return _orig_gdn_ws(size, device)
        return torch.empty(size, dtype=torch.uint8, device=device)

    @_get_gdn_ws_op.register_fake
    def _(nbytes: int, device: torch.device) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    _HAS_GDN_WS_CUSTOM_OP = True
except Exception:
    logger.warning("Failed to register sgl_xpu::get_gdn_ws_op", exc_info=True)
    _HAS_GDN_WS_CUSTOM_OP = False


def safe_get_moe_ws(
    name: str, shape: tuple, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    dev = _normalize_xpu_device(device)
    if _is_compiling() and _HAS_MOE_WS_CUSTOM_OP:
        return _get_moe_ws_op(name, list(shape), dtype, dev)
    if _is_xpu_capturing(dev):
        return _get_shared_graph_moe_ws(name, shape, dtype, dev)
    if _orig_moe_ws is not None:
        return _orig_moe_ws(name, shape, dtype, device)
    return torch.empty(shape, dtype=dtype, device=device)


def safe_get_gdn_workspace(nbytes: int, device: torch.device) -> torch.Tensor:
    dev = _normalize_xpu_device(device)
    if _is_compiling() and _HAS_GDN_WS_CUSTOM_OP:
        return _get_gdn_ws_op(nbytes, dev)
    if _is_xpu_capturing(dev):
        return _get_shared_graph_gdn_workspace(nbytes, dev)
    if _orig_gdn_ws is not None:
        return _orig_gdn_ws(nbytes, device)
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


def patch_sgl_kernel_scratchpad_buffers() -> None:
    """Patch sgl_kernel scratchpad allocators for XPU Graph safety."""
    global _orig_moe_ws, _orig_gdn_ws

    # 1. Independently patch sgl_kernel.moe
    try:
        import sgl_kernel.moe as sk_moe

        if (
            not getattr(sk_moe, "_XPU_GRAPH_SAFE", False)
            and not getattr(sk_moe, "_xpu_graph_safe_ws_patched", False)
            and hasattr(sk_moe, "_get_moe_ws")
            and sk_moe._get_moe_ws is not safe_get_moe_ws
        ):
            _orig_moe_ws = sk_moe._get_moe_ws
            sk_moe._get_moe_ws = safe_get_moe_ws
            sk_moe._xpu_graph_safe_ws_patched = True
    except Exception:
        pass

    # 2. Independently patch sgl_kernel.gdn_attn
    try:
        import sgl_kernel.gdn_attn as sk_gdn

        if (
            not getattr(sk_gdn, "_XPU_GRAPH_SAFE", False)
            and not getattr(sk_gdn, "_xpu_graph_safe_ws_patched", False)
            and hasattr(sk_gdn, "_get_gdn_workspace")
            and sk_gdn._get_gdn_workspace is not safe_get_gdn_workspace
        ):
            _orig_gdn_ws = sk_gdn._get_gdn_workspace
            sk_gdn._get_gdn_workspace = safe_get_gdn_workspace
            sk_gdn._xpu_graph_safe_ws_patched = True
    except Exception:
        pass
