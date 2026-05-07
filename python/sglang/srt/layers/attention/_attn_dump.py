"""Minimal per-layer attention I/O dumper for xpu_backend vs triton_backend diffs.

Activated only when SGLANG_DUMP_ATTN_DIR is set. Intended for investigation;
not a supported public API.

Env vars:
    SGLANG_DUMP_ATTN_DIR     base directory for dumps (required to enable)
    SGLANG_DUMP_ATTN_LAYERS  optional comma list, e.g. "0,4,8"; default = all
    SGLANG_DUMP_ATTN_MAX     optional per-(mode,layer) cap on dumps (default 4)

Layout:
    <dir>/<backend>/<mode>/rank<R>_layer<L>_step<S>.pt
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch

_LOCK = threading.Lock()
_COUNTERS: dict = {}


def _get_rank() -> int:
    for k in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None and v != "":
            try:
                return int(v)
            except ValueError:
                pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def maybe_dump_attn(
    backend_name: str,
    mode: str,
    layer_id: int,
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    o: Optional[torch.Tensor],
) -> None:
    out_dir = os.environ.get("SGLANG_DUMP_ATTN_DIR")
    if not out_dir:
        return

    layer_filter = os.environ.get("SGLANG_DUMP_ATTN_LAYERS")
    if layer_filter:
        allowed = {int(x) for x in layer_filter.split(",") if x.strip()}
        if layer_id not in allowed:
            return

    try:
        max_dumps = int(os.environ.get("SGLANG_DUMP_ATTN_MAX", "4"))
    except ValueError:
        max_dumps = 4

    rank = _get_rank()
    key = (backend_name, mode, layer_id, rank)
    with _LOCK:
        step = _COUNTERS.get(key, 0)
        if step >= max_dumps:
            return
        _COUNTERS[key] = step + 1

    sub = os.path.join(out_dir, backend_name, mode)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"rank{rank}_layer{layer_id:03d}_step{step:03d}.pt")

    def _copy(t: Optional[torch.Tensor]):
        if t is None:
            return None
        return t.detach().to("cpu", copy=True)

    payload = {
        "backend": backend_name,
        "mode": mode,
        "layer_id": layer_id,
        "step": step,
        "rank": rank,
        "q": _copy(q),
        "k": _copy(k),
        "v": _copy(v),
        "o": _copy(o),
    }
    torch.save(payload, path)
