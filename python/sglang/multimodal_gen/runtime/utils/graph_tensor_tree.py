# SPDX-License-Identifier: Apache-2.0
"""Tensor-tree helpers shared by the CUDA-graph runners (breakable and full).

A DiT's kwargs are not flat: tensors hide inside lists/tuples/dicts (RoPE
caches, per-branch controls) alongside non-tensor values that get baked into
the captured Python control flow. Graph capture therefore needs to (1) key on
the whole structure, (2) place a persistent static buffer at every tensor leaf,
and (3) hand the caller a copy of the output, since the static output buffer is
overwritten by the next replay.
"""

from __future__ import annotations

from typing import Any

import torch


def map_tensors(obj: Any, fn) -> Any:
    """Rebuild ``obj`` applying ``fn`` to every tensor leaf, recursing into
    list/tuple/dict containers; everything else passes through unchanged."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [map_tensors(o, fn) for o in obj]
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    return obj


def flatten_tensors(obj: Any, out: list) -> None:
    """Depth-first collect every tensor leaf into ``out`` (deterministic order:
    dicts traversed in sorted-key order to match across calls)."""
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            flatten_tensors(o, out)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            flatten_tensors(obj[k], out)


def flatten_kwargs(kwargs: dict[str, Any]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for name in sorted(kwargs):
        flatten_tensors(kwargs[name], out)
    return out


def signature_leaf(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return ("tensor", tuple(obj.shape), str(obj.dtype))
    if isinstance(obj, tuple):
        return ("tuple", tuple(signature_leaf(o) for o in obj))
    if isinstance(obj, list):
        return ("list", tuple(signature_leaf(o) for o in obj))
    if isinstance(obj, dict):
        return (
            "dict",
            tuple((k, signature_leaf(obj[k])) for k in sorted(obj)),
        )
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return ("const", obj)
    return ("object", type(obj).__module__, type(obj).__qualname__, id(obj))


def signature_kwargs(kwargs: dict[str, Any]) -> tuple:
    """Capture key for ``kwargs``: tensors by shape+dtype (values are copied
    into static buffers), non-tensors by value (they are baked into the
    captured control flow), mutable objects by identity (so a graph that closed
    over another request's state is never replayed)."""
    return tuple((name, signature_leaf(kwargs[name])) for name in sorted(kwargs))


def clone_output(out: Any) -> Any:
    """Copy a captured output off the static buffer so the caller can hold it
    across the next replay (serial CFG holds both branch results at once)."""
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, tuple):
        return tuple(clone_output(o) for o in out)
    if isinstance(out, list):
        return [clone_output(o) for o in out]
    if isinstance(out, dict):
        return {k: clone_output(v) for k, v in out.items()}
    return out


def static_buffer_like(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Persistent capture buffer for one tensor leaf. A CPU leaf (a host-built
    timestep or index tensor) gets a device buffer: leaving it on the host would
    put an unpinned host-to-device copy inside the captured region, which aborts
    capture."""
    if t.device.type == "cpu":
        buf = torch.empty(t.shape, dtype=t.dtype, device=device)
    else:
        buf = torch.empty_like(t)
    buf.copy_(t)
    return buf
