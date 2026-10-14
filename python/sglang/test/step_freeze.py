"""Checks that a scheduled step stays as built from finalize to commit, for tests.

After `freeze_step` (at finalize), setting a step field raises; `release_step`
(at commit) asserts that, down to `_MAX_DEPTH` below each field, no attribute
was rebound, no list, tuple, dict or set changed its members (set members by
identity), and no tensor was written in place.

The walk stops at what the step only points to: the engine-lifetime pools,
allocators and caches, the requests and what they own, and other steps. A
write that bypasses the tensor's version counter (a raw data_ptr write) is
not seen.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch

# id(step) -> (path -> fingerprint) at finalize
_frozen_steps: Dict[int, Dict[str, Tuple]] = {}

# How deep the walk goes below a step field.
_MAX_DEPTH = 6

# Fields the step only points to; nothing below them is step state.
_SHARED_FIELDS = frozenset(
    {
        "req_to_token_pool",
        "token_to_kv_pool_allocator",
        "tree_cache",
        "model_config",
    }
)

# Classes, by name, whose instances a step only points to: engine-lifetime
# state and request-owned objects.
_SHARED_CLASS_NAMES = frozenset(
    {
        "Req",
        "ScheduleBatch",
        "PenaltyStore",
        "BaseGrammarObject",
        "MultimodalInputs",
        "ModelConfig",
        "ReqToTokenPool",
        "BaseTokenToKVPoolAllocator",
        "BasePrefixCache",
        "KVCache",
    }
)

_ATOMS = (type(None), bool, int, float, complex, str, bytes, enum.Enum)


def _guarded_setattr(self: ScheduleBatch, name: str, value: Any) -> None:
    if id(self) in _frozen_steps:
        raise AttributeError(
            f"step {self.forward_iter} is frozen after finalize; set {name}"
        )
    object.__setattr__(self, name, value)


def _write_marker(tensor: torch.Tensor) -> Any:
    """What moves when the tensor is written in place: its version counter,
    or, for an inference tensor, which has none, a digest of its bytes."""
    if not tensor.is_inference():
        return tensor._version
    data = tensor.detach().reshape(-1).contiguous().view(torch.uint8)
    return hashlib.blake2b(data.cpu().numpy().tobytes()).hexdigest()


def _is_shared(value: Any) -> bool:
    if isinstance(
        value,
        (
            torch.nn.Module,
            torch.Stream,
            torch.Event,
            torch.cuda.Stream,
            torch.cuda.Event,
        ),
    ):
        return True
    return any(cls.__name__ in _SHARED_CLASS_NAMES for cls in type(value).__mro__)


def _attributes(value: Any) -> Iterator[Tuple[str, Any]]:
    """The attributes of a step-owned object, by name."""
    cls = type(value)
    if dataclasses.is_dataclass(value):
        names = [field.name for field in dataclasses.fields(value)]
    elif hasattr(cls, "__struct_fields__"):  # msgspec.Struct
        names = list(cls.__struct_fields__)
    elif hasattr(cls, "_fields"):  # NamedTuple
        names = list(cls._fields)
    else:
        names = list(getattr(value, "__dict__", {}))
        for klass in cls.__mro__:
            slots = klass.__dict__.get("__slots__", ())
            names.extend([slots] if isinstance(slots, str) else slots)
    missing = object()
    for name in dict.fromkeys(names):
        attr = getattr(value, name, missing)
        if attr is not missing:
            yield name, attr


def _walk(value: Any, path: str, depth: int, seen: set, out: Dict[str, Tuple]):
    """Record a fingerprint for value and everything below it the step owns."""
    if isinstance(value, _ATOMS):
        # repr, so that a NaN equals itself
        out[path] = ("atom", type(value), repr(value))
        return
    if isinstance(value, torch.Tensor):
        out[path] = (
            "tensor",
            id(value),
            value.data_ptr(),
            tuple(value.shape),
            value.dtype,
            _write_marker(value),
        )
        return
    if isinstance(value, np.ndarray):
        out[path] = ("ndarray", id(value), value.shape, value.tobytes())
        return
    if isinstance(value, (torch.dtype, torch.device)) or callable(value):
        out[path] = ("object", id(value))
        return
    if _is_shared(value) or depth >= _MAX_DEPTH or id(value) in seen:
        out[path] = ("object", id(value))
        return
    seen.add(id(value))
    if isinstance(value, (list, tuple)) and not hasattr(type(value), "_fields"):
        out[path] = (type(value).__name__, id(value), len(value))
        for i, item in enumerate(value):
            _walk(item, f"{path}[{i}]", depth + 1, seen, out)
        return
    if isinstance(value, dict):
        out[path] = ("dict", id(value), tuple(map(repr, value)))
        for key, item in value.items():
            _walk(item, f"{path}[{key!r}]", depth + 1, seen, out)
        return
    if isinstance(value, (set, frozenset)):
        out[path] = (type(value).__name__, id(value), frozenset(map(id, value)))
        return
    attrs = list(_attributes(value))
    out[path] = ("object", id(value), tuple(name for name, _ in attrs))
    for name, attr in attrs:
        _walk(attr, f"{path}.{name}", depth + 1, seen, out)


def _fingerprints(step: ScheduleBatch) -> Dict[str, Tuple]:
    out: Dict[str, Tuple] = {}
    seen = {id(step)}
    for field in dataclasses.fields(step):
        value = getattr(step, field.name)
        if field.name in _SHARED_FIELDS:
            out[field.name] = ("object", id(value))
        else:
            _walk(value, field.name, 0, seen, out)
    return out


def freeze_step(step: ScheduleBatch) -> None:
    ScheduleBatch.__setattr__ = _guarded_setattr
    _frozen_steps[id(step)] = _fingerprints(step)


def release_step(step: ScheduleBatch) -> None:
    fingerprints = _frozen_steps.pop(id(step), None)
    if fingerprints is None:
        return
    now = _fingerprints(step)
    changed = [
        f"{path}: {fingerprints.get(path)} -> {now.get(path)}"
        for path in dict.fromkeys([*fingerprints, *now])
        if fingerprints.get(path) != now.get(path)
    ]
    assert not changed, (
        f"step {step.forward_iter} changed after finalize:\n" + "\n".join(changed)
    )
