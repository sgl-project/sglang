"""Transactional verification for mlx-lm native request caches.

Gemma 4 mixes ordinary and rotating caches, and a rotating cache is no longer
trimmable after it wraps.  Verification therefore runs on a fully isolated
clone.  Commit replays only the accepted query prefix on the original cache;
if replay raises, a pre-replay clone restores every array and ring field.
"""

from __future__ import annotations

import copy
from enum import Enum, auto
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx


class _TransactionState(Enum):
    NEW = auto()
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()


def _clone_value(value: Any) -> Any:
    if isinstance(value, mx.array):
        # MLX arrays are immutable values.  Constructing a new array gives the
        # clone its own value when later slice assignments rebind cache fields.
        return mx.array(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_native_cache_entry(entry: Any) -> Any:
    """Clone one cache object, including provider-specific ring metadata."""

    clone = type(entry).__new__(type(entry))
    if hasattr(entry, "__dict__"):
        clone.__dict__.update(_clone_value(entry.__dict__))
    else:
        slots: list[str] = []
        for cls in type(entry).__mro__:
            declared = getattr(cls, "__slots__", ())
            if isinstance(declared, str):
                slots.append(declared)
            else:
                slots.extend(declared)
        for name in slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(entry, name):
                setattr(clone, name, _clone_value(getattr(entry, name)))
    return clone


def clone_native_cache(cache: Sequence[Any]) -> list[Any]:
    """Deep-clone all arrays, offsets, valid lengths, and ring metadata."""

    cloned = [clone_native_cache_entry(entry) for entry in cache]
    arrays = list(_iter_arrays(cloned))
    if arrays:
        mx.eval(*arrays)
    return cloned


def replace_native_cache_contents(
    destination: Sequence[Any], source: Sequence[Any]
) -> None:
    """Replace cache state while preserving destination entry identities."""

    if len(destination) != len(source):
        raise ValueError(
            f"native cache cardinality changed: {len(destination)} != {len(source)}"
        )
    for dest, src in zip(destination, source):
        if type(dest) is not type(src):
            raise TypeError(
                "native cache entry type changed during transaction: "
                f"{type(dest).__name__} != {type(src).__name__}"
            )
        cloned = clone_native_cache_entry(src)
        if hasattr(dest, "__dict__"):
            dest.__dict__.clear()
            dest.__dict__.update(cloned.__dict__)
            continue
        for cls in type(dest).__mro__:
            declared = getattr(cls, "__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            for name in declared:
                if name not in {"__dict__", "__weakref__"} and hasattr(cloned, name):
                    setattr(dest, name, getattr(cloned, name))


def _iter_arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_arrays(item)
    else:
        state = getattr(value, "state", None)
        if state is not None:
            yield from _iter_arrays(state)


ReplayForward = Callable[[Sequence[Any], tuple[int, ...]], Any]


class MlxNativeCacheTransaction:
    """Single-use clone/verify/replay transaction for one native request."""

    def __init__(
        self,
        cache: Sequence[Any],
        query_token_ids: Sequence[int],
        replay_forward: ReplayForward,
    ) -> None:
        if not query_token_ids:
            raise ValueError("a native-cache transaction requires at least one query")
        if any(int(token) < 0 for token in query_token_ids):
            raise ValueError("verification queries cannot contain negative token IDs")
        self._cache = cache
        self._queries = tuple(int(token) for token in query_token_ids)
        self._replay_forward = replay_forward
        self._state = _TransactionState.NEW
        self._speculative_cache: list[Any] | None = None
        self._committed_count: int | None = None

    @property
    def active(self) -> bool:
        return self._state is _TransactionState.ACTIVE

    @property
    def committed_count(self) -> int | None:
        return self._committed_count

    def begin(self) -> list[Any]:
        if self._state is not _TransactionState.NEW:
            raise RuntimeError("native-cache transaction begin() is single-use")
        self._speculative_cache = clone_native_cache(self._cache)
        self._state = _TransactionState.ACTIVE
        return self._speculative_cache

    def commit(self, count: int) -> Any:
        if self._state is not _TransactionState.ACTIVE:
            raise RuntimeError("only an active native-cache transaction can commit")
        if count < 0 or count > len(self._queries):
            raise ValueError(
                f"commit count {count} is outside [0, {len(self._queries)}]"
            )

        backup = clone_native_cache(self._cache)
        try:
            result = None
            if count:
                result = self._replay_forward(self._cache, self._queries[:count])
                arrays = list(_iter_arrays(result)) + list(_iter_arrays(self._cache))
                if arrays:
                    mx.eval(*arrays)
        except BaseException:
            replace_native_cache_contents(self._cache, backup)
            self._speculative_cache = None
            self._state = _TransactionState.ABORTED
            raise

        self._committed_count = count
        self._speculative_cache = None
        self._state = _TransactionState.COMMITTED
        return result

    def abort(self) -> None:
        if self._state is not _TransactionState.ACTIVE:
            raise RuntimeError("only an active native-cache transaction can abort")
        self._speculative_cache = None
        self._state = _TransactionState.ABORTED

    def __enter__(self) -> list[Any]:
        return self.begin()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._state is _TransactionState.ACTIVE:
            self.abort()
        return False
