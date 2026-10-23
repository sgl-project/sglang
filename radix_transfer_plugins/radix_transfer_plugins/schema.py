"""Types that cross the package's two boundaries.

``FieldSpec`` and ``StoreRef`` travel with a response as JSON. ``RawField`` and ``Stash``
stay in process: a stash turns its arrays into raw fields, and everything below it
(codecs, backends) sees only raw fields.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import msgspec


class FieldSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One stored field of a ref."""

    name: str
    # Decoded dtype as a name ("int32", "bfloat16"); the package never interprets it.
    dtype: str
    shape: tuple[int, ...]
    codec: str
    # Encoded size in the backend; 0 means nothing was stored for this field.
    nbytes: int


class StoreRef(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Everything a reader needs to read or delete one stored object."""

    backend: str
    # Full backend key, so a reader needs no key prefix of its own.
    key: str
    fields: tuple[FieldSpec, ...]

    def to_builtins(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)

    @classmethod
    def from_builtins(cls, value: Any) -> StoreRef:
        return msgspec.convert(value, cls)


class RawField(msgspec.Struct, frozen=True):
    """A field at the bytes boundary: its dtype name, shape, and bytes."""

    dtype: str
    shape: tuple[int, ...]
    # 1-D unsigned-byte view of the field's memory; not a copy.
    data: memoryview


class Stash(Protocol):
    """The outputs of one object, as one output kind's schema.

    Writers and readers share the stash class, so field names and their meaning live
    in one place.
    """

    def is_empty(self) -> bool:
        """Nothing to store."""
        ...

    def to_fields(self) -> dict[str, RawField]:
        """Raw fields to store; field names are the stored names."""
        ...

    @classmethod
    def from_fields(cls, fields: Mapping[str, RawField]) -> Stash:
        """Rebuild a stash from the raw fields a reader got back (possibly a subset)."""
        ...
