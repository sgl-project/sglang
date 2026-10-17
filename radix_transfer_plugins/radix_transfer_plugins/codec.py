"""Per-field byte transforms.

The writer picks a codec per field and records its id in the ref, so a reader decodes
from the ref alone. Codecs run on writer and reader threads next to latency-sensitive
event loops, so an implementation must release the GIL for its heavy work.
"""

from __future__ import annotations

from typing import Protocol

from radix_transfer_plugins.schema import FieldSpec, RawField

IDENTITY = "identity"


class Codec(Protocol):
    id: str

    def encode(self, field: RawField) -> memoryview:
        """Bytes to store for the field; a 1-D unsigned-byte view."""
        ...

    def decode(self, data: memoryview, *, spec: FieldSpec) -> RawField:
        """The field back from stored bytes and its spec."""
        ...


class IdentityCodec:
    """Stores the field's bytes as they are; no copy either way."""

    id = IDENTITY

    def encode(self, field: RawField) -> memoryview:
        return field.data

    def decode(self, data: memoryview, *, spec: FieldSpec) -> RawField:
        return RawField(dtype=spec.dtype, shape=spec.shape, data=data)


_CODECS: dict[str, Codec] = {codec.id: codec for codec in (IdentityCodec(),)}


def get_codec(codec_id: str) -> Codec:
    try:
        return _CODECS[codec_id]
    except KeyError:
        raise ValueError(
            f"unknown codec {codec_id!r}; available: {sorted(_CODECS)}"
        ) from None
