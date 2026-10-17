"""Writes and reads the stashes of responses through a storage backend.

Layers, top to bottom:

- A stash is one output kind's schema; writers and readers share it.
- ``OutputStore`` turns stashes into encoded named parts and builds ``StoreRef``s.
- A codec transforms one field's bytes; its id is recorded in the ref.
- A ``StorageBackend`` stores named parts under object keys.

Below the stash everything sees only raw fields, (dtype name, shape, 1-D bytes), so
the lower layers carry no torch or numpy types and can be rewritten in another
language.

Lifecycle:

- One ``put_many`` stores all objects of one response, all or nothing, as the last
  step of building that response that can fail. A ref exists only for a stored object.
- Until the response is sent, the writer owns its objects and deletes the ones whose
  response cannot be sent. After that, the reader side owns them: consumers may each
  read a subset of fields without deleting anything, and one owner deletes each ref
  once every consumer is done.
- Backends never evict an object; one lost after hand-off stays until the storage
  plane is torn down.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, TypeVar

from radix_transfer_plugins.backend import StorageBackend
from radix_transfer_plugins.codec import IDENTITY, get_codec
from radix_transfer_plugins.schema import FieldSpec, Stash, StoreRef

S = TypeVar("S", bound=Stash)

_EMPTY = memoryview(b"")


class OutputStore:
    def __init__(
        self,
        *,
        backend: StorageBackend,
        codecs: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._backend = backend
        # Field name -> codec id; fields not named here are stored as they are.
        self._codecs = dict(codecs or {})
        for codec_id in self._codecs.values():
            get_codec(codec_id)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def put_many(self, stashes: Sequence[Stash]) -> list[StoreRef]:
        """Store the stashes of one response, all or nothing."""
        encoded = [self._encode(stash) for stash in stashes]
        keys = self._backend.put_many([parts for _, parts in encoded])
        return [
            StoreRef(backend=self._backend.name, key=key, fields=specs)
            for key, (specs, _) in zip(keys, encoded)
        ]

    def get_many(
        self,
        refs: Sequence[StoreRef],
        *,
        stash_type: type[S],
        fields: Optional[Sequence[str]] = None,
    ) -> list[S]:
        """Read the stashes back; ``fields`` limits which fields are read."""
        self._check_backend(refs)
        wanted = [
            tuple(s for s in ref.fields if fields is None or s.name in fields)
            for ref in refs
        ]
        stored = self._backend.get_many(
            [ref.key for ref in refs],
            parts=[[(s.name, s.nbytes) for s in specs if s.nbytes] for specs in wanted],
        )
        return [
            stash_type.from_fields(
                {
                    s.name: get_codec(s.codec).decode(
                        data[s.name] if s.nbytes else _EMPTY, spec=s
                    )
                    for s in specs
                }
            )
            for specs, data in zip(wanted, stored)
        ]

    def delete_many(self, refs: Sequence[StoreRef]) -> None:
        """Delete every field of each ref."""
        self._check_backend(refs)
        self._backend.delete_many(
            [ref.key for ref in refs],
            parts=[[s.name for s in ref.fields if s.nbytes] for ref in refs],
        )

    def close(self) -> None:
        self._backend.close()

    def _encode(
        self, stash: Stash
    ) -> tuple[tuple[FieldSpec, ...], dict[str, memoryview]]:
        specs, parts = [], {}
        for name, field in stash.to_fields().items():
            if not name or "/" in name:
                raise ValueError(f"invalid field name {name!r}")
            codec = get_codec(self._codecs.get(name, IDENTITY))
            data = _as_bytes(codec.encode(field))
            specs.append(
                FieldSpec(
                    name=name,
                    dtype=field.dtype,
                    shape=field.shape,
                    codec=codec.id,
                    nbytes=data.nbytes,
                )
            )
            # Backends cannot store empty values; an empty field lives only in the ref.
            if data.nbytes:
                parts[name] = data
        return tuple(specs), parts

    def _check_backend(self, refs: Sequence[StoreRef]) -> None:
        for ref in refs:
            if ref.backend != self._backend.name:
                raise ValueError(
                    f"ref of backend {ref.backend!r} given to a {self._backend.name!r} store"
                )


def _as_bytes(data: memoryview) -> memoryview:
    return data if data.format == "B" and data.ndim == 1 else data.cast("B")
