"""Storage backends and the registry that is the only list of their names."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import ClassVar, Mapping, Optional, Sequence

import msgspec

# Backend name -> "module:class"; a backend's module is imported only when it is used.
_BACKENDS: dict[str, str] = {
    "mooncake": "radix_transfer_plugins.backends.mooncake:MooncakeBackend",
}


class StorageBackend(ABC):
    """Stores objects made of named byte parts.

    A backend is a client of storage it does not own; the deployment runs any server.
    How parts are laid out (a key per part, or several parts per key) is the backend's
    own business and never reaches a ref.
    """

    name: ClassVar[str]
    config_type: ClassVar[type[msgspec.Struct]]

    @abstractmethod
    def put_many(self, objects: Sequence[Mapping[str, memoryview]]) -> list[str]:
        """Store each object's parts; returns one new key per object.

        All or nothing: when it raises, nothing written by this call stays stored.
        """

    @abstractmethod
    def get_many(
        self, keys: Sequence[str], *, parts: Sequence[Sequence[tuple[str, int]]]
    ) -> list[dict[str, memoryview]]:
        """Read the named parts of each object; ``parts`` gives each part's size."""

    @abstractmethod
    def delete_many(
        self, keys: Sequence[str], *, parts: Sequence[Sequence[str]]
    ) -> None:
        """Remove each object; ``parts`` names every part the object was stored with."""

    @abstractmethod
    def close(self) -> None:
        """Disconnect; the backend is unusable afterwards."""


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def create_backend(name: str, extra_config: Optional[str]) -> StorageBackend:
    """Connect to backend ``name`` with ``extra_config``, a JSON object of its config."""
    try:
        location = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown output store backend {name!r}; available: {available_backends()}"
        ) from None
    module_name, class_name = location.split(":")
    backend_cls = getattr(importlib.import_module(module_name), class_name)
    try:
        config = msgspec.json.decode(extra_config or "{}", type=backend_cls.config_type)
    except msgspec.ValidationError as e:
        raise ValueError(f"invalid {name} output store config: {e}") from e
    return backend_cls(config)
