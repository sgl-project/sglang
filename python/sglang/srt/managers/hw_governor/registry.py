"""Resolves a built-in name or a ``package.module:ClassName`` spec to a class."""

from __future__ import annotations

import pkgutil
from typing import Mapping, Type, TypeVar

T = TypeVar("T")


def resolve_class(
    spec: str, *, builtins: Mapping[str, str], base: Type[T], kind: str
) -> Type[T]:
    target = builtins.get(spec)
    if target is None:
        if ":" not in spec:
            raise ValueError(
                f"Unknown {kind} {spec!r}: use one of {sorted(builtins)} "
                "or a 'package.module:ClassName' import path."
            )
        target = spec
    cls = pkgutil.resolve_name(target)
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(f"{kind} {spec!r} must name a subclass of {base.__name__}")
    return cls
