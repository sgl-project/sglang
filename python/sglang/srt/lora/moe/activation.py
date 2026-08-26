from __future__ import annotations

from enum import Enum


class ActivationFn(str, Enum):

    SILU = "silu"
    RELU2 = "relu2"

    @classmethod
    def parse(cls, name: str) -> ActivationFn:
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"activation {name!r} is not one of {tuple(fn.value for fn in cls)}"
            ) from None
