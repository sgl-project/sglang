from __future__ import annotations

from array import array
from typing import Iterable

_TYPECODE = "q"


def to_array(values: Iterable[int]) -> array:
    if isinstance(values, memoryview):
        result: array = array(_TYPECODE)
        result.frombytes(values.tobytes())
        return result
    return array(_TYPECODE, values)
