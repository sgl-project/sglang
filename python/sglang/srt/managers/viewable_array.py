from __future__ import annotations

from array import array
from typing import Iterable, Optional

_TYPECODE = "q"
_ITEMSIZE = 8
_MIN_CAPACITY = 64


def to_array(values: Iterable[int]) -> array:
    if isinstance(values, memoryview):
        result: array = array(_TYPECODE)
        result.frombytes(values.tobytes())
        return result
    return array(_TYPECODE, values)


class ViewableArray:
    __slots__ = ("_data", "_size", "_view", "_lock_prefix_len")

    def __init__(self, init: Optional[Iterable[int]] = None) -> None:
        values: array = array(_TYPECODE) if init is None else to_array(init)
        size: int = len(values)
        capacity: int = self._capacity_for(size)
        self._data: array = array(_TYPECODE, bytes(_ITEMSIZE * capacity))
        if size:
            self._data[:size] = values
        self._size: int = size
        self._view: memoryview = memoryview(self._data).toreadonly()
        self._lock_prefix_len: int = 0

    def __len__(self) -> int:
        return self._size

    def append(self, value: int) -> None:
        assert self._size >= self._lock_prefix_len, (self._size, self._lock_prefix_len)
        if self._size == len(self._data):
            self._reallocate(self._size + 1)
        self._data[self._size] = value
        self._size += 1

    def extend(self, values: Iterable[int]) -> None:
        materialized: array = to_array(values)
        added_size: int = len(materialized)
        if added_size == 0:
            return
        assert self._size >= self._lock_prefix_len, (self._size, self._lock_prefix_len)
        new_size: int = self._size + added_size
        if new_size > len(self._data):
            self._reallocate(new_size)
        self._data[self._size : new_size] = materialized
        self._size = new_size

    def truncate(self, size: int) -> None:
        assert 0 <= size <= self._size, (size, self._size)
        self._size = size

    def overwrite(self, index: int, value: int) -> None:
        assert 0 <= index < self._size, (index, self._size)
        assert index >= self._lock_prefix_len, (index, self._lock_prefix_len)
        self._data[index] = value

    def freeze_and_clone(self) -> ViewableArray:
        self._lock_prefix_len = self._size
        clone: ViewableArray = ViewableArray.__new__(ViewableArray)
        clone._data = self._data
        clone._size = self._size
        clone._view = memoryview(self._data).toreadonly()
        clone._lock_prefix_len = self._size
        return clone

    def readonly_view(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> memoryview:
        begin: int = 0 if start is None else start
        end: int = self._size if stop is None else stop
        assert 0 <= begin <= end <= self._size, (begin, end, self._size)
        return self._view[begin:end]

    def materialize(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> array:
        begin: int = 0 if start is None else start
        end: int = self._size if stop is None else stop
        assert 0 <= begin <= end <= self._size, (begin, end, self._size)
        return self._data[begin:end]

    def clone(self) -> ViewableArray:
        return ViewableArray(self.materialize())

    @staticmethod
    def _capacity_for(size: int) -> int:
        capacity: int = _MIN_CAPACITY
        while capacity < size:
            capacity *= 2
        return capacity

    def _reallocate(self, min_capacity: int) -> None:
        capacity: int = self._capacity_for(min_capacity)
        new_data: array = array(_TYPECODE, bytes(_ITEMSIZE * capacity))
        new_data[: self._size] = self._data[: self._size]
        self._data = new_data
        self._view = memoryview(self._data).toreadonly()
