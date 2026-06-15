# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from array import array
from typing import Iterable, Iterator, Optional, Union


class TokenArray:
    __slots__ = ("_data",)

    def __init__(self, data: Optional[array] = None) -> None:
        self._data: array = data if data is not None else array("q")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: Union[int, slice]) -> Union[int, array]:
        return self._data[idx]

    def __setitem__(
        self, idx: Union[int, slice], values: Union[int, Iterable[int]]
    ) -> None:
        self._data[idx] = values

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def extend(self, values: Iterable[int]) -> None:
        self._data.extend(values)

    def __iadd__(self, values: Iterable[int]) -> TokenArray:
        self._data.extend(values)
        return self

    def truncate(self, length: int) -> None:
        del self._data[length:]

    def readonly_prefix_view(self, length: Optional[int]) -> memoryview:
        capped = len(self._data) if length is None else min(length, len(self._data))
        return memoryview(self._data).toreadonly()[:capped]
