# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_full_cuda_graph_owners: ContextVar[Optional[list[Any]]] = ContextVar(
    "full_cuda_graph_owners", default=None
)


@contextmanager
def collect_full_cuda_graph_owners() -> Iterator[list[Any]]:
    """Collect tensor owners whose addresses are recorded by one full graph."""

    owners: list[Any] = []
    token = _full_cuda_graph_owners.set(owners)
    try:
        yield owners
    finally:
        _full_cuda_graph_owners.reset(token)


def retain_full_cuda_graph_owner(owner: Any) -> bool:
    """Retain ``owner`` when called from an active full-graph capture."""

    owners = _full_cuda_graph_owners.get()
    if owners is None:
        return False
    owners.append(owner)
    return True
