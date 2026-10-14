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
"""Allocator double for scheduler tests that admit requests.

`DecodePreallocQueue` asks the allocator to price a preallocation rather than
doing the arithmetic itself, so a bare `MagicMock` returns a truthy `Mock` and
the admission decision under test stops being made anywhere. Binding the real
separate-buffer implementations keeps the arithmetic live while leaving the
per-test stubs (`size_swa`, `swa_available_size`, ...) in charge of the state.
"""

from unittest.mock import MagicMock

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

# Bound on the double, so each reads the stubs the caller set on it.
_SEPARATE_BUFFER_METHODS = {
    "prealloc_fits_assumes_reclaim": BaseTokenToKVPoolAllocator.prealloc_fits_assumes_reclaim,
    "prealloc_ceiling_fits": BaseTokenToKVPoolAllocator.prealloc_ceiling_fits,
    "prealloc_fits": BaseTokenToKVPoolAllocator.prealloc_fits,
    "reclaim_for_prealloc": SWATokenToKVPoolAllocator.reclaim_for_prealloc,
    "swa_capacity_and_available": SWATokenToKVPoolAllocator.swa_capacity_and_available,
}


def bind_separate_buffer_capacity(allocator) -> None:
    """Make `allocator` price capacity like a pool whose sides own their own
    buffers. Call on any allocator double a `DecodePreallocQueue` will read."""
    for name, impl in _SEPARATE_BUFFER_METHODS.items():
        setattr(
            allocator,
            name,
            (lambda impl: lambda *args, **kwargs: impl(allocator, *args, **kwargs))(
                impl
            ),
        )


def separate_buffer_allocator_double(**attrs) -> MagicMock:
    """A `MagicMock` allocator that prices capacity as separate buffers."""
    allocator = MagicMock(**attrs)
    bind_separate_buffer_capacity(allocator)
    return allocator
