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
"""Graph-resident dump buffers.

Every buffer here has its `data_ptr()` baked into at least one recorded kernel,
which drives the whole design:

* a buffer may only be `copy_`-ed into, never reassigned;
* a buffer can never be grown -- so it is allocated at the largest shape it
  will ever see, and a later request for more elements is a hard error rather
  than a silent truncation;
* the identity of a buffer is `(fully-expanded dump name, occurrence index
  within one forward)`.  Sharing across shapes and phases is fine (the copy is
  re-recorded per graph); sharing across *occurrences* is not, because the
  second `copy_` would overwrite the first before the host ever reads it.

Buffers are flat.  Reconstructing the real shape needs one more piece of
information -- the shape that was live when the `copy_` was recorded -- which is
why `record()` takes a `shape_token` and stores a per-token layout.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Hashable, Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class DumpBudgetExceeded(RuntimeError):
    """Raised in strict mode when buffers would exceed `budget_mb`."""


class DumpBufferGrowthRejected(RuntimeError):
    """Raised when a tap needs more elements than its buffer was allocated for.

    Not recoverable: the small allocation's address is already baked into a
    recorded kernel, so it cannot be replaced.  Both capture loops iterate
    shapes largest-first (`reversed(capture_bs)` / `reversed(capture_num_tokens)`),
    so first-sight allocation is normally the max.  Seeing this means that
    ordering changed, or a shape does not scale monotonically with the bucket.
    """


@dataclass(frozen=True)
class BufferKey:
    name: str
    occurrence: int = 0

    def __str__(self) -> str:
        return self.name if self.occurrence == 0 else f"{self.name}#{self.occurrence}"


class BufferRegistry:
    """Allocates and hands out the flat graph-resident buffers.

    `accepts` is applied *before* allocation so that a narrow filter costs no
    device memory at all.
    """

    def __init__(
        self,
        *,
        budget_bytes: int,
        accepts: Callable[[str], bool],
        strict: bool = False,
    ) -> None:
        self._budget_bytes = budget_bytes
        self._accepts = accepts
        self._strict = strict
        self._buffers: dict[BufferKey, torch.Tensor] = {}
        self._addresses: dict[BufferKey, int] = {}
        self._layouts: dict[BufferKey, dict[Hashable, tuple[int, ...]]] = {}
        self._tag_of: dict[BufferKey, int] = {}
        self._key_of_tag: list[BufferKey] = []
        self._rejected: dict[BufferKey, str] = {}
        self._used_bytes = 0

    # ------------------------------- introspection --------------------------

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def num_buffers(self) -> int:
        return len(self._buffers)

    @property
    def rejected(self) -> dict[BufferKey, str]:
        return dict(self._rejected)

    def names(self) -> set[str]:
        return {key.name for key in self._buffers}

    # ------------------------------- tag mapping ----------------------------
    #
    # The T3 custom op cannot take a `BufferKey` (dynamo needs plain scalar
    # arguments), so keys are interned to small ints at trace time and resolved
    # back inside the op body at run time.

    def tag_for(self, key: BufferKey) -> int:
        tag = self._tag_of.get(key)
        if tag is None:
            tag = len(self._key_of_tag)
            self._key_of_tag.append(key)
            self._tag_of[key] = tag
        return tag

    def key_for_tag(self, tag: int) -> BufferKey:
        return self._key_of_tag[tag]

    # ------------------------------- recording ------------------------------

    def record(
        self, key: BufferKey, shape_token: Hashable, tensor: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return the flat slice `tensor` should be copied into, or None.

        None means "this name is not being captured" (filtered out, or the
        budget ran out in non-strict mode) and the caller should fall through.
        """
        if key in self._rejected:
            return None

        buf = self._buffers.get(key)
        numel = tensor.numel()
        if buf is None:
            if not self._accepts(key.name):
                self._rejected[key] = "filtered"
                return None
            buf = self._allocate(key, tensor)
            if buf is None:
                return None
        elif buf.dtype != tensor.dtype:
            raise DumpBufferGrowthRejected(
                f"dtype changed for {key}: buffer is {buf.dtype}, tap is {tensor.dtype}"
            )
        elif numel > buf.numel():
            raise DumpBufferGrowthRejected(
                f"{key} needs {numel} elements but its buffer holds only "
                f"{buf.numel()}; a recorded kernel already points at the "
                f"smaller allocation, so it cannot be replaced"
            )

        self._layouts.setdefault(key, {})[shape_token] = tuple(tensor.shape)
        return buf[:numel]

    def _allocate(self, key: BufferKey, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        nbytes = tensor.numel() * tensor.element_size()
        if self._budget_bytes >= 0 and self._used_bytes + nbytes > self._budget_bytes:
            message = (
                f"dump buffer budget of {self._budget_bytes / (1 << 20):.0f} MiB "
                f"exhausted at {self.num_buffers} buffers; {key} "
                f"(+{nbytes / (1 << 20):.1f} MiB) and any later name are dropped. "
                f"Narrow DUMPER_CUDA_GRAPH_FILTER or raise "
                f"DUMPER_CUDA_GRAPH_BUDGET_MB."
            )
            if self._strict:
                raise DumpBudgetExceeded(message)
            logger.warning(message)
            self._rejected[key] = "budget"
            return None

        buf = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        self._buffers[key] = buf
        self._addresses[key] = buf.data_ptr()
        self._used_bytes += nbytes
        return buf

    # ------------------------------- reading --------------------------------

    def views(self, shape_token: Hashable) -> Iterator[tuple[BufferKey, torch.Tensor]]:
        """Yield the buffers that were recorded for `shape_token`, reshaped.

        Keys with no layout for this token were never tapped under this shape
        (a conditionally executed module, or a different graph); they are
        skipped rather than reported as zeros.
        """
        for key, buf in self._buffers.items():
            shape = self._layouts.get(key, {}).get(shape_token)
            if shape is None:
                continue
            yield key, buf[: math.prod(shape)].view(shape)

    def assert_addresses_stable(self) -> None:
        """Acceptance check: no buffer was silently reallocated."""
        for key, buf in self._buffers.items():
            expected = self._addresses[key]
            assert buf.data_ptr() == expected, (
                f"buffer for {key} moved from {expected:#x} to "
                f"{buf.data_ptr():#x}; every recorded copy_ into it now writes "
                f"to freed memory"
            )
