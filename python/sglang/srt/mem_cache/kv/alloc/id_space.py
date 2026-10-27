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
"""How a sub-pool names its units: the id space, and the names consumers
ask for a token sub-pool by."""

from __future__ import annotations

import enum
from typing import Callable, Dict, Hashable, Optional

import msgspec
import torch


class IdSpaceKind(str, enum.Enum):
    """The names of the two token sub-pools today's consumers ask for by
    role. Id spaces are keyed by sub-pool name, so any sub-pool's name works
    wherever a kind does."""

    # Full-attention layers: the pool's own token ids.
    FULL = "full"
    # The sliding-window layers of a hybrid pool: the same token ids through
    # the sliding-window sub-pool's table.
    SLIDING_WINDOW = "swa"


class IdSpace(msgspec.Struct, frozen=True):
    """How one sub-pool names the iteration's slots.

    ``write`` maps the iteration's virtual token ids to this sub-pool's ids
    (None: they index it as they are). ``read_v2p`` is the page table its reads
    go through (None: reads stay in `req_to_token`, for the consumer to
    finish). ``table`` is the sub-pool's page table even when reads stay
    virtual (DCP), for builders that gather a rank's share themselves.
    Runners whose pools share a sub-pool share its ``key``, and with it a
    plan's ids for it."""

    key: Hashable
    write: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    read_v2p: Optional[torch.Tensor] = None
    table: Optional[torch.Tensor] = None

    def ids(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """``virtual_ids`` as this sub-pool indexes them."""
        return virtual_ids if self.write is None else self.write(virtual_ids)


def state_id_space(spaces: Dict[str, IdSpace], sub_pool: str) -> IdSpace:
    """How a recurrent-state sub-pool names the slots the request table holds:
    through its allocator's table when the allocator hands out virtual slot ids,
    else as they are."""
    space = spaces.get(sub_pool)
    return IdSpace(key=(sub_pool, None)) if space is None else space
