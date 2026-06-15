# Copyright 2023-2024 SGLang Team
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

"""
Eviction policies for LoRA adapter memory management.
"""

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set

logger = logging.getLogger(__name__)


class EvictionPolicy(ABC):
    """Abstract base class for LoRA adapter eviction policies."""

    def __init__(self):
        # Stores adapters in a conceptually ordered way as specified by the policy.
        # Key is the adapter's UID and value is unused.
        self.ordered_uids: OrderedDict[str, None] = OrderedDict()
        self.eviction_count = 0

    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None:
        """Marks an adapter as used."""
        pass

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Selects an adapter to evict from candidates."""
        pass

    def remove(self, uid: Optional[str]) -> None:
        """Removes an adapter from the policy's tracking."""
        if uid is not None:
            self.ordered_uids.pop(uid, None)
            logger.debug(f"Removed LoRA {uid} from eviction policy tracking")


class LRUEvictionPolicy(EvictionPolicy):
    """LRU eviction policy - evicts the least recently used adapter."""

    def mark_used(self, uid: Optional[str]) -> None:
        if uid is not None:
            # Remove and re-add to move to end (most recent)
            self.ordered_uids.pop(uid, None)
            self.ordered_uids[uid] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Select the least recently used adapter from candidates."""
        # Iterate through ordered_uids (oldest first) to find LRU victim
        for uid in self.ordered_uids:
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (LRU)")
                self.eviction_count += 1
                return uid

        # If no tracked UID found in candidates, check if None is available
        # This happens when the batch consists entirely of LoRA requests
        # and None (base model) is the only eviction candidate
        if None in candidates:
            logger.debug("Selected None (base model) for eviction")
            self.eviction_count += 1
            return None

        # Should never reach here if candidates is non-empty
        raise RuntimeError(f"Failed to select LRU victim from candidates: {candidates}")


class FIFOEvictionPolicy(EvictionPolicy):
    """FIFO eviction policy - for backward compatibility."""

    def mark_used(self, uid: Optional[str]) -> None:
        """For FIFO, we only track insertion order (not access time)."""
        if uid is not None and uid not in self.ordered_uids:
            self.ordered_uids[uid] = None

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Select the first inserted adapter from candidates."""
        # Iterate through ordered_uids (oldest first) to find FIFO victim
        for uid in self.ordered_uids:
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (FIFO)")
                self.eviction_count += 1
                return uid

        # If no tracked UID found in candidates, check if None is available
        # This happens when the batch consists entirely of LoRA requests
        # and None (base model) is the only eviction candidate
        if None in candidates:
            logger.debug("Selected None (base model) for eviction")
            self.eviction_count += 1
            return None

        # Should never reach here if candidates is non-empty
        raise RuntimeError(
            f"Failed to select FIFO victim from candidates: {candidates}"
        )


POLICIES = {
    "fifo": FIFOEvictionPolicy,
    "lru": LRUEvictionPolicy,
}


def get_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory function to create eviction policy instances."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown eviction policy: {policy_name}")
    return POLICIES[policy_name]()
