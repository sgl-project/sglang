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
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Set

logger = logging.getLogger(__name__)


class EvictionPolicy(ABC):
    """Abstract base class for LoRA adapter eviction policies."""

    @abstractmethod
    def mark_used(self, uid: Optional[str]) -> None:
        """Marks an adapter as used."""
        pass

    @abstractmethod
    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Selects an adapter to evict from candidates."""
        pass

    @abstractmethod
    def remove(self, uid: Optional[str]) -> None:
        """Removes an adapter from the policy's tracking."""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """LRU eviction policy - evicts the least recently used adapter."""

    def __init__(self):
        self.access_order = OrderedDict()  # key=uid, value=last_access_time
        self.total_accesses = 0
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:
        if uid is not None:
            current_time = time.monotonic()
            # Remove and re-add to move to end (most recent)
            self.access_order.pop(uid, None)
            self.access_order[uid] = current_time
            self.total_accesses += 1
            logger.debug(f"LoRA {uid} marked as used at {current_time}")

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Select the least recently used adapter from candidates."""
        # Base model (currently None, will be replaced with special UID in future)
        # always has lowest priority - evict it first if available
        BASE_MODEL_UID = None  # TODO: Replace with special UID constant
        if BASE_MODEL_UID in candidates:
            logger.debug(f"Selected base model for eviction (LRU)")
            self.eviction_count += 1
            return BASE_MODEL_UID

        # Iterate through access_order (oldest first) to find LRU victim
        for uid in list(self.access_order.keys()):
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (LRU)")
                self.eviction_count += 1
                return uid

        # Should never reach here if candidates is non-empty
        assert False, f"Failed to select LRU victim from candidates: {candidates}"

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.access_order.pop(uid, None)
            logger.debug(f"Removed LoRA {uid} from LRU tracking")


class FIFOEvictionPolicy(EvictionPolicy):
    """FIFO eviction policy - for backward compatibility."""

    def __init__(self):
        self.insertion_order = (
            OrderedDict()
        )  # key=uid, OrderedDict maintains insertion order
        self.eviction_count = 0

    def mark_used(self, uid: Optional[str]) -> None:
        """For FIFO, we only track insertion order (not access time)."""
        if uid is not None and uid not in self.insertion_order:
            self.insertion_order[uid] = (
                True  # Value unused, OrderedDict tracks insertion order
            )

    def select_victim(self, candidates: Set[Optional[str]]) -> Optional[str]:
        """Select the first inserted adapter from candidates."""
        # Base model (currently None, will be replaced with special UID in future)
        # always has lowest priority - evict it first if available
        BASE_MODEL_UID = None  # TODO: Replace with special UID constant
        if BASE_MODEL_UID in candidates:
            logger.debug(f"Selected base model for eviction (FIFO)")
            self.eviction_count += 1
            return BASE_MODEL_UID

        # Iterate through insertion_order (oldest first) to find FIFO victim
        for uid in list(self.insertion_order.keys()):
            if uid in candidates:
                logger.debug(f"Selected LoRA {uid} for eviction (FIFO)")
                self.eviction_count += 1
                return uid

        # Should never reach here if candidates is non-empty
        assert False, f"Failed to select FIFO victim from candidates: {candidates}"

    def remove(self, uid: Optional[str]) -> None:
        if uid is not None:
            self.insertion_order.pop(uid, None)


def get_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory function to create eviction policy instances."""
    policies = {
        "fifo": FIFOEvictionPolicy,
        "lru": LRUEvictionPolicy,
    }
    if policy_name not in policies:
        raise ValueError(f"Unknown eviction policy: {policy_name}")
    return policies[policy_name]()
