"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
KV caching events
"""

import enum
from typing import Optional, Union

import msgspec

# Re-exported for backward compatibility: the generic ZMQ event-publishing
# transport now lives in `sglang.srt.utils.event_publisher`. Existing importers
# (and external consumers) can keep importing these from `kv_events`.
from sglang.srt.utils.event_publisher import (  # noqa: F401
    EventBatch,
    EventPublisher,
    EventPublisherFactory,
    KVEventsConfig,
    NullEventPublisher,
    ZmqEventPublisher,
)


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


class OffloadedState:
    """
    OffloadedState represents the state of a KV cache block offloaded to the hicache.

    - prefill_len (int): The length of the prefill part of the KV cache block.
    - inc_len (int): The length of the incremental part of the KV cache block.
    - last_hash (Optional[str]): The hash of the last token in the KV cache block.
    """

    def __init__(
        self, prefill_len: int, inc_len: int = 0, last_hash: Optional[str] = None
    ):
        self.prefill_len = prefill_len
        self.inc_len = inc_len
        self.last_hash = last_hash


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]
