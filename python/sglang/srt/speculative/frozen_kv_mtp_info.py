# Copyright 2026 SGLang Team
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

from dataclasses import dataclass
from typing import Dict

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass(frozen=True)
class FrozenKVMTPContext:
    """Target KV pool + assistant-logical -> target-physical layer map."""

    target_token_to_kv_pool: KVCache
    physical_layer_ids: Dict[int, int]

    def get_physical_layer_id(self, idx: int) -> int:
        if idx not in self.physical_layer_ids:
            raise KeyError(
                f"FrozenKVMTPContext has no physical layer id for assistant "
                f"logical index {idx}; available: {sorted(self.physical_layer_ids)}"
            )
        return self.physical_layer_ids[idx]


@dataclass
class FrozenKVMTPDraftInput(EagleDraftInput):
    """Draft input for Frozen-KV MTP.

    Frozen-KV MTP currently reuses the EAGLE scheduler/attention contract, but
    has a dedicated type so algorithm-specific behavior can move here over time.
    """

    def __post_init__(self):
        SpecInput.__init__(self, SpecInputType.FROZEN_KV_MTP_DRAFT)


@dataclass
class FrozenKVMTPVerifyInput(EagleVerifyInput):
    """Verify input for Frozen-KV MTP."""

    def __post_init__(self):
        # Run EagleVerifyInput's width auto-fill, then correct the type tag.
        super().__post_init__()
        self.spec_input_type = SpecInputType.FROZEN_KV_MTP_VERIFY
