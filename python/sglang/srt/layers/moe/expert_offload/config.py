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
"""Configuration for UVM-based expert weight offloading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


@dataclass
class ExpertOffloadConfig:
    """Per-layer configuration for UVM expert offloading.

    Expert weights are stored in CUDA Unified Memory (cudaMallocManaged).
    Resident experts are advised PREFER_GPU (stay in VRAM).
    Offloaded experts are advised PREFER_CPU + ACCESSED_BY_GPU (live in CPU
    DRAM; the GPU reads them via PCIe without triggering a page fault).

    No LRU cache, no assembly buffer, no ID remapping -- UVM handles all of it.
    """

    layer_idx: int
    num_local_experts: int
    num_resident_experts: int  # experts whose pages are kept on GPU
    resident_selection: str  # "first_n" | "frequency" | "manual"
    resident_expert_ids: Optional[List[int]]  # explicit IDs for "manual" mode
    warmup_tokens: int = 16384  # routed tokens to collect before readvise
    prefetch_num: int = 0  # 0 = disabled; N = prefetch top-N hot offloaded experts
    prefetch_depth: int = 1  # how many layers ahead to prefetch (1 = next layer only)
    memory_headroom_gib: int = 2  # GiB reserved for UVM page swapping
    is_draft_worker: bool = False
    decode_debug_log: bool = False

    @property
    def num_offloaded_experts(self) -> int:
        return self.num_local_experts - self.num_resident_experts

    @property
    def worker_role(self) -> str:
        return "draft" if self.is_draft_worker else "target"

    @property
    def log_prefix(self) -> str:
        return f"[ExpertOffload][{self.worker_role}]"


def create_expert_offload_config_from_server_args(
    server_args: "ServerArgs",
    layer_id: int,
    num_local_experts: int,
    is_draft_worker: bool = False,
) -> Optional[ExpertOffloadConfig]:
    """Return an ExpertOffloadConfig if expert offloading is enabled, else None."""
    if server_args.expert_offload_num_resident < 0:
        return None
    if is_draft_worker and not server_args.expert_offload_draft_model:
        logger.debug(
            "[ExpertOffload][draft] Skipping draft-model expert offload for layer %s",
            layer_id,
        )
        return None

    num_resident = min(server_args.expert_offload_num_resident, num_local_experts)

    resident_ids: Optional[List[int]] = None
    if (
        server_args.expert_offload_resident_selection == "manual"
        and server_args.expert_offload_resident_ids is not None
    ):
        resident_ids = [
            int(x.strip())
            for x in server_args.expert_offload_resident_ids.split(",")
            if x.strip()
        ]

    return ExpertOffloadConfig(
        layer_idx=layer_id,
        num_local_experts=num_local_experts,
        num_resident_experts=num_resident,
        resident_selection=server_args.expert_offload_resident_selection,
        resident_expert_ids=resident_ids,
        prefetch_num=server_args.expert_offload_prefetch_num,
        prefetch_depth=server_args.expert_offload_prefetch_depth,
        memory_headroom_gib=server_args.expert_offload_memory_headroom,
        is_draft_worker=is_draft_worker,
        decode_debug_log=server_args.expert_offload_decode_debug_log,
    )
