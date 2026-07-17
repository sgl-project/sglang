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

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch
import torch.distributed
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _prefer_same_node_experts(server_args: ServerArgs) -> bool:
    from sglang.srt.elastic_ep.elastic_ep import elastic_expanded_world_enabled

    return server_args.ep_join_mode != "scale" and not elastic_expanded_world_enabled()


def _compute_elastic_expert_layout(
    base_num_physical_experts: int,
    initial_ep_size: int,
    effective_ep_size: int,
) -> tuple[int, int]:
    assert base_num_physical_experts % initial_ep_size == 0
    num_local_physical_experts = base_num_physical_experts // initial_ep_size
    return (
        num_local_physical_experts * effective_ep_size,
        num_local_physical_experts,
    )


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    ep_size: int
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]

    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(
        server_args: ServerArgs, model_config: ModelConfig, moe_ep_rank: int
    ):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        base_num_physical_experts = common["base_num_physical_experts"]
        physical_to_logical_map = (
            torch.arange(0, base_num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )
        physical_to_logical_map = append_trivial_expert_slots(
            physical_to_logical_map,
            num_physical_experts - base_num_physical_experts,
            num_logical_experts,
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
        moe_ep_rank: int = None,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        if common["num_physical_experts"] > common["base_num_physical_experts"]:
            if physical_to_logical_map.shape[-1] == common["base_num_physical_experts"]:
                physical_to_logical_map = append_trivial_expert_slots(
                    physical_to_logical_map,
                    common["num_physical_experts"]
                    - common["base_num_physical_experts"],
                    model_config_for_expert_location.num_logical_experts,
                )
            assert physical_to_logical_map.shape[-1] == common["num_physical_experts"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            server_args=server_args,
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
            ep_size=common["ep_size"],
            moe_ep_rank=moe_ep_rank,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            moe_ep_rank=moe_ep_rank,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        from sglang.srt.eplb import eplb_algorithms

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        if model_config_for_expert_location is None:
            return None

        base_num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        num_physical_experts = base_num_physical_experts
        initial_ep_size = server_args.elastic_ep_initial_size
        if initial_ep_size is not None:
            if server_args.ep_join_mode == "scale":
                ep_size = max(
                    ep_size,
                    server_args.ep_join_rank_offset + server_args.tp_size,
                )
            num_physical_experts, num_local_physical_experts = (
                _compute_elastic_expert_layout(
                    base_num_physical_experts,
                    initial_ep_size,
                    ep_size,
                )
            )
        else:
            assert num_physical_experts % ep_size == 0
            num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            base_num_physical_experts=base_num_physical_experts,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
        moe_ep_rank: Optional[int] = None,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            ep_size=ep_size,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    server_args=server_args,
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    ep_size=ep_size,
                    num_physical_experts=num_physical_experts,
                    ep_rank=(
                        moe_ep_rank
                        if moe_ep_rank is not None
                        else torch.distributed.get_rank() % ep_size
                    ),
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_cpu",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self,
        layer_id: int,
        logical_expert_id: int,
        require_global_experts: bool = False,
    ) -> List[int]:
        # Use CPU copy to avoid GPU→CPU sync on every call, which is expensive in update weights scenario
        cpu_map = self.logical_to_all_physical_map_cpu
        # Draft workers can query MoE layers whose layer_id lies beyond the
        # target-sized expert map; fall back to the identity mapping (no EPLB
        # rebalancing for those layers) instead of indexing out of range.
        if layer_id >= cpu_map.shape[0]:
            if require_global_experts:
                num_physical_experts = cpu_map.shape[-1]
                return list(
                    range(
                        logical_expert_id,
                        num_physical_experts,
                        self.num_logical_experts,
                    )
                )
            return [logical_expert_id]
        if require_global_experts:
            num_physical_experts = cpu_map[layer_id].shape[-1]
            return list(
                range(logical_expert_id, num_physical_experts, self.num_logical_experts)
            )
        return [
            physical_expert_id
            for physical_expert_id in cpu_map[layer_id, logical_expert_id].tolist()
            if physical_expert_id != -1
        ]


def format_expert_location_layout(
    metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if metadata is None:
        return "<none>"

    return format_physical_to_logical_map(
        metadata.physical_to_logical_map_cpu,
        ep_size=metadata.ep_size,
        layer_ids=layer_ids,
    )


def format_expert_location_layout_diff(
    old_metadata: Optional[ExpertLocationMetadata],
    new_metadata: Optional[ExpertLocationMetadata],
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    if old_metadata is None or new_metadata is None:
        return "<none>"

    old_map = old_metadata.physical_to_logical_map_cpu
    new_map = new_metadata.physical_to_logical_map_cpu
    if old_map.shape != new_map.shape:
        return f"shape_changed old_shape={tuple(old_map.shape)} new_shape={tuple(new_map.shape)}"

    layer_ids = _normalize_layer_ids(layer_ids, num_layers=old_map.shape[0])
    num_physical_experts = old_map.shape[1]

    changed_by_layer = []
    for layer_id in layer_ids:
        num_changed = torch.count_nonzero(old_map[layer_id] != new_map[layer_id]).item()
        if num_changed > 0:
            changed_by_layer.append((layer_id, num_changed))

    total_changed = sum(num_changed for _, num_changed in changed_by_layer)
    total_slots = len(layer_ids) * num_physical_experts
    lines = [f"changed_physical_slots={total_changed}/{total_slots}"]
    if not changed_by_layer:
        lines.append("changed_layers=[]")
        return "\n".join(lines)

    for layer_id, num_changed in changed_by_layer:
        lines.append(f"layer={layer_id}: changed={num_changed}/{num_physical_experts}")
    return "\n".join(lines)


def format_physical_to_logical_map(
    physical_to_logical_map: torch.Tensor,
    ep_size: int,
    layer_ids: Optional[Iterable[int]] = None,
) -> str:
    physical_to_logical_map = physical_to_logical_map.cpu()
    if physical_to_logical_map.numel() == 0:
        return "<empty>"

    layer_ids = _normalize_layer_ids(
        layer_ids, num_layers=physical_to_logical_map.shape[0]
    )
    num_physical_experts = physical_to_logical_map.shape[1]
    num_local_physical_experts, remainder = divmod(num_physical_experts, ep_size)

    lines = [
        "physical_to_logical_map "
        f"num_layers={physical_to_logical_map.shape[0]} "
        f"num_physical_experts={num_physical_experts} "
        f"ep_size={ep_size}"
    ]
    for layer_id in layer_ids:
        row = physical_to_logical_map[layer_id].tolist()
        if remainder != 0:
            lines.append(
                f"layer={layer_id}: "
                f"physical={json.dumps(row, separators=(',', ':'))}"
            )
            continue

        rank_chunks = []
        for ep_rank in range(ep_size):
            start = ep_rank * num_local_physical_experts
            end = start + num_local_physical_experts
            rank_chunks.append(
                f"ep{ep_rank}={json.dumps(row[start:end], separators=(',', ':'))}"
            )
        lines.append(f"layer={layer_id}: " + " ".join(rank_chunks))

    return "\n".join(lines)


def _normalize_layer_ids(
    layer_ids: Optional[Iterable[int]],
    num_layers: int,
) -> List[int]:
    if layer_ids is None:
        return list(range(num_layers))

    normalized_layer_ids = [int(layer_id) for layer_id in layer_ids]
    for layer_id in normalized_layer_ids:
        assert 0 <= layer_id < num_layers, f"{layer_id=} {num_layers=}"
    return normalized_layer_ids


def get_global_expert_location_metadata():
    from sglang.srt.runtime_context import get_resources

    return get_resources().expert_location_metadata


def set_global_expert_location_metadata(value, allow_overwrite=False):
    from sglang.srt.runtime_context import get_resources

    resources = get_resources()
    if not allow_overwrite:
        assert resources.expert_location_metadata is None
    resources.expert_location_metadata = value


def append_trivial_expert_slots(
    physical_to_logical_map: torch.Tensor,
    count: int,
    num_logical_experts: int,
    start: int = 0,
) -> torch.Tensor:
    if count <= 0:
        return physical_to_logical_map
    new_slots = torch.arange(
        start,
        start + count,
        dtype=physical_to_logical_map.dtype,
        device=physical_to_logical_map.device,
    ).unsqueeze(0)
    new_slots = new_slots.expand(physical_to_logical_map.shape[0], -1)
    return torch.cat([physical_to_logical_map, new_slots % num_logical_experts], dim=1)


def broadcast_global_expert_location_metadata(
    model_config: ModelConfig,
    moe_ep_rank: int,
    src_rank: int = 0,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> ExpertLocationMetadata:
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    metadata = get_global_expert_location_metadata()
    assert metadata is not None

    metadata.physical_to_logical_map = metadata.physical_to_logical_map.contiguous()
    torch.distributed.broadcast(
        metadata.physical_to_logical_map, src=src_rank, group=group
    )
    metadata = ExpertLocationMetadata.init_by_mapping(
        server_args,
        model_config,
        metadata.physical_to_logical_map,
        moe_ep_rank=moe_ep_rank,
    )
    set_global_expert_location_metadata(metadata, allow_overwrite=True)
    return metadata


def _compute_logical_to_all_physical_map(
    server_args: ServerArgs,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    # Find out the candidate physical experts for each logical expert on each layer
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    # Replace by the physical expert on local GPU or node if possible
    if moe_ep_rank is not None:
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        prefer_same_node = _prefer_same_node_experts(server_args)
        num_gpus_per_node = (
            server_args.ep_size // server_args.nnodes if prefer_same_node else None
        )
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
            if num_gpus_per_node is not None
            else None
        )
        for layer_id in range(num_layers):
            for logical_expert_id in range(num_logical_experts):
                # Try to find the nearest physical expert
                nearest_expert = _find_nearest_expert(
                    candidate_physical_expert_ids=logical_to_all_physical_map[layer_id][
                        logical_expert_id
                    ],
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                # Replace by the nearest physical expert
                if nearest_expert != -1:
                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
                        nearest_expert
                    ]

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    server_args: ServerArgs,
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    prefer_same_node = _prefer_same_node_experts(server_args)
    num_gpus_per_node = (
        server_args.ep_size // server_args.nnodes if prefer_same_node else None
    )
    num_local_node_physical_experts = (
        num_local_gpu_physical_experts * num_gpus_per_node
        if num_gpus_per_node is not None
        else None
    )
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                val = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            if remaining_ranks:
                choices = _fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_gpu_physical_experts: int
) -> int:
    return physical_expert_id // num_local_gpu_physical_experts


def _compute_node_id_of_physical_expert(
    physical_expert_id: int, num_local_host_physical_experts: int
) -> int:
    return physical_expert_id // num_local_host_physical_experts


def _find_nearest_expert(
    candidate_physical_expert_ids: List[int],
    num_local_gpu_physical_experts: int,
    moe_ep_rank: int,
    num_gpus_per_node: Optional[int],
    num_local_node_physical_experts: Optional[int],
) -> int:
    # 1. If only one candidate, return it directly
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]

    # 2. Prefer same-GPU experts
    same_gpu_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        == moe_ep_rank
    ]
    if len(same_gpu_physical_expert_ids) > 0:
        return same_gpu_physical_expert_ids[0]

    # Prefer same-node experts only when it narrows the candidate set.
    if num_gpus_per_node is not None and num_local_node_physical_experts is not None:
        node_rank = moe_ep_rank // num_gpus_per_node
        same_node_physical_expert_ids = [
            physical_expert_id
            for physical_expert_id in candidate_physical_expert_ids
            if _compute_node_id_of_physical_expert(
                physical_expert_id, num_local_node_physical_experts
            )
            == node_rank
        ]
        if 0 < len(same_node_physical_expert_ids) < len(candidate_physical_expert_ids):
            return same_node_physical_expert_ids[0]

    # 4. At last, leave it as -1 to indicate not found.
    return -1


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        from sglang.srt.model_loader import get_model_architecture

        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return None


def compute_initial_expert_location_metadata(
    server_args: ServerArgs,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> Optional[ExpertLocationMetadata]:
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(
            server_args, model_config, moe_ep_rank
        )

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            **data_dict,
            moe_ep_rank=moe_ep_rank,
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=data_dict["logical_count"]
        )
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )
