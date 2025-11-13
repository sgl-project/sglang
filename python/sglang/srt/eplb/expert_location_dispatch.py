# Copyright 2023-2025 SGLang Team
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

import random
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args

import os

max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 256))
DUMMY_MAX_TOPK_SHAPE = (max_batch_size, 8)  # (max batch size, topk)


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    precomputed_balanced_tensor_map: dict

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = get_global_server_args().ep_dispatch_algorithm
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if ep_dispatch_algorithm is None:
            return None

        precomputed_balanced_tensor_map = {}
        if ep_dispatch_algorithm == "fake":
            nnodes = get_global_server_args().nnodes
            node_rank = get_global_server_args().node_rank
            device = get_global_server_args().device
            num_physical_experts = expert_location_metadata.num_physical_experts
            dispatch_node = get_global_server_args().ep_dispatch_fake_num_node
            dummy_topk_ids_shape = DUMMY_MAX_TOPK_SHAPE
            balanced_tensor = generate_balanced_expert_selection(
                dummy_topk_ids_shape,
                node_rank=node_rank,
                nnodes=nnodes,
                total_experts=num_physical_experts,
                dispatch_node=dispatch_node,
                device=device,
            )
            precomputed_balanced_tensor_map[dummy_topk_ids_shape] = balanced_tensor

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
            precomputed_balanced_tensor_map=precomputed_balanced_tensor_map,
        )


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm == "dynamic":
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    if info.ep_dispatch_algorithm == "fake":
        return _topk_ids_logical_to_physical_fake(topk_ids, info)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_fake(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:

    topk_ids_original_shape = topk_ids.shape
    target_len = topk_ids_original_shape[0]
    base_shape, base_tensor = next(iter(info.precomputed_balanced_tensor_map.items()))
    base_len = base_shape[0]

    repeat_n = target_len // base_len
    remaining = target_len % base_len

    balanced_tensor = (
        torch.cat([base_tensor.repeat((repeat_n, 1)), base_tensor[:remaining]], dim=0)
        if target_len > 0
        else base_tensor[:0]
    )

    topk_ids = balanced_tensor.view(topk_ids_original_shape)

    return topk_ids


def generate_balanced_expert_selection(
    topk_ids_shape,
    node_rank=0,
    nnodes=1,
    total_experts=288,
    dispatch_node=1,
    device="cpu",
):
    num_elements = topk_ids_shape[0] * topk_ids_shape[1]
    # Number of experts per node
    assert total_experts % nnodes == 0
    experts_per_node = total_experts // nnodes

    # Number of tokens processed by each expert
    tokens_per_expert = num_elements // experts_per_node // dispatch_node
    remaining_tokens_per_expert = num_elements // experts_per_node % dispatch_node

    repeat = []
    remaining_list = []
    # Select experts from multiple nodes
    for i in range(dispatch_node):
        start = (node_rank + i) * experts_per_node
        end = (node_rank + i + 1) * experts_per_node
        base = torch.arange(start, end, device=device) % total_experts
        if remaining_tokens_per_expert != 0:
            remaining_list.append(base)
        repeat.append(base.repeat(tokens_per_expert))
    selected_experts = torch.cat(repeat)

    if remaining_tokens_per_expert != 0:
        sampled = random.sample(remaining_list, remaining_tokens_per_expert)
        random_remaining = torch.cat(sampled)
        selected_experts = torch.cat([selected_experts, random_remaining])

    return selected_experts.view(topk_ids_shape)
