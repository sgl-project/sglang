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

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.managers.schedule_batch import global_server_args_dict


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random", "dynamic", "fake", "probability", "test"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = global_server_args_dict["ep_dispatch_algorithm"]
        expert_location_metadata = get_global_expert_location_metadata()

        if ep_dispatch_algorithm is None:
            return None

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
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits = torch.randn_like(router_logits)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, 
    info: Optional[ExpertLocationDispatchInfo],
    logical_expert_probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    if info.ep_dispatch_algorithm == "probability" and logical_expert_probabilities is not None:
        return _topk_ids_logical_to_physical_probability(topk_ids, info, logical_expert_probabilities)
    if info.ep_dispatch_algorithm == "test":
        print("test ep dispatch algorithm probability")
        # Generate random probabilities with same shape as logical_to_all_physical_map
        logical_expert_probabilities = torch.rand_like(info.partial_logical_to_all_physical_map, dtype=torch.float32, device=topk_ids.device)
        print("Generated random probabilities")
        # Set probability to 0 where physical expert ID is -1
        logical_expert_probabilities[info.partial_logical_to_all_physical_map == -1] = 0
        
        return _topk_ids_logical_to_physical_probability(topk_ids, info, logical_expert_probabilities)
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
    print(f"Chosen dispatch index, topk_ids shape: {topk_ids.shape}, chosen_dispatch_index shape: {chosen_dispatch_index.shape}, partial_logical_to_all_physical_map shape: {info.partial_logical_to_all_physical_map.shape}")
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_probability(
    topk_ids: torch.Tensor, 
    info: ExpertLocationDispatchInfo,
    logical_expert_probabilities: torch.Tensor,
) -> torch.Tensor:
    """
    Select physical experts based on probability distribution for each logical expert.
    
    Args:
        topk_ids: Logical expert IDs (num_tokens, topk)
        info: Expert location dispatch information
        logical_expert_probabilities: Probability distribution tensor with same shape as 
                                     logical_to_all_physical_map (num_logical_experts, max_physical_per_logical)
                                     Each element represents probability of selecting that physical expert
                                     -1 indicates no such physical expert
        
    Returns:
        Physical expert IDs with same shape as topk_ids
    """
    print("Starting _topk_ids_logical_to_physical_probability")
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    # Get the mapping from logical to physical experts
    logical_to_all_physical_map = info.partial_logical_to_all_physical_map
    num_valid_physical = info.partial_logical_to_all_physical_map_num_valid
    
    # Verify probability tensor has correct shape
    expected_shape = logical_to_all_physical_map.shape
    if logical_expert_probabilities.shape != expected_shape:
        raise ValueError(
            f"Probability tensor shape {logical_expert_probabilities.shape} "
            f"does not match expected shape {expected_shape}"
        )
    print("Start masking probabilities")
    # Create mask for valid physical experts (-1 indicates invalid)
    valid_mask = logical_to_all_physical_map != -1
    
    # Zero out probabilities for invalid physical experts
    masked_probabilities = logical_expert_probabilities * valid_mask

    topk_probs = masked_probabilities[topk_ids]
    print("Masked probabilities")
    chosen_dispatch_index = torch.multinomial(topk_probs, 1).flatten()

    print(f"Chosen dispatch index, topk_ids shape: {topk_ids.shape}, chosen_dispatch_index shape: {chosen_dispatch_index.shape}, logical_to_all_physical_map shape: {logical_to_all_physical_map.shape}")
    topk_ids = logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    print("Converted to physical experts, topk_ids shape: ", topk_ids.shape)
    return topk_ids

    # # Create new physical expert selection
    # new_physical_ids = torch.zeros_like(topk_ids)
    
    # for i, logical_expert_id in enumerate(topk_ids):
    #     if logical_expert_id == -1:
    #         new_physical_ids[i] = -1
    #         continue
            
    #     # Get available physical experts and their probabilities for this logical expert
    #     available_physical = logical_to_all_physical_map[logical_expert_id]
    #     available_probabilities = logical_expert_probabilities[logical_expert_id]
    #     num_available = num_valid_physical[logical_expert_id]
        
    #     if num_available == 0:
    #         new_physical_ids[i] = -1
    #         continue
        
    #     # Filter valid physical experts and their probabilities
    #     valid_mask = available_physical != -1
    #     valid_physical = available_physical[valid_mask]
    #     valid_probabilities = available_probabilities[valid_mask]
        
    #     if len(valid_physical) == 0:
    #         new_physical_ids[i] = -1
    #         continue
        
    #     # Select physical expert based on probability
    #     if len(valid_physical) == 1:
    #         # Only one physical expert available
    #         selected_physical = valid_physical[0]
    #     else:
    #         # Multiple physical experts available - use weighted selection
    #         # Sample from multinomial distribution
    #         selected_idx = torch.multinomial(valid_probabilities, 1).item()
    #         selected_physical = valid_physical[selected_idx]
        
    #     new_physical_ids[i] = selected_physical
    
    # return new_physical_ids.view(topk_ids_original_shape)
