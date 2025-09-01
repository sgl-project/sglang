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

import numpy as np
import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.lp_matrices_prep import get_global_token_dispatch_metadata
from sglang.srt.managers.schedule_batch import global_server_args_dict


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random", "dynamic", "fake", "lp"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int
    # (num_logical_experts, X)
    # partial_logical_to_valid_physical_map: torch.Tensor
    lp_metadata: Optional[dict[str, np.ndarray]]

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = global_server_args_dict["ep_dispatch_algorithm"]
        expert_location_metadata = get_global_expert_location_metadata()
        if ep_dispatch_algorithm == "lp":
            lp_metadata = get_global_token_dispatch_metadata()
        else:
            lp_metadata = None

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
            # partial_logical_to_valid_physical_map=expert_location_metadata.logical_to_all_physical_map[
            #     layer_id,
            #     :,
            #     : expert_location_metadata.logical_to_all_physical_map_num_valid[
            #         layer_id, :
            #     ].max(),
            # ],
            lp_metadata=(
                {
                    "B1": lp_metadata.B1[layer_id],
                    "B2": lp_metadata.B2[layer_id],
                    "C": lp_metadata.C[layer_id],
                    "c": lp_metadata.c[layer_id],
                    "G": lp_metadata.G[layer_id],
                    "A": lp_metadata.A[layer_id],
                    "single_expert_array": lp_metadata.single_expert_array[layer_id],
                    "log_replicated_expert_array": lp_metadata.log_replicated_expert_array[
                        layer_id
                    ],
                    "phy_replicated_expert_array": lp_metadata.phy_replicated_expert_array[
                        layer_id
                    ],
                    "dims": lp_metadata.dims,
                    "ecos_opts": lp_metadata.ecos_opts,
                }
                if lp_metadata is not None
                else None
            ),
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits.uniform_(5, 10)
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
    if info.ep_dispatch_algorithm == "lp" and logical_expert_probabilities is not None:
        return _topk_ids_logical_to_physical_probability(
            topk_ids, info, logical_expert_probabilities
        )

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
    # topk_ids = torch.ones_like(topk_ids, device=device)
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
                                     0 indicates no such physical expert

    Returns:
        Physical expert IDs with same shape as topk_ids
    """
    topk_ids_original_shape = topk_ids.shape
    topk_ids = topk_ids.flatten()
    log2phy_map = info.partial_logical_to_all_physical_map

    topk_probs = logical_expert_probabilities[topk_ids]
    # topk_probs = torch.rand(topk_ids.shape[0], logical_expert_probabilities.shape[1], device=topk_ids.device)
    chosen_dispatch_index = _sample_from_probs_fast(topk_probs.float())
    topk_ids = log2phy_map[topk_ids, chosen_dispatch_index]
    topk_ids = topk_ids.view(topk_ids_original_shape)
    # topk_ids = torch.ones_like(topk_ids)
    return topk_ids


def _sample_from_probs_fast(probs: torch.Tensor) -> torch.Tensor:
    probs = probs / probs.sum(dim=1, keepdim=True)
    cumprobs = torch.cumsum(probs, dim=1)
    rand_vals = torch.rand(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)
    indices = torch.searchsorted(cumprobs, rand_vals, right=True)

    return indices.flatten()
