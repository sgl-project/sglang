from dataclasses import dataclass
from typing import Literal

import torch
from sglang.srt.utils import get_compiler_backend


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random"]


def topk_ids_logical_to_physical(topk_ids: torch.Tensor, info: ExpertLocationDispatchInfo) -> torch.Tensor:
    return TODO

    # TODO
    # expert_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor] = None,
    # forward_mode=None,
    # expert_logical_to_all_physical_map=None,
    # expert_logical_to_all_physical_map_num_valid=None,

    # TODO
    if expert_logical_to_rank_dispatch_physical_map is not None:
        # TODO optimize these things later
        if forward_mode.is_extend():
            topk_ids = _hack_expert_location_dispatch_random(
                topk_ids=topk_ids,
                expert_logical_to_all_physical_map=expert_logical_to_all_physical_map,
                expert_logical_to_all_physical_map_num_valid=expert_logical_to_all_physical_map_num_valid,
            )
        else:
            topk_ids = expert_logical_to_rank_dispatch_physical_map[topk_ids]


def _hack_expert_location_dispatch_random(
    topk_ids: torch.Tensor,
    expert_logical_to_all_physical_map: torch.Tensor,
    expert_logical_to_all_physical_map_num_valid: torch.Tensor,
):
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
                             % expert_logical_to_all_physical_map_num_valid[topk_ids])
    topk_ids = expert_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids
