from dataclasses import dataclass
from typing import Literal, Optional

import einops
import torch
from sglang.srt.managers.schedule_batch import (
    get_global_expert_location_metadata,
    global_server_args_dict,
)
from sglang.srt.utils import get_compiler_backend


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random"]
    partial_logical_to_rank_dispatch_physical_map: torch.Tensor
    partial_logical_to_all_physical_map: torch.Tensor
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
            partial_logical_to_rank_dispatch_physical_map=expert_location_metadata.logical_to_rank_dispatch_physical_map[
                                                          layer_id, :],
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                                                layer_id, :
                                                ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                                                          layer_id, :
                                                          ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm == "random":
        return _topk_ids_logical_to_physical_random(topk_ids, info)
    if info.ep_dispatch_algorithm == "fake_uniform":
        return _topk_ids_logical_to_physical_fake_uniform(topk_ids, info)
    if info.ep_dispatch_algorithm == "fake_grouped_uniform":
        return _topk_ids_logical_to_physical_fake_grouped_uniform(topk_ids, info)
    raise NotImplementedError


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_random(
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


def _topk_ids_logical_to_physical_fake_uniform(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # NOTE it will have probability to send one token to one expert multiple times
    return torch.randint(
        0,
        info.num_physical_experts,
        topk_ids.shape,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _topk_ids_logical_to_physical_fake_grouped_uniform(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # NOTE it will have probability to send one token to one expert multiple times
    # NOTE it will make each group have exactly two experts chosen

    num_tokens, num_topk = topk_ids.shape
    dtype = topk_ids.dtype
    device = topk_ids.device
    num_physical_experts = info.num_physical_experts

    n_group = 8
    topk_group = 4
    num_experts_per_group = num_physical_experts // n_group

    chosen_groups_of_token = torch.rand(num_tokens, n_group, device=device).argsort(
        dim=1
    )[:, :topk_group]
    delta_within_group = torch.randint(
        0,
        num_physical_experts // n_group,
        (num_tokens, num_topk),
        dtype=dtype,
        device=device,
    )
    chosen_groups_of_token_repeated = einops.repeat(
        chosen_groups_of_token,
        "num_tokens topk_group -> num_tokens (topk_group repeat_n)",
        repeat_n=num_topk // topk_group,
    )
    return chosen_groups_of_token_repeated * num_experts_per_group + delta_within_group
