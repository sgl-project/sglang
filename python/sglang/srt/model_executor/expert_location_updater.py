import logging
from typing import Dict, List

import torch
import torch.distributed
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata

logger = logging.getLogger(__name__)


def update_expert_location(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    new_expert_location_metadata: ExpertLocationMetadata,
):
    old_expert_location_metadata = get_global_expert_location_metadata()
    _update_expert_weights(routed_experts_weights_of_layer, old_expert_location_metadata, new_expert_location_metadata)
    old_expert_location_metadata.update(new_expert_location_metadata)


def _update_expert_weights(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
):
    temp_buffers = create_temp_buffers(next(iter(routed_experts_weights_of_layer.values())))
    for layer_id in sorted(routed_experts_weights_of_layer.keys()):
        update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_expert_location_metadata.physical_to_logical_map[layer_id],
            new_physical_to_logical_map=new_expert_location_metadata.physical_to_logical_map[layer_id],
        )


def create_temp_buffers(sample_tensors):
    return [torch.empty_like(tensor) for tensor in sample_tensors]


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],
    temp_buffers: List[torch.Tensor],
    old_physical_to_logical_map: torch.Tensor,  # (num_global_physical_Experts,)
    new_physical_to_logical_map: torch.Tensor,  # (num_global_physical_Experts,)
    num_local_physical_experts: int,
    rank: int,
):
    assert all(tensor.shape[0] == num_local_physical_experts for tensor in routed_experts_weights)
    old_physical_to_logical_map = old_physical_to_logical_map.tolist()
    new_physical_to_logical_map = new_physical_to_logical_map.tolist()

    def _handle_dst_expert_location(dst_expert_location: int):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            return

        # case 2: same-gpu
        for src_expert_location in range(
            rank * num_local_physical_experts,
            (rank + 1) * num_local_physical_experts,
        ):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                TODO
                break
        TODO_early_return

    def _entrypoint():
        for dst_expert_location in range(
            rank * num_local_physical_experts,
            (rank + 1) * num_local_physical_experts,
        ):
            _handle_dst_expert_location(dst_expert_location=dst_expert_location)
            TODO

        for src_expert_location in range(
            rank * num_local_physical_experts,
            (rank + 1) * num_local_physical_experts,
        ):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]
            TODO

        reqs = torch.distributed.batch_isend_irecv(TODO)
        for req in reqs:
            req.wait()

        for copy_back_info in TODO:
            TODO

    _entrypoint()
