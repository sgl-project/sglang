import logging
from typing import Dict, List, Tuple

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

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        # List[Tuple[src_temp_buffers_expert_location, dst_routed_experts_weights_expert_location]]
        copy_back_infos: List[Tuple[int, int]] = []

        for dst_expert_location in range(*local_expert_location_range):
            _handle_dst_expert_location(dst_expert_location, copy_back_infos)
            TODO

        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]
            TODO

        reqs = torch.distributed.batch_isend_irecv(TODO)
        for req in reqs:
            req.wait()

        for copy_back_info in TODO:
            TODO

    def _handle_dst_expert_location(dst_expert_location: int, copy_back_infos):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(len(routed_experts_weights)):
                    temp_buffers[i][to_local(dst_expert_location)].copy_(
                        routed_experts_weights[i][to_local(src_expert_location)])
                copy_back_infos.append((dst_expert_location, dst_expert_location))
                return

        # case 3: free-rider
        for src_expert_location in range(rank * num_local_physical_experts, dst_expert_location):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                copy_back_infos.append((src_expert_location, dst_expert_location))
                return

        # case 4: same-node
        TODO

        # case 5: cross-node
        TODO

    _entrypoint()
