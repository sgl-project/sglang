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
    old_physical_to_logical_map: torch.Tensor,  # (num_physical_Experts,)
    new_physical_to_logical_map: torch.Tensor,  # (num_physical_Experts,)
    num_local_physical_experts: int,
    rank: int,
):
    assert all(tensor.shape[0] == num_local_physical_experts for tensor in routed_experts_weights)

    num_physical_experts, = old_physical_to_logical_map.shape
    num_gpu_per_node = TODO

    self_node_id = rank // num_gpu_per_node

    old_physical_to_logical_map = old_physical_to_logical_map.tolist()
    new_physical_to_logical_map = new_physical_to_logical_map.tolist()

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        # List[Tuple[src_temp_buffers_expert_location, dst_routed_experts_weights_expert_location]]
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        for dst_expert_location in range(*local_expert_location_range):
            _handle_dst_expert_location(dst_expert_location, buffer2weight_copy_infos)
            TODO

        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]
            TODO

        reqs = torch.distributed.batch_isend_irecv(TODO)
        for req in reqs:
            req.wait()

        for copy_back_info in TODO:
            TODO

    def _handle_dst_expert_location(dst_expert_location: int, buffer2weight_copy_infos):
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
                buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))
                return

        # case 3: free-rider
        for src_expert_location in range(rank * num_local_physical_experts, dst_expert_location):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append((src_expert_location, dst_expert_location))
                return

        all_src_ranks = _deduplicate_ordered([
            x // num_local_physical_experts
            for x in range(num_physical_experts)
            if old_physical_to_logical_map[x] == logical_expert_id
        ])
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [x for x in all_src_ranks if x // num_gpu_per_node == self_node_id]

        need_p2p_dst_ranks = _deduplicate_ordered([
            x // num_local_physical_experts
            for x in range(num_physical_experts)
            if new_physical_to_logical_map[x] == logical_expert_id
            and x // num_local_physical_experts not in all_src_ranks
        ])

        # case 4: same-node
        TODO

        # case 5: cross-node
        TODO

    _entrypoint()


def _deduplicate_ordered(arr: List[int]):
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output
