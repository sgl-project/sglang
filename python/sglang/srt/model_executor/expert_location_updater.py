import logging
from typing import Dict, List, Tuple

import torch
import torch.distributed
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata
from torch.distributed import P2POp

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
    num_tensors = len(routed_experts_weights)

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
        # List[Tuple[logical_expert_id, P2POp]]
        p2p_op_infos: List[Tuple[int, P2POp]] = []

        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_dst_expert_location(dst_expert_location, buffer2weight_copy_infos, p2p_op_infos)

        _create_isend_ops(p2p_op_infos)

        _execute_p2p_ops(p2p_op_infos)
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

        for copy_back_info in TODO:
            TODO

    def _handle_recv_dst_expert_location(dst_expert_location: int, buffer2weight_copy_infos, p2p_op_infos):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    temp_buffers[i][to_local(dst_expert_location)].copy_(
                        routed_experts_weights[i][to_local(src_expert_location)])
                buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))
                return

        # case 3: free-rider
        for src_expert_location in range(rank * num_local_physical_experts, dst_expert_location):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append((src_expert_location, dst_expert_location))
                return

        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = _compute_comm_info(
            logical_expert_id=logical_expert_id)

        # case 4: same-node
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(element_value=rank)
            for i in range(num_tensors):
                p2p_op_infos.append((TODO, TODO))
            buffer2weight_copy_infos.append((TODO, TODO))
            return

        # case 5: cross-node
        # Future work: can optimize when there are multiple ranks in the same dst node that uses the same logical expert
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(element_value=rank)
        for i in range(num_tensors):
            p2p_op_infos.append((TODO, TODO))
        buffer2weight_copy_infos.append((TODO, TODO))
        return

    def _create_isend_ops(p2p_op_infos):
        logical_expert_ids = sorted(set(
            old_physical_to_logical_map[src_expert_location]
            for src_expert_location in range(*local_expert_location_range)
        ))
        for logical_expert_id in logical_expert_ids:
            _create_isend_ops_of_logical_expert_id(logical_expert_id, p2p_op_infos)

    def _create_isend_ops_of_logical_expert_id(logical_expert_id, p2p_op_infos):
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = _compute_comm_info(
            logical_expert_id=logical_expert_id)

        # a. same-node
        chosen_dst_ranks = same_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        p2p_op_infos.append((TODO, TODO))

        # b. cross-node
        chosen_dst_ranks = cross_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        p2p_op_infos.append((TODO, TODO))

    def _compute_comm_info(logical_expert_id: int):
        all_src_ranks = _deduplicate_ordered([
            x // num_local_physical_experts
            for x in range(num_physical_experts) if old_physical_to_logical_map[x] == logical_expert_id
        ])
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [x for x in all_src_ranks if x // num_gpu_per_node == self_node_id]

        need_comm_dst_ranks = _deduplicate_ordered([
            x // num_local_physical_experts
            for x in range(num_physical_experts) if new_physical_to_logical_map[x] == logical_expert_id
            and x // num_local_physical_experts not in all_src_ranks
        ])
        need_comm_self_node_dst_ranks = [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id]
        need_comm_cross_node_dst_ranks = [x for x in need_comm_dst_ranks if
                                          (x // num_gpu_per_node) not in all_src_nodes]

        same_node_mapping = _ChunkUtils(
            chunk_values=self_node_src_ranks,
            element_values=need_comm_self_node_dst_ranks,
        )

        cross_node_mapping = _ChunkUtils(
            chunk_values=all_src_ranks,
            element_values=need_comm_cross_node_dst_ranks,
        )

        return same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks

    def _execute_p2p_ops(p2p_op_infos):
        sorted_infos = sorted(p2p_op_infos, key=lambda info: info[0])
        p2p_ops = [op for _, op in sorted_infos]
        reqs = torch.distributed.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        for src_expert_location, dst_expert_location in buffer2weight_copy_infos:
            for i in range(num_tensors):
                routed_experts_weights[i][to_local(dst_expert_location)].copy_(
                    temp_buffers[i][to_local(src_expert_location)])

    _entrypoint()


class _ChunkUtils:
    def __init__(self, *, chunk_values: List, element_values: List):
        self.chunk_values = chunk_values
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        return TODO

    def element_values_from_chunk_value(self, chunk_value) -> List:
        return TODO

    @staticmethod
    def _chunk_index_from_element_index(num_elements: int, num_chunks: int, element_index: int) -> int:
        return TODO

    @staticmethod
    def _element_slice_from_chunk_index(num_elements: int, num_chunks: int, chunk_index: int) -> slice:
        return TODO


def _deduplicate_ordered(arr: List[int]):
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output
