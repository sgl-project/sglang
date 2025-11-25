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
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import einops
import torch
import torch.distributed
from torch.distributed import P2POp

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


_LOG_INPUT = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT")


class ExpertLocationUpdater:
    def __init__(self):
        self._first_execution = True

    def update(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
        nnodes: int,
        rank: int,
        model_runner: "ModelRunner",
    ):
        if self._first_execution:
            self._first_execution = False
            torch.get_device_module().empty_cache()

        old_expert_location_metadata = get_global_expert_location_metadata()
        assert old_expert_location_metadata is not None

        all_missing_logical_experts = _update_expert_weights(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=nnodes,
            rank=rank,
        )
        old_expert_location_metadata.update(
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
        )

        if len(all_missing_logical_experts) > 0:
            logger.info(
                f"[ExpertLocationUpdater] All missing logical experts: {all_missing_logical_experts}"
            )

            # Load the missing expert weights from disk
            if callable(
                getattr(model_runner.model, "generate_weight_name_filter", None)
            ):
                # Filter and load only missing expert weights
                weight_name_filter = model_runner.model.generate_weight_name_filter(
                    all_missing_logical_experts
                )
            else:
                # Do a full reload from disk
                logger.info(
                    "[ExpertLocationUpdater] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            model_runner.update_weights_from_disk(
                model_runner.server_args.model_path,
                model_runner.server_args.load_format,
                weight_name_filter=weight_name_filter,
            )


def _update_expert_weights(**kwargs):
    if get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_CANARY"):
        return _update_expert_weights_with_canary(**kwargs)
    else:
        return _update_expert_weights_raw(**kwargs)


# can add watchdog as well
def _update_expert_weights_with_canary(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
):
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts

    def _get_canary_value(meta: ExpertLocationMetadata, layer_id: int):
        return meta.physical_to_logical_map_cpu[
            layer_id,
            num_local_physical_experts * rank : num_local_physical_experts * (rank + 1),
        ]

    routed_experts_weights_of_layer = {
        k: [x for x in v] for k, v in routed_experts_weights_of_layer.items()
    }
    for layer_id in update_layer_ids:
        canary_tensor = (
            _get_canary_value(old_expert_location_metadata, layer_id)
            .clone()
            .to(device=get_global_server_args().device, non_blocking=True)
        )
        routed_experts_weights_of_layer[layer_id].append(canary_tensor)

    all_missing_logical_experts = _update_expert_weights_raw(
        routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        old_expert_location_metadata=old_expert_location_metadata,
        new_expert_location_metadata=new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=rank,
    )

    for layer_id in update_layer_ids:
        # can optimize speed if needed
        expect_value = _get_canary_value(new_expert_location_metadata, layer_id)
        actual_value = routed_experts_weights_of_layer[layer_id][-1].cpu()
        assert torch.all(expect_value == actual_value), (
            f"{expect_value=} {actual_value=} {layer_id=} "
            f"{old_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
            f"{new_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
        )

    return all_missing_logical_experts


def _update_expert_weights_raw(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
):
    log_metrics = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS")

    temp_buffers = create_temp_buffers(
        routed_experts_weights_of_layer[update_layer_ids[0]]
    )

    world_size = torch.distributed.get_world_size()
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts
    num_gpu_per_node = world_size // nnodes

    # Due to rank failures, some logical experts could not be transferred P2P between GPUs.
    # The ModelRunner will reload these experts from disk.
    all_missing_logical_experts: List[Tuple[int, List[int]]] = []

    for layer_id in update_layer_ids:
        _, missing_logical_experts = update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            new_physical_to_logical_map=new_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=rank,
            world_size=world_size,
            log_metrics=log_metrics,
        )
        if len(missing_logical_experts) > 0:
            all_missing_logical_experts.append((layer_id, missing_logical_experts))
    return all_missing_logical_experts


def create_temp_buffers(sample_tensors):
    return [torch.empty_like(tensor) for tensor in sample_tensors]


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],
    temp_buffers: List[torch.Tensor],
    old_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    new_physical_to_logical_map: List[int],  # (num_physical_Experts,)
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: Optional[int] = None,
    debug: bool = False,
    log_metrics: bool = False,
):
    assert all(
        tensor.shape[0] == num_local_physical_experts
        for tensor in routed_experts_weights
    ), f"{num_local_physical_experts=} {[x.shape for x in routed_experts_weights]=}"
    assert isinstance(old_physical_to_logical_map, list)
    assert isinstance(new_physical_to_logical_map, list)

    if _LOG_INPUT:
        logger.info(
            "update_expert_weights_single_layer "
            f"{[x.shape for x in routed_experts_weights]=} "
            f"{[x.shape for x in temp_buffers]=} "
            f"{old_physical_to_logical_map=} "
            f"{new_physical_to_logical_map=} "
            f"{num_local_physical_experts=} "
            f"{num_gpu_per_node=} "
            f"{rank=} "
            f"{world_size=} "
        )

    output_logs = [] if debug else None

    num_physical_experts = len(old_physical_to_logical_map)
    num_tensors = len(routed_experts_weights)

    self_node_id = rank // num_gpu_per_node

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    elastic_ep_state = ElasticEPStateManager.instance()
    if elastic_ep_state is None:
        active_ranks = [1] * torch.distributed.get_world_size()
    else:
        active_ranks = elastic_ep_state.active_ranks_cpu

    missing_logical_experts: List[int] = []

    def _entrypoint():
        # List[Tuple[logical_expert_id, List[P2POp]]]
        p2p_op_infos: List[Tuple[int, List[P2POp]]] = []
        # List[Tuple[temp_buffers_expert_location, routed_experts_weights_expert_location]]
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        _create_isend_ops(p2p_op_infos)
        _execute_p2p_ops(p2p_op_infos)
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

        if log_metrics:
            _log_p2p_op_metrics(
                p2p_op_infos,
                world_size=world_size,
                num_gpu_per_node=num_gpu_per_node,
                self_node_id=self_node_id,
            )

        if debug:
            output_logs.append(f"{p2p_op_infos=}")
            output_logs.append(f"{buffer2weight_copy_infos=}")

    def _handle_recv(buffer2weight_copy_infos, p2p_op_infos):
        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_of_dst_expert_location(
                dst_expert_location, buffer2weight_copy_infos, p2p_op_infos
            )

    def _handle_recv_of_dst_expert_location(
        dst_expert_location: int, buffer2weight_copy_infos, p2p_op_infos
    ):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=unchanged"
                )
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    _get_tensor(temp_buffers, i, dst_expert_location).copy_(
                        _get_tensor(routed_experts_weights, i, src_expert_location)
                    )
                buffer2weight_copy_infos.append(
                    (dst_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-gpu {src_expert_location=}"
                    )
                return

        # case 3: free-rider
        for src_expert_location in range(
            rank * num_local_physical_experts, dst_expert_location
        ):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append(
                    (src_expert_location, dst_expert_location)
                )
                if debug:
                    output_logs.append(
                        f"handle_recv_of_dst_expert_location {dst_expert_location=} case=free-rider {src_expert_location=}"
                    )
                return

        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        # case 4: same-node
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(
                element_value=rank
            )
            _create_p2p_recv_and_buffer2weight_copy(
                buffer2weight_copy_infos,
                p2p_op_infos,
                src_rank=chosen_src_rank,
                logical_expert_id=logical_expert_id,
                dst_expert_location=dst_expert_location,
            )
            if debug:
                output_logs.append(
                    f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-node {chosen_src_rank=}"
                )
            return

        # case 5: cross-node
        # Future work: can optimize when there are multiple ranks in the same dst node that uses the same logical expert
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(
            element_value=rank
        )
        _create_p2p_recv_and_buffer2weight_copy(
            buffer2weight_copy_infos,
            p2p_op_infos,
            src_rank=chosen_src_rank,
            logical_expert_id=logical_expert_id,
            dst_expert_location=dst_expert_location,
        )
        if debug:
            output_logs.append(
                f"handle_recv_of_dst_expert_location {dst_expert_location=} case=cross-node {chosen_src_rank=}"
            )
        return

    def _create_p2p_recv_and_buffer2weight_copy(
        buffer2weight_copy_infos,
        p2p_op_infos,
        *,
        logical_expert_id: int,
        src_rank: int,
        dst_expert_location: int,
    ):
        if not active_ranks[src_rank]:
            # The logical expert cannot be loaded from peers
            missing_logical_experts.append(logical_expert_id)
            return
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.irecv,
                        tensor=_get_tensor(temp_buffers, i, dst_expert_location),
                        peer=src_rank,
                    )
                    for i in range(num_tensors)
                ],
            )
        )
        buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))

    def _create_isend_ops(p2p_op_infos):
        handled_logical_expert_ids = set()
        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]

            if logical_expert_id in handled_logical_expert_ids:
                continue
            handled_logical_expert_ids.add(logical_expert_id)

            _create_isend_ops_of_logical_expert_id(
                logical_expert_id, src_expert_location, p2p_op_infos
            )

    def _create_isend_ops_of_logical_expert_id(
        logical_expert_id, src_expert_location, p2p_op_infos
    ):
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        same_node_dst_ranks = same_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        cross_node_dst_ranks = cross_node_mapping.element_values_from_chunk_value(
            chunk_value=rank
        )
        all_dst_ranks = same_node_dst_ranks + cross_node_dst_ranks

        if debug:
            output_logs.append(
                f"create_isend_ops_of_logical_expert_id {logical_expert_id=} {src_expert_location=} {same_node_dst_ranks=} {cross_node_dst_ranks=}"
            )

        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.isend,
                        tensor=_get_tensor(
                            routed_experts_weights, i, src_expert_location
                        ),
                        peer=dst_rank,
                    )
                    for dst_rank in all_dst_ranks
                    if active_ranks[dst_rank]
                    for i in range(num_tensors)
                ],
            )
        )

    def _compute_comm_info(logical_expert_id: int):
        all_src_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if old_physical_to_logical_map[x] == logical_expert_id
            ]
        )
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [
            x for x in all_src_ranks if x // num_gpu_per_node == self_node_id
        ]

        need_comm_dst_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if new_physical_to_logical_map[x] == logical_expert_id
                and x // num_local_physical_experts not in all_src_ranks
            ]
        )
        need_comm_self_node_dst_ranks = (
            [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id]
            if len(self_node_src_ranks) > 0
            else []
        )
        need_comm_cross_node_dst_ranks = [
            x
            for x in need_comm_dst_ranks
            if (x // num_gpu_per_node) not in all_src_nodes
        ]

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
        p2p_ops = [op for _, ops in sorted_infos for op in ops]
        if len(p2p_ops) == 0:
            return

        reqs = torch.distributed.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        for (
            temp_buffers_expert_location,
            routed_experts_weights_expert_location,
        ) in buffer2weight_copy_infos:
            for i in range(num_tensors):
                _get_tensor(
                    routed_experts_weights, i, routed_experts_weights_expert_location
                ).copy_(_get_tensor(temp_buffers, i, temp_buffers_expert_location))

    def _get_tensor(tensors, tensor_index: int, expert_location: int) -> torch.Tensor:
        return tensors[tensor_index][_get_local_expert_location(expert_location)]

    def _get_local_expert_location(expert_location: int) -> int:
        assert (
            local_expert_location_range[0]
            <= expert_location
            < local_expert_location_range[1]
        )
        return expert_location % num_local_physical_experts

    _entrypoint()

    return output_logs, missing_logical_experts


class _ChunkUtils:
    def __init__(self, *, chunk_values: List, element_values: List):
        self.chunk_values = chunk_values
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        chunk_index = self._chunk_index_from_element_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            element_index=self.element_values.index(element_value),
        )
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value) -> List:
        if len(self.element_values) == 0:
            return []
        element_slice = self._element_slice_from_chunk_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            chunk_index=self.chunk_values.index(chunk_value),
        )
        return self.element_values[element_slice]

    @staticmethod
    def _chunk_index_from_element_index(
        num_elements: int, num_chunks: int, element_index: int
    ) -> int:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        num_elements_for_long_chunks = num_long_chunks * (short_chunk_size + 1)
        if element_index < num_elements_for_long_chunks:
            return element_index // (short_chunk_size + 1)
        else:
            return (
                num_long_chunks
                + (element_index - num_elements_for_long_chunks) // short_chunk_size
            )

    @staticmethod
    def _element_slice_from_chunk_index(
        num_elements: int, num_chunks: int, chunk_index: int
    ) -> slice:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        start = chunk_index * short_chunk_size + min(chunk_index, num_long_chunks)
        end = start + short_chunk_size + int(chunk_index < num_long_chunks)
        return slice(start, end)


def _deduplicate_ordered(arr: List[int]):
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output


def _log_p2p_op_metrics(
    p2p_op_infos: List[Tuple[int, List[P2POp]]],
    num_gpu_per_node: int,
    world_size: int,
    self_node_id: int,
):
    text = ""
    all_ops = [op for _, ops in p2p_op_infos for op in ops]

    for direction, ops in _group_by(all_ops, _get_direction_from_op).items():
        nbytes_of_gpu = [0] * world_size
        for op in ops:
            nbytes_of_gpu[op.peer] += op.tensor.nbytes
        nbytes_of_gpu = torch.tensor(nbytes_of_gpu, dtype=torch.int64)

        nbytes_of_node = einops.reduce(
            nbytes_of_gpu,
            "(num_nodes num_gpu_per_node) -> num_nodes",
            num_gpu_per_node=num_gpu_per_node,
            reduction="sum",
        )

        nbytes_curr_node = nbytes_of_node[self_node_id]
        nbytes_cross_node = torch.sum(nbytes_of_node) - nbytes_curr_node

        text += (
            f"{direction}_nbytes_of_gpu={nbytes_of_gpu.tolist()} "
            f"{direction}_nbytes_of_node={nbytes_of_node.tolist()} "
            f"{direction}_nbytes_curr_node={nbytes_curr_node.item()} "
            f"{direction}_nbytes_cross_node={nbytes_cross_node.item()} "
        )

    logger.info(f"[ExpertLocationUpdater] {text}")


def _get_direction_from_op(op: P2POp):
    if op.op == torch.distributed.isend:
        return "isend"
    if op.op == torch.distributed.irecv:
        return "irecv"
    raise NotImplementedError


def _group_by(items, keyfunc):
    ans = defaultdict(list)
    for item in items:
        ans[keyfunc(item)].append(item)
    return dict(ans)
