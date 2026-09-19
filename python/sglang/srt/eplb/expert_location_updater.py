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
import atexit
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import einops
import torch
import torch.distributed
from torch.distributed import P2POp

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.cpp_async_runtime import (
    create_eplb_async_runtime,
)
from sglang.srt.eplb.cpp_async_runtime import create_layer_plan as create_cpp_layer_plan
from sglang.srt.eplb.cpp_async_runtime import (
    create_metadata_field_pair,
)
from sglang.srt.eplb.cpp_async_runtime import (
    create_prepared_plan as create_cpp_prepared_plan,
)
from sglang.srt.eplb.cpp_async_runtime import (
    create_reset_tensor_spec,
)
from sglang.srt.eplb.eplb_async_host_mirror import (
    get_global_eplb_async_host_mirror_manager,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.runtime_context import get_device
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


_NCCL_DTYPE_ALIAS = {
    torch.uint16: torch.bfloat16,
}


def _comm_view(t: torch.Tensor) -> torch.Tensor:
    """Return a NCCL-friendly view of `t` for P2P send/recv. If the dtype is
    in `_NCCL_DTYPE_ALIAS`, reinterpret as the aliased signed integer type;
    otherwise return `t` unchanged. The underlying storage is shared, so
    after irecv the original-dtype tensor reads the correct bytes."""
    alias = _NCCL_DTYPE_ALIAS.get(t.dtype)
    if alias is None:
        return t
    return t.view(alias)


_LOG_INPUT = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT")
_LOG_ASYNC_SYNC_DEBUG = get_bool_env_var("SGLANG_EPLB_ASYNC_SYNC_DEBUG")
_GLOBAL_EXPERT_LOCATION_UPDATER = None


def get_global_expert_location_updater():
    return _GLOBAL_EXPERT_LOCATION_UPDATER


def set_global_expert_location_updater(updater):
    global _GLOBAL_EXPERT_LOCATION_UPDATER
    _GLOBAL_EXPERT_LOCATION_UPDATER = updater


@dataclass
class _AsyncLayerUpdatePlan:
    layer_id: int
    routed_experts_weights: List[torch.Tensor]
    copy_pairs: List[Tuple[int, int]]


class ExpertLocationUpdater:
    def __init__(self, enable_async: bool = False):
        self._first_execution = True
        self._enable_async = enable_async
        self._prepared_async_layer_ids: Set[int] = set()
        self._current_step = 0
        self._registered_atexit = False
        self._runtime = None
        self.async_eplb_stream = torch.cuda.Stream()
        self.layer_id = None
        self._skip_current_layer = False
        if self._enable_async and torch.cuda.is_available():
            self._runtime = create_eplb_async_runtime(torch.cuda.current_device())
            self._maybe_register_atexit()

    def _maybe_register_atexit(self):
        if self._registered_atexit:
            return
        self._registered_atexit = True
        atexit.register(self.close)

    def close(self):
        if self._runtime is None:
            return
        self._runtime.shutdown()
        self._runtime = None

    def _disable_async_due_to_missing_layers(self):
        logger.warning_once(
            "EPLB async is disabled because the model does not expose "
            "routed_experts_weights_of_layer. Expert location updates will "
            "fall back to no-op behavior for this model."
        )
        self.close()
        self._enable_async = False

    # Register layers with the C++ async runtime.
    def prepare_async_layers(
        self, routed_experts_weights_of_layer: Optional[Dict[int, List[torch.Tensor]]]
    ):
        if not self._enable_async or not torch.cuda.is_available():
            return
        if not routed_experts_weights_of_layer:
            self._disable_async_due_to_missing_layers()
            return

        for layer_id in sorted(routed_experts_weights_of_layer):
            if layer_id in self._prepared_async_layer_ids:
                continue
            assert self._runtime is not None
            self._runtime.register_layer(layer_id)
            self._prepared_async_layer_ids.add(layer_id)

        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] prepared_async_layers layer_ids=%s",
                sorted(self._prepared_async_layer_ids),
            )

    # give owner to GPU before forward
    def on_capture_start(self):
        if not self._enable_async or not torch.cuda.is_available():
            return

        target_step = self._current_step + 1
        assert self._runtime is not None
        self._runtime.prepare_capture_step(target_step)

        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] capture_prepare step=%s prepared_layers=%s",
                target_step,
                sorted(self._prepared_async_layer_ids),
            )

    # Transfer iteration info to eplb_async_runtime
    def on_forward_pass_start(self):
        self._current_step += 1
        if not self._enable_async or not torch.cuda.is_available():
            return

        target_step = self._current_step
        assert self._runtime is not None
        self._runtime.start_iter(
            target_step,
            get_global_expert_distribution_recorder().recording,
        )
        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info("[EPLBAsyncSync] start_iter step=%s", target_step)

    def set_current_layer_id(self, layer_id: int, is_nextn: bool = False):
        self.layer_id = layer_id
        self._skip_current_layer = is_nextn

    def wait_gpu_stage(self):
        if not self._enable_async or not torch.cuda.is_available():
            return
        if self._skip_current_layer:
            return
        self._ensure_async_layer_prepared(self.layer_id)
        assert self._runtime is not None
        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] layer_wait_gpu_before layer_id=%s", self.layer_id
            )
        # Launch signal kernel directly on the current stream.
        # The spin-wait kernel MUST complete before subsequent operations
        # (e.g. TopK) on the same stream, ensuring correct ordering.
        self._runtime.wait_gpu_stage(self.layer_id)
        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] layer_wait_gpu_after layer_id=%s", self.layer_id
            )

    def set_cpu_stage(self):
        if not self._enable_async or not torch.cuda.is_available():
            return
        if self._skip_current_layer:
            return
        self._ensure_async_layer_prepared(self.layer_id)
        assert self._runtime is not None
        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] layer_set_cpu_before layer_id=%s", self.layer_id
            )
        # SGLANG_EPLB_KERNEL_OVERLAP_WITH_COMBINE=True: dispatch set_cpu_stage
        # to async_eplb_stream so it can overlap with subsequent operations.
        # SGLANG_EPLB_KERNEL_OVERLAP_WITH_COMBINE=False (default): launch
        # directly on the current CUDA stream, avoiding extra inter-stream sync
        # but executing sequentially.
        if envs.SGLANG_EPLB_KERNEL_OVERLAP_WITH_COMBINE.get():
            current_stream = torch.cuda.current_stream()
            self.async_eplb_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.async_eplb_stream):
                self._runtime.set_cpu_stage(self.layer_id)
            current_stream.wait_stream(self.async_eplb_stream)
        else:
            self._runtime.set_cpu_stage(self.layer_id)
        if _LOG_ASYNC_SYNC_DEBUG:
            logger.info(
                "[EPLBAsyncSync] layer_set_cpu_after layer_id=%s", self.layer_id
            )

    def _ensure_async_layer_prepared(self, layer_id: int):
        if layer_id not in self._prepared_async_layer_ids:
            raise RuntimeError(
                f"Async EPLB layer is not prepared for layer_id={layer_id}."
            )

    def _get_host_expert_tensors(self, layer_id: int, logical_expert_id: int):
        host_mirror = get_global_eplb_async_host_mirror_manager()
        if host_mirror is None:
            raise RuntimeError("EPLB async host mirror manager is not initialized.")
        return host_mirror.get_expert_tensors(layer_id, logical_expert_id)

    def update(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
        nnodes: int,
        rank: int,
        use_flat_topology: bool = False,
    ):
        """
        Update experts' physical location after EPLB.

        Returns a map of layer_id to expert_ids that are missing due to rank
        failures during fault conditions when elastic EP is enabled.
        """
        if self._first_execution:
            self._first_execution = False
            torch.get_device_module().empty_cache()

        old_expert_location_metadata = get_global_expert_location_metadata()
        assert old_expert_location_metadata is not None
        topology_num_nodes = 1 if use_flat_topology else nnodes

        missing_logical_experts_by_layers = _update_expert_weights(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=topology_num_nodes,
            rank=rank,
            enable_async=self._enable_async,
        )
        if not self._enable_async:
            old_expert_location_metadata.update(
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
        return missing_logical_experts_by_layers

    def submit_prepared_async_plan(
        self,
        *,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
        layer_plans: Dict[int, _AsyncLayerUpdatePlan],
    ):
        if not self._enable_async or not torch.cuda.is_available():
            return
        assert self._runtime is not None
        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        if _LOG_ASYNC_SYNC_DEBUG:
            copy_pairs_per_layer = {
                layer_id: len(layer_plans[layer_id].copy_pairs)
                for layer_id in update_layer_ids
            }
            metadata_equal_per_layer = {
                layer_id: bool(
                    torch.equal(
                        current_metadata.physical_to_logical_map_cpu[layer_id],
                        new_expert_location_metadata.physical_to_logical_map_cpu[
                            layer_id
                        ],
                    )
                )
                for layer_id in update_layer_ids
            }
            logger.info(
                "[EPLBAsyncSync] submit_async_plan update_layers=%s layer_plan_keys=%s copy_pairs_per_layer=%s metadata_equal_per_layer=%s",
                list(update_layer_ids),
                sorted(layer_plans.keys()),
                copy_pairs_per_layer,
                metadata_equal_per_layer,
            )
        cpp_layer_plans = []
        for layer_id in update_layer_ids:
            if layer_id not in layer_plans:
                raise RuntimeError(
                    "Async EPLB prepared plan is missing layer "
                    f"{layer_id}. update_layer_ids={update_layer_ids}, "
                    f"layer_plan_keys={sorted(layer_plans.keys())}"
                )
            plan = layer_plans[layer_id]
            host_expert_tensors_per_copy = [
                list(self._get_host_expert_tensors(layer_id, logical_expert_id))
                for logical_expert_id, _ in plan.copy_pairs
            ]
            cpp_layer_plans.append(
                create_cpp_layer_plan(
                    layer_id=layer_id,
                    routed_experts_weights=plan.routed_experts_weights,
                    host_expert_tensors_per_copy=host_expert_tensors_per_copy,
                    dst_slots=[dst_slot for _, dst_slot in plan.copy_pairs],
                )
            )
        gpu_metadata_fields = []
        cpu_metadata_fields = []
        for current_field, next_field in current_metadata.iter_metadata_field_pairs(
            new_expert_location_metadata
        ):
            pair = create_metadata_field_pair(current_field, next_field)
            if current_field.is_cuda:
                gpu_metadata_fields.append(pair)
            else:
                cpu_metadata_fields.append(pair)

        gpu_reset_tensors = []
        cpu_reset_tensors = []
        for (
            tensor,
            layer_dim,
        ) in get_global_expert_distribution_recorder().get_async_runtime_reset_tensor_specs():
            spec = create_reset_tensor_spec(tensor, layer_dim)
            if tensor.is_cuda:
                gpu_reset_tensors.append(spec)
            else:
                cpu_reset_tensors.append(spec)

        self._runtime.submit_plan(
            create_cpp_prepared_plan(
                update_layer_ids=list(update_layer_ids),
                layer_plans=cpp_layer_plans,
                gpu_metadata_fields=gpu_metadata_fields,
                cpu_metadata_fields=cpu_metadata_fields,
                gpu_reset_tensors=gpu_reset_tensors,
                cpu_reset_tensors=cpu_reset_tensors,
            )
        )


def _update_expert_weights(**kwargs):
    enable_async = kwargs.pop("enable_async", False)
    if get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_CANARY"):
        if enable_async:
            assert False, (
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY is not compatible with EPLB Async"
            )
        return _update_expert_weights_with_canary(**kwargs)
    elif enable_async:
        return _update_expert_weights_raw_async(**kwargs)
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
            .to(device=get_device().device, non_blocking=True)
        )
        routed_experts_weights_of_layer[layer_id].append(canary_tensor)

    missing_logical_experts_by_layers = _update_expert_weights_raw(
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

    return missing_logical_experts_by_layers


def _update_expert_weights_raw_async(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    rank: int,
):
    """Async version of expert weights update using EPLB async runtime."""
    log_metrics = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS")
    world_size = torch.distributed.get_world_size()
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts
    num_gpu_per_node = world_size // nnodes
    host_mirror = get_global_eplb_async_host_mirror_manager()
    if host_mirror is None:
        raise RuntimeError("EPLB async host mirror manager is not initialized.")

    layer_plans: Dict[int, _AsyncLayerUpdatePlan] = {}
    for layer_id in update_layer_ids:
        common_kwargs = dict(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
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
        layer_plans[layer_id] = _AsyncLayerUpdatePlan(
            layer_id=layer_id,
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            copy_pairs=build_async_h2d_copy_plan_single_layer(**common_kwargs),
        )

    updater = get_global_expert_location_updater()
    if updater is None:
        raise RuntimeError("Global expert location updater is not initialized.")
    updater.submit_prepared_async_plan(
        new_expert_location_metadata=new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        layer_plans=layer_plans,
    )


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

    missing_logical_experts_by_layers: Dict[int, List[int]] = {}

    for layer_id in update_layer_ids:
        missing_logical_experts_info: List[int] = []
        update_expert_weights_single_layer(
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
            missing_logical_experts_info=missing_logical_experts_info,
            log_metrics=log_metrics,
        )
        if len(missing_logical_experts_info) > 0:
            missing_logical_experts_by_layers[layer_id] = missing_logical_experts_info
    return missing_logical_experts_by_layers


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
    missing_logical_experts_info: Optional[List[int]] = None,
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

    def _entrypoint():
        # List[Tuple[logical_expert_id, List[P2POp]]]
        p2p_op_infos: List[Tuple[int, List[P2POp]]] = []
        # List[Tuple[temp_buffers_expert_location, routed_experts_weights_expert_location]]
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        _create_isend_ops(p2p_op_infos)
        _filter_p2p_ops(p2p_op_infos)
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
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    P2POp(
                        op=torch.distributed.irecv,
                        tensor=_comm_view(
                            _get_tensor(temp_buffers, i, dst_expert_location)
                        ),
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
                        tensor=_comm_view(
                            _get_tensor(routed_experts_weights, i, src_expert_location)
                        ),
                        peer=dst_rank,
                    )
                    for dst_rank in all_dst_ranks
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

    def _filter_p2p_ops(p2p_op_infos):
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and missing_logical_experts_info is not None:
            # Filter out inactive P2P ops and record missing expert IDs in missing_logical_experts_info
            is_active = elastic_ep_state.active_ranks_cpu
            for i, (logical_expert_id, ops) in enumerate(p2p_op_infos):
                has_isend = any(op.op == torch.distributed.isend for op in ops)
                has_irecv = any(op.op == torch.distributed.irecv for op in ops)
                assert not (has_isend and has_irecv), (
                    "Each p2p_op_infos entry is expected to contain only send "
                    "or only recv ops."
                )

                if has_isend:
                    p2p_op_infos[i] = (
                        logical_expert_id,
                        [op for op in ops if is_active[op.peer]],
                    )
                elif has_irecv:
                    if any(not is_active[op.peer] for op in ops):
                        missing_logical_experts_info.append(logical_expert_id)
                        p2p_op_infos[i] = (logical_expert_id, [])

    def _execute_p2p_ops(p2p_op_infos):
        sorted_infos = sorted(p2p_op_infos, key=lambda info: info[0])
        p2p_ops = [op for _, ops in sorted_infos for op in ops]
        if len(p2p_ops) == 0:
            return

        # Submit P2P ops in batches to prevent NCCL/RCCL GPU-side accumulation
        # hangs on large rebalances. All ranks use the same expert_id ranges
        # (based on num_physical_experts) so matching send/recv pairs land in
        # the same batch. Set batch_chunk_size >= num_physical_experts to disable.
        batch_chunk_size = envs.SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE.get()
        ops_by_expert = {eid: ops for eid, ops in sorted_infos}
        for start in range(0, num_physical_experts, batch_chunk_size):
            batch_ops = []
            for eid in range(
                start, min(start + batch_chunk_size, num_physical_experts)
            ):
                if eid in ops_by_expert:
                    batch_ops.extend(ops_by_expert[eid])
            if batch_ops:
                reqs = torch.distributed.batch_isend_irecv(batch_ops)
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

    return output_logs


def build_async_h2d_copy_plan_single_layer(
    routed_experts_weights: List[torch.Tensor],
    old_physical_to_logical_map: List[int],
    new_physical_to_logical_map: List[int],
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: Optional[int] = None,
    debug: bool = False,
    log_metrics: bool = False,
) -> List[Tuple[int, int]]:
    del routed_experts_weights, num_gpu_per_node, world_size, debug, log_metrics
    copy_pairs = []
    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )
    for dst_expert_location in range(*local_expert_location_range):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            continue
        dst_slot = dst_expert_location % num_local_physical_experts
        copy_pairs.append((logical_expert_id, dst_slot))
    return copy_pairs


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
