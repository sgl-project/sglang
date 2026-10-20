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
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import msgspec
import torch
import torch.distributed

from sglang.srt.eplb.expert_location import EplbPlacement, ExpertLocationMetadata
from sglang.srt.eplb.expert_location_updater import (
    LayerUpdatePlan,
    create_temp_buffers,
    execute_buffer2weight_copies,
    execute_p2p_ops,
    execute_weight2buffer_copies,
    plan_expert_weights_single_layer,
)
from sglang.srt.runtime_context import get_device, get_parallel

logger = logging.getLogger(__name__)


class AsyncRebalanceJob(msgspec.Struct):
    layer_ids: List[int]
    old_metadata: ExpertLocationMetadata
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]]
    compute_placement: Callable[[torch.Tensor], Optional[EplbPlacement]]
    logical_count_host: torch.Tensor
    count_event: torch.Event
    time_start: float
    logical_count_device: Optional[torch.Tensor] = None
    thread: Optional[threading.Thread] = None
    error: Optional[BaseException] = None
    placement: Optional[EplbPlacement] = None
    plans: Dict[int, LayerUpdatePlan] = {}
    new_metadata: Optional[ExpertLocationMetadata] = None
    transfer_event: Optional[torch.Event] = None
    layers_issued: int = 0
    layers_installed: int = 0


class AsyncExpertMigrator:
    def __init__(self):
        self._device = get_device().device
        self._device_module = torch.get_device_module(self._device)
        self._stream = self._device_module.Stream()
        self._group = torch.distributed.new_group(
            backend=torch.distributed.get_backend(),
            group_desc="eplb_async_migrator",
        )
        self._temp_buffers: Optional[List[torch.Tensor]] = None
        self._install_event = self._device_module.Event()
        self._install_event.record()
        self._warm_up()

    def _warm_up(self):
        world_size = torch.distributed.get_world_size(self._group)
        rank = torch.distributed.get_rank(self._group)
        with self._device_module.stream(self._stream):
            probe = torch.ones(1, device=self._device)
            torch.distributed.all_reduce(probe, group=self._group)
            ops = []
            for peer in range(world_size):
                if peer == rank:
                    continue
                ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        torch.ones(1, device=self._device),
                        peer,
                        group=self._group,
                    )
                )
                ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        torch.empty(1, device=self._device),
                        peer,
                        group=self._group,
                    )
                )
            if ops:
                for req in torch.distributed.batch_isend_irecv(ops):
                    req.wait()
        self._device_module.synchronize()

    def start(
        self,
        *,
        logical_count: torch.Tensor,
        old_metadata: ExpertLocationMetadata,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        compute_placement: Callable[[torch.Tensor], Optional[EplbPlacement]],
    ) -> AsyncRebalanceJob:
        layer_ids = sorted(routed_experts_weights_of_layer.keys())
        if self._temp_buffers is None:
            self._temp_buffers = create_temp_buffers(
                routed_experts_weights_of_layer[layer_ids[0]]
            )

        logical_count_host = torch.empty(
            logical_count.shape,
            dtype=logical_count.dtype,
            device="cpu",
            pin_memory=torch.device(self._device).type != "cpu",
        )
        count_event = self._device_module.Event()
        with self._device_module.stream(self._stream):
            self._stream.wait_stream(self._device_module.current_stream())
            logical_count_host.copy_(logical_count, non_blocking=True)
            count_event.record(self._stream)

        return AsyncRebalanceJob(
            layer_ids=layer_ids,
            old_metadata=old_metadata,
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            compute_placement=compute_placement,
            logical_count_host=logical_count_host,
            count_event=count_event,
            time_start=time.time(),
            logical_count_device=logical_count,
        )

    def poll_placement(self, job: AsyncRebalanceJob) -> bool:
        if job.thread is None:
            if not job.count_event.query():
                return False
            job.logical_count_device = None
            job.thread = threading.Thread(
                target=self._run_planning,
                args=(job,),
                name="eplb-async-planning",
                daemon=True,
            )
            job.thread.start()
            return False
        if job.thread.is_alive():
            return False
        if job.error is not None:
            raise RuntimeError("EPLB async placement failed") from job.error
        return True

    def _run_planning(self, job: AsyncRebalanceJob):
        try:
            placement = job.compute_placement(job.logical_count_host)
            if placement is None:
                raise RuntimeError("EPLB produced no placement for this model")
            job.placement = placement
            new_physical_to_logical_map_cpu = placement.physical_to_logical_map.cpu()
            world_size = torch.distributed.get_world_size(self._group)
            rank = get_parallel().tp_rank
            num_gpu_per_node = world_size // get_parallel().nnodes
            for layer_id in job.layer_ids:
                job.plans[layer_id] = plan_expert_weights_single_layer(
                    routed_experts_weights=job.routed_experts_weights_of_layer[
                        layer_id
                    ],
                    temp_buffers=self._temp_buffers,
                    old_physical_to_logical_map=job.old_metadata.physical_to_logical_map_cpu[
                        layer_id
                    ].tolist(),
                    new_physical_to_logical_map=new_physical_to_logical_map_cpu[
                        layer_id
                    ].tolist(),
                    num_local_physical_experts=job.old_metadata.num_local_physical_experts,
                    num_gpu_per_node=num_gpu_per_node,
                    rank=rank,
                    group=self._group,
                )
        except BaseException as e:
            job.error = e

    def build_metadata(self, job: AsyncRebalanceJob) -> ExpertLocationMetadata:
        assert job.placement is not None
        job.new_metadata = ExpertLocationMetadata.init_by_placement(
            job.placement, ep_size=job.old_metadata.ep_size
        )
        return job.new_metadata

    def issue_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = job.plans[layer_id]
        weights = job.routed_experts_weights_of_layer[layer_id]
        transfer_event = self._device_module.Event()
        with self._device_module.stream(self._stream):
            self._stream.wait_event(self._install_event)
            execute_weight2buffer_copies(plan, weights, self._temp_buffers)
            execute_p2p_ops(plan)
            transfer_event.record(self._stream)
        job.transfer_event = transfer_event
        job.layers_issued += 1

    def install_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = job.plans[layer_id]
        weights = job.routed_experts_weights_of_layer[layer_id]
        assert job.new_metadata is not None and job.transfer_event is not None
        current_stream = self._device_module.current_stream()
        current_stream.wait_event(job.transfer_event)
        execute_buffer2weight_copies(plan, weights, self._temp_buffers)
        job.old_metadata.update(job.new_metadata, update_layer_ids=[layer_id])
        self._install_event.record(current_stream)
        job.layers_installed += 1

    def cancel(self, job: Optional[AsyncRebalanceJob]):
        if job is not None and job.thread is not None and job.thread.is_alive():
            job.thread.join()
        self._device_module.synchronize()
