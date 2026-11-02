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
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import msgspec
import torch
import torch.distributed

from sglang.srt.eplb.expert_location import EplbPlacement, ExpertLocationMetadata
from sglang.srt.runtime_context import get_device, get_parallel

logger = logging.getLogger(__name__)


class LayerPullPlan(msgspec.Struct):
    weight2buffer_copies: List[Tuple[int, int]]
    buffer2weight_copies: List[Tuple[int, int]]
    reads: Dict[int, List[Tuple[int, int]]]


def plan_layer_pull(
    old_physical_to_logical_map: List[int],
    new_physical_to_logical_map: List[int],
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
) -> LayerPullPlan:
    lo = rank * num_local_physical_experts
    hi = lo + num_local_physical_experts
    node_of_rank = rank // num_gpu_per_node
    weight2buffer_copies: List[Tuple[int, int]] = []
    buffer2weight_copies: List[Tuple[int, int]] = []
    reads: Dict[int, List[Tuple[int, int]]] = {}

    for dst in range(lo, hi):
        logical_expert_id = new_physical_to_logical_map[dst]
        if old_physical_to_logical_map[dst] == logical_expert_id:
            continue

        src = next(
            (
                p
                for p in range(lo, hi)
                if old_physical_to_logical_map[p] == logical_expert_id
            ),
            None,
        )
        if src is not None:
            weight2buffer_copies.append((src - lo, dst - lo))
            buffer2weight_copies.append((dst - lo, dst - lo))
            continue

        src = next(
            (
                p
                for p in range(lo, dst)
                if new_physical_to_logical_map[p] == logical_expert_id
            ),
            None,
        )
        if src is not None:
            buffer2weight_copies.append((src - lo, dst - lo))
            continue

        holders = [
            p
            for p, e in enumerate(old_physical_to_logical_map)
            if e == logical_expert_id
        ]
        same_node = [
            p
            for p in holders
            if (p // num_local_physical_experts) // num_gpu_per_node == node_of_rank
        ]
        candidates = same_node or holders
        p = candidates[(dst - lo) % len(candidates)]
        reads.setdefault(p // num_local_physical_experts, []).append(
            (p % num_local_physical_experts, dst - lo)
        )
        buffer2weight_copies.append((dst - lo, dst - lo))

    return LayerPullPlan(
        weight2buffer_copies=weight2buffer_copies,
        buffer2weight_copies=buffer2weight_copies,
        reads=reads,
    )


class ExpertTransport:
    def prepare(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        staging: List[torch.Tensor],
    ) -> None:
        raise NotImplementedError

    def read_experts(
        self,
        src_rank: int,
        layer_id: int,
        slot_pairs: List[Tuple[int, int]],
        staging: List[torch.Tensor],
    ) -> None:
        raise NotImplementedError


class MooncakeExpertTransport(ExpertTransport):
    def __init__(self):
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        self._engine = get_mooncake_transfer_engine()
        if self._engine is None:
            raise RuntimeError(
                "--eplb-async-rebalance moves experts through the Mooncake "
                "transfer engine, which is not initialized in this process."
            )
        self._remote: List[Tuple[str, Dict[int, List[Tuple[int, int]]]]] = []

    def _register(self, tensor: torch.Tensor) -> None:
        assert tensor.is_contiguous()
        ret = self._engine.engine.register_memory(
            tensor.data_ptr(), tensor.numel() * tensor.element_size()
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake register_memory failed (ret={ret})")

    def prepare(self, routed_experts_weights_of_layer, staging):
        from sglang.srt.distributed.parallel_state import get_world_group

        local: Dict[int, List[Tuple[int, int]]] = {}
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            descs = []
            for tensor in tensors:
                self._register(tensor)
                descs.append(
                    (tensor.data_ptr(), tensor[0].numel() * tensor.element_size())
                )
            local[layer_id] = descs
        for tensor in staging:
            self._register(tensor)

        cpu_group = get_world_group().cpu_group
        gathered = [None] * torch.distributed.get_world_size(cpu_group)
        torch.distributed.all_gather_object(
            gathered, (self._engine.get_session_id(), local), group=cpu_group
        )
        self._remote = gathered

    def read_experts(self, src_rank, layer_id, slot_pairs, staging):
        session_id, descs_of_layer = self._remote[src_rank]
        descs = descs_of_layer[layer_id]
        local_ptrs, remote_ptrs, lengths = [], [], []
        for src_slot, dst_slot in slot_pairs:
            for tensor, (remote_base, nbytes) in zip(staging, descs, strict=True):
                local_ptrs.append(tensor.data_ptr() + dst_slot * nbytes)
                remote_ptrs.append(remote_base + src_slot * nbytes)
                lengths.append(nbytes)
        ret = self._engine.engine.batch_transfer_sync_read(
            session_id, local_ptrs, remote_ptrs, lengths
        )
        if ret != 0:
            raise RuntimeError(
                f"Mooncake batch_transfer_sync_read from rank {src_rank} "
                f"for layer {layer_id} failed (ret={ret})"
            )


class AsyncRebalanceJob(msgspec.Struct):
    layer_ids: List[int]
    old_metadata: ExpertLocationMetadata
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]]
    compute_placement: Callable[[torch.Tensor], Optional[EplbPlacement]]
    logical_count_host: torch.Tensor
    count_event: torch.Event
    time_start: float
    logical_count_device: Optional[torch.Tensor] = None
    placement_thread: Optional[threading.Thread] = None
    placement_error: Optional[BaseException] = None
    placement: Optional[EplbPlacement] = None
    plans: Dict[int, LayerPullPlan] = {}
    new_metadata: Optional[ExpertLocationMetadata] = None
    layers_done: Dict[int, bool] = {}
    layers_installed: int = 0


class AsyncExpertMigrator:
    def __init__(self, transport: ExpertTransport):
        self._device = get_device().device
        self._device_module = torch.get_device_module(self._device)
        self._transport = transport
        self._staging: Optional[List[torch.Tensor]] = None
        self._cpu_group = None
        self._install_event = self._device_module.Event()
        self._install_event.record()
        self._read_queue: queue.Queue[Optional[Tuple[AsyncRebalanceJob, int]]] = (
            queue.Queue()
        )
        self._read_thread: Optional[threading.Thread] = None
        self._read_error: Optional[BaseException] = None

    def prepare(
        self,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        cpu_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if cpu_group is None:
            from sglang.srt.distributed.parallel_state import get_world_group

            cpu_group = get_world_group().cpu_group

        layer_ids = sorted(routed_experts_weights_of_layer.keys())
        self._staging = [
            torch.empty_like(t) for t in routed_experts_weights_of_layer[layer_ids[0]]
        ]
        self._transport.prepare(routed_experts_weights_of_layer, self._staging)
        self._cpu_group = cpu_group
        self._read_thread = threading.Thread(
            target=self._read_loop, name="eplb-async-reads", daemon=True
        )
        self._read_thread.start()

    @property
    def prepared(self) -> bool:
        return self._staging is not None

    def start(
        self,
        *,
        logical_count: torch.Tensor,
        old_metadata: ExpertLocationMetadata,
        routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
        compute_placement: Callable[[torch.Tensor], Optional[EplbPlacement]],
    ) -> AsyncRebalanceJob:
        assert self.prepared
        logical_count_host = torch.empty(
            logical_count.shape,
            dtype=logical_count.dtype,
            device="cpu",
            pin_memory=torch.device(self._device).type != "cpu",
        )
        logical_count_host.copy_(logical_count, non_blocking=True)
        count_event = self._device_module.Event()
        count_event.record()
        return AsyncRebalanceJob(
            layer_ids=sorted(routed_experts_weights_of_layer.keys()),
            old_metadata=old_metadata,
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            compute_placement=compute_placement,
            logical_count_host=logical_count_host,
            count_event=count_event,
            time_start=time.time(),
            logical_count_device=logical_count,
        )

    def agree(self, local_flag: bool) -> bool:
        flag = torch.tensor([int(local_flag)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=self._cpu_group
        )
        return bool(flag.item())

    def placement_ready(self, job: AsyncRebalanceJob) -> bool:
        if job.placement_thread is None:
            if not job.count_event.query():
                return self.agree(False)
            job.logical_count_device = None
            job.placement_thread = threading.Thread(
                target=self._run_placement,
                args=(job,),
                name="eplb-async-placement",
                daemon=True,
            )
            job.placement_thread.start()
            return self.agree(False)
        local_ready = not job.placement_thread.is_alive()
        ready = self.agree(local_ready)
        if job.placement_error is not None:
            raise RuntimeError("EPLB async placement failed") from job.placement_error
        return ready

    def _run_placement(self, job: AsyncRebalanceJob):
        try:
            placement = job.compute_placement(job.logical_count_host)
            if placement is None:
                raise RuntimeError("EPLB produced no placement for this model")
            job.placement = placement
            new_map_cpu = placement.physical_to_logical_map.cpu()
            world_size = torch.distributed.get_world_size(self._cpu_group)
            rank = get_parallel().tp_rank
            num_gpu_per_node = world_size // get_parallel().nnodes
            for layer_id in job.layer_ids:
                job.plans[layer_id] = plan_layer_pull(
                    old_physical_to_logical_map=job.old_metadata.physical_to_logical_map_cpu[
                        layer_id
                    ].tolist(),
                    new_physical_to_logical_map=new_map_cpu[layer_id].tolist(),
                    num_local_physical_experts=job.old_metadata.num_local_physical_experts,
                    num_gpu_per_node=num_gpu_per_node,
                    rank=rank,
                )
        except BaseException as e:
            job.placement_error = e

    def build_metadata(self, job: AsyncRebalanceJob) -> ExpertLocationMetadata:
        assert job.placement is not None
        job.new_metadata = ExpertLocationMetadata.init_by_placement(
            job.placement, ep_size=job.old_metadata.ep_size
        )
        return job.new_metadata

    def issue_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = job.plans[layer_id]
        weights = job.routed_experts_weights_of_layer[layer_id]
        for src_slot, dst_slot in plan.weight2buffer_copies:
            for weight, buffer in zip(weights, self._staging, strict=True):
                buffer[dst_slot].copy_(weight[src_slot])
        job.layers_done[layer_id] = False
        self._read_queue.put((job, layer_id))

    def _read_loop(self):
        while True:
            item = self._read_queue.get()
            if item is None:
                return
            job, layer_id = item
            try:
                self._install_event.synchronize()
                plan = job.plans[layer_id]
                for src_rank, slot_pairs in plan.reads.items():
                    self._transport.read_experts(
                        src_rank, layer_id, slot_pairs, self._staging
                    )
                job.layers_done[layer_id] = True
            except BaseException as e:
                self._read_error = e
                job.layers_done[layer_id] = True

    def layer_ready(self, job: AsyncRebalanceJob, layer_id: int) -> bool:
        ready = self.agree(job.layers_done.get(layer_id, False))
        if self._read_error is not None:
            raise RuntimeError("EPLB async expert read failed") from self._read_error
        return ready

    def install_layer(self, job: AsyncRebalanceJob, layer_id: int):
        plan = job.plans[layer_id]
        weights = job.routed_experts_weights_of_layer[layer_id]
        assert job.new_metadata is not None
        for src_slot, dst_slot in plan.buffer2weight_copies:
            for weight, buffer in zip(weights, self._staging, strict=True):
                weight[dst_slot].copy_(buffer[src_slot])
        job.old_metadata.update(job.new_metadata, update_layer_ids=[layer_id])
        self._install_event = self._device_module.Event()
        self._install_event.record()
        job.layers_installed += 1

    def cancel(self, job: Optional[AsyncRebalanceJob]):
        if job is not None and job.placement_thread is not None:
            job.placement_thread.join()
        while not self._read_queue.empty():
            try:
                self._read_queue.get_nowait()
            except queue.Empty:
                break
        self._device_module.synchronize()
