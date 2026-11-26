"""CPU-only shadow planning. Only the scheduler thread enters collectives."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from enum import IntEnum
from threading import BoundedSemaphore

import torch
import torch.distributed as dist

from sglang.srt.eplb import eplb_algorithms
from sglang.srt.eplb.expert_location import compute_static_dispatch_maps
from sglang.srt.eplb.interval_policy import ShadowLayout

logger = logging.getLogger(__name__)


class ReplayStatus(IntEnum):
    WINDOW_INTERRUPTED = 0
    OVER_BUDGET = 1
    FAILED = 2
    LATE = 3
    READY = 4


@torch.inference_mode()
def _plan_packets(demand, slots, ep_size, num_groups, algorithm):
    # No live model, recorder, or process group is accessed by this worker.
    with torch.device("cpu"):
        physical, replicas, _ = eplb_algorithms.rebalance_experts(
            demand.unsqueeze(0), slots, slots // ep_size, num_groups, 1, algorithm
        )
        # All supported ranks share one node, so a same-node preference
        # cannot narrow a replica set. Keep runtime context off this thread.
        maps = compute_static_dispatch_maps(
            replicas, ep_size, slots, prefer_same_node=False
        )
        layers, experts = demand.shape
        packets = torch.empty((ep_size, layers, slots + experts), dtype=torch.int32)
        packets[:, :, :slots].copy_(physical)
        packets[:, :, slots:].copy_(maps.permute(2, 0, 1))
        return packets


class ShadowPlanner:
    def __init__(self, ep_size, rank, group, num_groups, algorithm):
        self.ep_size = ep_size
        self.rank = rank
        self.group = group
        self.num_groups = num_groups
        self.algorithm = algorithm
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-plan")
            if rank == 0
            else None
        )
        self.capacity = BoundedSemaphore(1)
        self.future = None
        self.plan_step = None

    def submit(self, step, ring, slots):
        # sum_steps() owns its storage. The live ring may wrap or reset as
        # soon as this reduction completes.
        demand = ring.sum_steps().cpu()
        dist.reduce(demand, dst=0, group=self.group)
        layers, experts = demand.shape
        self.packet_shape = (layers, slots + experts)
        self.slots = slots
        if self.rank != 0 or not self.capacity.acquire(blocking=False):
            return
        self.plan_step = step
        try:
            self.future = self.executor.submit(
                _plan_packets,
                demand,
                slots,
                self.ep_size,
                self.num_groups,
                self.algorithm,
            )
        except RuntimeError as exc:
            self.future = Future()
            self.future.set_exception(exc)
        capacity = self.capacity
        self.future.add_done_callback(lambda _: capacity.release())

    def discard(self):
        # A running job still holds the capacity token. Dropping our reference
        # lets its packets be freed on completion even if the worker stays pinned.
        if self.future is not None:
            self.future.cancel()
        self.future = self.plan_step = None

    def status(self, step):
        if self.rank != 0:
            return ReplayStatus.READY
        if self.plan_step != step or self.future is None or not self.future.done():
            return ReplayStatus.LATE
        if (error := self.future.exception()) is not None:
            logger.error("Discarding failed shadow EPLB plan", exc_info=error)
            return ReplayStatus.FAILED
        return ReplayStatus.READY

    def receive(self, routed_layers):
        # Called only after all ranks voted READY for this exact boundary.
        packet = torch.empty(self.packet_shape, dtype=torch.int32, device="cpu")
        packets = None
        if self.rank == 0:
            packets = list(self.future.result().unbind())
            self.future = None
        dist.scatter(packet, packets, src=0, group=self.group)
        return ShadowLayout.from_slots(
            packet[routed_layers, : self.slots],
            packet[routed_layers, self.slots :],
            self.slots // self.ep_size,
        )
