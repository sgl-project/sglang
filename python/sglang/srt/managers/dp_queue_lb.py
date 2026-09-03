# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""DP waiting-queue load balancing (queue_lb feature).

Under DP attention with a consistent-hashing / dp-aware router, requests are
routed to DP ranks by prefix key. This gives good prefix-cache locality but
tends to make the per-rank waiting queues badly imbalanced (e.g. dp0 has 100
queued reqs while dp6 has 3). Because DP attention is hard lock-step (the EP
all-to-all synchronizes every rank each step), the most-loaded rank dictates the
step time, so instantaneous queue imbalance directly creates bubbles.

This module implements the runtime rebalancer:

* ``DPQueueBalancer`` (runs inside the DataParallelController): reads the per-rank
  ``LoadSnapshot`` it already collects and, with a naive threshold policy,
  decides how many requests to move and where (``MigrationOrder``).
* ``SchedulerMigrationAgent`` (runs inside each Scheduler): owns the peer-to-peer
  ZMQ mesh used to actually ship migrated requests between DP ranks.

Phase 1 (this skeleton): only requests that are still purely queued (not yet
prefilled, no KV allocated) are migrated, so no KV cache has to travel with the
request. Phase 2 will add KV migration through the shared UMBP DRAM pool.

See queue_lb/PLAN.md for the full design.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import zmq

from sglang.srt.managers.io_struct import MigrateBatchReq, sock_recv, sock_send
from sglang.srt.utils.network import get_zmq_socket

if TYPE_CHECKING:
    from sglang.srt.managers.load_snapshot import LoadSnapshot
    from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def migration_endpoint(port_args: "PortArgs", dp_rank: int) -> str:
    """Deterministic ipc:// endpoint for a DP rank's migration PULL socket.

    Every DP-rank scheduler must derive the SAME endpoint for a given peer, so
    the seed has to be a PortArgs field that is identical across all ranks.

    ``scheduler_input_ipc_name`` and ``nccl_port`` are per-rank unique (each
    scheduler gets its own via ``PortArgs.init_new``), so seeding on them makes
    every rank bind/connect a different path and migrated requests are silently
    dropped. Only ``tokenizer_ipc_name`` / ``detokenizer_ipc_name`` /
    ``instance_id`` are explicitly kept shared across ranks
    (see DataParallelController.launch_dp_schedulers), so seed on those.
    """
    seed = (
        getattr(port_args, "tokenizer_ipc_name", None)
        or getattr(port_args, "instance_id", None)
        or "sglang"
    )
    digest = hashlib.sha1(str(seed).encode("utf-8")).hexdigest()[:12]
    return f"ipc:///tmp/sglang_dp_migrate_{digest}_{dp_rank}.sock"


@dataclass
class MigrationOrder:
    src_dp_rank: int
    dst_dp_rank: int
    count: int


class DPQueueBalancer:
    """Controller-side decision engine (naive threshold policy)."""

    def __init__(self, server_args: "ServerArgs", dp_size: int):
        self.dp_size = dp_size
        self.threshold = server_args.dp_queue_balance_threshold
        self.min_abs = server_args.dp_queue_balance_min_abs
        self.cap_per_round = server_args.dp_queue_balance_cap_per_round
        self.interval_s = server_args.dp_queue_balance_interval_ms / 1000.0
        self._last_time = 0.0

    def maybe_plan(
        self, loads: List["LoadSnapshot"], now: Optional[float] = None
    ) -> List[MigrationOrder]:
        """Throttled entry point: returns [] until ``interval_ms`` has elapsed."""
        now = now if now is not None else time.perf_counter()
        if now - self._last_time < self.interval_s:
            return []
        self._last_time = now
        return self.plan(loads)

    def plan(self, loads: List["LoadSnapshot"]) -> List[MigrationOrder]:
        """Compute migration orders from the current per-rank load snapshots.

        Threshold policy:
          * mean = average waiting-queue length over active ranks
          * donor:     waiting > mean*(1+thr) and waiting >= min_abs
          * recipient: waiting < mean*(1-thr)
          * per donor budget = min(donor_excess, cap_per_round)
          * per pair move = min(remaining_budget, recipient_deficit)
        Greedy match: largest donors -> smallest recipients.
        """
        # waiting[rank] = queue length; None => no fresh snapshot for that rank.
        waiting: Dict[int, int] = {}
        for s in loads:
            if 0 <= s.dp_rank < self.dp_size:
                waiting[s.dp_rank] = int(s.num_waiting_reqs)
        if len(waiting) < 2:
            return []

        values = list(waiting.values())
        mean = sum(values) / len(values)
        if mean <= 0:
            return []

        hi = mean * (1.0 + self.threshold)
        lo = mean * (1.0 - self.threshold)

        donors = sorted(
            (
                (rank, q)
                for rank, q in waiting.items()
                if q > hi and q >= self.min_abs
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        recipients = sorted(
            ((rank, q) for rank, q in waiting.items() if q < lo),
            key=lambda kv: kv[1],
        )
        if not donors or not recipients:
            return []

        # Mutable deficits/excess so a single round is internally consistent.
        excess = {rank: int(q - mean) for rank, q in donors}
        deficit = {rank: int(mean - q) for rank, q in recipients}

        orders: List[MigrationOrder] = []
        r_idx = 0
        recipient_ranks = [rank for rank, _ in recipients]
        for src, _ in donors:
            src_budget = min(excess[src], self.cap_per_round)
            while src_budget > 0 and r_idx < len(recipient_ranks):
                dst = recipient_ranks[r_idx]
                if deficit[dst] <= 0:
                    r_idx += 1
                    continue
                move = min(src_budget, deficit[dst])
                if move <= 0:
                    break
                orders.append(MigrationOrder(src, dst, move))
                src_budget -= move
                excess[src] -= move
                deficit[dst] -= move
            if r_idx >= len(recipient_ranks):
                break

        if orders:
            logger.info("DPQueueBalancer plan (waiting=%s): %s", waiting, orders)
        return orders


class SchedulerMigrationAgent:
    """Scheduler-side owner of the peer-to-peer migration ZMQ mesh.

    Each DP-rank leader binds one PULL socket at ``migration_endpoint(rank)`` and
    lazily opens PUSH sockets to peers when it needs to ship requests.
    """

    def __init__(
        self,
        context: zmq.Context,
        port_args: "PortArgs",
        dp_rank: int,
        dp_size: int,
        enabled: bool,
    ):
        self.enabled = enabled
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.port_args = port_args
        self.context = context
        self.recv_socket: Optional[zmq.Socket] = None
        self._send_sockets: Dict[int, zmq.Socket] = {}

        if not self.enabled:
            return
        endpoint = migration_endpoint(port_args, dp_rank)
        # bind=True: this rank owns (creates) its own inbound endpoint.
        self.recv_socket = get_zmq_socket(context, zmq.PULL, endpoint, True)
        logger.info("SchedulerMigrationAgent dp_rank=%s listening on %s", dp_rank, endpoint)

    def _get_send_socket(self, dst_rank: int) -> zmq.Socket:
        sock = self._send_sockets.get(dst_rank)
        if sock is None:
            endpoint = migration_endpoint(self.port_args, dst_rank)
            sock = get_zmq_socket(self.context, zmq.PUSH, endpoint, False)
            self._send_sockets[dst_rank] = sock
        return sock

    def send_batch(self, msg: MigrateBatchReq) -> None:
        if not self.enabled:
            return
        sock_send(self._get_send_socket(msg.dst_dp_rank), msg)

    def poll(self) -> List[MigrateBatchReq]:
        """Drain all pending migrated batches addressed to this rank."""
        if not self.enabled or self.recv_socket is None:
            return []
        out: List[MigrateBatchReq] = []
        while True:
            try:
                msg = sock_recv(self.recv_socket, flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            if isinstance(msg, MigrateBatchReq):
                out.append(msg)
            else:
                logger.warning("Migration mesh got unexpected message: %r", type(msg))
        return out

    def close(self) -> None:
        for sock in self._send_sockets.values():
            try:
                sock.close(0)
            except Exception:
                pass
        if self.recv_socket is not None:
            try:
                self.recv_socket.close(0)
            except Exception:
                pass
