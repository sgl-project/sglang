from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import zmq
import zmq.asyncio

from sglang.srt.fault_tolerance.constants import (
    FT_OPERATION_RETRY,
    FT_OPERATION_SCALE_DOWN,
)
from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.protocol import FaultToleranceApplyRequest
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    FaultToleranceCommandReqInput,
    FaultToleranceCommandReqOutput,
    FaultToleranceDPCShutdownReqInput,
    FaultToleranceRankFaultOutput,
    ProcessActiveRanksOutput,
    RouteUpdateAckOutput,
    WatchdogHeartbeatOutput,
    async_sock_send,
)
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

WATCHDOG_LEASE_SWEEP_INTERVAL_SEC = 1
WATCHDOG_LEASE_TIMEOUT_SEC = 60
FT_REQUEST_ACCEPTED_MESSAGE = (
    "Request accepted; poll /v1/fault_tolerance/status for updates."
)


@dataclasses.dataclass
class PendingFTCommand:
    remaining_ranks: set[int]
    future: asyncio.Future

    def acknowledge(self, rank: int) -> None:
        self.remaining_ranks.discard(rank)
        if not self.remaining_ranks:
            self.future.set_result(None)


class FaultToleranceManager:
    def __init__(self, *, server_args, zmq_context, send_to_scheduler):
        self.server_args = server_args
        self._zmq_context = zmq_context
        self.send_to_scheduler = send_to_scheduler
        self.send_to_watchdog: Dict[int, zmq.asyncio.Socket] = {}
        self._watchdog_endpoints: Dict[int, str] = {}
        self.state = FaultToleranceState(
            dp_size=server_args.dp_size,
            strategy=server_args.fault_tolerance_on_error_strategy,
            global_rank_count=server_args.tp_size,
        )
        self.event_loop = None
        self.asyncio_tasks = None
        self._pending_commands: Dict[str, PendingFTCommand] = {}
        self._pending_active_rank_updates: Dict[str, asyncio.Future] = {}
        self._shutdown_waiter: Optional[Tuple[set[int], asyncio.Future]] = None
        self._watchdog_leases: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        self._watchdog_lease_task: Optional[asyncio.Task] = None
        self._route_dp_mask = [True] * server_args.dp_size
        self._last_ft_request_id: Optional[str] = None
        self._ft_error: Optional[str] = None

    def bind_event_loop(self, loop) -> None:
        if self.event_loop is loop:
            return
        if self.event_loop is not None:
            raise RuntimeError(
                "fault tolerance manager is already bound to an event loop"
            )
        self.event_loop = loop
        self.asyncio_tasks = set()

    def init_request_dispatcher(self) -> TypeBasedDispatcher:
        return TypeBasedDispatcher(
            [
                (RouteUpdateAckOutput, self.handle_route_update_ack),
                (FaultToleranceCommandReqOutput, self.handle_command_output),
                (FaultToleranceRankFaultOutput, self.handle_rank_fault),
                (WatchdogHeartbeatOutput, self.observe_watchdog_heartbeat),
            ]
        )

    def status(self) -> tuple[int, dict]:
        body = self.state.status_response()
        if self._last_ft_request_id is not None:
            body["last_ft_request_id"] = self._last_ft_request_id
            if self._ft_error is not None:
                body["ft_error"] = self._ft_error
        return 200, body

    def submit(self, request: FaultToleranceApplyRequest) -> tuple[int, dict]:
        if request.instruction == FT_OPERATION_SCALE_DOWN and any(
            rank >= self.state.dp_size for rank in request.params.removed_dp_ranks
        ):
            return 400, {"message": "'removed_dp_ranks' contains a rank out of range."}
        # One centralized SGLang request controls all DP engines atomically.
        if self.state.ft_operation_in_progress:
            return 409, {"message": "ft_operation_in_progress"}

        self.state.ft_operation_in_progress = True
        try:
            self._create_task(self._run_submitted_apply(request))
        except Exception:
            self.state.ft_operation_in_progress = False
            raise
        return 202, {
            "message": FT_REQUEST_ACCEPTED_MESSAGE,
            "request_id": request.request_id,
        }

    async def _run_submitted_apply(self, request: FaultToleranceApplyRequest) -> None:
        timeout = self.server_args.fault_tolerance_timeout

        if not self.state.has_unresolved_expected_dp_fault():
            self._finish_submitted_apply(
                request.request_id,
                f"{request.instruction}_requires_unresolved_expected_dp_fault",
            )
            return

        if request.instruction == FT_OPERATION_RETRY:
            error = await self._apply_retry(timeout)
        elif request.instruction == FT_OPERATION_SCALE_DOWN:
            error = await self._apply_scale_down(
                request.params.removed_dp_ranks, timeout
            )

        if error is None:
            self.state.unhealthy_dp_ranks.clear()
            self.state.cluster_paused = False
        self._finish_submitted_apply(request.request_id, error)

    def _finish_submitted_apply(self, request_id: str, error: Optional[str]) -> None:
        self._last_ft_request_id = request_id
        self._ft_error = error
        self.state.ft_operation_in_progress = False

    async def _apply_retry(self, timeout: int) -> Optional[str]:
        st = self.state
        process_alive_dp_mask = st.process_alive_dp_mask()
        if any(
            expected and not process_alive_dp_mask[rank]
            for rank, expected in enumerate(st.expected_dp_mask)
        ):
            return "retry_requires_all_expected_processes_alive"

        await self._send_command_collect(
            command=FT_OPERATION_RETRY,
            target_ranks=st.expected_dp_ranks(),
            timeout_sec=timeout,
        )
        await self._publish_route_dp_mask(st.expected_dp_mask, timeout)
        return None

    async def _apply_scale_down(self, ranks: List[int], timeout: int) -> Optional[str]:
        st = self.state
        requested = set(ranks)
        if not requested:
            return "scale_down_requires_ranks"
        if any(not st.expected_dp_mask[rank] for rank in requested):
            return "scale_down_requires_expected_ranks"

        # DP ranks that remain active after the scale-down.
        candidate_dp_mask = [
            expected and rank not in requested
            for rank, expected in enumerate(st.expected_dp_mask)
        ]
        if not any(candidate_dp_mask):
            return "cannot_scale_down_all_expected_ranks"

        await self._shutdown_dp_processes(requested, timeout)
        await self._send_command_collect(
            command=FT_OPERATION_SCALE_DOWN,
            target_ranks=[
                rank for rank, active in enumerate(candidate_dp_mask) if active
            ],
            timeout_sec=timeout,
            active_global_rank_mask=st.expand_dp_mask_to_global_rank_mask(
                candidate_dp_mask
            ),
        )
        await self._publish_route_dp_mask(candidate_dp_mask, timeout)
        for rank in requested:
            st.expected_dp_mask[rank] = False
        return None

    def admission_error(self, routed_dp_rank: Optional[int]) -> Optional[str]:
        if routed_dp_rank is not None and not self._route_dp_mask[routed_dp_rank]:
            return f"routed_dp_rank={routed_dp_rank} is not active"
        if self.state.is_global_admission_blocked(self._route_dp_mask):
            return "fault_tolerance_paused"
        return None

    def _route_dp_update(
        self, route_dp_mask: List[bool]
    ) -> Optional[ActiveRanksOutput]:
        # Cache a changed route mask and return the update for TokenizerManager.
        if route_dp_mask == self._route_dp_mask:
            return None
        self._route_dp_mask = list(route_dp_mask)
        return ActiveRanksOutput(status=route_dp_mask)

    def _auto_recover_ready_dps(self) -> List[int]:
        # Re-add removed DPs only after process and runtime recovery complete.
        st = self.state
        if st.ft_operation_in_progress:
            return []

        process_alive_dp_mask = st.process_alive_dp_mask()
        recovered_dp_ranks = []
        for dp_rank in range(st.dp_size):
            if st.expected_dp_mask[dp_rank]:
                continue
            if (
                process_alive_dp_mask[dp_rank]
                and st.runtime_active_dp_mask[dp_rank]
                and not any(
                    member in st.pending_recovery_global_ranks
                    for member in st.global_ranks_for_dp(dp_rank)
                )
            ):
                st.expected_dp_mask[dp_rank] = True
                recovered_dp_ranks.append(dp_rank)
        return recovered_dp_ranks

    def _update_route_after_observation(
        self, recovered_dp_ranks: List[int]
    ) -> Optional[ActiveRanksOutput]:
        # Apply observed process/runtime availability to the admission route.
        st = self.state
        if st.strategy == "continue":
            process_alive_dp_mask = st.process_alive_dp_mask()
            return self._route_dp_update(
                [
                    expected
                    and process_alive_dp_mask[dp_rank]
                    and st.runtime_active_dp_mask[dp_rank]
                    for dp_rank, expected in enumerate(st.expected_dp_mask)
                ]
            )

        if not recovered_dp_ranks:
            return None
        route_dp_mask = list(self._route_dp_mask)
        for dp_rank in recovered_dp_ranks:
            route_dp_mask[dp_rank] = True
        return self._route_dp_update(route_dp_mask)

    def observe_active_ranks(
        self, ranks: ActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        # This Scheduler report is runtime/backend readiness, not the route mask.
        self.state.observe_runtime_active_dp_mask(list(ranks.status))
        return self._update_route_after_observation(self._auto_recover_ready_dps())

    def observe_process_active_ranks(
        self, ranks: ProcessActiveRanksOutput
    ) -> Optional[ActiveRanksOutput]:
        self.state.observe_process_active_ranks(ranks.ranks, active=ranks.active)
        self._finish_shutdown_if_ready()
        return self._update_route_after_observation(self._auto_recover_ready_dps())

    def observe_watchdog_heartbeat(self, heartbeat: WatchdogHeartbeatOutput) -> None:
        existing = self._watchdog_leases.get(heartbeat.node_rank)
        ranks = tuple(sorted(set(heartbeat.ranks))) if existing is None else existing[1]
        self._watchdog_leases[heartbeat.node_rank] = (time.monotonic(), ranks)
        if self._watchdog_lease_task is None or self._watchdog_lease_task.done():
            self._watchdog_lease_task = self._create_task(
                self._watchdog_lease_sweep_loop()
            )

        endpoint = heartbeat.control_endpoint
        if (
            not endpoint
            or self._watchdog_endpoints.get(heartbeat.node_rank) == endpoint
        ):
            return

        new = get_zmq_socket(self._zmq_context, zmq.PUSH, endpoint, False)
        old = self.send_to_watchdog.get(heartbeat.node_rank)
        self.send_to_watchdog[heartbeat.node_rank] = new
        self._watchdog_endpoints[heartbeat.node_rank] = endpoint
        if old is not None:
            old.close(linger=0)

    async def _send_watchdog_command(
        self, nodes: List[int], request: FaultToleranceDPCShutdownReqInput
    ) -> None:
        for node in nodes:
            await async_sock_send(self.send_to_watchdog[node], request)

    async def _watchdog_lease_sweep_loop(self) -> None:
        try:
            while self._watchdog_leases:
                await asyncio.sleep(WATCHDOG_LEASE_SWEEP_INTERVAL_SEC)
                await self._sweep_expired_watchdog_leases()
        finally:
            self._watchdog_lease_task = None

    async def _sweep_expired_watchdog_leases(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        expired = [
            node
            for node, (last_seen, _) in self._watchdog_leases.items()
            if now - last_seen >= WATCHDOG_LEASE_TIMEOUT_SEC
        ]
        inactive = set()
        for node in expired:
            _, ranks = self._watchdog_leases.pop(node)
            inactive.update(ranks)
        if not inactive:
            return
        logger.warning(
            "FT watchdog lease expired: nodes=%s global_ranks=%s",
            sorted(expired),
            sorted(inactive),
        )
        update = self.observe_process_active_ranks(
            ProcessActiveRanksOutput(ranks=sorted(inactive), active=False)
        )
        if update is not None:
            await self.send_to_scheduler(update)

    def handle_command_output(self, output: FaultToleranceCommandReqOutput) -> None:
        pending = self._pending_commands.get(output.request_id)
        if pending is None:
            logger.warning("Unknown fault tolerance command ack: rank=%s", output.rank)
            return
        pending.acknowledge(output.rank)

    def handle_route_update_ack(self, output: RouteUpdateAckOutput) -> None:
        future = self._pending_active_rank_updates.get(output.request_id)
        if future is None or future.done():
            return
        future.set_result(None)

    def handle_rank_fault(self, event: FaultToleranceRankFaultOutput) -> None:
        self.state.observe_rank_fault(event.rank)
        logger.warning(
            "FT observed scheduler exception on rank %s: %s", event.rank, event.message
        )

    def _create_task(self, coro):
        if self.event_loop is None:
            coro.close()
            raise RuntimeError("fault tolerance manager has no bound event loop")
        task = self.event_loop.create_task(self._fatal_task_wrapper(coro))
        self.asyncio_tasks.add(task)
        task.add_done_callback(self.asyncio_tasks.discard)
        return task

    async def _fatal_task_wrapper(self, coro):
        try:
            await coro
        except Exception:
            self._failstop(
                f"FaultToleranceManager hit an exception: {get_exception_traceback()}"
            )

    async def _publish_route_dp_mask(
        self, route_dp_mask: List[bool], timeout_sec: int
    ) -> None:
        request_id = uuid.uuid4().hex
        future = self.event_loop.create_future()
        self._pending_active_rank_updates[request_id] = future
        try:
            await self.send_to_scheduler(
                ActiveRanksOutput(status=list(route_dp_mask), request_id=request_id)
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
            self._route_dp_mask = list(route_dp_mask)
        finally:
            self._pending_active_rank_updates.pop(request_id, None)

    async def _send_command_collect(
        self,
        *,
        command: str,
        target_ranks: List[int],
        timeout_sec: int,
        active_global_rank_mask: Optional[List[bool]] = None,
    ) -> None:
        targets = set(target_ranks)
        if not targets:
            return
        request_id = uuid.uuid4().hex
        pending = PendingFTCommand(
            remaining_ranks=targets, future=self.event_loop.create_future()
        )
        self._pending_commands[request_id] = pending
        req = FaultToleranceCommandReqInput(
            request_id=request_id,
            command=command,
            target_ranks=sorted(targets),
            # This command is consumed by per-scheduler Elastic EP control, so
            # its active_mask remains a physical scheduler/global-rank mask.
            active_mask=active_global_rank_mask,
        )
        await self.send_to_scheduler(req)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout_sec)
        finally:
            self._pending_commands.pop(request_id, None)

    def _finish_shutdown_if_ready(self) -> None:
        if self._shutdown_waiter is None:
            return
        targets, future = self._shutdown_waiter
        if not future.done() and not any(
            self.state.process_alive_global_rank_mask[rank] for rank in targets
        ):
            future.set_result(None)

    async def _shutdown_dp_processes(
        self, target_dp_ranks: set[int], timeout_sec: int
    ) -> None:
        targets = set(self.state.global_ranks_for_dps(target_dp_ranks))
        live_targets = {
            rank for rank in targets if self.state.process_alive_global_rank_mask[rank]
        }
        if not live_targets:
            return
        nodes = [
            node
            for node, (_, owned) in self._watchdog_leases.items()
            if live_targets.intersection(owned)
        ]
        future = self.event_loop.create_future()
        self._shutdown_waiter = (targets, future)
        try:
            await self._send_watchdog_command(
                nodes,
                FaultToleranceDPCShutdownReqInput(
                    target_dp_ranks=sorted(target_dp_ranks)
                ),
            )
            await asyncio.wait_for(future, timeout=timeout_sec)
        finally:
            self._shutdown_waiter = None

    @staticmethod
    def _failstop(message: str) -> None:
        logger.error(message)
        kill_process_tree(os.getpid(), include_parent=True)
        raise RuntimeError(message)
