"""FFN-side AFD lifecycle and event loops for staged and Direct sessions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch
import zmq

from sglang.srt.afd.afd_config import (
    AFDRole,
    is_direct_afd_backend,
)
from sglang.srt.afd.afd_handshake import (
    AFDGraphCapturePlan,
    AFDHandshakeResult,
    AFDHandshakeServer,
    afd_handshake_port,
    build_afd_ffn_capacity_plan,
    build_afd_graph_capture_plan,
)
from sglang.srt.afd.afd_sync import (
    AFDDirectCommonPlan,
    AFDDirectControlServer,
    AFDDirectProposal,
    AFDSyncMsg,
    AFDSyncMsgReceiver,
)
from sglang.srt.afd.comm import (
    AFDCommBackend,
)
from sglang.srt.afd.dispatcher import (
    is_afd_fixed_shape_mode,
    select_token_bucket,
)
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
    gather_nvshmem_topology,
    init_afd_comm_backend,
    should_log_step,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    FreezeGCReq,
    ProfileReq,
    ProfileReqOutput,
    ShutdownReq,
    sock_recv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_serving,
)
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.afd.ffn.executor import AFDFFNExecutor
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


def init_ffn_model_worker_state(scheduler: Scheduler) -> None:
    """``Scheduler.init_model_worker`` branch for the ffn role.

    The native model was already loaded by ``init_tp_model_worker``; export
    the runtime state the ffn event loop needs, then let the caller return
    before memory pools, attention backends and CUDA graphs.
    """
    from sglang.srt.distributed.parallel_state import get_tp_group

    if scheduler.server_args.is_startup_weight_load_overlap:
        scheduler.tp_worker.start_startup_weight_load()
        scheduler.tp_worker.finalize_startup_weight_load()

    scheduler.draft_worker = None
    scheduler.model_worker = scheduler.tp_worker

    scheduler.device = scheduler.tp_worker.device
    scheduler.forward_stream = scheduler.tp_worker.model_runner.forward_stream
    scheduler.tp_group = get_tp_group()
    scheduler.tp_cpu_group = scheduler.tp_group.cpu_group
    scheduler.dp_tp_group = scheduler.tp_group
    scheduler.dp_tp_cpu_group = scheduler.tp_cpu_group

    # No-KV semantics for get_init_info(): the ffn instance serves no
    # generation requests and allocates no token pool.
    scheduler.max_total_num_tokens = 0
    scheduler.max_req_input_len = 0
    scheduler.kv_cache_allocation_time = 0.0


def finish_ffn_scheduler_init(scheduler: Scheduler) -> None:
    """``Scheduler.__init__`` early-return branch for the ffn role.

    Completes the minimal lifecycle state used by ``run_event_loop`` and the
    process exit path; KV cache, radix cache, schedule policy and request
    components are never built.
    """
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )

    # run_event_loop creates the schedule stream from device_module (normally
    # set in init_overlap, which the ffn role skips).
    scheduler.device_module = torch.get_device_module(scheduler.device)

    # Prepare expert compute now; graph capture runs after the A/F handshake.
    from sglang.srt.afd.ffn.compute import AFDFFNCompute

    scheduler.afd_ffn_compute = AFDFFNCompute(scheduler)

    # run_scheduler_process's exit path shuts down the metrics reporter.
    scheduler.metrics_reporter = SchedulerMetricsReporter(
        scheduler=scheduler,
        tp_rank=scheduler.ps.tp_rank,
        pp_rank=scheduler.ps.pp_rank,
        dp_rank=scheduler.ps.dp_rank,
        metrics_collector_context=scheduler.metrics_collector_context,
        metrics_collector=scheduler.metrics_collector,
    )

    scheduler.forward_ct = 0
    scheduler.gracefully_exit = False
    scheduler.init_profiler()

    scheduler.is_initializing = False
    scheduler.init_startup_timing_summary()


def _handle_ffn_profile_request(
    scheduler: Scheduler, req: ProfileReq
) -> ProfileReqOutput:
    if req.start_step is not None or req.num_steps is not None or req.profile_by_stage:
        return ProfileReqOutput(
            success=False,
            message=(
                "AFD ffn profiling supports manual /start_profile and "
                "/stop_profile only; start_step, num_steps, and "
                "profile_by_stage are not supported."
            ),
        )
    return scheduler.profiler_manager._profile(req)


def _handle_ffn_scheduler_request(
    scheduler: Scheduler, req: object, is_rank0: bool
) -> bool:
    if isinstance(req, FreezeGCReq):
        scheduler.handle_freeze_gc(req)
        return False
    if isinstance(req, ProfileReq):
        output = _handle_ffn_profile_request(scheduler, req)
        if is_rank0:
            scheduler.ipc_channels.send_to_tokenizer.send_output(output, req)
        return False
    if isinstance(req, ShutdownReq):
        scheduler.gracefully_exit = True
        return True
    if is_rank0:
        logger.warning(
            "AFD ffn scheduler ignores unsupported request: %s",
            type(req).__name__,
        )
    return False


@dataclass
class _AFDFFNSession:
    """One attn instance served by this ffn instance."""

    session_id: int
    handshake: AFDHandshakeResult
    plan: AFDGraphCapturePlan
    backend: AFDCommBackend
    executor: AFDFFNExecutor
    receiver: Optional[AFDSyncMsgReceiver] = None  # rank 0 only
    ffn_graph_variants: dict[int, int] = field(default_factory=dict)
    ffn_parent_graph_variants: dict[tuple[int, ...], tuple[int, ...]] = field(
        default_factory=dict
    )


class _AFDFFNCohortSelector:
    def __init__(self, join_timeout_us: int):
        self._join_timeout_ns = join_timeout_us * 1_000
        self._pending: dict[int, AFDSyncMsg] = {}
        self._previous_members: tuple[int, ...] = ()
        self._join_deadline_ns: int | None = None

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def has_pending_session(self, session_id: int) -> bool:
        return session_id in self._pending

    def add(self, session_id: int, msg: AFDSyncMsg) -> None:
        if self.has_pending_session(session_id):
            raise RuntimeError(
                f"AFD ffn session {session_id} has multiple pending steps"
            )
        self._pending[session_id] = msg

    def select(
        self,
        now_ns: int,
        *,
        can_cohort: Callable[[int, AFDSyncMsg], bool],
        cohort_cost: Callable[[int, AFDSyncMsg], tuple[int, ...]],
        cohort_capacity: int | None,
    ) -> list[tuple[int, AFDSyncMsg]] | None:
        if not self._pending:
            return None

        anchor_id = next(iter(self._pending))
        if not can_cohort(anchor_id, self._pending[anchor_id]):
            return self._take((anchor_id,))

        compatible = [anchor_id]
        used_capacity = cohort_cost(anchor_id, self._pending[anchor_id])
        for session_id, msg in self._pending.items():
            if session_id == anchor_id or not can_cohort(session_id, msg):
                continue
            session_cost = cohort_cost(session_id, msg)
            if len(session_cost) != len(used_capacity):
                continue
            next_capacity = tuple(
                used + added
                for used, added in zip(
                    used_capacity,
                    session_cost,
                    strict=True,
                )
            )
            if cohort_capacity is not None and any(
                cost > cohort_capacity for cost in next_capacity
            ):
                continue
            compatible.append(session_id)
            used_capacity = next_capacity
        if len(compatible) >= 2:
            action = self._take(tuple(compatible))
            if self._pending:
                self._join_deadline_ns = now_ns + self._join_timeout_ns
            return action

        if len(self._pending) > 1:
            return self._take((anchor_id,))
        # Preserve single-session latency when the same singleton keeps
        # producing work; join only after the active session set changes.
        if self._previous_members == (anchor_id,):
            return self._take((anchor_id,))

        if self._join_deadline_ns is None:
            self._join_deadline_ns = now_ns + self._join_timeout_ns
        if now_ns < self._join_deadline_ns:
            return None
        return self._take((anchor_id,))

    def _take(self, session_ids: tuple[int, ...]) -> list[tuple[int, AFDSyncMsg]]:
        action = [
            (session_id, self._pending.pop(session_id)) for session_id in session_ids
        ]
        self._previous_members = session_ids
        self._join_deadline_ns = None
        return action


def _build_ffn_session(
    scheduler: Scheduler, handshake: AFDHandshakeResult, compute
) -> _AFDFFNSession:
    from sglang.srt.afd.ffn.executor import AFDFFNExecutor

    plan = build_afd_graph_capture_plan(handshake)
    backend = init_afd_comm_backend(scheduler, AFDRole.FFN, handshake)
    executor = AFDFFNExecutor(scheduler, backend, compute, handshake)
    executor.capture_communication_graphs(plan)
    ffn_graph_variants = dict(plan.ffn_graph_variants)
    ffn_graph_variants.update(
        {
            variant.child_wire_capacities[0]: variant.child_compute_capacities[0]
            for variant in plan.attn_graph_variants
            if len(variant.child_wire_capacities) == 1
            and variant.child_compute_capacities[0] is not None
        }
    )
    ffn_parent_graph_variants = {
        variant.child_wire_capacities: variant.child_compute_capacities
        for variant in plan.attn_graph_variants
        if len(variant.child_wire_capacities) > 1
        and all(capacity is not None for capacity in variant.child_compute_capacities)
    }
    return _AFDFFNSession(
        session_id=handshake.session_id,
        handshake=handshake,
        plan=plan,
        backend=backend,
        executor=executor,
        ffn_graph_variants=ffn_graph_variants,
        ffn_parent_graph_variants=ffn_parent_graph_variants,
    )


def _log_ffn_step(session: _AFDFFNSession, compute, msg: AFDSyncMsg, rank: int) -> None:
    handshake = session.handshake
    backend = session.backend
    forward_mode = ForwardMode(msg.forward_mode)
    graph_capacity = None
    if (
        is_afd_fixed_shape_mode(forward_mode)
        and msg.wire_capacity > 0
        and not msg.child_wire_capacities
    ):
        if (
            is_direct_afd_backend(handshake.local.afd_backend)
            and msg.communication_graph
        ):
            graph_capacity = msg.wire_capacity
        elif backend.decode_transport is not None or msg.communication_graph:
            graph_capacity = session.ffn_graph_variants.get(msg.wire_capacity)
        elif not session.plan.enabled:
            graph_capacity = compute.graph_capacity_for(msg.wire_capacity)
    if msg.child_wire_capacities:
        child_compute_capacities = session.ffn_parent_graph_variants.get(
            tuple(msg.child_wire_capacities)
        )
        if msg.communication_graph and child_compute_capacities:
            graph_capacity = max(child_compute_capacities)
        if graph_capacity is not None:
            compute_capacity = graph_capacity
        elif handshake.local.afd_ffn_moe_backend == "deepep":
            attn_ranks = backend.config.attn_ranks_for_ffn(
                rank,
                scattered=True,
            )
            compute_capacity = len(attn_ranks) * max(msg.child_wire_capacities)
        else:
            compute_capacity = max(msg.child_wire_capacities)
    elif graph_capacity is not None:
        compute_capacity = graph_capacity
    elif handshake.local.afd_ffn_moe_backend == "deepep":
        attn_ranks = backend.config.attn_ranks_for_ffn(
            rank,
            scattered=True,
        )
        compute_capacity = (
            len(attn_ranks) * msg.wire_capacity
            if is_afd_fixed_shape_mode(forward_mode)
            else sum(msg.tokens_per_rank[index] for index in attn_ranks)
        )
    else:
        compute_capacity = msg.wire_capacity
    compute_graph = graph_capacity is not None or (
        is_direct_afd_backend(handshake.local.afd_backend) and msg.communication_graph
    )
    path = "direct" if is_direct_afd_backend(handshake.local.afd_backend) else "staged"
    logger.info(
        "AFD step: side=ffn path=%s session=%d session_step=%d mode=%s "
        "tokens_per_rank=%s total_tokens=%d wire=%d children=%s "
        "communication_graph=%s compute_graph=%s compute_capacity=%d",
        path,
        session.session_id,
        msg.step_id,
        forward_mode.name,
        msg.tokens_per_rank,
        sum(msg.tokens_per_rank),
        msg.wire_capacity,
        msg.child_wire_capacities,
        msg.communication_graph,
        compute_graph,
        compute_capacity,
    )


def _event_loop_afd_ffn_cohort(
    scheduler: Scheduler,
    sessions: list[_AFDFFNSession],
    poller: Optional[zmq.Poller],
    scheduler_input_socket,
) -> None:
    from sglang.srt.afd.ffn.cohort import AFDFFNCohortExecutor

    rank = scheduler.ps.tp_rank
    is_rank0 = rank == 0
    join_timeout_us = envs.SGLANG_AFD_STAGED_COHORT_JOIN_TIMEOUT_US.get()
    cohort_executor = AFDFFNCohortExecutor([session.executor for session in sessions])
    cohort_executor.capture_communication_graphs()
    if is_rank0:
        logger.info(
            "AFD ffn cohort event loop started: sessions=%d join_timeout_us=%d",
            len(sessions),
            join_timeout_us,
        )

    selector = None
    completion_event = None
    has_submitted_action = False
    at_busy_boundary = False
    next_session_index = 0
    if is_rank0:
        assert poller is not None
        selector = _AFDFFNCohortSelector(join_timeout_us)
        completion_event = torch.cuda.Event()

    def receive_controls(
        *, block: bool
    ) -> tuple[Optional[list[tuple[int, object]]], bool]:
        nonlocal next_session_index
        assert is_rank0 and poller is not None and selector is not None
        events = dict(poller.poll() if block else poller.poll(0))
        if not events:
            return None, False
        if scheduler_input_socket in events:
            return [(-1, sock_recv(scheduler_input_socket))], True

        consumed = False
        num_sessions = len(sessions)
        start_session_index = next_session_index
        for probe in range(num_sessions):
            session = sessions[(start_session_index + probe) % num_sessions]
            # Native overlap may queue the next control for this session while
            # its current step is waiting for cohort selection. Leave that
            # control in ZMQ until the pending step has been submitted.
            if session.receiver.socket not in events or selector.has_pending_session(
                session.session_id
            ):
                continue
            msg = session.receiver.recv()
            consumed = True
            next_session_index = (session.session_id + 1) % num_sessions
            if not isinstance(msg, AFDSyncMsg) or (
                sum(msg.tokens_per_rank) == 0 or ForwardMode(msg.forward_mode).is_idle()
            ):
                return [(session.session_id, msg)], True
            selector.add(session.session_id, msg)
        return None, consumed

    def drain_ready_controls() -> Optional[list[tuple[int, object]]]:
        while True:
            immediate, consumed = receive_controls(block=False)
            if immediate is not None:
                return immediate
            if not consumed:
                return None

    def select_pending_action() -> Optional[list[tuple[int, object]]]:
        assert is_rank0 and selector is not None
        while selector.has_pending:
            immediate = drain_ready_controls()
            if immediate is not None:
                return immediate
            action = selector.select(
                time.monotonic_ns(),
                can_cohort=cohort_executor.can_cohort,
                cohort_cost=cohort_executor.cohort_cost,
                cohort_capacity=cohort_executor.cohort_capacity,
            )
            if action is not None:
                return action
        return None

    def next_action() -> list[tuple[int, object]]:
        nonlocal at_busy_boundary, has_submitted_action
        assert is_rank0 and selector is not None and completion_event is not None
        while True:
            if has_submitted_action and not at_busy_boundary:
                completion_event.synchronize()
                immediate = drain_ready_controls()
                if immediate is not None:
                    return immediate
                if selector.has_pending:
                    at_busy_boundary = True
                else:
                    has_submitted_action = False

            if at_busy_boundary:
                action = select_pending_action()
                if action is not None:
                    return action
                continue

            if not selector.has_pending:
                immediate, _ = receive_controls(block=True)
                if immediate is not None:
                    return immediate
                immediate = drain_ready_controls()
                if immediate is not None:
                    return immediate
            action = select_pending_action()
            if action is not None:
                return action

    while True:
        if is_rank0:
            action = next_action()
            if scheduler.ps.tp_size > 1:
                broadcast_pyobj(action, rank, scheduler.tp_cpu_group, src=0)
        else:
            action = broadcast_pyobj([], rank, scheduler.tp_cpu_group, src=0)

        if len(action) == 1 and action[0][0] == -1:
            if _handle_ffn_scheduler_request(scheduler, action[0][1], is_rank0):
                break
            continue

        if any(not isinstance(msg, AFDSyncMsg) for _, msg in action):
            if is_rank0:
                logger.warning(
                    "AFD ffn session ignores unsupported control action: %s",
                    [type(msg).__name__ for _, msg in action],
                )
            continue

        active_action = any(sum(msg.tokens_per_rank) > 0 for _, msg in action)
        if len(action) == 1:
            session_id, msg = action[0]
            if is_rank0 and should_log_step(msg.step_id):
                _log_ffn_step(
                    sessions[session_id], scheduler.afd_ffn_compute, msg, rank
                )
            sessions[session_id].executor.run_step(msg)
        else:
            compute_graph_mode = cohort_executor.run_step(action)
            if is_rank0 and any(should_log_step(msg.step_id) for _, msg in action):
                logger.info(
                    "AFD step: side=ffn path=staged_cohort sessions=%s "
                    "session_steps=%s modes=%s total_tokens=%s wire=%s "
                    "children=%s merged_wire_capacity=%d "
                    "communication_graphs=%s compute_graph_mode=%s",
                    [session_id for session_id, _ in action],
                    [msg.step_id for _, msg in action],
                    [ForwardMode(msg.forward_mode).name for _, msg in action],
                    [sum(msg.tokens_per_rank) for _, msg in action],
                    [msg.wire_capacity for _, msg in action],
                    [msg.child_wire_capacities for _, msg in action],
                    sum(msg.wire_capacity for _, msg in action),
                    [msg.communication_graph for _, msg in action],
                    compute_graph_mode,
                )

        if is_rank0 and active_action:
            completion_event.record()
            has_submitted_action = True
            at_busy_boundary = False


def _build_direct_union_action(
    session: _AFDFFNSession,
    proposals: Sequence[AFDDirectProposal],
    selected_sessions: Sequence[int],
    union_step_id: int,
) -> tuple[AFDDirectCommonPlan, AFDSyncMsg]:
    selected = frozenset(selected_sessions)
    if not selected:
        raise RuntimeError("AFD Direct union action requires active work")

    selected_proposals = [proposals[index] for index in selected_sessions]
    fixed_shape = is_afd_fixed_shape_mode(
        ForwardMode(selected_proposals[0].forward_mode)
    )

    communication_graph = False
    child_wire_capacities: tuple[int, ...] = ()
    if fixed_shape:
        minimum_wire_capacity = max(
            max(proposal.tokens_per_rank) for proposal in selected_proposals
        )
        parent_wire_capacity = select_token_bucket(
            session.handshake.peer.decode_token_buckets,
            minimum_wire_capacity,
        )
        if all(proposal.can_run_graph for proposal in selected_proposals):
            graph_variant = min(
                (
                    variant
                    for variant in session.plan.attn_graph_variants
                    if variant.attn_wire_capacity >= minimum_wire_capacity
                ),
                key=lambda variant: variant.attn_wire_capacity,
                default=None,
            )
            if graph_variant is not None and (
                not session.handshake.peer.disable_cuda_graph_padding
                or all(
                    max(proposal.tokens_per_rank) == graph_variant.attn_wire_capacity
                    for proposal in selected_proposals
                )
            ):
                communication_graph = True
                parent_wire_capacity = graph_variant.attn_wire_capacity
                if len(graph_variant.child_wire_capacities) > 1:
                    child_wire_capacities = graph_variant.child_wire_capacities
        if not communication_graph:
            child_plans = [
                proposal.child_wire_capacities for proposal in selected_proposals
            ]
            num_microbatches = session.handshake.peer.decode_num_microbatches
            if num_microbatches > 1 and all(
                len(children) == num_microbatches for children in child_plans
            ):
                child_wire_capacities = tuple(
                    max(capacities) for capacities in zip(*child_plans, strict=True)
                )
        wire_capacity = (
            sum(child_wire_capacities)
            if child_wire_capacities
            else parent_wire_capacity
        )
        plan_forward_mode = ForwardMode.DECODE
    else:
        parent_wire_capacity = max(
            proposal.wire_capacity for proposal in selected_proposals
        )
        wire_capacity = parent_wire_capacity
        plan_forward_mode = ForwardMode.EXTEND

    active_mask = tuple(index in selected for index in range(len(proposals)))
    plan = AFDDirectCommonPlan(
        union_step_id=union_step_id,
        observed_proposal_ids=tuple(proposal.proposal_id for proposal in proposals),
        active_mask=active_mask,
        forward_mode=int(plan_forward_mode),
        parent_wire_capacity=parent_wire_capacity,
        wire_capacity=wire_capacity,
        communication_graph=communication_graph,
        child_wire_capacities=child_wire_capacities,
    )
    tokens_per_rank = [
        num_tokens if active else 0
        for proposal, active in zip(proposals, active_mask, strict=True)
        for num_tokens in proposal.tokens_per_rank
    ]
    msg = AFDSyncMsg(
        step_id=union_step_id,
        tokens_per_rank=tokens_per_rank,
        forward_mode=int(plan_forward_mode),
        wire_capacity=wire_capacity,
        communication_graph=communication_graph,
        child_wire_capacities=child_wire_capacities,
    )
    return plan, msg


def _select_direct_union_sessions(
    proposals: Sequence[AFDDirectProposal],
) -> list[int]:
    active_sessions = [
        index
        for index, proposal in enumerate(proposals)
        if sum(proposal.tokens_per_rank) > 0
    ]
    prefill_sessions = [
        index
        for index in active_sessions
        if not is_afd_fixed_shape_mode(ForwardMode(proposals[index].forward_mode))
    ]
    return prefill_sessions or active_sessions


def _event_loop_afd_ffn_direct_union(
    scheduler: Scheduler,
    handshakes: list[AFDHandshakeResult],
) -> None:
    rank = scheduler.ps.tp_rank
    is_rank0 = rank == 0
    session = _build_ffn_session(
        scheduler,
        handshakes[0],
        scheduler.afd_ffn_compute,
    )
    scheduler_input_socket = None
    poller = None
    controls: list[AFDDirectControlServer] = []
    if is_rank0:
        scheduler_input_socket = scheduler.ipc_channels.recv_from_tokenizer
        poller = zmq.Poller()
        poller.register(scheduler_input_socket, zmq.POLLIN)
        for handshake in handshakes:
            control = AFDDirectControlServer(
                f"tcp://*:{handshake.sync_port}",
                handshake.session_id,
            )
            controls.append(control)
            poller.register(control.socket, zmq.POLLIN)
        logger.info(
            "AFD Direct union event loop started: sessions=%d attn_ranks=%d",
            len(handshakes),
            session.backend.config.num_attn_ranks,
        )

    pending: dict[int, AFDDirectProposal] = {}
    union_step_id = 0
    try:
        while True:
            if is_rank0:
                assert poller is not None
                while True:
                    events = dict(poller.poll())
                    if scheduler_input_socket in events:
                        action = (None, sock_recv(scheduler_input_socket))
                        break
                    for session_id, control in enumerate(controls):
                        if control.socket in events:
                            proposal = control.recv_proposal()
                            pending[session_id] = proposal
                    if len(pending) != len(controls) or not any(
                        sum(proposal.tokens_per_rank) > 0
                        for proposal in pending.values()
                    ):
                        continue

                    proposals = [pending[index] for index in range(len(controls))]
                    selected_sessions = _select_direct_union_sessions(proposals)
                    plan, msg = _build_direct_union_action(
                        session,
                        proposals,
                        selected_sessions,
                        union_step_id,
                    )
                    for control in controls:
                        control.send_plan(plan)
                    pending.clear()
                    union_step_id += 1
                    action = (plan, msg)
                    break
                if scheduler.ps.tp_size > 1:
                    broadcast_pyobj(
                        [action],
                        rank,
                        scheduler.tp_cpu_group,
                        src=0,
                    )
            else:
                (action,) = broadcast_pyobj(
                    [],
                    rank,
                    scheduler.tp_cpu_group,
                    src=0,
                )

            plan, msg = action
            if plan is None:
                if _handle_ffn_scheduler_request(scheduler, msg, is_rank0):
                    break
                continue

            if is_rank0 and should_log_step(msg.step_id):
                logger.info(
                    "AFD step: side=ffn path=direct union_step=%d active=%s "
                    "mode=%s tokens_per_rank=%s total_tokens=%d "
                    "parent_wire=%d wire=%d children=%s "
                    "communication_graph=%s compute_graph=%s",
                    msg.step_id,
                    plan.active_mask,
                    ForwardMode(msg.forward_mode).name,
                    msg.tokens_per_rank,
                    sum(msg.tokens_per_rank),
                    plan.parent_wire_capacity,
                    msg.wire_capacity,
                    msg.child_wire_capacities,
                    msg.communication_graph,
                    msg.communication_graph,
                )
            session.executor.run_step(msg)
    finally:
        for control in controls:
            control.close()
    if is_rank0:
        logger.info("AFD Direct union event loop exited")


def event_loop_afd_ffn(scheduler: Scheduler) -> None:
    """Ffn-side event loop for AFD steps and scheduler management requests.

    Serves ``--afd-attn-instances`` sessions: all handshakes are collected
    before serving (static membership), then per-session control messages
    are multiplexed over one shared ffn compute stream.
    """
    rank = scheduler.ps.tp_rank
    is_rank0 = rank == 0
    num_sessions = get_disagg().afd_attn_instances
    compute = scheduler.afd_ffn_compute
    rank_node_ids, rank_nvshmem_fabric_keys = gather_nvshmem_topology(scheduler)

    # Static membership lets joint-world backends build [F][A0][A1]... from
    # the same handshake roster on every process.
    if is_rank0:
        local_info = build_local_handshake_info(
            scheduler,
            AFDRole.FFN,
            rank_node_ids,
            rank_nvshmem_fabric_keys,
        )
        handshake_server = AFDHandshakeServer(afd_handshake_port(get_serving().port))
        try:
            handshakes = handshake_server.accept_all(
                local_info,
                num_sessions,
                use_deepep_low_latency=(get_exec().moe.deepep_mode != "normal"),
            )
        finally:
            handshake_server.close()
        if scheduler.ps.tp_size > 1:
            broadcast_pyobj([handshakes], rank, scheduler.tp_cpu_group, src=0)
    else:
        # Every ffn rank needs the handshake results (topology / negotiated
        # ports) to build its data-plane comm backends.
        (handshakes,) = broadcast_pyobj([], rank, scheduler.tp_cpu_group, src=0)

    backend = handshakes[0].local.afd_backend
    if backend == "direct_deepep":
        if get_exec().moe.deepep_mode != "normal":
            ffn_ranks = handshakes[0].local.num_ranks
            union_fan_in = (handshakes[0].total_attn_ranks + ffn_ranks - 1) // ffn_ranks
            dispatch_limit = max(
                max(handshakes[0].peer.decode_token_buckets) * union_fan_in,
                envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
            )
            if dispatch_limit > 1024:
                raise RuntimeError(
                    "AFD Direct DeepEP requires at most 1024 native DeepEP "
                    f"dispatch tokens per F rank; required={dispatch_limit}"
                )
            compute.configure_native_deepep_dispatch_limit(dispatch_limit)
    elif not is_direct_afd_backend(backend):
        capacity_plan = build_afd_ffn_capacity_plan(
            handshakes[0].local,
            tuple(handshake.peer for handshake in handshakes),
            get_exec().moe.deepep_mode != "normal",
        )
        compute.configure_staged_capacity(capacity_plan)

    if is_direct_afd_backend(backend) and num_sessions > 1:
        _event_loop_afd_ffn_direct_union(scheduler, handshakes)
        return

    sessions: list[_AFDFFNSession] = [
        _build_ffn_session(scheduler, handshake, compute) for handshake in handshakes
    ]
    if num_sessions > 1 or not sessions[0].plan.enabled:
        # Capture after every backend and executor has initialized communication
        # and selected the compute SM budget.
        compute.capture_compute_graphs()

    poller = None
    scheduler_input_socket = None
    if is_rank0:
        scheduler_input_socket = scheduler.ipc_channels.recv_from_tokenizer
        poller = zmq.Poller()
        poller.register(scheduler_input_socket, zmq.POLLIN)
        for session in sessions:
            session.receiver = AFDSyncMsgReceiver(
                f"tcp://*:{session.handshake.sync_port}"
            )
            poller.register(session.receiver.socket, zmq.POLLIN)

    if num_sessions > 1:
        _event_loop_afd_ffn_cohort(
            scheduler,
            sessions,
            poller,
            scheduler_input_socket,
        )
        if is_rank0:
            logger.info("AFD ffn event loop exited")
        return

    if is_rank0:
        logger.info("AFD ffn event loop started: sessions=%d", num_sessions)
    session = sessions[0]
    while True:
        if is_rank0:
            events = dict(poller.poll())
            if scheduler_input_socket in events:
                session_id = -1
                msg = sock_recv(scheduler_input_socket)
            else:
                session_id = session.session_id
                msg = session.receiver.recv()
            if scheduler.ps.tp_size > 1:
                broadcast_pyobj(
                    [(session_id, msg)], rank, scheduler.tp_cpu_group, src=0
                )
        else:
            ((session_id, msg),) = broadcast_pyobj(
                [], rank, scheduler.tp_cpu_group, src=0
            )

        if session_id == -1:
            if _handle_ffn_scheduler_request(scheduler, msg, is_rank0):
                break
            continue

        if not isinstance(msg, AFDSyncMsg):
            if is_rank0:
                logger.warning(
                    "AFD ffn session %d ignores unsupported control message: %s",
                    session_id,
                    type(msg).__name__,
                )
            continue

        if is_rank0 and should_log_step(msg.step_id):
            _log_ffn_step(session, compute, msg, rank)
        # The executor keeps step order but skips data-plane I/O for a global
        # 0-token batch, matching AFDAttnMoE.
        session.executor.run_step(msg)

    if is_rank0:
        logger.info("AFD ffn event loop exited")
