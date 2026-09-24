"""FFN-side AFD lifecycle and shared-worker session scheduling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import zmq

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.afd_handshake import (
    AFDGraphCapturePlan,
    AFDHandshakeResult,
    AFDHandshakeServer,
    afd_handshake_port,
    build_afd_ffn_capacity_plan,
    build_afd_graph_capture_plan,
)
from sglang.srt.afd.afd_sync import AFDSyncMsg, AFDSyncMsgReceiver
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.dispatcher import is_afd_fixed_shape_mode
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
    gather_nvshmem_topology,
    init_afd_comm_backend,
    should_log_step,
)
from sglang.srt.managers.io_struct import (
    FreezeGCReq,
    ProfileReq,
    ProfileReqOutput,
    ShutdownReq,
    sock_recv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_disagg, get_exec, get_serving
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def init_ffn_model_worker_state(scheduler: Scheduler) -> None:
    """Export worker state without allocating attention or KV-cache resources."""
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
    scheduler.max_total_num_tokens = 0
    scheduler.max_req_input_len = 0
    scheduler.kv_cache_allocation_time = 0.0


def finish_ffn_scheduler_init(scheduler: Scheduler) -> None:
    """Finish the lifecycle state needed by the FFN event loop and process exit."""
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )

    scheduler.device_module = torch.get_device_module(scheduler.device)
    scheduler.afd_ffn_compute = AFDFFNCompute(scheduler)
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
            "AFD ffn scheduler ignores unsupported request: %s", type(req).__name__
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


def _build_ffn_session(
    scheduler: Scheduler, handshake: AFDHandshakeResult, compute
) -> _AFDFFNSession:
    plan = build_afd_graph_capture_plan(handshake)
    backend = init_afd_comm_backend(scheduler, AFDRole.FFN, handshake)
    executor = AFDFFNExecutor(scheduler, backend, compute, handshake)
    if not plan.enabled:
        compute.capture_compute_graphs()
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
        if backend.decode_transport is not None or msg.communication_graph:
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
    compute_graph = graph_capacity is not None
    logger.info(
        "AFD step: side=ffn path=staged session=%d session_step=%d mode=%s "
        "tokens_per_rank=%s total_tokens=%d wire=%d children=%s "
        "communication_graph=%s compute_graph=%s compute_capacity=%d",
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

    rank_node_ids, fabric_keys = gather_nvshmem_topology(scheduler)

    # Negotiate shared capacity before any session captures graphs.
    if is_rank0:
        local_info = build_local_handshake_info(
            scheduler,
            AFDRole.FFN,
            rank_node_ids,
            fabric_keys,
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

    capacity_plan = build_afd_ffn_capacity_plan(
        handshakes[0].local,
        tuple(handshake.peer for handshake in handshakes),
        use_deepep_low_latency=get_exec().moe.deepep_mode != "normal",
    )
    compute.configure_staged_capacity(capacity_plan)

    sessions: list[_AFDFFNSession] = [
        _build_ffn_session(scheduler, handshake, compute) for handshake in handshakes
    ]

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

    if is_rank0:
        logger.info("AFD ffn event loop started: sessions=%d", num_sessions)
    try:
        next_session_index = 0
        while True:
            if is_rank0:
                events = dict(poller.poll())
                if scheduler_input_socket in events:
                    session_id = -1
                    msg = sock_recv(scheduler_input_socket)
                else:
                    # Round-robin among ready sessions so one busy attn instance
                    # cannot starve the others.
                    session_id = None
                    for probe in range(num_sessions):
                        session = sessions[(next_session_index + probe) % num_sessions]
                        if session.receiver.socket in events:
                            session_id = session.session_id
                            next_session_index = (session_id + 1) % num_sessions
                            msg = session.receiver.recv()
                            break
                    if session_id is None:
                        raise RuntimeError("AFD ffn poller returned no readable socket")
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

            session = sessions[session_id]
            if is_rank0 and should_log_step(msg.step_id):
                _log_ffn_step(session, compute, msg, rank)
            # The executor keeps step order but skips data-plane I/O for a global
            # 0-token batch, matching AFDAttnMoE.
            session.executor.run_step(msg)

    finally:
        for session in sessions:
            if session.receiver is not None:
                session.receiver.close()
            session.backend.close()

    if is_rank0:
        logger.info("AFD ffn event loop exited")
