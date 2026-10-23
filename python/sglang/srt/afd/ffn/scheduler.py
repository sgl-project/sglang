"""FFN-side AFD lifecycle and single-session event loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import zmq

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.afd_handshake import (
    AFDHandshakeServer,
    afd_handshake_port,
    build_afd_ffn_capacity_plan,
)
from sglang.srt.afd.afd_sync import AFDSyncMsgReceiver
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
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
from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_serving
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def init_ffn_model_worker_state(scheduler: Scheduler) -> None:
    """Export worker state without allocating attention or KV-cache resources."""
    if get_model().is_startup_weight_load_overlap:
        scheduler.tp_worker.start_startup_weight_load()
        scheduler.tp_worker.finalize_startup_weight_load()
    scheduler.draft_worker = None
    scheduler.model_worker = scheduler.tp_worker
    scheduler.device = scheduler.tp_worker.device
    scheduler.forward_stream = scheduler.tp_worker.model_runner.forward_stream
    scheduler.tp_group = get_parallel().tp_group
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
    scheduler.schedule_stream = scheduler.forward_stream
    scheduler.afd_ffn_compute = AFDFFNCompute(scheduler)
    scheduler.metrics_reporter = SchedulerMetricsReporter(
        scheduler=scheduler,
        metrics_collector_context=scheduler.metrics_collector_context,
        metrics_collector=scheduler.metrics_collector,
    )
    scheduler.rust_server = None
    scheduler.async_d2h_worker = None
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


def event_loop_afd_ffn(scheduler: Scheduler) -> None:
    rank = get_parallel().tp_rank
    is_rank0 = rank == 0
    if is_rank0:
        handshake_server = AFDHandshakeServer(afd_handshake_port(get_serving().port))
        try:
            handshake = handshake_server.accept(
                build_local_handshake_info(scheduler, AFDRole.FFN)
            )
        finally:
            handshake_server.close()
        if get_parallel().tp_size > 1:
            broadcast_pyobj([handshake], rank, scheduler.tp_cpu_group, src=0)
    else:
        (handshake,) = broadcast_pyobj([], rank, scheduler.tp_cpu_group, src=0)

    capacity_plan = build_afd_ffn_capacity_plan(
        handshake.local,
        handshake.peer,
        use_deepep_low_latency=get_exec().moe.deepep_mode != "normal",
    )
    scheduler.afd_ffn_compute.configure_native_deepep_dispatch_limit(
        capacity_plan.deepep_dispatch_limit
    )
    backend = init_afd_comm_backend(scheduler, AFDRole.FFN, handshake)
    executor = AFDFFNExecutor(scheduler, backend, scheduler.afd_ffn_compute)
    backend.startup_barrier()

    receiver = None
    poller = None
    scheduler_input_socket = None
    if is_rank0:
        receiver = AFDSyncMsgReceiver(f"tcp://*:{handshake.sync_port}")
        scheduler_input_socket = scheduler.ipc_channels.recv_from_tokenizer
        poller = zmq.Poller()
        poller.register(scheduler_input_socket, zmq.POLLIN)
        poller.register(receiver.socket, zmq.POLLIN)
        logger.info("AFD ffn event loop started")

    try:
        while True:
            if is_rank0:
                events = dict(poller.poll())
                if scheduler_input_socket in events:
                    action = (False, sock_recv(scheduler_input_socket))
                else:
                    action = (True, receiver.recv())
                if get_parallel().tp_size > 1:
                    broadcast_pyobj([action], rank, scheduler.tp_cpu_group, src=0)
            else:
                (action,) = broadcast_pyobj([], rank, scheduler.tp_cpu_group, src=0)

            is_step, msg = action
            if not is_step:
                if _handle_ffn_scheduler_request(scheduler, msg, is_rank0):
                    break
                continue
            if is_rank0 and should_log_step(msg.step_id):
                logger.info(
                    "AFD step: side=ffn step=%d mode=%s tokens_per_rank=%s wire=%d",
                    msg.step_id,
                    ForwardMode(msg.forward_mode).name,
                    msg.tokens_per_rank,
                    msg.wire_capacity,
                )
            executor.run_step(msg)
    finally:
        if receiver is not None:
            receiver.close()
        backend.close()
    if is_rank0:
        logger.info("AFD ffn event loop exited")
