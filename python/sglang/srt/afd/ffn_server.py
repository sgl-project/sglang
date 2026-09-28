"""Standalone FFN role: one process per lane, each a rank of the FFN TP/EP group."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .config import AFDExecutionMode, execution_mode_from_server_args
from .contracts import AFDError
from .integration import build_ffn_pipeline

logger = logging.getLogger(__name__)


def _resolved_args_receipt(server_args: Any, lane: int, resolved_rank: int) -> dict:
    """Report the actual worker configuration, independently of planned argv."""
    from sglang.srt.arg_groups.overrides import resolving_view

    server_args = resolving_view(server_args)
    return {
        "role": "ffn",
        "lane": lane,
        "rank": resolved_rank,
        "pid": os.getpid(),
        "source": "worker_server_args_after_model_load",
        "speculative_algorithm": server_args.speculative_algorithm,
        "enable_dp_lm_head": server_args.enable_dp_lm_head,
        "moe_runner_backend": server_args.moe_runner_backend,
        "moe_a2a_backend": server_args.moe_a2a_backend,
        "context_length": server_args.context_length,
        "page_size": server_args.page_size,
    }


def _node_split(server_args: Any) -> tuple[int, int, tuple[int, ...]]:
    """Which of the N FFN lanes this host owns.

    N is a property of the MoE layout, not of one machine, so it can exceed a
    host's GPU count -- N=8 needs two 4-GPU hosts. Host h owns the contiguous
    block [h*per_node, (h+1)*per_node), which keeps the global lane ordinal equal
    to the global TP rank and so leaves the wire groups, keyed by FFN ordinal,
    untouched.
    """

    lanes = server_args.afd_config.lanes
    nnodes = int(getattr(server_args, "nnodes", 1) or 1)
    node_rank = int(getattr(server_args, "node_rank", 0) or 0)
    if nnodes < 1 or lanes % nnodes:
        raise AFDError(
            "AFD_FFN_NODE_SPLIT_UNSUPPORTED",
            f"lanes={lanes} nnodes={nnodes}",
        )
    if not 0 <= node_rank < nnodes:
        raise AFDError(
            "AFD_FFN_NODE_RANK_INVALID",
            f"node_rank={node_rank} nnodes={nnodes}",
        )
    per_node = lanes // nnodes
    base = node_rank * per_node
    return nnodes, per_node, tuple(range(base, base + per_node))


def _local_slot(server_args: Any, lane: int) -> int:
    """This host's GPU slot for a global lane ordinal."""

    _, _, owned = _node_split(server_args)
    if lane not in owned:
        raise AFDError(
            "AFD_FFN_LANE_NOT_LOCAL",
            f"lane={lane} owned={owned}",
        )
    return lane - owned[0]


def _init_distributed(*, server_args: Any, device: Any, lane: int) -> None:
    import torch

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # This role never goes through init_torch_distributed, so reuse its flag
    # setter rather than let parallel_state's defaults decide the all-reduce
    # implementation behind the operator's back.
    from sglang.srt.distributed.bootstrap import _set_all_reduce_flags
    from sglang.srt.layers.dp_attention import (
        init_dp_gathered_buffer,
        initialize_dp_attention,
    )
    from sglang.srt.layers.moe import initialize_moe_config

    # Publishing server_args does not initialize the MoE runtime flags. This
    # standalone role must do it before constructing any model or dispatcher.
    initialize_moe_config()
    lanes = server_args.afd_config.lanes
    torch.cuda.set_device(device)
    nnodes, _, _ = _node_split(server_args)
    nccl_port = (
        server_args.nccl_port
        if server_args.nccl_port is not None
        else server_args.afd_config.rendezvous_port + lanes + 1
    )
    if nnodes > 1:
        # The loopback cannot rendezvous a group that spans hosts. The operator
        # supplies the address because the port has to stay clear of both the
        # wire groups and the attention role's own tp group, and that budget
        # already lives with the launcher.
        addr = getattr(server_args, "dist_init_addr", None)
        if not addr:
            raise AFDError(
                "AFD_FFN_DIST_INIT_ADDR_REQUIRED",
                f"lanes={lanes} nnodes={nnodes}",
            )
        init_method = f"tcp://{addr}"
    else:
        init_method = f"tcp://127.0.0.1:{nccl_port}"
    init_distributed_environment(
        backend="nccl",
        world_size=lanes,
        rank=lane,
        local_rank=device.index,
        distributed_init_method=init_method,
        timeout=server_args.dist_timeout,
    )
    _set_all_reduce_flags()
    # The capability gate requires expert sharding across all FFN ranks.
    initialize_model_parallel()
    model_config = ModelConfig.from_server_args(server_args)
    initialize_dp_attention(server_args=server_args)
    init_dp_gathered_buffer(model_config)


def _load_model(*, server_args: Any, device: Any, lane: int) -> tuple[Any, Any]:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.eplb.expert_location import (
        compute_initial_expert_location_metadata,
        set_global_expert_location_metadata,
    )
    from sglang.srt.model_executor.model_runner_components.load_model_utils import (
        build_load_config,
    )
    from sglang.srt.model_loader import get_model

    model_config = ModelConfig.from_server_args(server_args)
    set_global_expert_location_metadata(
        compute_initial_expert_location_metadata(
            model_config=model_config,
            moe_ep_rank=lane,
        )
    )
    model = get_model(
        model_config=model_config,
        load_config=build_load_config(
            server_args=server_args,
            tp_rank=lane,
            remote_instance_weight_transporter_engine=None,
            remote_instance_weight_transporter_session_id="",
            draft_model_idx=None,
            weight_cache_mode="off",
            weight_cache_socket=None,
        ),
        device_config=DeviceConfig("cuda", device.index),
    )
    return model, model_config.dtype


def _close_within_deadline(pipeline: Any, *, seconds: int) -> None:
    """Close the pipeline, or let the kernel end this process trying.

    The graph teardown finishes in a CUDAGraph destructor, a C++ call reached
    through Py_DECREF that never releases the GIL, so neither a joinable worker nor
    a Timer can preempt it -- the idiom transport.close() uses for ncclCommAbort
    works only because ctypes does release the GIL. A signal left at its default
    disposition is delivered by the kernel and needs no interpreter, so it is the
    one deadline that still holds here. Measured cost of having none: four GB300
    GPUs pinned at 100% util for 12 h 22 m after a single broken wave, recoverable
    only by killing the container from outside.

    Dying on the alarm is the good outcome: the exit status is non-zero, so
    _run_lanes reports AFD_FFN_LANE_EXITED instead of waiting forever, and the
    driver reclaims the context.
    """

    import signal

    # The aggregate close already carries two sequential close_timeout_seconds
    # deadlines of its own -- the ack recv in exchange_close and the ncclCommAbort
    # join in transport.close -- so a backstop equal to one of them would fire on a
    # close that was still going to succeed. Give the one unbounded step, the graph
    # teardown, a third unit of the same budget.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, 3 * seconds))
    try:
        pipeline.close()
    finally:
        signal.alarm(0)


def run_lane(server_args: Any, lane: int) -> None:
    """Serve one lane until its attention peer closes the pair."""

    import torch

    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
    from sglang.srt.utils import configure_logger

    # A spawned interpreter inherits no logging configuration, so without this
    # every logger.info in the lane is dropped -- including the role graph's
    # usage snapshots, which makes this role's graph state unobservable.
    configure_logger(server_args, prefix=f" FFN{lane}")
    gpu_id = server_args.base_gpu_id + _local_slot(server_args, lane)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(world_rank=lane, gpu_id=gpu_id),
    )
    device = torch.device("cuda", gpu_id)
    _init_distributed(server_args=server_args, device=device, lane=lane)
    model, dtype = _load_model(server_args=server_args, device=device, lane=lane)
    resolved = get_parallel().tp_group.rank_in_group
    if resolved != lane:
        raise AFDError(
            "AFD_FFN_LANE_RANK_MISMATCH",
            f"lane={lane} tp_rank={resolved}",
        )
    logger.info(
        "AFD_FFN_RESOLVED_ARGS_JSON %s",
        json.dumps(_resolved_args_receipt(server_args, lane, resolved), sort_keys=True),
    )
    pipeline = build_ffn_pipeline(
        model=model,
        config=server_args.afd_config,
        device=device,
        dtype=dtype,
        lane=lane,
    )
    logger.info("AFD FFN lane %d ready", lane)
    try:
        while pipeline.run_once():
            pass
    except BaseException:
        # Emit the original failure before collective teardown can block or
        # replace it with a secondary close error.
        logger.exception("AFD_FFN_RUN_FAILED lane=%d", lane)
        raise
    else:
        logger.info("AFD_FFN_PEER_CLOSE lane=%d", lane)
    finally:
        _close_within_deadline(
            pipeline,
            seconds=server_args.afd_config.close_timeout_seconds,
        )


def _stop_workers(workers: list[Any], *, seconds: float) -> None:
    """Reap only our started children, with a shared deadline per signal phase."""
    import time

    started = [worker for worker in workers if worker.pid is not None]
    for action in ("terminate", "kill"):
        for worker in started:
            if worker.exitcode is None:
                getattr(worker, action)()
        deadline = time.monotonic() + seconds
        for worker in started:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if all(worker.exitcode is not None for worker in started):
            return
    raise AFDError(
        "AFD_FFN_WORKERS_NOT_REAPED",
        str([worker.name for worker in started if worker.exitcode is None]),
    )


def _run_lanes(server_args: Any, *, owned: tuple[int, ...]) -> None:
    """Serve indefinitely; bound startup failure and draining after a lane exits."""
    import multiprocessing
    import time
    from multiprocessing.connection import wait

    context = multiprocessing.get_context("spawn")
    workers = []
    seconds = max(1, server_args.afd_config.close_timeout_seconds)
    try:
        for lane in owned:
            worker = context.Process(target=run_lane, args=(server_args, lane))
            workers.append(worker)
            worker.start()
        pending = {worker.sentinel: worker for worker in workers}
        drain_deadline = None
        while pending:
            timeout = (
                None
                if drain_deadline is None
                else max(0.0, drain_deadline - time.monotonic())
            )
            ready = wait(list(pending), timeout=timeout)
            if not ready:
                raise AFDError("AFD_FFN_DRAIN_TIMEOUT", str(list(pending)))
            for sentinel in ready:
                worker = pending.pop(sentinel)
                if worker.exitcode:
                    raise AFDError(
                        "AFD_FFN_LANE_EXITED",
                        f"worker={worker.name} exitcode={worker.exitcode}",
                    )
            # A peer has finished serving; the others may still be closing their
            # wire and graph resources. Match the lane's three-phase backstop.
            if drain_deadline is None:
                drain_deadline = time.monotonic() + 3 * seconds
    except BaseException:
        try:
            _stop_workers(workers, seconds=seconds)
        except BaseException:
            logger.exception("AFD_FFN_WORKER_CLEANUP_FAILED")
        raise
    else:
        _stop_workers(workers, seconds=seconds)


def launch_server(server_args: Any) -> None:
    """Load only FFN weights and serve until the attention role closes."""

    server_args.check_server_args()
    if execution_mode_from_server_args(server_args) != AFDExecutionMode.FFN:
        raise AFDError("AFD_FFN_EXECUTION_MODE_REQUIRED")
    from sglang.srt.entrypoints.engine import _set_envs_and_config

    # Match attention's runtime defaults before any NCCL initialization. Spawned
    # lanes must inherit the same cuMem/IPC policy as the in-process lane.
    _set_envs_and_config(server_args)
    nnodes, per_node, owned = _node_split(server_args)
    logger.info(
        "AFD FFN role: %d lanes over %d node(s), this host owns %s",
        server_args.afd_config.lanes,
        nnodes,
        list(owned),
    )
    if per_node == 1:
        run_lane(server_args, owned[0])
        return
    _run_lanes(server_args, owned=owned)
