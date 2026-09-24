"""Attention-side AFD initialization, batch synchronization, and graph decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.afd.afd_config import AFDRole, is_afd_attn_moe_scattered
from sglang.srt.afd.afd_handshake import (
    AFDGraphCapturePlan,
    AFDHandshakeResult,
    build_afd_graph_capture_plan,
    run_handshake_attn,
)
from sglang.srt.afd.afd_sync import AFDSyncMsg, AFDSyncMsgSender
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.dispatcher import (
    AFDDispatcher,
    get_afd_dispatcher,
    select_token_bucket,
    set_afd_dispatcher,
)
from sglang.srt.afd.layers import collect_afd_moe_layers
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
    init_afd_comm_backend,
    should_log_step,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _capture_afd_attn_communication_graphs(
    scheduler: Scheduler,
    backend: AFDCommBackend,
    dispatcher: AFDDispatcher,
    plan: AFDGraphCapturePlan,
) -> None:
    model_runner = scheduler.tp_worker.model_runner
    if not plan.enabled:
        model_runner.init_cuda_graphs()
        return

    if plan.attn_graph_variants:
        dispatcher.enable_communication_graph_variants(plan.attn_graph_variants)

    backend.startup_barrier()
    model_runner.init_cuda_graphs()
    backend.startup_barrier()
    logger.info(
        "AFD attn communication graphs ready: full_steps=%d",
        len(plan.attn_graph_variants),
    )


@dataclass(frozen=True)
class AFDAttnGraphDecision:
    use_local_model_graph: bool
    use_paired_communication_graph: bool


class AFDSyncSender:
    """Attn side. Owned by the global sender rank; None elsewhere."""

    def __init__(self, handshake: AFDHandshakeResult):
        self._msg_sender = AFDSyncMsgSender(
            f"tcp://{handshake.dist_host}:{handshake.sync_port}"
        )
        self._pending: Optional[AFDSyncMsg] = None

    def prepare_for_batch(
        self,
        step_id: int,
        tokens_per_rank: list[int],
        forward_mode: ForwardMode,
        wire_capacity: int,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("AFD control message from the previous batch is pending")
        self._pending = AFDSyncMsg(
            step_id=step_id,
            tokens_per_rank=tokens_per_rank,
            forward_mode=int(forward_mode),
            wire_capacity=wire_capacity,
        )

    def send_prepared(self, communication_graph: bool) -> None:
        if self._pending is None:
            raise RuntimeError("AFD has no prepared control message to send")
        msg = self._pending
        msg.communication_graph = communication_graph
        if should_log_step(msg.step_id):
            logger.info(
                "AFD step: side=attn step=%d mode=%s tokens_per_rank=%s "
                "wire=%d communication_graph=%s",
                msg.step_id,
                ForwardMode(msg.forward_mode).name,
                msg.tokens_per_rank,
                msg.wire_capacity,
                msg.communication_graph,
            )
        self._msg_sender.send(msg)
        self._pending = None


def _decide_afd_attn_graph(
    can_run_graph: bool,
    dispatcher: AFDDispatcher,
) -> AFDAttnGraphDecision:
    use_paired_communication_graph = dispatcher.select_paired_communication_graph(
        can_run_graph
    )
    return AFDAttnGraphDecision(
        use_local_model_graph=use_paired_communication_graph,
        use_paired_communication_graph=use_paired_communication_graph,
    )


def _tokens_per_rank(batch: ScheduleBatch, num_attn_ranks: int) -> list[int]:
    # Only mirror cross-DP padding when the model uses that layout. A replayed
    # graph may add a separate fixed-stride layer through wire_capacity.
    model_uses_cross_dp_layout = (
        batch.global_num_tokens is not None and len(batch.global_num_tokens) > 1
    )
    if batch.global_num_tokens_across_dp is not None:
        tokens = list(batch.global_num_tokens_across_dp)
    elif batch.global_num_tokens is not None:
        tokens = list(batch.global_num_tokens)
    elif batch.forward_mode.is_idle():
        tokens = [0]
    elif batch.forward_mode.is_decode():
        tokens = [batch.batch_size()]
    else:
        tokens = [batch.extend_num_tokens]

    if len(tokens) != num_attn_ranks:
        raise RuntimeError(
            f"tokens_per_rank has {len(tokens)} entries but the attn instance "
            f"has {num_attn_ranks} ranks"
        )
    if model_uses_cross_dp_layout and len(tokens) > 1:
        from sglang.srt.layers.dp_attention import DpPaddingMode

        padding_mode = DpPaddingMode.get_dp_padding_mode(
            batch.is_extend_in_batch, tokens
        )
        if padding_mode.is_max_len():
            tokens = [max(tokens)] * len(tokens)
    return tokens


def _effective_forward_mode(
    batch: ScheduleBatch, tokens_per_rank: Sequence[int]
) -> ForwardMode:
    if batch.global_forward_mode is not None:
        return ForwardMode(batch.global_forward_mode)
    if batch.global_num_tokens is None:
        return batch.forward_mode
    if batch.is_extend_in_batch:
        return ForwardMode.EXTEND
    if sum(tokens_per_rank) > 0:
        return ForwardMode.DECODE
    return ForwardMode.IDLE


def prepare_afd_attn_batch(scheduler: Scheduler, batch: ScheduleBatch) -> None:
    num_attn_ranks = scheduler.afd_handshake.local.num_ranks
    tokens_per_rank = _tokens_per_rank(batch, num_attn_ranks)
    forward_mode = _effective_forward_mode(batch, tokens_per_rank)
    scheduler.tp_worker.model_runner.afd_cuda_graph_batch_size = (
        max(tokens_per_rank) if forward_mode.is_decode() else None
    )
    scattered = is_afd_attn_moe_scattered()
    dp_capacity = (
        tokens_per_rank[scheduler.ps.tp_rank] if scattered else sum(tokens_per_rank)
    )
    if forward_mode.is_decode():
        minimum_wire_capacity = (
            max(tokens_per_rank)
            if scattered
            else (
                max(tokens_per_rank) * num_attn_ranks
                if get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
                else dp_capacity
            )
        )
        wire_capacity = select_token_bucket(
            scheduler.afd_handshake.local.decode_token_buckets,
            minimum_wire_capacity,
        )
        control_wire_capacity = wire_capacity
    else:
        wire_capacity = dp_capacity
        control_wire_capacity = (
            max(tokens_per_rank, default=0) if scattered else wire_capacity
        )
    step_id = get_afd_dispatcher().begin_batch(
        forward_mode=forward_mode,
        dp_capacity=dp_capacity,
        wire_capacity=wire_capacity,
    )
    if scheduler._afd_sync_sender is not None:
        scheduler._afd_sync_sender.prepare_for_batch(
            step_id=step_id,
            tokens_per_rank=tokens_per_rank,
            forward_mode=forward_mode,
            wire_capacity=control_wire_capacity,
        )


def init_afd_attn(
    scheduler: Scheduler,
) -> tuple[AFDHandshakeResult, Optional[AFDSyncSender]]:
    rank = scheduler.ps.tp_rank
    if rank == 0:
        handshake = run_handshake_attn(
            get_disagg().afd_ffn_addr,
            build_local_handshake_info(scheduler, AFDRole.ATTN),
        )
        if scheduler.ps.tp_size > 1:
            broadcast_pyobj([handshake], rank, get_parallel().tp_group.cpu_group, src=0)
    else:
        (handshake,) = broadcast_pyobj(
            [], rank, get_parallel().tp_group.cpu_group, src=0
        )

    plan = build_afd_graph_capture_plan(handshake)
    backend = init_afd_comm_backend(scheduler, AFDRole.ATTN, handshake)
    model_runner = scheduler.tp_worker.model_runner
    dispatcher = AFDDispatcher(
        comm=backend,
        layers=collect_afd_moe_layers(model_runner.model, AFDRole.ATTN),
        token_buckets=handshake.local.decode_token_buckets,
        dtype=model_runner.dtype,
        device=torch.device(backend.config.device),
        scattered=handshake.local.afd_ffn_moe_backend == "deepep",
    )
    set_afd_dispatcher(dispatcher)
    _capture_afd_attn_communication_graphs(
        scheduler,
        backend,
        dispatcher,
        plan,
    )
    sender = AFDSyncSender(handshake) if rank == 0 else None

    def select_local_model_graph(can_run_graph: bool) -> bool:
        decision = _decide_afd_attn_graph(can_run_graph, dispatcher)
        if sender is not None:
            sender.send_prepared(decision.use_paired_communication_graph)
        return decision.use_local_model_graph

    # Every A rank makes the same dispatcher decision. Only rank 0 owns the
    # control-plane sender and publishes it to F.
    model_runner.afd_control_callback = select_local_model_graph
    return handshake, sender
