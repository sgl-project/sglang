"""Attention-side AFD initialization and batch synchronization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.afd.afd_config import AFDRole, is_afd_attn_moe_scattered
from sglang.srt.afd.afd_handshake import AFDHandshakeResult, run_handshake_attn
from sglang.srt.afd.afd_sync import AFDSyncMsg, AFDSyncMsgSender
from sglang.srt.afd.dispatcher import (
    AFDDispatcher,
    get_afd_dispatcher,
    set_afd_dispatcher,
)
from sglang.srt.afd.layers import collect_afd_moe_layers
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
    init_afd_comm_backend,
    should_log_step,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class AFDSyncSender:
    """Send one control message per batch from attention rank zero."""

    def __init__(self, handshake: AFDHandshakeResult):
        self._msg_sender = AFDSyncMsgSender(
            f"tcp://{handshake.dist_host}:{handshake.sync_port}"
        )

    def send(self, msg: AFDSyncMsg) -> None:
        if should_log_step(msg.step_id):
            logger.info(
                "AFD step: side=attn step=%d mode=%s tokens_per_rank=%s wire=%d",
                msg.step_id,
                ForwardMode(msg.forward_mode).name,
                msg.tokens_per_rank,
                msg.wire_capacity,
            )
        self._msg_sender.send(msg)


def _tokens_per_rank(batch: ScheduleBatch, num_attn_ranks: int) -> list[int]:
    # Mirror the native MoE cross-DP layout, including max-length padding.
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
    tokens_per_rank = _tokens_per_rank(batch, scheduler.afd_handshake.local.num_ranks)
    forward_mode = _effective_forward_mode(batch, tokens_per_rank)
    scattered = is_afd_attn_moe_scattered()
    dp_capacity = (
        tokens_per_rank[get_parallel().tp_rank] if scattered else sum(tokens_per_rank)
    )
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
        scheduler._afd_sync_sender.send(
            AFDSyncMsg(
                step_id=step_id,
                tokens_per_rank=tokens_per_rank,
                forward_mode=int(forward_mode),
                wire_capacity=control_wire_capacity,
            )
        )


def init_afd_attn(
    scheduler: Scheduler,
) -> tuple[AFDHandshakeResult, Optional[AFDSyncSender]]:
    rank = get_parallel().tp_rank
    if rank == 0:
        handshake = run_handshake_attn(
            get_disagg().afd_ffn_addr,
            build_local_handshake_info(scheduler, AFDRole.ATTN),
        )
        if get_parallel().tp_size > 1:
            broadcast_pyobj([handshake], rank, get_parallel().tp_group.cpu_group, src=0)
    else:
        (handshake,) = broadcast_pyobj(
            [], rank, get_parallel().tp_group.cpu_group, src=0
        )

    backend = init_afd_comm_backend(scheduler, AFDRole.ATTN, handshake)
    model_runner = scheduler.tp_worker.model_runner
    set_afd_dispatcher(
        AFDDispatcher(
            comm=backend,
            layers=collect_afd_moe_layers(model_runner.model, AFDRole.ATTN),
            dtype=model_runner.dtype,
            device=torch.device(backend.config.device),
            scattered=handshake.local.afd_ffn_moe_backend == "deepep",
        )
    )
    backend.startup_barrier()
    sender = AFDSyncSender(handshake) if rank == 0 else None
    return handshake, sender
