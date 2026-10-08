"""Attention-side AFD scheduler hooks, graph decisions, and Direct coordination."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.afd.afd_config import (
    AFDRole,
    is_direct_afd_backend,
    resolve_overlap_sm_partition,
)
from sglang.srt.afd.afd_handshake import (
    AFDGraphCapturePlan,
    AFDHandshakeResult,
    build_afd_attn_local_graph_variants,
    build_afd_graph_capture_plan,
    run_handshake_attn,
)
from sglang.srt.afd.afd_sync import (
    AFDDirectCommonPlan,
    AFDDirectControlClient,
    AFDDirectProposal,
    AFDSyncMsg,
    AFDSyncMsgSender,
)
from sglang.srt.afd.comm import (
    AFDCommBackend,
)
from sglang.srt.afd.dispatcher import (
    AFDDispatcher,
    get_afd_dispatcher,
    is_afd_fixed_shape_mode,
    select_token_bucket,
    set_afd_dispatcher,
    split_afd_batch_size,
)
from sglang.srt.afd.layers import collect_afd_moe_layers
from sglang.srt.afd.scheduler_common import (
    build_local_handshake_info,
    gather_nvshmem_topology,
    init_afd_comm_backend,
    should_log_step,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_parallel,
)
from sglang.srt.utils.common import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.afd.afd_handshake import AFDGraphCaptureVariant
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


def _init_afd_attn_dispatcher(
    scheduler: Scheduler,
    backend: AFDCommBackend,
    handshake: AFDHandshakeResult,
) -> AFDDispatcher:
    layers = collect_afd_moe_layers(
        scheduler.tp_worker.model_runner.model, AFDRole.ATTN
    )
    direct_deepep_lanes = ()
    direct_megamoe = None
    if handshake.local.afd_backend == "direct_deepep":
        from sglang.srt.afd.direct_deepep import (
            build_afd_direct_deepep_adapters,
        )

        direct_deepep_lanes = build_afd_direct_deepep_adapters(
            comm=backend,
            layers=layers,
            token_buckets=handshake.local.decode_token_buckets,
            num_lanes=handshake.local.decode_num_microbatches,
            tokens_per_batch=handshake.local.decode_num_tokens_per_bs,
            dtype=scheduler.tp_worker.model_runner.dtype,
            comm_sm_budget=handshake.peer.afd_comm_sm_budget,
            nvfp4_input_scales=handshake.peer.direct_nvfp4_input_scales,
            fp8_scale_ue8m0=handshake.peer.direct_deepep_fp8_scale_ue8m0,
            expert_alignment=handshake.peer.direct_deepep_expert_alignment,
            allow_hybrid_mode=handshake.peer.direct_deepep_allow_hybrid,
        )
    elif handshake.local.afd_backend == "direct_megamoe":
        from sglang.srt.afd.direct_megamoe import (
            build_afd_direct_megamoe_adapter,
        )

        direct_megamoe = build_afd_direct_megamoe_adapter(
            comm=backend,
            layers=layers,
            token_buckets=handshake.local.decode_token_buckets,
            num_lanes=handshake.local.decode_num_microbatches,
            dtype=scheduler.tp_worker.model_runner.dtype,
            comm_sm_budget=handshake.peer.afd_comm_sm_budget,
            nvfp4_input_scales=handshake.peer.direct_nvfp4_input_scales,
        )

    compute_num_sms = None
    num_microbatches = handshake.local.decode_num_microbatches
    if direct_deepep_lanes and num_microbatches > 1:
        overlap_sm_budget = handshake.peer.afd_comm_sm_budget
        assert overlap_sm_budget is not None
        total_num_sms = torch.cuda.get_device_properties(
            backend.config.device
        ).multi_processor_count
        compute_num_sms, communication_num_sms = resolve_overlap_sm_partition(
            total_num_sms=total_num_sms,
            num_microbatches=num_microbatches,
            overlap_sm_budget=overlap_sm_budget,
            direct_deepep_communication_num_sms_per_lane=(
                direct_deepep_lanes[0].overlap_communication_num_sms
            ),
        )
        if scheduler.ps.tp_rank == 0:
            logger.info(
                "AFD attn overlap SM partition: backend=%s compute=%d communication=%d",
                handshake.local.afd_backend,
                compute_num_sms,
                communication_num_sms,
            )

    dispatcher = AFDDispatcher(
        comm=backend,
        layers=layers,
        token_buckets=handshake.local.decode_token_buckets,
        dtype=scheduler.tp_worker.model_runner.dtype,
        device=torch.device(backend.config.device),
        scattered=handshake.local.afd_ffn_moe_backend == "deepep",
        num_slots=handshake.local.decode_num_microbatches,
        direct_deepep_lanes=direct_deepep_lanes,
        direct_megamoe=direct_megamoe,
        compute_num_sms=compute_num_sms,
    )
    set_afd_dispatcher(dispatcher)
    return dispatcher


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
    transport = backend.decode_transport

    if plan.attn_graph_variants:
        dispatcher.enable_communication_graph_variants(plan.attn_graph_variants)

    backend.startup_barrier()
    model_runner.init_cuda_graphs()
    backend.startup_barrier()

    if transport is None:
        logger.info(
            "AFD attn communication graphs ready: full_steps=%d ffn_participants=0",
            len(plan.attn_graph_variants),
        )
        return

    attn_capacities = {
        variant.attn_wire_capacity for variant in plan.attn_graph_variants
    }
    ffn_only_capacities = [
        wire_capacity
        for wire_capacity, _ in plan.ffn_graph_variants
        if wire_capacity not in attn_capacities
    ]
    temporary_graphs = []
    if ffn_only_capacities:
        with torch.inference_mode(), graph_capture() as ctx:
            for wire_capacity in ffn_only_capacities:
                for _ in range(2):
                    scheduler.device_module.synchronize()
                    get_parallel().tp_group.barrier()
                    dispatcher.warmup_decode_transport(wire_capacity)
                temporary_graphs.append(
                    transport.capture_wait_graph(
                        dispatcher.num_layers,
                        ctx.stream,
                    )
                )
        scheduler.device_module.synchronize()
        get_parallel().tp_group.barrier()
    backend.startup_barrier()

    with graph_capture() as ctx:
        helper = transport.capture_wait_graph(1, ctx.stream)
    scheduler.device_module.synchronize()
    get_parallel().tp_group.barrier()
    transport.install_wait_helper(helper)
    backend.startup_barrier()

    logger.info(
        "AFD attn communication graphs ready: full_steps=%d ffn_participants=%d",
        len(plan.attn_graph_variants),
        len(temporary_graphs),
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
        self._session_id = handshake.session_id
        self._path = (
            "direct" if is_direct_afd_backend(handshake.local.afd_backend) else "staged"
        )
        self._pending: Optional[AFDSyncMsg] = None

    def prepare_for_batch(
        self,
        step_id: int,
        tokens_per_rank: list[int],
        forward_mode: ForwardMode,
        wire_capacity: int,
        child_wire_capacities: tuple[int, ...] = (),
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("AFD control message from the previous batch is pending")
        self._pending = AFDSyncMsg(
            step_id=step_id,
            tokens_per_rank=tokens_per_rank,
            forward_mode=int(forward_mode),
            wire_capacity=wire_capacity,
            child_wire_capacities=child_wire_capacities,
        )

    def send_prepared(self, communication_graph: bool) -> None:
        if self._pending is None:
            raise RuntimeError("AFD has no prepared control message to send")
        msg = self._pending
        msg.communication_graph = communication_graph
        if should_log_step(msg.step_id):
            logger.info(
                "AFD step: side=attn path=%s session=%d session_step=%d "
                "mode=%s tokens_per_rank=%s total_tokens=%d wire=%d children=%s "
                "communication_graph=%s",
                self._path,
                self._session_id,
                msg.step_id,
                ForwardMode(msg.forward_mode).name,
                msg.tokens_per_rank,
                sum(msg.tokens_per_rank),
                msg.wire_capacity,
                msg.child_wire_capacities,
                msg.communication_graph,
            )
        self._msg_sender.send(msg)
        self._pending = None


@dataclass(frozen=True)
class _AFDPreparedAttnStep:
    tokens_per_rank: tuple[int, ...]
    forward_mode: ForwardMode
    dp_capacity: int
    wire_capacity: int
    control_wire_capacity: int
    child_wire_capacities: tuple[int, ...]
    paired_graph_preflight: bool


class _AFDDirectAttnCoordinator:
    def __init__(
        self,
        scheduler: Scheduler,
        handshake: AFDHandshakeResult,
        dispatcher: AFDDispatcher,
    ) -> None:
        if handshake.attn_members is None or len(handshake.attn_members) <= 1:
            raise RuntimeError("AFD Direct coordinator requires multiple A instances")
        self._scheduler = scheduler
        self._handshake = handshake
        self._dispatcher = dispatcher
        self._rank = scheduler.ps.tp_rank
        self._session_id = handshake.session_id
        self._client = (
            AFDDirectControlClient(
                f"tcp://{handshake.dist_host}:{handshake.sync_port}",
                handshake.session_id,
            )
            if self._rank == 0
            else None
        )
        self._graph_eligible = torch.empty(1, dtype=torch.int32)
        self._pending: Optional[_AFDPreparedAttnStep] = None

    def stage(self, prepared: _AFDPreparedAttnStep) -> None:
        if self._pending is not None:
            raise RuntimeError("AFD Direct attn step is already pending")
        self._pending = prepared

    def coordinate(self, can_run_graph: bool) -> bool:
        prepared = self._pending
        if prepared is None:
            raise RuntimeError("AFD Direct has no prepared attn step")
        proposal_can_run_graph = can_run_graph and prepared.paired_graph_preflight
        if self._scheduler.ps.tp_size > 1:
            self._graph_eligible.fill_(int(proposal_can_run_graph))
            self._scheduler.tp_cpu_group.allreduce(
                self._graph_eligible,
                torch.distributed.ReduceOp.MIN,
            ).wait()
            proposal_can_run_graph = bool(self._graph_eligible.item())
        if self._client is not None:
            self._send_active_proposal(prepared, proposal_can_run_graph)

        while True:
            plan = self._broadcast_plan(block=True)
            assert plan is not None
            if plan.active_mask[self._session_id]:
                self._pending = None
                return self._begin_selected_step(prepared, plan)
            self._run_inactive_plan(plan)
            if self._client is not None and not self._client.has_outstanding_proposal:
                self._send_active_proposal(prepared, proposal_can_run_graph)

    def on_idle(self) -> None:
        if self._pending is not None:
            raise RuntimeError("AFD Direct cannot enter idle with pending work")
        if self._client is not None:
            if not self._client.has_outstanding_proposal:
                self._send_idle_proposal()
        plan = self._broadcast_plan(block=False)
        if plan is not None:
            if plan.active_mask[self._session_id]:
                raise RuntimeError("AFD Direct selected an idle A instance")
            self._run_inactive_plan(plan)

    def _broadcast_plan(self, *, block: bool) -> Optional[AFDDirectCommonPlan]:
        if self._client is not None:
            plan = (
                self._client.recv_plan() if block or self._client.poll_plan(1) else None
            )
        else:
            plan = None
        if self._scheduler.ps.tp_size > 1:
            if self._rank == 0:
                (plan,) = broadcast_pyobj(
                    [plan],
                    self._rank,
                    self._scheduler.tp_cpu_group,
                    src=0,
                )
            else:
                (plan,) = broadcast_pyobj(
                    [],
                    self._rank,
                    self._scheduler.tp_cpu_group,
                    src=0,
                )
        if self._rank == 0 and plan is not None and should_log_step(plan.union_step_id):
            pending_mode = (
                self._pending.forward_mode
                if self._pending is not None
                else ForwardMode.IDLE
            )
            pending_tokens = (
                self._pending.tokens_per_rank
                if self._pending is not None
                else (0,) * self._handshake.local.num_ranks
            )
            logger.info(
                "AFD step: side=attn path=direct session=%d union_step=%d "
                "active=%s "
                "union_mode=%s pending_mode=%s pending_tokens_per_rank=%s "
                "parent_wire=%d wire=%d children=%s communication_graph=%s",
                self._session_id,
                plan.union_step_id,
                plan.active_mask[self._session_id],
                ForwardMode(plan.forward_mode).name,
                pending_mode.name,
                pending_tokens,
                plan.parent_wire_capacity,
                plan.wire_capacity,
                plan.child_wire_capacities,
                plan.communication_graph,
            )
        return plan

    def _send_active_proposal(
        self,
        prepared: _AFDPreparedAttnStep,
        can_run_graph: bool,
    ) -> None:
        assert self._client is not None
        self._client.send_proposal(
            AFDDirectProposal(
                proposal_id=self._client.next_proposal_id,
                tokens_per_rank=prepared.tokens_per_rank,
                forward_mode=int(prepared.forward_mode),
                wire_capacity=prepared.control_wire_capacity,
                can_run_graph=can_run_graph,
                child_wire_capacities=prepared.child_wire_capacities,
            )
        )

    def _send_idle_proposal(self) -> None:
        assert self._client is not None
        self._client.send_proposal(
            AFDDirectProposal(
                proposal_id=self._client.next_proposal_id,
                tokens_per_rank=(0,) * self._handshake.local.num_ranks,
                forward_mode=int(ForwardMode.IDLE),
                wire_capacity=0,
            )
        )

    def _begin_selected_step(
        self,
        prepared: _AFDPreparedAttnStep,
        plan: AFDDirectCommonPlan,
    ) -> bool:
        if not is_afd_fixed_shape_mode(prepared.forward_mode):
            self._dispatcher.begin_batch(
                forward_mode=prepared.forward_mode,
                dp_capacity=prepared.dp_capacity,
                wire_capacity=prepared.wire_capacity,
            )
            return False

        child_wire_capacities = plan.child_wire_capacities
        graph_variant = None
        if plan.communication_graph and len(child_wire_capacities) > 1:
            graph_variant = self._dispatcher.select_graph_variant(
                plan.parent_wire_capacity
            )
            if (
                graph_variant is None
                or graph_variant.child_wire_capacities != child_wire_capacities
            ):
                raise RuntimeError(
                    "AFD Direct common plan has no matching local graph variant: "
                    f"children={child_wire_capacities}"
                )
        self._dispatcher.begin_batch(
            forward_mode=prepared.forward_mode,
            dp_capacity=prepared.dp_capacity,
            wire_capacity=plan.wire_capacity,
            child_wire_capacities=child_wire_capacities,
            graph_variant=graph_variant,
        )
        if plan.communication_graph:
            num_tokens_per_bs = self._handshake.local.decode_num_tokens_per_bs
            graph_batch_size = (
                graph_variant.attn_batch_size
                if graph_variant is not None
                else plan.wire_capacity // num_tokens_per_bs
            )
            self._scheduler.tp_worker.model_runner.afd_cuda_graph_batch_size = (
                graph_batch_size
            )
        return _decide_afd_attn_graph(
            plan.communication_graph,
            self._dispatcher,
            False,
        ).use_local_model_graph

    def _run_inactive_plan(self, plan: AFDDirectCommonPlan) -> None:
        forward_mode = ForwardMode(plan.forward_mode)
        if is_afd_fixed_shape_mode(forward_mode):
            # Inactive ranks preserve the active graphs' direct collective
            # order without N1 host polling.
            self._dispatcher.run_inactive_direct_step(
                forward_mode,
                plan.wire_capacity,
                plan.child_wire_capacities,
                do_cpu_sync=False if plan.communication_graph else None,
            )


def _decide_afd_attn_graph(
    can_run_graph: bool,
    dispatcher: AFDDispatcher,
    force_eager_communication: bool,
) -> AFDAttnGraphDecision:
    use_paired_communication_graph = dispatcher.select_paired_communication_graph(
        False if force_eager_communication else can_run_graph
    )
    if dispatcher.current_num_children > 1:
        use_local_model_graph = dispatcher.select_parent_graph_variant(
            can_run_graph
            if force_eager_communication
            else use_paired_communication_graph
        )
    else:
        use_local_model_graph = (
            can_run_graph
            if force_eager_communication
            else use_paired_communication_graph
        )
    return AFDAttnGraphDecision(
        use_local_model_graph=use_local_model_graph,
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


def _select_child_wire_capacities(
    tokens_per_rank: Sequence[int],
    token_buckets: Sequence[int],
    num_microbatches: int,
    *,
    scattered: bool,
    num_tokens_per_bs: int = 1,
) -> tuple[int, ...]:
    logical_batch_sizes = [
        num_tokens // num_tokens_per_bs for num_tokens in tokens_per_rank
    ]
    if any(batch_size < num_microbatches for batch_size in logical_batch_sizes):
        return ()
    per_rank_child_sizes = [
        tuple(
            child_batch_size * num_tokens_per_bs
            for child_batch_size in split_afd_batch_size(batch_size, num_microbatches)
        )
        for batch_size in logical_batch_sizes
    ]
    return tuple(
        select_token_bucket(
            token_buckets,
            max(child_sizes) if scattered else sum(child_sizes),
        )
        for child_sizes in zip(*per_rank_child_sizes, strict=True)
    )


def _prepare_afd_attn_step(
    scheduler: Scheduler,
    *,
    tokens_per_rank: Sequence[int],
    forward_mode: ForwardMode,
    can_run_decode_cuda_graph: bool,
) -> bool:
    dispatcher = get_afd_dispatcher()
    num_attn_ranks = scheduler.afd_handshake.local.num_ranks
    tokens_per_rank = list(tokens_per_rank)
    if len(tokens_per_rank) != num_attn_ranks:
        raise RuntimeError(
            f"tokens_per_rank has {len(tokens_per_rank)} entries but the attn "
            f"instance has {num_attn_ranks} ranks"
        )
    if any(num_tokens < 0 for num_tokens in tokens_per_rank):
        raise RuntimeError("AFD tokens_per_rank cannot contain negative values")
    total_tokens = sum(tokens_per_rank)

    model_runner = scheduler.tp_worker.model_runner
    num_tokens_per_bs = scheduler.afd_handshake.local.decode_num_tokens_per_bs
    model_runner.afd_cuda_graph_batch_size = (
        max(tokens_per_rank) if forward_mode.is_decode() else None
    )

    scattered = get_disagg().afd_ffn_moe_backend == "deepep"
    dp_capacity = tokens_per_rank[scheduler.ps.tp_rank] if scattered else total_tokens
    if is_afd_fixed_shape_mode(forward_mode):
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
    child_wire_capacities: tuple[int, ...] = ()
    parent_graph_variant: Optional[AFDGraphCaptureVariant] = None
    num_microbatches = get_disagg().afd_decode_num_microbatches
    fixed_shape_decode = forward_mode.is_decode()
    # A captured parent bucket owns its child topology. Low-load DP ranks mask
    # padded children; only the eager fallback requires nonempty children.
    # Without padding, the runner requires an exact logical batch-size match.
    if num_microbatches > 1 and fixed_shape_decode and can_run_decode_cuda_graph:
        graph_variant = dispatcher.select_graph_variant(minimum_wire_capacity)
        if (
            graph_variant is not None
            and len(graph_variant.child_wire_capacities) == num_microbatches
            and (
                not get_exec().graph.disable_cuda_graph_padding
                or graph_variant.attn_batch_size
                == model_runner.afd_cuda_graph_batch_size
            )
        ):
            parent_graph_variant = graph_variant
            child_wire_capacities = graph_variant.child_wire_capacities

    can_split_eager = (
        num_microbatches > 1
        and fixed_shape_decode
        and all(num_tokens >= num_microbatches for num_tokens in tokens_per_rank)
    )
    eager_child_wire_capacities = (
        _select_child_wire_capacities(
            tokens_per_rank,
            scheduler.afd_handshake.local.decode_token_buckets,
            num_microbatches,
            scattered=scattered,
        )
        if can_split_eager
        else ()
    )
    if not child_wire_capacities:
        child_wire_capacities = eager_child_wire_capacities
    uses_microbatch_overlap = len(child_wire_capacities) > 1
    if uses_microbatch_overlap:
        wire_capacity = sum(child_wire_capacities)
        control_wire_capacity = wire_capacity
    else:
        child_wire_capacities = ()
        parent_graph_variant = None

    paired_graph_preflight = False
    if fixed_shape_decode and total_tokens > 0:
        if uses_microbatch_overlap:
            paired_graph_preflight = parent_graph_variant is not None
        elif control_wire_capacity in (
            scheduler.afd_handshake.local.decode_graph_token_buckets
        ):
            if num_microbatches == 1:
                paired_graph_preflight = True
            else:
                graph_wire_tokens_per_bs = num_tokens_per_bs * (
                    1 if scattered else num_attn_ranks
                )
                graph_batch_size = control_wire_capacity // graph_wire_tokens_per_bs
                paired_graph_preflight = graph_batch_size < num_microbatches

    direct_coordinator = scheduler._afd_attn_coordinator
    if direct_coordinator is not None:
        direct_coordinator.stage(
            _AFDPreparedAttnStep(
                tokens_per_rank=tuple(tokens_per_rank),
                forward_mode=forward_mode,
                dp_capacity=dp_capacity,
                wire_capacity=wire_capacity,
                control_wire_capacity=control_wire_capacity,
                child_wire_capacities=eager_child_wire_capacities,
                paired_graph_preflight=paired_graph_preflight,
            )
        )
        return paired_graph_preflight

    step_id = dispatcher.begin_batch(
        forward_mode=forward_mode,
        dp_capacity=dp_capacity,
        wire_capacity=wire_capacity,
        child_wire_capacities=child_wire_capacities,
        graph_variant=parent_graph_variant,
    )
    if scheduler._afd_sync_sender is not None:
        scheduler._afd_sync_sender.prepare_for_batch(
            step_id=step_id,
            tokens_per_rank=tokens_per_rank,
            forward_mode=forward_mode,
            wire_capacity=control_wire_capacity,
            child_wire_capacities=child_wire_capacities,
        )

    return False


def prepare_afd_attn_batch(scheduler: Scheduler, batch: ScheduleBatch) -> None:

    num_attn_ranks = scheduler.afd_handshake.local.num_ranks
    tokens_per_rank = _tokens_per_rank(batch, num_attn_ranks)
    forward_mode = _effective_forward_mode(batch, tokens_per_rank)
    _prepare_afd_attn_step(
        scheduler,
        tokens_per_rank=tokens_per_rank,
        forward_mode=forward_mode,
        can_run_decode_cuda_graph=batch.can_run_decode_cuda_graph,
    )


def init_afd_attn(
    scheduler: Scheduler,
) -> tuple[AFDHandshakeResult, Optional[AFDSyncSender]]:
    """Attn-side init: rank 0 handshakes, every rank gets the result.

    Ranks other than the global sender rank need the handshake result too
    (topology / negotiated ports) to build their data-plane comm backend.
    Returns (handshake, sync_sender); the sender is None off the sender rank.
    """
    rank = scheduler.ps.tp_rank
    is_sender_rank = rank == 0  # pp_size == 1 is enforced for AFD
    rank_node_ids, rank_nvshmem_fabric_keys = gather_nvshmem_topology(scheduler)
    if is_sender_rank:
        handshake = run_handshake_attn(
            get_disagg().afd_ffn_addr,
            build_local_handshake_info(
                scheduler,
                AFDRole.ATTN,
                rank_node_ids,
                rank_nvshmem_fabric_keys,
            ),
        )
        if scheduler.ps.tp_size > 1:
            broadcast_pyobj([handshake], rank, get_parallel().tp_group.cpu_group, src=0)
    else:
        (handshake,) = broadcast_pyobj(
            [], rank, get_parallel().tp_group.cpu_group, src=0
        )

    force_eager_communication = handshake.requires_eager_decode_communication
    decode_config = get_exec().graph.cuda_graph_config.decode
    if force_eager_communication and decode_config.backend == Backend.FULL:
        logger.info(
            "AFD NCCL cohort: switching the attn decode CUDA graph backend "
            "from full to breakable for eager remote-expert communication"
        )
        get_context().override(
            "afd_cohort_decode_graph",
            cuda_graph_config=with_phase(
                get_exec().graph.cuda_graph_config,
                Phase.DECODE,
                backend=Backend.BREAKABLE,
            ),
        )

    plan = build_afd_graph_capture_plan(handshake)
    backend = init_afd_comm_backend(scheduler, AFDRole.ATTN, handshake)
    dispatcher = _init_afd_attn_dispatcher(scheduler, backend, handshake)
    direct_multi_attn = (
        is_direct_afd_backend(handshake.local.afd_backend)
        and handshake.attn_members is not None
        and len(handshake.attn_members) > 1
    )
    local_graph_variants = build_afd_attn_local_graph_variants(handshake)
    if local_graph_variants:
        dispatcher.enable_local_parent_graph_variants(local_graph_variants)
        logger.info(
            "AFD attn local microbatch-overlap Breakable Graph enabled: variants=%d",
            len(local_graph_variants),
        )
    _capture_afd_attn_communication_graphs(
        scheduler,
        backend,
        dispatcher,
        plan,
    )

    direct_coordinator = None
    if direct_multi_attn:
        direct_coordinator = _AFDDirectAttnCoordinator(
            scheduler,
            handshake,
            dispatcher,
        )
        scheduler._afd_attn_coordinator = direct_coordinator
        sync_sender = None
    else:
        sync_sender = AFDSyncSender(handshake) if is_sender_rank else None

    def select_local_model_graph(can_run_graph: bool) -> bool:
        if direct_coordinator is not None:
            return direct_coordinator.coordinate(can_run_graph)
        decision = _decide_afd_attn_graph(
            can_run_graph,
            dispatcher,
            force_eager_communication,
        )
        if sync_sender is not None:
            sync_sender.send_prepared(decision.use_paired_communication_graph)
        return decision.use_local_model_graph

    # Every A rank makes the same dispatcher decision. Only rank 0 owns the
    # control-plane sender and publishes it to F.
    scheduler.tp_worker.model_runner.afd_control_callback = select_local_model_graph
    return handshake, sync_sender
