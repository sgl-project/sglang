"""Scheduler-side handlers for steppable_engine messages."""

from __future__ import annotations

import pickle
import types
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.api import get_canary_runner
from sglang.srt.managers.io_struct import RpcReqOutput
from sglang.srt.steppable_engine.messages import (
    ActiveReqsReq,
    ActiveReqsResp,
    AllocatorStatsReq,
    AllocatorStatsResp,
    BlockTableReq,
    BlockTableResp,
    CanaryOverheadPctReq,
    CanaryOverheadPctResp,
    CanaryViolationsReq,
    CanaryViolationsResp,
    InjectPerturbationReq,
    IsActiveReq,
    IsActiveResp,
    LastWritePlanReq,
    LastWritePlanResp,
    OutputHistoryReq,
    OutputHistoryResp,
    _ApplyPrFixTogglesReq,
)
from sglang.srt.steppable_engine.views import (
    CanaryViolationView,
    CanaryWritePlanView,
    SteppableReqHandle,
)
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


_FAIL_REASON_BIT_CHAIN_HASH = 1 << 0
_FAIL_REASON_BIT_POSITION = 1 << 1
_FAIL_REASON_BIT_REAL_KV_HASH = 1 << 2


def install_steppable_handlers(scheduler: "Scheduler") -> None:
    if getattr(scheduler, "_steppable_handlers_installed", False):
        return

    handler_specs = [
        (CanaryViolationsReq, _handle_canary_violations),
        (AllocatorStatsReq, _handle_allocator_stats),
        (BlockTableReq, _handle_block_table),
        (OutputHistoryReq, _handle_output_history),
        (IsActiveReq, _handle_is_active),
        (ActiveReqsReq, _handle_active_reqs),
        (LastWritePlanReq, _handle_last_write_plan),
        (CanaryOverheadPctReq, _handle_canary_overhead_pct),
        (InjectPerturbationReq, _handle_inject_perturbation),
        (_ApplyPrFixTogglesReq, _handle_apply_pr_fix_toggles),
    ]

    bound_pairs = [
        (msg_type, _wrap_steppable_handler(scheduler, fn))
        for msg_type, fn in handler_specs
    ]
    scheduler._request_dispatcher += TypeBasedDispatcher(bound_pairs)

    scheduler._steppable_handlers_installed = True


def _wrap_steppable_handler(
    scheduler: "Scheduler", fn: Callable[["Scheduler", Any], Any]
) -> Callable[[Any], None]:
    bound = types.MethodType(fn, scheduler)

    def wrapper(req: Any) -> None:
        resp = bound(req)
        if resp is None:
            resp = RpcReqOutput(success=True, message="")
        if scheduler.recv_from_rpc is not None:
            scheduler.recv_from_rpc.send_pyobj(resp)
        return None

    return wrapper


def _handle_canary_violations(
    self: "Scheduler", req: CanaryViolationsReq
) -> CanaryViolationsResp:
    runner = get_canary_runner(self.tp_worker.model_runner)
    views: List[CanaryViolationView] = []
    if runner is None:
        return CanaryViolationsResp(violations_pickled=pickle.dumps(views))

    violation_log = runner._device_state.violation_log
    write_index = int(violation_log.violation_write_index.cpu().item())
    if write_index == 0:
        return CanaryViolationsResp(violations_pickled=pickle.dumps(views))

    ring = violation_log.violation_ring.cpu()
    valid_count = min(write_index, int(ring.shape[0]))
    step_id = int(runner._step_counter)

    for row_idx in range(valid_count):
        row = ring[row_idx].tolist()
        (
            kernel_kind,
            slot_idx,
            position,
            stored_token,
            expected_token,
            _stored_chain_hash,
            _expected_aux,
            fail_reason_bits,
        ) = row
        try:
            tag_name = CanaryLaunchTag(int(kernel_kind)).name
        except ValueError:
            tag_name = f"unknown({int(kernel_kind)})"
        views.append(
            CanaryViolationView(
                fail_reason_name=_fail_reason_name(int(fail_reason_bits)),
                fail_reason_bits=int(fail_reason_bits),
                req_pool_idx=int(slot_idx),
                position=int(position),
                expected=int(expected_token),
                actual=int(stored_token),
                runner_kind=tag_name,
                step_id=step_id,
            )
        )
    return CanaryViolationsResp(violations_pickled=pickle.dumps(views))


def _fail_reason_name(bits: int) -> str:
    reasons: List[str] = []
    if bits & _FAIL_REASON_BIT_CHAIN_HASH:
        reasons.append("chain_hash")
    if bits & _FAIL_REASON_BIT_POSITION:
        reasons.append("position")
    if bits & _FAIL_REASON_BIT_REAL_KV_HASH:
        reasons.append("real_kv_hash")
    return "|".join(reasons) if reasons else "none"


def _handle_allocator_stats(
    self: "Scheduler", req: AllocatorStatsReq
) -> AllocatorStatsResp:
    allocator = self.token_to_kv_pool_allocator
    total = int(allocator.size)
    free = int(allocator.available_size())
    held_slots = _collect_held_slots(self)
    held = len(held_slots)
    used = total - free
    return AllocatorStatsResp(
        free=free,
        used=used,
        held=held,
        total=total,
        held_slots_pickled=pickle.dumps(frozenset(held_slots)),
    )


def _collect_held_slots(scheduler: "Scheduler") -> List[int]:
    req_to_token_pool = scheduler.req_to_token_pool
    held: List[int] = []
    reqs = _iter_active_reqs(scheduler)
    for r in reqs:
        if r.req_pool_idx is None:
            continue
        seqlen = int(r.seqlen)
        if seqlen <= 0:
            continue
        row = req_to_token_pool.req_to_token[r.req_pool_idx, :seqlen]
        held.extend(int(v) for v in row.cpu().tolist())
    return held


def _iter_active_reqs(scheduler: "Scheduler") -> List:
    reqs = list(scheduler.waiting_queue)
    if scheduler.running_batch is not None and not scheduler.running_batch.is_empty():
        reqs.extend(scheduler.running_batch.reqs)
    return reqs


def _handle_block_table(self: "Scheduler", req: BlockTableReq) -> BlockTableResp:
    candidate = self._steppable_lookup_req(req.rid)
    if candidate is None:
        raise RuntimeError(f"block_table: req {req.rid!r} not found")
    if candidate.req_pool_idx is None:
        raise RuntimeError(f"block_table: req {req.rid!r} has no req_pool_idx")
    seqlen = int(candidate.seqlen)
    row = self.req_to_token_pool.req_to_token[candidate.req_pool_idx, :seqlen]
    return BlockTableResp(slot_indices=[int(v) for v in row.cpu().tolist()])


def _handle_output_history(
    self: "Scheduler", req: OutputHistoryReq
) -> OutputHistoryResp:
    candidate = self._steppable_lookup_req(req.rid)
    if candidate is None:
        return OutputHistoryResp(tokens=[])
    return OutputHistoryResp(tokens=list(candidate.output_ids))


def _handle_is_active(self: "Scheduler", req: IsActiveReq) -> IsActiveResp:
    candidate = self._steppable_lookup_req(req.rid)
    if candidate is None:
        return IsActiveResp(active=False)
    return IsActiveResp(active=not candidate.finished())


def _handle_active_reqs(self: "Scheduler", req: ActiveReqsReq) -> ActiveReqsResp:
    handles: List[SteppableReqHandle] = [
        SteppableReqHandle(
            rid=r.rid,
            prompt_len=len(r.origin_input_ids),
            max_new_tokens=int(r.sampling_params.max_new_tokens),
        )
        for r in _iter_active_reqs(self)
    ]
    return ActiveReqsResp(handles_pickled=pickle.dumps(handles))


def _handle_last_write_plan(
    self: "Scheduler", req: LastWritePlanReq
) -> LastWritePlanResp:
    plan = getattr(self, "_last_canary_write_plan", None)
    if plan is None:
        return LastWritePlanResp(plan_pickled=None)

    expected_tokens: Optional[List[int]] = (
        list(plan.expected_input_tokens)
        if plan.expected_input_tokens is not None
        else None
    )
    expected_positions: Optional[List[int]] = (
        list(plan.expected_input_positions)
        if plan.expected_input_positions is not None
        else None
    )
    view = CanaryWritePlanView(
        num_write=int(plan.num_write),
        num_verify=int(plan.num_verify),
        write_slot_indices=list(plan.write_slot_indices),
        verify_slot_indices=list(plan.verify_slot_indices),
        expected_input_tokens=expected_tokens,
        expected_input_positions=expected_positions,
    )
    return LastWritePlanResp(plan_pickled=pickle.dumps(view))


def _handle_canary_overhead_pct(
    self: "Scheduler", req: CanaryOverheadPctReq
) -> CanaryOverheadPctResp:
    return CanaryOverheadPctResp(
        pct=float(getattr(self, "_last_canary_overhead_pct", 0.0))
    )


def _handle_inject_perturbation(self: "Scheduler", req: InjectPerturbationReq) -> None:
    from sglang.srt.steppable_engine.perturb import arm_one_shot

    arm_one_shot(self, channel=req.channel, kind=req.kind, rank=req.rank)


def _handle_apply_pr_fix_toggles(self: "Scheduler", req: _ApplyPrFixTogglesReq) -> None:
    from sglang.srt.steppable_engine.pr_fix_toggle import apply_pr_fix_toggles

    choices = pickle.loads(req.choices_pickled)
    apply_pr_fix_toggles(self, choices=choices)
