"""Per-rank canary orchestrator.

Owns:

- three (:class:`VerifyPlan`, :class:`WritePlan`) pairs — one per caller path (per-forward, running-sweep,
  radix-orphan-sweep). Each pair is allocated once with capacity sized for its worst-case batch and reused
  in place so cuda-graph capture can pin its addresses.
- one :class:`ViolationLog` shared across all (head|tail|sweep) × (K|V) × (FULL|SWA) launches on this
  runner.
- one ``slot_run_counter`` + one ``kernel_run_counter`` per (head, tail, sweep) — health monitoring.
- the list of :class:`CanaryEndpoint` instances bound to the canary buffer group this runner protects.
- the side-stream async D2H pump that watches ``ViolationLog.violation_write_index``.

Public API (kernels.md §5):

- :meth:`CanaryRunner.run_head` / :meth:`CanaryRunner.run_tail` — per-forward launches.
- :meth:`CanaryRunner.run_sweep` — combined running + radix-orphan sweep.
- :meth:`CanaryRunner.end_of_forward` — async D2H pump + raise path + cross-rank allreduce.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary_plan import canary_plan_step
from sglang.jit_kernel.kv_cache_canary_verify import (
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_cache_canary_write import (
    CanaryPseudoMode,
    WritePlan,
)
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.endpoint import CanaryEndpoint
from sglang.srt.kv_cache_canary.plan_input import (
    PlanInput,
    build_plan_input_per_forward,
)
from sglang.srt.kv_cache_canary.sweep_planner import (
    build_radix_orphan_input,
    build_running_sweep_input,
    collect_running_reqs_for_sweep,
)
from sglang.srt.kv_cache_canary.violation_state import ViolationLog

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


_LOG_RATE_LIMIT_SECONDS: float = 5.0


@dataclass(frozen=True, slots=True, kw_only=True)
class _LaunchPlans:
    """One ``(VerifyPlan, WritePlan)`` pair sized for one caller path.

    Held on the runner so capture-time addresses stay stable; the plan kernel refills the tensors in-place
    on every call.
    """

    verify_plan: VerifyPlan
    write_plan: WritePlan


@dataclass(frozen=True, slots=True, kw_only=True)
class _HealthCounters:
    """Per-kind health counters shared by every launch flavor of one (head|tail|sweep)."""

    slot_run_counter: torch.Tensor
    kernel_run_counter: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class _PseudoBuffers:
    """In-place destinations for caller-supplied pseudo expectations.

    Always allocated — when ``pseudo_mode == OFF`` the kernel ignores both tensors but their pointers must
    still be valid for the cuda-graph ABI.
    """

    expected_tokens: torch.Tensor
    expected_positions: torch.Tensor


class CanaryRunner:
    """Per-pool canary orchestrator. One instance per attached :class:`CanaryBufferGroup`.

    Two callers (:meth:`run_head`, :meth:`run_tail`) drive the per-forward path; :meth:`run_sweep` drives
    the periodic sweep. :meth:`end_of_forward` runs the post-forward bookkeeping (async D2H pump, raise
    path, allreduce, periodic health check).

    .. warning::
        Three pieces are critical safety logic and must NOT be removed by an automated /simplify pass:

        - the side-stream + event-based async D2H pump
          (:meth:`_record_poll_events` / :meth:`_pull_latest_from_events`) — keeps the hot path
          non-blocking,
        - the health monitoring in :meth:`_maybe_health_check` — detects a dead canary kernel by checking
          that ``kernel_run_counter`` is advancing after warmup,
        - the unconditional cross-rank allreduce in :meth:`_cross_rank_max`, called from every
          :meth:`end_of_forward`.

        Removing any of them silently disables the canary or produces TP deadlock when raising. Do not
        simplify away.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        buffer_group: CanaryBufferGroup,
        device: torch.device,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        running_sweep_verify_capacity: int,
        radix_sweep_verify_capacity: int,
        radix_sweep_extras_capacity: int,
        per_forward_extras_capacity: int = 1,
        running_sweep_extras_capacity: int = 1,
        pseudo_token_capacity: int = 1,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
        radix_cache: Optional["BasePrefixCache"] = None,
        tp_rank: int = 0,
    ) -> None:
        self._config = config
        self._device = device
        self._pool_kind = buffer_group.kind
        self._buffer_group = buffer_group
        self._tp_rank: int = int(tp_rank)
        self._req_to_token_pool: Optional["ReqToTokenPool"] = req_to_token_pool
        self._radix_cache: Optional["BasePrefixCache"] = radix_cache
        self._per_forward_extras_capacity: int = max(
            1, int(per_forward_extras_capacity)
        )
        self._running_sweep_extras_capacity: int = max(
            1, int(running_sweep_extras_capacity)
        )
        self._radix_sweep_extras_capacity: int = max(
            1, int(radix_sweep_extras_capacity)
        )
        self._pseudo_token_capacity: int = max(1, int(pseudo_token_capacity))

        self._violation_log = ViolationLog.allocate(
            ring_capacity=config.violation_ring_capacity, device=device
        )

        self._head_counters = _allocate_health_counters(device=device)
        self._tail_counters = _allocate_health_counters(device=device)
        self._sweep_counters = _allocate_health_counters(device=device)

        self._per_forward_plans = _allocate_launch_plans(
            verify_capacity=per_forward_verify_capacity,
            write_req_capacity=per_forward_write_req_capacity,
            device=device,
        )
        self._running_sweep_plans = _allocate_launch_plans(
            verify_capacity=running_sweep_verify_capacity,
            write_req_capacity=1,
            device=device,
        )
        self._radix_sweep_plans = _allocate_launch_plans(
            verify_capacity=radix_sweep_verify_capacity,
            write_req_capacity=1,
            device=device,
        )

        self._pseudo_buffers = _allocate_pseudo_buffers(
            capacity=self._pseudo_token_capacity, device=device
        )

        self._head_endpoints: List[CanaryEndpoint] = _build_endpoints_for_phase(
            buffer_group=buffer_group, config=config, phase="head"
        )
        self._tail_endpoints: List[CanaryEndpoint] = _build_endpoints_for_phase(
            buffer_group=buffer_group, config=config, phase="tail"
        )
        self._sweep_endpoints: List[CanaryEndpoint] = _build_endpoints_for_phase(
            buffer_group=buffer_group, config=config, phase="sweep"
        )

        self._sweep_every_n: int = int(config.real_data_sweep_every_n_steps)
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._violation_write_index_host: torch.Tensor = torch.zeros(
            1, dtype=torch.int32, pin_memory=torch.cuda.is_available()
        )
        self._counters_host: torch.Tensor = torch.zeros(
            6, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        self._violation_write_index_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )
        self._counters_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )

        self._latest_violation_write_index: int = 0
        self._latest_counters: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
        self._raise_latch: bool = False
        self._last_log_time: float = 0.0
        self._forward_step: int = 0
        self._poll_armed: bool = False
        self._warmup_check_done: bool = False
        self._last_forward_batch: Optional["ForwardBatch"] = None
        self._last_per_forward_plan_input: Optional[PlanInput] = None

    @property
    def config(self) -> CanaryConfig:
        return self._config

    @property
    def pool_kind(self) -> PoolKind:
        return self._pool_kind

    @property
    def tp_rank(self) -> int:
        return self._tp_rank

    @property
    def buffer_group(self) -> CanaryBufferGroup:
        return self._buffer_group

    @property
    def violation_log(self) -> ViolationLog:
        return self._violation_log

    @property
    def per_forward_plans(self) -> _LaunchPlans:
        return self._per_forward_plans

    def attach_req_to_token_pool(self, req_to_token_pool: "ReqToTokenPool") -> None:
        """Bind the scheduler's ``req_to_token`` pool for sweep planning."""
        self._req_to_token_pool = req_to_token_pool

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        """Bind the radix cache for the radix-orphan sweep path."""
        self._radix_cache = radix_cache

    def set_last_forward_batch(self, forward_batch: "ForwardBatch") -> None:
        """Stash the current forward batch so sweep / perturb hooks can read it."""
        self._last_forward_batch = forward_batch

    def run_head(self, *, forward_batch: "ForwardBatch") -> None:
        """Per-forward HEAD launch: build plan input, run plan kernel, then fire every head endpoint."""
        if not self._config.enabled:
            return
        if self._req_to_token_pool is None:
            return

        plan_input = build_plan_input_per_forward(
            forward_batch=forward_batch,
            req_to_token_pool=self._req_to_token_pool,
            extras_capacity=self._per_forward_extras_capacity,
        )
        if plan_input is None:
            return
        self._last_per_forward_plan_input = plan_input

        self._fill_pseudo_buffers_for_per_forward(forward_batch=forward_batch)
        self._invoke_plan_kernel(
            plan_input=plan_input,
            plans=self._per_forward_plans,
        )
        self._launch_per_forward_endpoints(
            endpoints=self._head_endpoints,
            counters=self._head_counters,
            forward_batch=forward_batch,
        )

    def run_tail(self, *, forward_batch: "ForwardBatch") -> None:
        """Per-forward TAIL launch: reuses the same per-forward plans from :meth:`run_head`."""
        if not self._config.enabled:
            return
        if self._last_per_forward_plan_input is None:
            return
        self._launch_per_forward_endpoints(
            endpoints=self._tail_endpoints,
            counters=self._tail_counters,
            forward_batch=forward_batch,
        )

    def run_sweep(self) -> None:
        """Run one sweep cycle: running-reqs first, then radix orphans.

        Each leg builds its own :class:`PlanInput`, runs the plan kernel into the matching plan pair, and
        fires every sweep endpoint against the resulting :class:`VerifyPlan`. Skipping is per-leg: missing
        running reqs only skips that leg; missing radix cache only skips orphans.
        """
        if not self._config.enabled or self._req_to_token_pool is None:
            return

        running = collect_running_reqs_for_sweep(forward_batch=self._last_forward_batch)
        if running is not None:
            running_input = build_running_sweep_input(
                req_to_token_pool=self._req_to_token_pool,
                running=running,
                extras_capacity=self._running_sweep_extras_capacity,
            )
            self._invoke_plan_kernel(
                plan_input=running_input,
                plans=self._running_sweep_plans,
            )
            self._launch_sweep_endpoints(
                verify_plan=self._running_sweep_plans.verify_plan
            )

        if self._radix_cache is not None:
            radix_input = build_radix_orphan_input(
                req_to_token_pool=self._req_to_token_pool,
                radix_cache=self._radix_cache,
                extras_capacity=self._radix_sweep_extras_capacity,
                swa_index_lut=self._buffer_group.swa_index_lut,
            )
            if radix_input is not None:
                self._invoke_plan_kernel(
                    plan_input=radix_input,
                    plans=self._radix_sweep_plans,
                )
                self._launch_sweep_endpoints(
                    verify_plan=self._radix_sweep_plans.verify_plan
                )

    def end_of_forward(self) -> None:
        """Called once per forward (after run_tail) on the compute stream.

        Records D2H copies on the side stream, queries the recorded event non-blockingly, refreshes the
        host-side cached error flag + health counters, increments the step counter, and unconditionally
        all-reduces the local error flag so every rank decides to raise in lock-step (no single-rank raise
        → no NCCL deadlock).

        Skipped during cuda graph capture: side-stream / event / sync ops are unsafe inside captured
        regions.
        """
        if not self._config.enabled:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return

        self._record_poll_events()
        self._pull_latest_from_events()
        self._forward_step += 1

        if self._sweep_every_n > 0 and self._forward_step % self._sweep_every_n == 0:
            self.run_sweep()

        local_flag = 1 if self._latest_violation_write_index >= 1 else 0
        global_flag = self._cross_rank_max(local_flag)
        self._maybe_health_check()
        if global_flag:
            self._handle_violation_global()

    def is_errored(self) -> bool:
        """Host-cached error flag — refreshed by the async D2H pump after every forward."""
        return self._latest_violation_write_index >= 1

    def _launch_per_forward_endpoints(
        self,
        *,
        endpoints: List[CanaryEndpoint],
        counters: _HealthCounters,
        forward_batch: "ForwardBatch",
    ) -> None:
        if not endpoints:
            return
        for endpoint in endpoints:
            endpoint.launch_per_forward(
                verify_plan=self._per_forward_plans.verify_plan,
                write_plan=self._per_forward_plans.write_plan,
                fb_input_ids=forward_batch.input_ids,
                fb_positions=forward_batch.positions,
                fb_out_cache_loc=forward_batch.out_cache_loc,
                pseudo_mode=self._config.pseudo_mode,
                pseudo_expected_tokens=self._pseudo_buffers.expected_tokens,
                pseudo_expected_positions=self._pseudo_buffers.expected_positions,
                violation_log=self._violation_log,
                slot_run_counter=counters.slot_run_counter,
                kernel_run_counter=counters.kernel_run_counter,
            )

    def _launch_sweep_endpoints(self, *, verify_plan: VerifyPlan) -> None:
        for endpoint in self._sweep_endpoints:
            endpoint.launch_sweep(
                verify_plan=verify_plan,
                violation_log=self._violation_log,
                slot_run_counter=self._sweep_counters.slot_run_counter,
                kernel_run_counter=self._sweep_counters.kernel_run_counter,
            )

    def _invoke_plan_kernel(
        self, *, plan_input: PlanInput, plans: _LaunchPlans
    ) -> None:
        swa_window = (
            int(self._config.swa_window_size)
            if (
                self._pool_kind is PoolKind.SWA
                and self._config.swa_window_size is not None
            )
            else 0
        )
        canary_plan_step(
            verify_plan_out=plans.verify_plan,
            write_plan_out=plans.write_plan,
            fb_req_pool_indices=plan_input.fb_req_pool_indices,
            fb_prefix_lens=plan_input.fb_prefix_lens,
            fb_extend_seq_lens=plan_input.fb_extend_seq_lens,
            req_to_token=plan_input.req_to_token,
            extra_verify_slot_indices=plan_input.extra_verify_slot_indices,
            extra_verify_positions=plan_input.extra_verify_positions,
            extra_verify_prev_slot_indices=plan_input.extra_verify_prev_slot_indices,
            extra_verify_num_valid=plan_input.extra_verify_num_valid,
            swa_window_size=swa_window,
            full_to_swa_index_mapping=self._buffer_group.swa_index_lut,
        )

    def _fill_pseudo_buffers_for_per_forward(
        self, *, forward_batch: "ForwardBatch"
    ) -> None:
        """Populate ``pseudo_expected_*`` in-place when ``pseudo_mode == ON``.

        Caller-provided oracle returns ``(expected_tokens, expected_positions)``; both copied into the
        runner's pre-allocated tensors so cuda-graph capture stays valid. When OFF the buffers are left at
        zero (kernel ignores them via the pseudo_mode toggle).
        """
        if self._config.pseudo_mode is CanaryPseudoMode.OFF:
            return
        oracle = self._config.pseudo_oracle
        if oracle is None:
            return
        expected_tokens, expected_positions = oracle(forward_batch)
        n_tokens = min(self._pseudo_token_capacity, int(expected_tokens.shape[0]))
        if n_tokens <= 0:
            return
        self._pseudo_buffers.expected_tokens[:n_tokens].copy_(
            expected_tokens[:n_tokens].to(torch.int32)
        )
        self._pseudo_buffers.expected_positions[:n_tokens].copy_(
            expected_positions[:n_tokens].to(torch.int32)
        )

    def _record_poll_events(self) -> None:
        if self._side_stream is None:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        compute_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._side_stream):
            self._side_stream.wait_stream(compute_stream)
            if self._violation_write_index_event is not None:
                self._violation_write_index_host.copy_(
                    self._violation_log.violation_write_index, non_blocking=True
                )
                self._violation_write_index_event.record(stream=self._side_stream)
            if self._counters_event is not None:
                pairs = (
                    self._head_counters.kernel_run_counter,
                    self._tail_counters.kernel_run_counter,
                    self._head_counters.slot_run_counter,
                    self._tail_counters.slot_run_counter,
                    self._sweep_counters.kernel_run_counter,
                    self._sweep_counters.slot_run_counter,
                )
                for host_slot, device_counter in zip(self._counters_host, pairs):
                    host_slot.copy_(device_counter.flatten()[0], non_blocking=True)
                self._counters_event.record(stream=self._side_stream)
            self._poll_armed = True

    def _pull_latest_from_events(self) -> None:
        """Non-blocking ``event.query()`` — refresh host-cached values when ready."""
        try:
            if (
                self._violation_write_index_event is not None
                and self._poll_armed
                and self._violation_write_index_event.query()
            ):
                self._latest_violation_write_index = int(
                    self._violation_write_index_host.item()
                )
            if (
                self._counters_event is not None
                and self._poll_armed
                and self._counters_event.query()
            ):
                self._latest_counters = tuple(  # type: ignore[assignment]
                    int(x) for x in self._counters_host.tolist()
                )
        except Exception:
            logger.exception("kv-canary: event poll failed")

    def _cross_rank_max(self, local_flag: int) -> int:
        """Unconditional all-reduce-MAX on the local 1-byte flag.

        Critical: every forward must enter this allreduce so all peers agree on whether ANY rank saw a
        violation. If only the offending rank all-reduced, the others would block in the next NCCL
        collective.
        """
        if not torch.distributed.is_initialized():
            return local_flag
        try:
            from sglang.srt.distributed.parallel_state import get_tp_group

            tp_group = get_tp_group()
        except Exception:
            logger.exception(
                "kv-canary: TP group unavailable; falling back to local flag only"
            )
            return local_flag

        flag = torch.tensor([local_flag], dtype=torch.int32, device=self._device)
        try:
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                group=tp_group.device_group,
            )
        except Exception:
            logger.exception(
                "kv-canary: allreduce failed; falling back to local flag only"
            )
            return local_flag
        return int(flag.item())

    def _maybe_health_check(self) -> None:
        """Counter-zero detection (after warmup) + periodic liveness print."""
        step = self._forward_step
        period = max(1, self._config.health_print_every_n_forwards)
        (
            kernel_head,
            kernel_tail,
            slot_head,
            slot_tail,
            kernel_sweep,
            slot_sweep,
        ) = self._latest_counters

        if (
            not self._warmup_check_done
            and step >= self._config.counter_zero_warmup_forwards
        ):
            local_unhealthy = 1 if (kernel_head == 0 or kernel_tail == 0) else 0
            global_unhealthy = self._cross_rank_max(local_unhealthy)
            self._warmup_check_done = True
            if global_unhealthy:
                message = (
                    f"kv-canary: kernel never ran after warmup "
                    f"(step={step}, kernel_head={kernel_head}, kernel_tail={kernel_tail}). "
                    "The canary is not actually executing — refusing to continue silently."
                )
                logger.error(message)
                if self._config.mode is CanaryMode.RAISE:
                    raise RuntimeError(message)

        if (
            self._sweep_every_n > 0
            and step >= self._sweep_every_n * 2
            and kernel_sweep == 0
        ):
            message = (
                f"kv-canary: sweep-every-n-steps={self._sweep_every_n} configured "
                f"but kernel_run_counter_sweep stayed at 0 after {step} forwards. "
                "The sweep path is not executing — refusing to continue silently."
            )
            logger.error(message)
            if self._config.mode is CanaryMode.RAISE:
                raise RuntimeError(message)

        if step > 0 and step % period == 0:
            logger.info(
                "kv-canary[%s]: protected %d forwards "
                "(head=%d kernels / %d slots, tail=%d kernels / %d slots, "
                "sweep=%d kernels / %d slots)",
                self._pool_kind.name,
                step,
                kernel_head,
                slot_head,
                kernel_tail,
                slot_tail,
                kernel_sweep,
                slot_sweep,
            )

    def _handle_violation_global(self) -> None:
        if self._config.mode is CanaryMode.LOG:
            self._handle_violation_log()
            return
        if self._config.mode is CanaryMode.RAISE:
            if self._raise_latch:
                return
            self._raise_latch = True
            self._raise_with_first_violation()

    def _pull_first_violation(self) -> Tuple[List[int], int]:
        """Synchronous D2H pull of the first violation row + total count from the global ring."""
        first_violation = self._violation_log.violation_ring[0].cpu().tolist()
        write_index = int(self._violation_log.violation_write_index.cpu().item())
        return first_violation, write_index

    def _handle_violation_log(self) -> None:
        now = time.time()
        first_violation, write_index = self._pull_first_violation()
        if now - self._last_log_time >= _LOG_RATE_LIMIT_SECONDS:
            self._last_log_time = now
            logger.error(_format_violation(first_violation, write_index))
        self._violation_log.clear()
        self._latest_violation_write_index = 0

    def _raise_with_first_violation(self) -> None:
        first_violation, write_index = self._pull_first_violation()
        if write_index == 0:
            raise RuntimeError(
                "kv-canary: raise path entered but no violation rows found "
                "(peer rank may have observed the violation)"
            )
        raise RuntimeError(_format_violation(first_violation, write_index))


def _allocate_health_counters(*, device: torch.device) -> _HealthCounters:
    return _HealthCounters(
        slot_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, device=device),
    )


def _allocate_launch_plans(
    *,
    verify_capacity: int,
    write_req_capacity: int,
    device: torch.device,
) -> _LaunchPlans:
    return _LaunchPlans(
        verify_plan=VerifyPlan.allocate(verify_capacity=verify_capacity, device=device),
        write_plan=WritePlan.allocate(
            write_req_capacity=write_req_capacity, device=device
        ),
    )


def _allocate_pseudo_buffers(*, capacity: int, device: torch.device) -> _PseudoBuffers:
    return _PseudoBuffers(
        expected_tokens=torch.zeros(capacity, dtype=torch.int32, device=device),
        expected_positions=torch.zeros(capacity, dtype=torch.int32, device=device),
    )


_HEAD_TAGS: Dict[PoolKind, Tuple[CanaryLaunchTag, Optional[CanaryLaunchTag]]] = {
    PoolKind.FULL: (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.HEAD_V_FULL),
    PoolKind.SWA: (CanaryLaunchTag.HEAD_K_SWA, CanaryLaunchTag.HEAD_V_SWA),
}
_TAIL_TAGS: Dict[PoolKind, Tuple[CanaryLaunchTag, Optional[CanaryLaunchTag]]] = {
    PoolKind.FULL: (CanaryLaunchTag.TAIL_K_FULL, CanaryLaunchTag.TAIL_V_FULL),
    PoolKind.SWA: (CanaryLaunchTag.TAIL_K_SWA, CanaryLaunchTag.TAIL_V_SWA),
}
_SWEEP_TAGS: Dict[PoolKind, Tuple[CanaryLaunchTag, Optional[CanaryLaunchTag]]] = {
    PoolKind.FULL: (CanaryLaunchTag.SWEEP_K_FULL, CanaryLaunchTag.SWEEP_V_FULL),
    PoolKind.SWA: (CanaryLaunchTag.SWEEP_K_SWA, CanaryLaunchTag.SWEEP_V_SWA),
}


def _build_endpoints_for_phase(
    *,
    buffer_group: CanaryBufferGroup,
    config: CanaryConfig,
    phase: str,
) -> List[CanaryEndpoint]:
    """Construct the K and V endpoint pair for one (head | tail | sweep) phase.

    Sweep uses the tail canary buffer because tail captures post-write fingerprints (kernels.md §6.2 says
    sweep must read the post-step state of every alive slot).
    """
    if phase == "head":
        tags = _HEAD_TAGS[buffer_group.kind]
        k_buf = buffer_group.k_head
        v_buf = buffer_group.v_head
    elif phase == "tail":
        tags = _TAIL_TAGS[buffer_group.kind]
        k_buf = buffer_group.k_tail
        v_buf = buffer_group.v_tail
    elif phase == "sweep":
        tags = _SWEEP_TAGS[buffer_group.kind]
        k_buf = buffer_group.k_tail
        v_buf = buffer_group.v_tail
    else:
        raise ValueError(f"kv-canary: unknown phase {phase!r}")

    endpoints: List[CanaryEndpoint] = [
        CanaryEndpoint(
            canary_buf=k_buf,
            kernel_kind=tags[0],
            real_kv_sources=buffer_group.real_kv_sources_k,
            real_kv_hash_mode=config.real_kv_hash_mode,
            full_to_swa_index_mapping=buffer_group.swa_index_lut,
        )
    ]
    if buffer_group.has_v_half and v_buf is not None and tags[1] is not None:
        endpoints.append(
            CanaryEndpoint(
                canary_buf=v_buf,
                kernel_kind=tags[1],
                real_kv_sources=buffer_group.real_kv_sources_v,
                real_kv_hash_mode=config.real_kv_hash_mode,
                full_to_swa_index_mapping=buffer_group.swa_index_lut,
            )
        )
    return endpoints


def _format_violation(first_violation: List[int], write_index: int) -> str:
    """Format a violation row as a labelled, multi-line error message.

    Decodes ``kernel_kind`` into its :class:`CanaryLaunchTag` name when possible; falls back to the raw int
    for forward-compat with unknown tags.
    """
    (
        kernel_kind,
        slot_idx,
        position,
        stored_token,
        expected_token,
        stored_chain_hash,
        expected_aux,
        fail_reason_bits,
    ) = first_violation
    u64_mask = (1 << 64) - 1
    try:
        tag = CanaryLaunchTag(int(kernel_kind))
        tag_label = tag.name
    except ValueError:
        tag_label = f"unknown({int(kernel_kind)})"

    return "\n".join(
        [
            "kv-canary violation:",
            f"  kernel_kind:        {tag_label}",
            f"  slot_idx:           {int(slot_idx)}",
            f"  position:           {int(position)}",
            f"  token:              stored={int(stored_token)} expected={int(expected_token)}",
            f"  stored_chain_hash:  {int(stored_chain_hash) & u64_mask:#018x}",
            f"  expected_aux:       {int(expected_aux) & u64_mask:#018x}",
            f"  fail_reason_bits:   {int(fail_reason_bits):#010b}",
            f"  total_violations:   {write_index} (since last reset)",
        ]
    )
