from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import (
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    FailReason,
    canary_step,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_cache_canary.host_state import (
    VIOLATION_KIND_HEAD_K,
    VIOLATION_KIND_HEAD_V,
    VIOLATION_KIND_TAIL_K,
    VIOLATION_KIND_TAIL_V,
    VIOLATION_KINDS,
    BatchPlan,
    CanaryDeviceState,
    CanaryHostState,
    CanaryLaunchBuffers,
)
from sglang.srt.kv_cache_canary.pool_patch import (
    PoolKind,
    attach_shadow_buffers,
    get_shadow_buffers,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


_LOG_RATE_LIMIT_SECONDS: float = 5.0


class CanaryRunner:
    """Top-level orchestrator for KV cache canary.

    One instance lives on each rank. Owns the host-side request state, the
    GPU-side violation buffer + counters, the side stream that asynchronously
    copies the ``is_errored`` flag and health counters back to host, and the
    log/raise policy.

    .. warning::
        Several pieces in this class are critical safety logic and must NOT
        be simplified away by an automated pass:

        - the side-stream + event-based async D2H pump
          (``_record_poll_events`` / ``_pull_latest_from_events``) — the hot
          path stays non-blocking, the README §4 design,
        - the §5 health monitoring: ``_maybe_health_check``,
        - the unconditional cross-rank allreduce in ``_cross_rank_max``,
          called from every ``end_of_forward``.

        Removing any of them silently disables the canary or produces TP
        deadlock when raising. Do not simplify away.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        pool: "KVCache",
        num_req_slots: int,
        device: torch.device,
        pool_kind: PoolKind = PoolKind.FULL,
        launch_capacity: int,
    ) -> None:
        self._config = config
        self._device = device
        self._pool_kind = pool_kind

        attach_shadow_buffers(pool, pool_kind=pool_kind)
        self._k_slot_stride_bytes = pool.canary_k_slot_stride_bytes
        self._v_slot_stride_bytes = pool.canary_v_slot_stride_bytes
        self._k_head, self._k_tail, self._v_head, self._v_tail = get_shadow_buffers(
            pool
        )
        self._has_v_half: bool = pool.canary_has_v_half

        self.host_state = CanaryHostState(config=config, num_req_slots=num_req_slots)
        self._device_state = CanaryDeviceState.allocate(
            device=device, ring_capacity=config.violation_ring_capacity
        )
        # Pre-allocated fixed-address launch buffers. Cuda graph capture
        # records these specific addresses; replay-side host code refills
        # them in-place before ``graph.replay()`` so the recorded kernel
        # launches see the correct expected_* / slot_indices for the batch.
        self._launch_capacity: int = int(launch_capacity)
        self._launch = CanaryLaunchBuffers.allocate(
            device=device, capacity=self._launch_capacity
        )
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        # Per-kind pinned host buffers; the side stream copies each kind's
        # is_errored into its own slot so the main thread can OR them in
        # ``end_of_forward`` without losing the K/V distinction.
        self._is_errored_host_per_kind: Dict[str, torch.Tensor] = {
            kind: torch.zeros(
                1, dtype=torch.int32, pin_memory=torch.cuda.is_available()
            )
            for kind in VIOLATION_KINDS
        }
        self._counters_host = torch.zeros(
            4, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        self._is_errored_event_per_kind: Dict[str, Optional[torch.cuda.Event]] = {
            kind: (torch.cuda.Event() if torch.cuda.is_available() else None)
            for kind in VIOLATION_KINDS
        }
        self._counters_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )

        # Latest values pulled by the side-stream events; main inference
        # thread reads these inside ``end_of_forward``.
        self._latest_is_errored_per_kind: Dict[str, int] = {
            kind: 0 for kind in VIOLATION_KINDS
        }
        self._latest_counters: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._raise_latch: bool = False
        self._last_log_time_per_reason: Dict[Tuple[str, int], float] = {}
        self._forward_step: int = 0
        # Whether ``_record_poll_events`` has armed the cudaEvent at least
        # once. ``event.query()`` on an unrecorded event is undefined; skip
        # until we've recorded once.
        self._poll_armed: bool = False
        # README §5 warmup zero-check runs exactly once, after the latch flips.
        # Tracked explicitly so a step skipped by cuda graph paths can't cause
        # the check to silently leapfrog past the warmup boundary.
        self._warmup_check_done: bool = False

    @property
    def config(self) -> CanaryConfig:
        return self._config

    @property
    def pool_kind(self) -> PoolKind:
        return self._pool_kind

    def run_head(self, *, plan: BatchPlan) -> None:
        self._run_kernel_pair(plan=plan, kernel_kind=KERNEL_KIND_HEAD)

    def run_tail(self, *, plan: BatchPlan) -> None:
        self._run_kernel_pair(plan=plan, kernel_kind=KERNEL_KIND_TAIL)

    def end_of_forward(self) -> None:
        """Called once per forward (after run_tail) on the compute stream.

        Records D2H copies on the side stream, queries the recorded event
        non-blockingly, refreshes the host-side cached error flag + health
        counters, increments the step counter, and unconditionally
        allreduces the local error flag so every rank decides to raise in
        lock-step (no single-rank raise → no NCCL deadlock; README §3
        decision #3).

        Skipped during cuda graph capture: side-stream / event / sync ops
        are unsafe inside captured regions.
        """
        if not self._config.enabled:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        self._record_poll_events()
        self._pull_latest_from_events()
        self._forward_step += 1
        local_flag = 1 if any(self._latest_is_errored_per_kind.values()) else 0
        global_flag = self._cross_rank_max(local_flag)
        self._maybe_health_check()
        if global_flag:
            self._handle_violation_global()

    def _pull_latest_from_events(self) -> None:
        """Non-blocking event.query() — if done, refresh host-cached values.

        Called from the main inference thread only (never from another
        thread) so we never query while another thread is mid-capture.
        """
        try:
            for kind in VIOLATION_KINDS:
                event = self._is_errored_event_per_kind[kind]
                if event is not None and self._poll_armed and event.query():
                    self._latest_is_errored_per_kind[kind] = int(
                        self._is_errored_host_per_kind[kind].item()
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

    def _record_poll_events(self) -> None:
        if self._side_stream is None:
            return
        # Skip during cuda graph capture: side-stream operations and
        # event.record / event.query interact badly with the captured stream
        # (CUDA refuses these ops mid-capture).
        if torch.cuda.is_current_stream_capturing():
            return
        compute_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._side_stream):
            self._side_stream.wait_stream(compute_stream)
            for kind in VIOLATION_KINDS:
                event = self._is_errored_event_per_kind[kind]
                if event is None:
                    continue
                self._is_errored_host_per_kind[kind].copy_(
                    self._device_state.get_violation_slot(kind).is_errored,
                    non_blocking=True,
                )
                event.record(stream=self._side_stream)
            self._poll_armed = True
            if self._counters_event is not None and (
                self._forward_step % max(1, self._config.health_print_every_n_forwards)
                == 0
            ):
                self._counters_host[0].copy_(
                    self._device_state.kernel_run_counter_head.flatten()[0],
                    non_blocking=True,
                )
                self._counters_host[1].copy_(
                    self._device_state.kernel_run_counter_tail.flatten()[0],
                    non_blocking=True,
                )
                self._counters_host[2].copy_(
                    self._device_state.slot_run_counter_head.flatten()[0],
                    non_blocking=True,
                )
                self._counters_host[3].copy_(
                    self._device_state.slot_run_counter_tail.flatten()[0],
                    non_blocking=True,
                )
                self._counters_event.record(stream=self._side_stream)

    def prepare_for_replay(self, *, plan: BatchPlan) -> None:
        """Refill the fixed launch buffers in-place for a replay forward.

        Called from the pre-replay hook on ``CudaGraphRunner.replay``. The
        captured graph holds references to ``self._launch.*`` tensors at
        their fixed addresses; this method copies the current batch's
        expected_* / slot_indices into those tensors so the replayed kernel
        launches verify and write the right slots.
        """
        self._launch.fill_from_plan(plan)

    def reset_launch_buffers_to_skip_sentinel(self) -> None:
        """Reset the entire fixed launch buffer to the skip-sentinel state.

        Replay-side pre-fill calls this when there is no plan (e.g. the
        batch is degenerate / out_cache_loc is empty) so the recorded
        canary kernel becomes a no-op for this replay rather than reusing
        stale data from a prior forward.
        """
        self._launch.verify_mask.fill_(-1)
        self._launch.verify_seq_positions.fill_(-1)

    def launch_for_capture(self, *, kernel_kind: int) -> None:
        """Capture-only: record one kernel launch as a no-op (skip-sentinel).

        Called separately for head (before original_forward) and tail
        (after) so the relative order of canary launches against the real
        model forward matches the eager path. Replay refills the buffers
        BEFORE ``graph.replay()``, so the same recorded launches become
        real verify+write work on every replay forward.

        Resets the buffers to skip-sentinel just before each launch so
        capture never accidentally verifies / writes — replay-side
        ``fill_from_plan`` overwrites the buffer contents at replay time
        before ``graph.replay()`` re-executes this launch.
        """
        if not self._config.enabled:
            return
        self.reset_launch_buffers_to_skip_sentinel()
        self._launch_kernel_only(kernel_kind=kernel_kind)

    def _launch_kernel_only(self, *, kernel_kind: int) -> None:
        """Launch a single kernel reading from the current fixed-buffer contents.

        Shared by capture-time recording (skip-sentinel contents) and
        replay-graph re-launch (real contents after pre-fill).

        K-half and V-half each get their own ``CanaryViolationSlot`` so the
        first-violation latch / ring / is_errored never cross-pollinate.
        """
        if kernel_kind == KERNEL_KIND_HEAD:
            src_buf_k, dst_buf_k = self._k_tail, self._k_head
            src_buf_v, dst_buf_v = self._v_tail, self._v_head
            slot_run_counter = self._device_state.slot_run_counter_head
            kernel_run_counter = self._device_state.kernel_run_counter_head
            kind_k = VIOLATION_KIND_HEAD_K
            kind_v = VIOLATION_KIND_HEAD_V
        else:
            src_buf_k, dst_buf_k = self._k_head, self._k_tail
            src_buf_v, dst_buf_v = self._v_head, self._v_tail
            slot_run_counter = self._device_state.slot_run_counter_tail
            kernel_run_counter = self._device_state.kernel_run_counter_tail
            kind_k = VIOLATION_KIND_TAIL_K
            kind_v = VIOLATION_KIND_TAIL_V

        buf_specs: List[Tuple[torch.Tensor, torch.Tensor, int, str]] = [
            (src_buf_k, dst_buf_k, self._k_slot_stride_bytes, kind_k)
        ]
        if self._has_v_half and src_buf_v is not None and dst_buf_v is not None:
            buf_specs.append((src_buf_v, dst_buf_v, self._v_slot_stride_bytes, kind_v))
        for src_buf, dst_buf, stride, kind in buf_specs:
            slot = self._device_state.get_violation_slot(kind)
            canary_step(
                src_buf=src_buf.view(torch.uint8).flatten(),
                dst_buf=dst_buf.view(torch.uint8).flatten(),
                slot_stride_bytes=stride,
                slot_indices=self._launch.slot_indices,
                expected_req_ids=self._launch.expected_req_ids,
                expected_token_ids=self._launch.expected_token_ids,
                expected_positions=self._launch.expected_positions,
                expected_prev_hashes=self._launch.expected_prev_hashes,
                verify_mask=self._launch.verify_mask,
                verify_seq_positions=self._launch.verify_seq_positions,
                violation_ring=slot.violation_ring,
                violation_ring_valid=slot.violation_ring_valid,
                violation_write_index=slot.violation_write_index,
                first_violation=slot.first_violation,
                first_violation_set=slot.first_violation_set,
                is_errored=slot.is_errored,
                slot_run_counter=slot_run_counter,
                kernel_run_counter=kernel_run_counter,
                kernel_kind=kernel_kind,
            )

    def _run_kernel_pair(
        self,
        *,
        plan: BatchPlan,
        kernel_kind: int,
    ) -> None:
        """Eager-path launch: fill the fixed buffers from the plan, then launch.

        The replay path bypasses this and goes through ``prepare_for_replay``
        + recorded-graph launches instead.
        """
        if not self._config.enabled:
            return
        total = plan.num_verify + plan.num_write
        if total == 0:
            return
        self._launch.fill_from_plan(plan)
        self._launch_kernel_only(kernel_kind=kernel_kind)

    def _cross_rank_max(self, local_flag: int) -> int:
        """Unconditional all-reduce-MAX on the local 1-byte flag.

        Critical: every forward must enter this allreduce so all peers agree
        on whether ANY rank saw a violation. If only the offending rank
        all-reduced, the others would block in the next NCCL collective
        (README §3 decision #3).

        Returns the post-reduce global flag.
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
        """README §5 — counter-zero detection + periodic print.

        The warmup zero-check uses a latch + ``>=`` trigger (not ``==``) so a
        step that's skipped (e.g. cuda graph capture / replay does not advance
        ``_forward_step``) cannot leapfrog past the warmup boundary and miss
        the check.

        Any raise path here MUST be cross-rank synchronized: a single-rank
        raise on a TP group would deadlock peers in the next NCCL collective.
        We allreduce-MAX a health flag across the TP group and only raise if
        every rank agrees.
        """
        step = self._forward_step
        period = max(1, self._config.health_print_every_n_forwards)
        kernel_head, kernel_tail, slot_head, slot_tail = self._latest_counters

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

        if step > 0 and step % period == 0:
            logger.info(
                "kv-canary: protected %d forwards "
                "(full.head=%d kernels / %d slots, full.tail=%d kernels / %d slots)",
                step,
                kernel_head,
                slot_head,
                kernel_tail,
                slot_tail,
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

    def _pull_first_violation(self, kind: str) -> Tuple[List[int], int]:
        """Synchronous D2H pull of one kind's first-violation row + write count.

        Only called on the error path; the hot path stays asynchronous via
        the side-stream event pump.
        """
        slot = self._device_state.get_violation_slot(kind)
        first_violation = slot.first_violation.cpu().tolist()
        write_index = int(slot.violation_write_index.cpu().item())
        return first_violation, write_index

    def _kinds_with_violation(self) -> List[str]:
        return [
            kind
            for kind in VIOLATION_KINDS
            if self._latest_is_errored_per_kind.get(kind, 0)
        ]

    def _handle_violation_log(self) -> None:
        now = time.time()
        for kind in self._kinds_with_violation():
            first_violation, write_index = self._pull_first_violation(kind)
            fail_reason = int(first_violation[1])
            key = (kind, fail_reason)
            last_log = self._last_log_time_per_reason.get(key, 0.0)
            if now - last_log >= _LOG_RATE_LIMIT_SECONDS:
                self._last_log_time_per_reason[key] = now
                logger.error(self._format_violation(kind, first_violation, write_index))

        # Reset GPU-side flags/rings so subsequent NEW violations surface.
        # Without this, each kind's device flag stays 1 forever (every
        # subsequent forward re-triggers a sync D2H + allreduce);
        # ``first_violation_set`` stays latched (new violations silently
        # masked); the ring fills up and CAS-fails all future writes.
        # Host-cached ``_latest_is_errored_per_kind`` is reset to 0 so the
        # next ``end_of_forward`` doesn't re-enter this handler before the
        # next async event poll catches up.
        self._device_state.reset_violation_state()
        for kind in VIOLATION_KINDS:
            self._latest_is_errored_per_kind[kind] = 0

    def _raise_with_first_violation(self) -> None:
        kinds = self._kinds_with_violation()
        if not kinds:
            # Defensive: global flag was set (e.g. peer rank fired the
            # allreduce) but our own pumps haven't surfaced any per-kind
            # flag yet. Fall back to scanning every kind synchronously.
            kinds = list(VIOLATION_KINDS)
        messages: List[str] = []
        for kind in kinds:
            first_violation, write_index = self._pull_first_violation(kind)
            if int(first_violation[1]) == 0 and write_index == 0:
                continue
            messages.append(self._format_violation(kind, first_violation, write_index))
        if not messages:
            messages.append(
                "kv-canary: raise path entered but no per-kind violation rows "
                "found (peer rank may have observed the violation)"
            )
        raise RuntimeError("\n".join(messages))

    @staticmethod
    def _format_violation(
        kind: str, first_violation: List[int], write_index: int
    ) -> str:
        (
            kernel_kind,
            fail_reason,
            slot_idx,
            req_id,
            token_id,
            position,
            expected_hash,
            actual_hash,
        ) = first_violation
        u64_mask = (1 << 64) - 1
        try:
            reason_name = FailReason(int(fail_reason)).name
        except ValueError:
            reason_name = f"unknown({int(fail_reason)})"
        return (
            f"kv-canary violation: kind={kind} kernel_kind={int(kernel_kind)} "
            f"fail_reason={reason_name} slot_idx={int(slot_idx)} "
            f"req_id={int(req_id)} token_id={int(token_id)} position={int(position)} "
            f"expected_hash={int(expected_hash) & u64_mask:#x} "
            f"actual_hash={int(actual_hash) & u64_mask:#x} "
            f"(total violations recorded: {write_index})"
        )
