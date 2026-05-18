from __future__ import annotations

import logging
import threading
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
    BatchPlan,
    CanaryDeviceState,
    CanaryHostState,
)
from sglang.srt.kv_cache_canary.pool_patch import (
    attach_shadow_buffers,
    get_shadow_buffers,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


_LOG_RATE_LIMIT_SECONDS: float = 5.0


class CanaryRunner:
    """Top-level orchestrator for KV cache canary.

    One instance lives on each rank. Owns the host-side request state, the
    GPU-side violation buffer + counters, the side stream and background
    daemon that polls the ``is_errored`` flag and health counters, and the
    log/raise policy.

    .. warning::
        Several pieces in this class are critical safety logic and must NOT
        be simplified away by an automated pass:

        - the daemon poll thread (README §4),
        - the §5 health monitoring (zero-counter → raise),
        - the unconditional cross-rank allreduce in ``maybe_raise_on_step``.

        Removing any of them silently disables the canary or produces TP
        deadlock when raising. Do not simplify away.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        pool: "MHATokenToKVPool",
        num_req_slots: int,
        device: torch.device,
    ) -> None:
        self._config = config
        self._device = device

        attach_shadow_buffers(pool)
        self._slot_stride_bytes = pool.canary_slot_stride_bytes
        self._k_head, self._k_tail, self._v_head, self._v_tail = get_shadow_buffers(
            pool
        )

        self.host_state = CanaryHostState(config=config, num_req_slots=num_req_slots)
        self._device_state = CanaryDeviceState.allocate(
            device=device, ring_capacity=config.violation_ring_capacity
        )
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._is_errored_host = torch.zeros(
            1, dtype=torch.int32, pin_memory=torch.cuda.is_available()
        )
        self._counters_host = torch.zeros(
            4, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        self._is_errored_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )
        self._counters_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )

        self._daemon_lock = threading.Lock()
        # Latest values pulled by the daemon — main thread reads these.
        self._latest_is_errored: int = 0
        self._latest_counters: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._raise_latch: bool = False
        self._last_log_time_per_reason: Dict[int, float] = {}
        self._forward_step: int = 0

        self._daemon_stop = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
        self._start_daemon()

    @property
    def config(self) -> CanaryConfig:
        return self._config

    def run_head(self, *, plan: BatchPlan, slot_indices: torch.Tensor) -> None:
        self._run_kernel_pair(
            plan=plan, write_slot_indices=slot_indices, kernel_kind=KERNEL_KIND_HEAD
        )

    def run_tail(self, *, plan: BatchPlan, slot_indices: torch.Tensor) -> None:
        self._run_kernel_pair(
            plan=plan, write_slot_indices=slot_indices, kernel_kind=KERNEL_KIND_TAIL
        )

    def end_of_forward(self) -> None:
        """Called once per forward (after run_tail) on the compute stream.

        Records events for the daemon to query, increments the step counter,
        and unconditionally allreduces the (host-cached) error flag so every
        rank decides to raise in lock-step (no single-rank raise → no NCCL
        deadlock; README §3 decision #3).
        """
        if not self._config.enabled:
            return
        self._record_poll_events()
        self._forward_step += 1
        with self._daemon_lock:
            local_flag = self._latest_is_errored
        global_flag = self._cross_rank_max(local_flag)
        self._maybe_health_check()
        if global_flag:
            self._handle_violation_global()

    def shutdown(self) -> None:
        self._daemon_stop.set()
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=2.0)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def _record_poll_events(self) -> None:
        if self._side_stream is None or self._is_errored_event is None:
            return
        compute_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._side_stream):
            self._side_stream.wait_stream(compute_stream)
            self._is_errored_host.copy_(
                self._device_state.is_errored, non_blocking=True
            )
            self._is_errored_event.record(stream=self._side_stream)
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

    def _start_daemon(self) -> None:
        if not self._config.enabled or self._side_stream is None:
            return
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            name="kv-canary-poll",
            daemon=True,
        )
        self._daemon_thread.start()

    def _daemon_loop(self) -> None:
        poll_interval = max(0.001, self._config.daemon_poll_seconds)
        while not self._daemon_stop.is_set():
            try:
                if (
                    self._is_errored_event is not None
                    and self._is_errored_event.query()
                ):
                    flag = int(self._is_errored_host.item())
                    with self._daemon_lock:
                        self._latest_is_errored = flag
                if self._counters_event is not None and self._counters_event.query():
                    counters = tuple(int(x) for x in self._counters_host.tolist())
                    with self._daemon_lock:
                        self._latest_counters = counters  # type: ignore[assignment]
            except Exception:
                logger.exception("kv-canary: daemon poll iteration failed")
            self._daemon_stop.wait(poll_interval)

    def _run_kernel_pair(
        self,
        *,
        plan: BatchPlan,
        write_slot_indices: torch.Tensor,
        kernel_kind: int,
    ) -> None:
        if not self._config.enabled:
            return
        total = plan.num_verify + plan.num_write
        if total == 0:
            return

        slot_indices = self._make_slot_indices_tensor(
            plan=plan, write_slot_indices=write_slot_indices
        )

        if kernel_kind == KERNEL_KIND_HEAD:
            src_buf_k, dst_buf_k = self._k_tail, self._k_head
            src_buf_v, dst_buf_v = self._v_tail, self._v_head
            slot_run_counter = self._device_state.slot_run_counter_head
            kernel_run_counter = self._device_state.kernel_run_counter_head
        else:
            src_buf_k, dst_buf_k = self._k_head, self._k_tail
            src_buf_v, dst_buf_v = self._v_head, self._v_tail
            slot_run_counter = self._device_state.slot_run_counter_tail
            kernel_run_counter = self._device_state.kernel_run_counter_tail

        expected_req_ids = _to_int64(plan.expected_req_ids, self._device)
        expected_token_ids = _to_int64(plan.expected_token_ids, self._device)
        expected_positions = _to_int64(plan.expected_positions, self._device)
        expected_prev_hashes = _to_int64(plan.expected_prev_hashes, self._device)
        verify_mask = _to_int32(plan.verify_mask, self._device)
        verify_seq_positions = _to_int64(plan.verify_seq_positions, self._device)

        for src_buf, dst_buf in ((src_buf_k, dst_buf_k), (src_buf_v, dst_buf_v)):
            canary_step(
                src_buf=src_buf.view(torch.uint8).flatten(),
                dst_buf=dst_buf.view(torch.uint8).flatten(),
                slot_stride_bytes=self._slot_stride_bytes,
                slot_indices=slot_indices,
                expected_req_ids=expected_req_ids,
                expected_token_ids=expected_token_ids,
                expected_positions=expected_positions,
                expected_prev_hashes=expected_prev_hashes,
                verify_mask=verify_mask,
                verify_seq_positions=verify_seq_positions,
                violation_ring=self._device_state.violation_ring,
                violation_ring_valid=self._device_state.violation_ring_valid,
                violation_write_index=self._device_state.violation_write_index,
                first_violation=self._device_state.first_violation,
                first_violation_set=self._device_state.first_violation_set,
                is_errored=self._device_state.is_errored,
                slot_run_counter=slot_run_counter,
                kernel_run_counter=kernel_run_counter,
                kernel_kind=kernel_kind,
            )

    def _make_slot_indices_tensor(
        self, *, plan: BatchPlan, write_slot_indices: torch.Tensor
    ) -> torch.Tensor:
        write_part = write_slot_indices.to(device=self._device, dtype=torch.int64)
        if plan.num_verify == 0:
            return write_part
        verify_part = torch.tensor(
            plan.verify_slot_indices, dtype=torch.int64, device=self._device
        )
        return torch.cat([verify_part, write_part])

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
        """README §5 — counter-zero detection + periodic print."""
        step = self._forward_step
        period = max(1, self._config.health_print_every_n_forwards)
        with self._daemon_lock:
            counters = self._latest_counters
        kernel_head, kernel_tail, slot_head, slot_tail = counters

        if step == self._config.counter_zero_warmup_forwards:
            if kernel_head == 0 or kernel_tail == 0:
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

    def _handle_violation_log(self) -> None:
        # Pull the first-violation row (synchronous D2H — only happens on the
        # error path; the hot path stays asynchronous via the daemon).
        first_violation = self._device_state.first_violation.cpu().tolist()
        write_index = int(self._device_state.violation_write_index.cpu().item())
        fail_reason = int(first_violation[1])

        now = time.time()
        last_log = self._last_log_time_per_reason.get(fail_reason, 0.0)
        if now - last_log < _LOG_RATE_LIMIT_SECONDS:
            return
        self._last_log_time_per_reason[fail_reason] = now
        logger.error(self._format_violation(first_violation, write_index))

    def _raise_with_first_violation(self) -> None:
        first_violation = self._device_state.first_violation.cpu().tolist()
        write_index = int(self._device_state.violation_write_index.cpu().item())
        raise RuntimeError(self._format_violation(first_violation, write_index))

    @staticmethod
    def _format_violation(first_violation: List[int], write_index: int) -> str:
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
            f"kv-canary violation: kernel_kind={int(kernel_kind)} "
            f"fail_reason={reason_name} slot_idx={int(slot_idx)} "
            f"req_id={int(req_id)} token_id={int(token_id)} position={int(position)} "
            f"expected_hash={int(expected_hash) & u64_mask:#x} "
            f"actual_hash={int(actual_hash) & u64_mask:#x} "
            f"(total violations recorded: {write_index})"
        )


def _to_int64(values: List[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _to_int32(values: List[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)
