from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.jit_kernel.kv_cache_canary import (
    KERNEL_KIND_HEAD,
    KERNEL_KIND_SWEEP,
    KERNEL_KIND_TAIL,
    FailReason,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlanGpu
from sglang.srt.kv_cache_canary.config import (
    CanaryConfig,
    CanaryMode,
    real_kv_hash_mode_to_int,
    real_kv_hash_read_bytes,
)
from sglang.srt.kv_cache_canary.endpoint import (
    CanaryEndpoint,
    _empty_real_kv_buf,
)
from sglang.srt.kv_cache_canary.host_state import (
    VIOLATION_KIND_HEAD_K,
    VIOLATION_KIND_HEAD_V,
    VIOLATION_KIND_SWEEP_K,
    VIOLATION_KIND_SWEEP_V,
    VIOLATION_KIND_TAIL_K,
    VIOLATION_KIND_TAIL_V,
    VIOLATION_KINDS,
    BatchPlan,
    CanaryDeviceState,
    allocate_batch_plan_gpu,
    fill_batch_plan_gpu_from_plan,
    reset_batch_plan_gpu_to_skip_sentinel,
    translate_alive_slots_for_swa,
)
from sglang.srt.kv_cache_canary.pool_patch import (
    CanaryBufferGroup,
    PoolKind,
)
from sglang.srt.kv_cache_canary.sweep import (
    build_sweep_plan,
    compute_alive_owned_slots,
)

logger = logging.getLogger(__name__)


_LOG_RATE_LIMIT_SECONDS: float = 5.0

_FAIL_REASON_DESCRIPTIONS: Dict[int, str] = {
    int(FailReason.NONE): "no failure",
    int(FailReason.TOKEN_ID): "slot's stored token_id does not match the write entry",
    int(FailReason.POSITION): "slot's stored position does not match the write entry",
    int(FailReason.HASH): "slot's chain hash diverged from splitmix64 recomputation",
    int(
        FailReason.POSITION_MONOTONIC
    ): "verify-time position does not match what was written",
    int(FailReason.REAL_KV_HASH): "real-KV slot bytes changed underneath the canary",
    int(
        FailReason.INPUT_TOKEN_MISMATCH
    ): "write-entry token_id diverged from the oracle's expected input token",
    int(
        FailReason.INPUT_POSITION_MISMATCH
    ): "write-entry position diverged from the oracle's expected input position",
}


def _fail_reason_description(reason: int) -> str:
    return _FAIL_REASON_DESCRIPTIONS.get(reason, "unknown")


@dataclass(frozen=True, slots=True, kw_only=True)
class _RealKvKernelArgs:
    """Resolved real-KV kernel arguments shared by head + tail endpoints."""

    buf: torch.Tensor
    read_bytes: int
    hash_mode_int: int


def _resolve_real_kv_kernel_args(
    *,
    buffer_group: CanaryBufferGroup,
    config: CanaryConfig,
    device: torch.device,
) -> _RealKvKernelArgs:
    """Resolve the real-KV kernel arguments for this canary buffer group + config.

    When the real-KV-hash feature is disabled (or the canary buffer group has no
    real KV source) the kernel receives a 2D placeholder tensor and
    zero read_bytes so it short-circuits on the OFF early-out.
    """
    hash_mode_int = real_kv_hash_mode_to_int(config.real_kv_hash_mode)
    if (
        buffer_group.real_kv_source is not None
        and hash_mode_int != 0
        and buffer_group.real_kv_slot_stride_bytes > 0
    ):
        real_kv_source = buffer_group.real_kv_source.contiguous()
        num_slots = int(real_kv_source.shape[0])
        buf = real_kv_source.view(torch.uint8).view(num_slots, -1)
        read_bytes = real_kv_hash_read_bytes(
            config.real_kv_hash_mode, buffer_group.real_kv_slot_stride_bytes
        )
        return _RealKvKernelArgs(
            buf=buf,
            read_bytes=read_bytes,
            hash_mode_int=hash_mode_int,
        )
    return _RealKvKernelArgs(
        buf=_empty_real_kv_buf(device),
        read_bytes=0,
        hash_mode_int=hash_mode_int,
    )


class CanaryRunner:
    """Per-rank orchestrator: owns the GPU violation buffer + counters,
    the side stream that async-D2Hs the ``is_errored`` flag + counters,
    and the log / raise policy.

    .. warning::
        These three pieces are critical safety logic and must NOT be
        removed by an automated /simplify pass:

        - the side-stream + event-based async D2H pump
          (``_record_poll_events`` / ``_pull_latest_from_events``) — keeps
          the hot path non-blocking,
        - the health monitoring in ``_maybe_health_check`` — detects a
          dead canary kernel by checking that ``kernel_run_counter`` is
          advancing after warmup,
        - the unconditional cross-rank allreduce in ``_cross_rank_max``,
          called from every ``end_of_forward``.

        Removing any of them silently disables the canary or produces TP
        deadlock when raising. Do not simplify away.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        buffer_group: CanaryBufferGroup,
        device: torch.device,
        verify_capacity: int,
        write_capacity: int,
        write_req_capacity: int,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
        tp_rank: int = 0,
    ) -> None:
        self._config = config
        self._device = device
        self._pool_kind = buffer_group.kind
        self._has_v_half: bool = buffer_group.has_v_half
        self._buffer_group = buffer_group
        self._tp_rank: int = int(tp_rank)

        self._device_state = CanaryDeviceState.allocate(
            device=device, ring_capacity=config.violation_ring_capacity
        )
        real_kv = _resolve_real_kv_kernel_args(
            buffer_group=buffer_group, config=config, device=device
        )
        self._head_endpoint = self._make_endpoint(
            kernel_kind=KERNEL_KIND_HEAD,
            buffer_group=buffer_group,
            violation_kind_k=VIOLATION_KIND_HEAD_K,
            violation_kind_v=VIOLATION_KIND_HEAD_V,
            slot_run_counter=self._device_state.slot_run_counter_head,
            kernel_run_counter=self._device_state.kernel_run_counter_head,
            use_head=True,
            real_kv=real_kv,
        )
        self._tail_endpoint = self._make_endpoint(
            kernel_kind=KERNEL_KIND_TAIL,
            buffer_group=buffer_group,
            violation_kind_k=VIOLATION_KIND_TAIL_K,
            violation_kind_v=VIOLATION_KIND_TAIL_V,
            slot_run_counter=self._device_state.slot_run_counter_tail,
            kernel_run_counter=self._device_state.kernel_run_counter_tail,
            use_head=False,
            real_kv=real_kv,
        )
        # Sweep verifies post-model-write real_kv_hash for every alive slot,
        # so it must read the tail buffer (which stores post-write hashes).
        # Sweep plans have num_write == 0 so the buffer is never written.
        # K-half / V-half each gets its own sweep violation slot for the
        # same reason head/tail do.
        self._sweep_endpoint = self._make_endpoint(
            kernel_kind=KERNEL_KIND_SWEEP,
            buffer_group=buffer_group,
            violation_kind_k=VIOLATION_KIND_SWEEP_K,
            violation_kind_v=VIOLATION_KIND_SWEEP_V,
            slot_run_counter=self._device_state.slot_run_counter_sweep,
            kernel_run_counter=self._device_state.kernel_run_counter_sweep,
            use_head=False,
            real_kv=real_kv,
        )
        # Pre-allocated fixed-address launch buffers. Cuda graph capture
        # records these specific addresses; replay-side host code refills
        # them in-place before ``graph.replay()`` so the recorded kernel
        # launches see the correct expected_* / slot_indices for the batch.
        self._launch = allocate_batch_plan_gpu(
            device=device,
            verify_capacity=verify_capacity,
            write_capacity=write_capacity,
            write_req_capacity=write_req_capacity,
        )
        # Sweep launch buffers are sized to verify every alive slot in the
        # pool (= the canary buffer's slot count). Independent tensor
        # instances so they cannot conflict with the per-step buffers that
        # are baked into cuda-graph capture.
        sweep_verify_capacity = int(buffer_group.k_head.shape[0])
        self._sweep_launch: Optional[BatchPlanGpu] = (
            allocate_batch_plan_gpu(
                device=device,
                verify_capacity=sweep_verify_capacity,
                write_capacity=1,
                write_req_capacity=1,
            )
            if config.real_data_sweep_every_n_steps > 0 and sweep_verify_capacity > 0
            else None
        )
        self._sweep_every_n: int = int(config.real_data_sweep_every_n_steps)
        self._req_to_token_pool: Optional["ReqToTokenPool"] = req_to_token_pool
        self._last_forward_batch: Optional["ForwardBatch"] = None
        self._last_plan: Optional[BatchPlan] = None
        self._side_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._is_errored_host_per_kind: Dict[str, torch.Tensor] = {
            kind: torch.zeros(
                1, dtype=torch.int32, pin_memory=torch.cuda.is_available()
            )
            for kind in VIOLATION_KINDS
        }
        self._counters_host = torch.zeros(
            6, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        self._is_errored_event_per_kind: Dict[str, Optional[torch.cuda.Event]] = {
            kind: (torch.cuda.Event() if torch.cuda.is_available() else None)
            for kind in VIOLATION_KINDS
        }
        self._counters_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )

        self._latest_is_errored_per_kind: Dict[str, int] = {
            kind: 0 for kind in VIOLATION_KINDS
        }
        self._latest_counters: Tuple[int, int, int, int, int, int] = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        self._raise_latch: bool = False
        self._last_log_time_per_reason: Dict[Tuple[str, int], float] = {}
        self._forward_step: int = 0
        self._poll_armed: bool = False
        self._warmup_check_done: bool = False

    def _make_endpoint(
        self,
        *,
        kernel_kind: int,
        buffer_group: CanaryBufferGroup,
        violation_kind_k: str,
        violation_kind_v: str,
        slot_run_counter: torch.Tensor,
        kernel_run_counter: torch.Tensor,
        use_head: bool,
        real_kv: "_RealKvKernelArgs",
    ) -> CanaryEndpoint:
        return CanaryEndpoint(
            kernel_kind=kernel_kind,
            k_canary_buf=buffer_group.k_head if use_head else buffer_group.k_tail,
            v_canary_buf=buffer_group.v_head if use_head else buffer_group.v_tail,
            k_violation=self._device_state.get_violation_slot(violation_kind_k),
            v_violation=(
                self._device_state.get_violation_slot(violation_kind_v)
                if self._has_v_half
                else None
            ),
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_buf=real_kv.buf,
            real_kv_read_bytes=real_kv.read_bytes,
            real_kv_hash_mode=real_kv.hash_mode_int,
        )

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

    def run_head(self, *, plan: BatchPlan) -> None:
        self._last_plan = plan
        self._run_kernel_pair(plan=plan, kernel_kind=KERNEL_KIND_HEAD)

    def run_tail(self, *, plan: BatchPlan) -> None:
        self._run_kernel_pair(plan=plan, kernel_kind=KERNEL_KIND_TAIL)

    def set_last_forward_batch(self, forward_batch: "ForwardBatch") -> None:
        """Stash the current forward batch for sweep / perturb hooks.

        Called by the api-level run_head wrapper so :meth:`_run_sweep` and
        future self-test perturb code can pull alive req metadata without
        threading the forward_batch through every kernel path.
        """
        self._last_forward_batch = forward_batch

    def attach_req_to_token_pool(self, req_to_token_pool: "ReqToTokenPool") -> None:
        """Bind the scheduler's ``req_to_token`` pool for sweep's alive-set lookup."""
        self._req_to_token_pool = req_to_token_pool

    @property
    def last_plan(self) -> Optional[BatchPlan]:
        return self._last_plan

    @property
    def device_state(self) -> CanaryDeviceState:
        return self._device_state

    def end_of_forward(self) -> None:
        """Called once per forward (after run_tail) on the compute stream.

        Records D2H copies on the side stream, queries the recorded event
        non-blockingly, refreshes the host-side cached error flag + health
        counters, increments the step counter, and unconditionally
        allreduces the local error flag so every rank decides to raise in
        lock-step (no single-rank raise → no NCCL deadlock).

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

        # Local import to break the test_utils ↔ runner module cycle.
        from sglang.srt.kv_cache_canary.test_utils import maybe_perturb_real_kv_bytes

        maybe_perturb_real_kv_bytes(
            runner=self,
            req_to_token_pool=self._req_to_token_pool,
            forward_batch=self._last_forward_batch,
        )

        if self._sweep_every_n > 0 and self._forward_step % self._sweep_every_n == 0:
            self._run_sweep()

        local_flag = 1 if any(self._latest_is_errored_per_kind.values()) else 0
        global_flag = self._cross_rank_max(local_flag)
        self._maybe_health_check()
        if global_flag:
            self._handle_violation_global()

    def _run_sweep(self) -> None:
        """Verify every alive slot's real_kv_hash. Skip the chain hash check.

        Reuses the per-step kernel via :data:`KERNEL_KIND_SWEEP`; the sweep
        endpoint feeds into independent sweep_k / sweep_v violation slots
        and its own slot/kernel run counters so the per-step health
        accounting stays clean.
        """
        if self._sweep_launch is None:
            return
        if self._req_to_token_pool is None or self._last_forward_batch is None:
            return
        alive_slots = compute_alive_owned_slots(
            req_to_token_pool=self._req_to_token_pool,
            forward_batch=self._last_forward_batch,
        )
        if alive_slots.numel() == 0:
            return
        lut = self._buffer_group.swa_index_lut
        if lut is not None:
            alive_slots = translate_alive_slots_for_swa(
                alive_slots=alive_slots, lut=lut
            )
            if alive_slots.numel() == 0:
                return
        plan = build_sweep_plan(
            canary_buf=self._buffer_group.k_tail,
            alive_slot_indices=alive_slots,
        )
        if plan.num_verify == 0:
            return
        fill_batch_plan_gpu_from_plan(launch=self._sweep_launch, plan=plan)
        self._launch_kernel_only(
            kernel_kind=KERNEL_KIND_SWEEP,
            launch_buffers=self._sweep_launch,
        )

    def _pull_latest_from_events(self) -> None:
        """Non-blocking event.query() — if done, refresh host-cached values."""
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
            # Counter D2H is intentionally decoupled from the periodic print
            # interval (``health_print_every_n_forwards``). Refreshing every
            # forward lets the warmup health check see fresh values by
            # ``counter_zero_warmup_forwards`` (~64); tying the D2H to the
            # 1024-forward print period would leave ``_latest_counters`` stuck
            # at the step-0 snapshot and trip a spurious "kernel never ran".
            if self._counters_event is not None:
                ds = self._device_state
                for host_slot, device_counter in zip(
                    self._counters_host,
                    (
                        ds.kernel_run_counter_head,
                        ds.kernel_run_counter_tail,
                        ds.slot_run_counter_head,
                        ds.slot_run_counter_tail,
                        ds.kernel_run_counter_sweep,
                        ds.slot_run_counter_sweep,
                    ),
                ):
                    host_slot.copy_(device_counter.flatten()[0], non_blocking=True)
                self._counters_event.record(stream=self._side_stream)

    def prepare_for_replay(self, *, plan: BatchPlan) -> None:
        """Refill the fixed launch buffers in-place for a replay forward."""
        fill_batch_plan_gpu_from_plan(launch=self._launch, plan=plan)

    def reset_launch_buffers_to_skip_sentinel(self) -> None:
        """Reset the launch buffers to skip-sentinel state (kernel no-op)."""
        reset_batch_plan_gpu_to_skip_sentinel(self._launch)

    def launch_for_capture(self, *, kernel_kind: int) -> None:
        """Capture-only: record one kernel launch as a no-op (skip-sentinel).

        Crucially, this does NOT call ``reset_launch_buffers_to_skip_sentinel``:
        that helper issues ``zero_()`` ops on the compute stream, and any op
        on the compute stream during cuda graph capture gets recorded INTO
        the graph. A captured zero would then wipe the active-mask buffer
        the replay-side ``prepare_for_replay`` had just filled with the
        real plan, leaving every replay launching with all-zero masks (the
        kernel skips every entry, ``slot_run_counter`` stays at 0, the
        canary never actually verifies anything).

        The buffers are guaranteed to already be all zero at capture time:
        :class:`BatchPlanGpu` is allocated with ``torch.zeros`` and
        nothing writes a non-skip-sentinel value to it before
        ``init_device_graphs`` runs (the canary attaches BEFORE graph
        init, and no eager forward with a real plan can fire before then).
        The captured kernel therefore records a no-op grid by virtue of
        the initial state, with no captured zeroing op needed.
        """
        if not self._config.enabled:
            return
        self._launch_kernel_only(kernel_kind=kernel_kind)

    def _launch_kernel_only(
        self,
        *,
        kernel_kind: int,
        launch_buffers: Optional[BatchPlanGpu] = None,
    ) -> None:
        """Launch one kernel pair from the current fixed-buffer contents.

        Shared by capture-time recording (skip-sentinel contents) and
        replay-graph re-launch (real contents after pre-fill). K-half and
        V-half each get their own ``CanaryViolationSlot`` so the
        first-violation latch / ring / is_errored never cross-pollinate;
        head/tail symmetry is captured by :class:`CanaryEndpoint`. Each
        endpoint is self-verifying — it reads and writes its own canary
        buffer, no cross-shadow link.

        ``kernel_kind == KERNEL_KIND_SWEEP`` runs the verify-only sweep
        endpoint. Sweep fires at ``end_of_forward`` (post-model-write) and
        verifies real_kv_hash for every alive slot, including the new write
        positions of this forward. The sweep endpoint is wired to the
        tail buffers (which capture post-model-write real_kv_hash) so the
        check sees consistent post-write bytes for every alive slot.
        """
        if kernel_kind == KERNEL_KIND_HEAD:
            endpoint = self._head_endpoint
        elif kernel_kind == KERNEL_KIND_TAIL:
            endpoint = self._tail_endpoint
        elif kernel_kind == KERNEL_KIND_SWEEP:
            endpoint = self._sweep_endpoint
        else:
            raise ValueError(f"kv-canary: unknown kernel_kind {kernel_kind}")
        buffers = launch_buffers if launch_buffers is not None else self._launch
        endpoint.launch(plan=buffers, seed=int(self._config.seed))

    def _run_kernel_pair(
        self,
        *,
        plan: BatchPlan,
        kernel_kind: int,
    ) -> None:
        """Eager-path launch: fill the fixed buffers from the plan, then launch.

        Always launches the kernel — even when the plan has zero verify
        entries and zero write entries — so the liveness counter
        (``kernel_run_counter``) advances on every forward. The kernel's
        unconditional entry-side atomicAdd is what the health monitor uses
        to detect "canary is actually executing"; gating the host launch
        on plan size would silently zero the counter during no-work
        forwards and trip a spurious "kernel never ran" panic.
        """
        if not self._config.enabled:
            return
        fill_batch_plan_gpu_from_plan(launch=self._launch, plan=plan)
        self._launch_kernel_only(kernel_kind=kernel_kind)

    def _cross_rank_max(self, local_flag: int) -> int:
        """Unconditional all-reduce-MAX on the local 1-byte flag.

        Critical: every forward must enter this allreduce so all peers agree
        on whether ANY rank saw a violation. If only the offending rank
        all-reduced, the others would block in the next NCCL collective.

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
                self._pool_kind.value,
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

    def _pull_first_violation(self, kind: str) -> Tuple[List[int], int]:
        """Synchronous D2H pull of one kind's first-violation row + write count."""
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

        self._device_state.reset_violation_state()
        for kind in VIOLATION_KINDS:
            self._latest_is_errored_per_kind[kind] = 0

    def _raise_with_first_violation(self) -> None:
        kinds = self._kinds_with_violation()
        if not kinds:
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
        """Format a violation row as a labelled, multi-line error message.

        Violation row carries both actual fields (read from the slot at
        violation time) and expected fields (computed by the canary from
        the live verify plan), so every fail_reason can print a full
        expected vs actual diff.
        """
        (
            kernel_kind,
            fail_reason,
            slot_idx,
            token_id,
            position,
            expected_hash,
            actual_hash,
            expected_position,
        ) = first_violation
        u64_mask = (1 << 64) - 1
        reason_int = int(fail_reason)
        try:
            reason = FailReason(reason_int)
            reason_name = reason.name
        except ValueError:
            reason = None
            reason_name = f"unknown({reason_int})"
        kernel_label = {
            KERNEL_KIND_HEAD: "HEAD",
            KERNEL_KIND_TAIL: "TAIL",
            KERNEL_KIND_SWEEP: "SWEEP",
        }.get(int(kernel_kind), str(int(kernel_kind)))

        lines = [
            "kv-canary violation:",
            f"  canary_kind:       {kind} (one of head_k/head_v/tail_k/tail_v/sweep_k/sweep_v)",
            f"  kernel_kind:       {kernel_label}",
            f"  fail_reason:       {reason_name} ({_fail_reason_description(reason_int)})",
            f"  slot_idx:          {int(slot_idx)}",
            f"  position:          expected={int(expected_position)} actual={int(position)}",
        ]
        if reason is FailReason.INPUT_TOKEN_MISMATCH:
            lines.append(
                f"  token_id:          expected={int(expected_hash)} "
                f"actual={int(actual_hash)}"
            )
        else:
            lines.append(f"  actual token_id:   {int(token_id)}")

        if reason in (FailReason.HASH, FailReason.REAL_KV_HASH):
            lines += [
                f"  expected_hash:     {int(expected_hash) & u64_mask:#018x}",
                f"  actual_hash:       {int(actual_hash) & u64_mask:#018x}",
                f"  hash_xor_diff:     {(int(expected_hash) ^ int(actual_hash)) & u64_mask:#018x}",
            ]
        lines.append(f"  total_violations:  {write_index} (since last reset)")
        return "\n".join(lines)
