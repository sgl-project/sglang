from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.distributed as dist

from sglang.jit_kernel.kv_cache_canary_plan import canary_plan_step
from sglang.jit_kernel.kv_cache_canary_verify import (
    CanaryLaunchTag,
    VerifyPlan,
)
from sglang.jit_kernel.kv_cache_canary_write import WritePlan
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_cache_canary.plan_input import (
    AliveReqSnapshot,
    PlanInput,
    build_plan_input_per_forward,
    build_plan_input_radix_sweep,
    build_plan_input_running_sweep,
)
from sglang.srt.kv_cache_canary.pool_patch import attach_canary_buffers
from sglang.srt.kv_cache_canary.violation_state import CanaryDeviceState

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_HEALTH_CHECK_EVERY_N_STEPS: int = 1000
_HEALTH_CHECK_WARMUP_STEPS: int = 100


class CanaryRunner:
    """Owns all canary state for one ModelRunner. Constructed once during install_canary, lives until
    server shutdown.

    Internal state (private — never touched outside this class):
        config: CanaryConfig
        device_state: CanaryDeviceState
        endpoints_per_pool: tuple[tuple[CanaryEndpoint, ...], ...]  # one tuple per pool
        verify_plan_per_forward / write_plan_per_forward: VerifyPlan / WritePlan sized for per-forward
            capacity (= max_batch_size × max_seq_len for verify, max_batch_size for write).
        verify_plan_sweep / write_plan_sweep: sized for sweep capacity (= total pool slots).
        step_counter: int, host-side, bumped per forward.
        last_sweep_step: int, host-side.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        pools: list["KVCache"],
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
        radix_cache: Optional["BasePrefixCache"] = None,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
        sweep_verify_capacity: int,
        swa_window_size: int = 0,
    ) -> None:
        self.config = config
        self._device = device
        self._tp_group = tp_group
        self._req_to_token_pool = req_to_token_pool
        self._radix_cache = radix_cache
        self._swa_window_size = int(swa_window_size)

        groups_per_pool: list[tuple[CanaryBufferGroup, ...]] = []
        for pool in pools:
            groups_per_pool.append(
                attach_canary_buffers(pool=pool, config=config, device=device)
            )
        self._groups_per_pool: tuple[tuple[CanaryBufferGroup, ...], ...] = tuple(
            groups_per_pool
        )

        self._device_state = CanaryDeviceState.allocate(
            config=config, device=device, num_tags=len(CanaryLaunchTag)
        )

        endpoints_per_pool: list[tuple[CanaryEndpoint, ...]] = []
        for groups in self._groups_per_pool:
            pool_endpoints: list[CanaryEndpoint] = []
            for group in groups:
                pool_endpoints.extend(
                    build_endpoints_from_group(
                        group=group, device_state=self._device_state
                    )
                )
            endpoints_per_pool.append(tuple(pool_endpoints))
        self._endpoints_per_pool: tuple[tuple[CanaryEndpoint, ...], ...] = tuple(
            endpoints_per_pool
        )

        self._verify_plan_per_forward = VerifyPlan.allocate(
            verify_capacity=max(1, per_forward_verify_capacity), device=device
        )
        self._write_plan_per_forward = WritePlan.allocate(
            write_req_capacity=max(1, per_forward_write_req_capacity), device=device
        )
        self._verify_plan_sweep_running = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._verify_plan_sweep_radix = VerifyPlan.allocate(
            verify_capacity=max(1, sweep_verify_capacity), device=device
        )
        self._write_plan_sweep = WritePlan.allocate(write_req_capacity=1, device=device)

        write_entry_capacity = max(1, per_forward_write_entry_capacity)
        self._expected_input_tokens = torch.zeros(
            write_entry_capacity, dtype=torch.int32, device=device
        )
        self._expected_input_positions = torch.zeros(
            write_entry_capacity, dtype=torch.int32, device=device
        )

        self._pump_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if torch.cuda.is_available() else None
        )
        self._pump_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if torch.cuda.is_available() else None
        )
        self._previous_pump_event: Optional[torch.cuda.Event] = None

        self._step_counter: int = 0
        self._last_sweep_step: int = -1
        self._sweep_passes: int = 0
        self._raised: bool = False
        self._perturb_undo: Optional[tuple[int, int, int]] = None
        self._last_forward_batch: Optional["ForwardBatch"] = None

        active: set[CanaryLaunchTag] = set()
        for endpoints in self._endpoints_per_pool:
            for endpoint in endpoints:
                active.add(endpoint.kernel_kind)
        self._active_tags: tuple[CanaryLaunchTag, ...] = tuple(
            sorted(active, key=lambda tag: tag.value)
        )

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def forward_step_before_model(self, forward_batch: "ForwardBatch") -> None:
        """SOT §6.2 step 1: perturb hook + plan + head launches. Stashes plan for the tail launches."""
        if self.config.mode == "off":
            return
        if self._req_to_token_pool is None:
            return

        self.perturb_hook(forward_batch)
        self._last_forward_batch = forward_batch

        for pool_idx, groups in enumerate(self._groups_per_pool):
            for group in groups:
                plan_input = build_plan_input_per_forward(
                    forward_batch=forward_batch,
                    swa_window_size=(
                        self._swa_window_size if group.kind is PoolKind.SWA else 0
                    ),
                    full_to_swa_index_mapping=group.swa_index_lut,
                )
                self._invoke_plan(
                    plan_input=plan_input,
                    verify_plan=self._verify_plan_per_forward,
                    write_plan=self._write_plan_per_forward,
                    group=group,
                )
                self._launch_endpoints(
                    pool_idx=pool_idx,
                    group=group,
                    tag_filter=_is_head_tag,
                    verify_plan=self._verify_plan_per_forward,
                    forward_batch=forward_batch,
                )

    def forward_step_after_model(self) -> None:
        """SOT §6.2 step 3: tail launches reusing the plan stashed by forward_step_before_model."""
        if self.config.mode == "off":
            return
        forward_batch = self._last_forward_batch
        if forward_batch is None:
            return

        for pool_idx, groups in enumerate(self._groups_per_pool):
            for group in groups:
                self._launch_endpoints(
                    pool_idx=pool_idx,
                    group=group,
                    tag_filter=_is_tail_tag,
                    verify_plan=self._verify_plan_per_forward,
                    forward_batch=forward_batch,
                )

    def forward_step(
        self,
        forward_batch: "ForwardBatch",
        run_model: Callable[[], object],
    ) -> object:
        """Convenience wrapper: head launches -> run_model() -> tail launches -> end_of_step.

        SOT §6.2 sequence. Callers that need finer control (e.g. cuda-graph replay) should call
        forward_step_before_model / forward_step_after_model / end_of_step explicitly.
        """
        self.forward_step_before_model(forward_batch)
        output = run_model()
        self.forward_step_after_model()
        self.end_of_step()
        return output

    def end_of_step(self) -> None:
        """SOT §6.4: sweep + async D2H pump + step bump + drain previous pump + allreduce + raise."""
        if self.config.mode == "off":
            return

        self.maybe_run_sweep()

        if self._pump_stream is not None and self._pump_event is not None:
            violation_log = self._device_state.violation_log
            default_stream = torch.cuda.current_stream(self._device)
            self._pump_stream.wait_stream(default_stream)
            with torch.cuda.stream(self._pump_stream):
                signal = (violation_log.violation_write_index > 0).to(torch.uint8)
                self._device_state.violation_signal_host.copy_(
                    signal.view(-1)[:1], non_blocking=True
                )
                self._pump_event.record()

        self._step_counter += 1

        if self._previous_pump_event is not None:
            self._previous_pump_event.synchronize()
            local_errored = bool(int(self._device_state.violation_signal_host.item()))
        else:
            local_errored = False
        self._previous_pump_event = self._pump_event
        if torch.cuda.is_available():
            self._pump_event = torch.cuda.Event()

        any_rank_errored = local_errored
        allreduce_buf = self._device_state.allreduce_buf
        if (
            self.config.allreduce_violation_signal
            and allreduce_buf is not None
            and self._tp_group is not None
            and dist.is_initialized()
        ):
            allreduce_buf.fill_(int(local_errored))
            dist.all_reduce(
                allreduce_buf,
                op=dist.ReduceOp.MAX,
                group=self._tp_group.device_group,
            )
            any_rank_errored = bool(int(allreduce_buf.item()))

        self.health_check_step()
        self._print_periodic_stats()

        if self._perturb_undo is not None and self._req_to_token_pool is not None:
            row, col, original = self._perturb_undo
            self._req_to_token_pool.req_to_token[row, col] = original
            self._perturb_undo = None

        if any_rank_errored and not self._raised:
            self._raise_violation()

    def maybe_run_sweep(self) -> None:
        if self.config.sweep_every_n_steps == 0:
            return
        if (
            self._last_sweep_step >= 0
            and self._step_counter - self._last_sweep_step
            < self.config.sweep_every_n_steps
        ):
            return
        self._last_sweep_step = self._step_counter

        running_snapshot: Optional[AliveReqSnapshot] = None
        forward_batch = self._last_forward_batch
        if (
            forward_batch is not None
            and forward_batch.req_pool_indices is not None
            and forward_batch.seq_lens is not None
        ):
            running_snapshot = AliveReqSnapshot(
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
            )

        for pool_idx, groups in enumerate(self._groups_per_pool):
            for group in groups:
                window = self._swa_window_size if group.kind is PoolKind.SWA else 0
                if running_snapshot is not None and self._req_to_token_pool is not None:
                    running_input = build_plan_input_running_sweep(
                        req_to_token_pool=self._req_to_token_pool,
                        alive_reqs=running_snapshot,
                        swa_window_size=window,
                        full_to_swa_index_mapping=group.swa_index_lut,
                    )
                    self._invoke_plan(
                        plan_input=running_input,
                        verify_plan=self._verify_plan_sweep_running,
                        write_plan=self._write_plan_sweep,
                        group=group,
                    )
                    self._launch_endpoints(
                        pool_idx=pool_idx,
                        group=group,
                        tag_filter=_is_sweep_tag,
                        verify_plan=self._verify_plan_sweep_running,
                        forward_batch=None,
                    )

                if self._radix_cache is not None:
                    radix_input = build_plan_input_radix_sweep(
                        radix_cache=self._radix_cache,
                        swa_window_size=window,
                        full_to_swa_index_mapping=group.swa_index_lut,
                    )
                    self._invoke_plan(
                        plan_input=radix_input,
                        verify_plan=self._verify_plan_sweep_radix,
                        write_plan=self._write_plan_sweep,
                        group=group,
                    )
                    self._launch_endpoints(
                        pool_idx=pool_idx,
                        group=group,
                        tag_filter=_is_sweep_tag,
                        verify_plan=self._verify_plan_sweep_radix,
                        forward_batch=None,
                    )

        self._sweep_passes += 1

    def _raise_violation(self) -> None:
        violation_log = self._device_state.violation_log
        write_index = int(violation_log.violation_write_index.cpu().item())
        if write_index == 0:
            return
        ring = violation_log.violation_ring.cpu()
        ring_overflow = write_index > int(ring.shape[0])
        message = _format_violation(
            row=ring[0].tolist(),
            total=write_index,
            ring_overflow=ring_overflow,
            step_when_pumped=self._step_counter,
        )
        if self.config.mode == "on":
            logger.error(message)
            return
        self._raised = True
        raise RuntimeError(message)

    def health_check_step(self) -> None:
        if self._step_counter < _HEALTH_CHECK_WARMUP_STEPS:
            return
        if self._step_counter % _HEALTH_CHECK_EVERY_N_STEPS != 0:
            return

        if not self._active_tags:
            return
        counters = self._device_state.kernel_run_counters.detach().cpu().tolist()
        zero_tags = [tag for tag in self._active_tags if int(counters[tag.value]) == 0]
        if zero_tags:
            names = ", ".join(tag.name for tag in zero_tags)
            raise RuntimeError(
                f"kv-canary: kernel_run_counter is zero after warmup for tags=[{names}] "
                f"at step={self._step_counter}; canary path is not executing"
            )

    def perturb_hook(self, forward_batch: "ForwardBatch") -> None:
        if self.config.perturb_req_to_token_prob <= 0.0:
            return
        if self._req_to_token_pool is None:
            return
        table = self._req_to_token_pool.req_to_token
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            return

        if torch.rand((), device="cpu").item() >= self.config.perturb_req_to_token_prob:
            return

        rows, cols = int(table.shape[0]), int(table.shape[1])
        if rows <= 1 or cols <= 1:
            return

        row = int(torch.randint(1, rows, (1,)).item())
        col = int(torch.randint(0, cols, (1,)).item())
        original = int(table[row, col].item())
        new_value = (original + 1) & 0x7FFFFFFF
        if new_value == original:
            new_value = (original + 2) & 0x7FFFFFFF
        table[row, col] = new_value
        self._perturb_undo = (row, col, original)

    def _print_periodic_stats(self) -> None:
        period = self.config.stats_print_every_n_steps
        if period <= 0:
            return
        if self._step_counter == 0 or self._step_counter % period != 0:
            return
        protected = int(self._device_state.slot_run_counters.sum().item())
        violations = int(self._device_state.violation_log.violation_write_index.item())
        active = len(self._active_tags)
        logger.info(
            "[canary] step=%d protected_tokens=%d sweep_passes=%d violations=%d "
            "launch_tags_active=%d/%d",
            self._step_counter,
            protected,
            self._sweep_passes,
            violations,
            active,
            len(CanaryLaunchTag),
        )

    def _invoke_plan(
        self,
        *,
        plan_input: PlanInput,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        group: CanaryBufferGroup,
    ) -> None:
        if self._req_to_token_pool is None:
            return
        window = self._swa_window_size if group.kind is PoolKind.SWA else 0
        canary_plan_step(
            verify_plan_out=verify_plan,
            write_plan_out=write_plan,
            fb_req_pool_indices=plan_input.fb_req_pool_indices,
            fb_prefix_lens=plan_input.fb_prefix_lens,
            fb_extend_seq_lens=plan_input.fb_extend_seq_lens,
            req_to_token=self._req_to_token_pool.req_to_token,
            extra_verify_slot_indices=plan_input.extra_verify_slot_indices,
            extra_verify_positions=plan_input.extra_verify_positions,
            extra_verify_prev_slot_indices=plan_input.extra_verify_prev_slot_indices,
            extra_verify_num_valid=plan_input.extra_verify_num_valid,
            swa_window_size=window,
            full_to_swa_index_mapping=group.swa_index_lut,
        )

    def _launch_endpoints(
        self,
        *,
        pool_idx: int,
        group: CanaryBufferGroup,
        tag_filter: Callable[[CanaryLaunchTag], bool],
        verify_plan: VerifyPlan,
        forward_batch: Optional["ForwardBatch"],
    ) -> None:
        violation_log = self._device_state.violation_log
        positions: Optional[torch.Tensor] = None
        if forward_batch is not None:
            positions = forward_batch.positions
            if positions.dtype != torch.int32:
                positions = positions.to(torch.int32)

        for endpoint in self._endpoints_per_pool[pool_idx]:
            if not _endpoint_belongs_to_group(endpoint, group):
                continue
            if not tag_filter(endpoint.kernel_kind):
                continue
            if _is_sweep_tag(endpoint.kernel_kind):
                endpoint.launch_sweep(
                    verify_plan=verify_plan,
                    violation_log=violation_log,
                    real_kv_hash_mode=self.config.real_kv_hash_mode,
                )
                continue
            assert forward_batch is not None and positions is not None
            endpoint.launch_per_forward(
                verify_plan=verify_plan,
                write_plan=self._write_plan_per_forward,
                fb_input_ids=forward_batch.input_ids,
                fb_positions=positions,
                fb_out_cache_loc=forward_batch.out_cache_loc,
                input_check_mode=self.config.input_check_mode,
                expected_input_tokens=self._expected_input_tokens,
                expected_input_positions=self._expected_input_positions,
                violation_log=violation_log,
                real_kv_hash_mode=self.config.real_kv_hash_mode,
            )


def _is_head_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.HEAD_K_FULL,
        CanaryLaunchTag.HEAD_V_FULL,
        CanaryLaunchTag.HEAD_K_SWA,
        CanaryLaunchTag.HEAD_V_SWA,
    )


def _is_tail_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.TAIL_K_FULL,
        CanaryLaunchTag.TAIL_V_FULL,
        CanaryLaunchTag.TAIL_K_SWA,
        CanaryLaunchTag.TAIL_V_SWA,
    )


def _is_sweep_tag(tag: CanaryLaunchTag) -> bool:
    return tag in (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    )


def _endpoint_belongs_to_group(
    endpoint: CanaryEndpoint, group: CanaryBufferGroup
) -> bool:
    suffix = endpoint.kernel_kind.name.rsplit("_", 1)[1]
    return suffix == group.kind.name


def _format_violation(
    *,
    row: list[int],
    total: int,
    ring_overflow: bool,
    step_when_pumped: int,
) -> str:
    (
        kernel_kind,
        slot_idx,
        position,
        stored_token,
        expected_token,
        stored_chain_hash,
        expected_aux,
        fail_reason_bits,
    ) = row
    try:
        tag_label = CanaryLaunchTag(int(kernel_kind)).name
    except ValueError:
        tag_label = f"unknown({int(kernel_kind)})"
    bits = int(fail_reason_bits)
    reasons: list[str] = []
    if bits & 0x1:
        reasons.append("chain_hash")
    if bits & 0x2:
        reasons.append("position")
    if bits & 0x4:
        reasons.append("real_kv_hash")
    u64_mask = (1 << 64) - 1
    stored_prev_hash = int(stored_chain_hash) & u64_mask
    expected_prev_hash = int(expected_aux) & u64_mask

    return "\n".join(
        [
            (
                f"KV cache canary violation detected (kernel_kind={tag_label}, "
                f"slot_idx={int(slot_idx)}, position={int(position)})"
            ),
            f"  fail_reasons: {' '.join(reasons) if reasons else 'none'}",
            (
                f"  stored:   token_id={int(stored_token)}   position={int(position)} "
                f"prev_hash={stored_prev_hash:#018x} real_kv_hash={0:#018x}"
            ),
            (
                f"  expected: token_id={int(expected_token)}   position={int(position)} "
                f"prev_hash={expected_prev_hash:#018x} real_kv_hash={0:#018x}"
            ),
            (
                f"  total_violations={total} ring_overflow={ring_overflow} "
                f"step_when_pumped={step_when_pumped}"
            ),
        ]
    )
