from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.jit_kernel.kv_canary.plan import canary_plan_step
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.srt.kv_canary.pool_patch.api import attach_canary_buffers
from sglang.srt.kv_canary.runner.health import HealthAndStats
from sglang.srt.kv_canary.runner.per_forward import (
    PerForwardOrchestrator,
    _endpoint_belongs_to_group,
    _is_sweep_tag,
)
from sglang.srt.kv_canary.runner.perturb import PerturbHook
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation import ViolationReporter
from sglang.srt.kv_canary.violation_state import CanaryDeviceState

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryRunner:
    """Owns all canary state for one ModelRunner. Constructed once during install_canary, lives until
    server shutdown.

    Internal state (private — never touched outside this class):
        config: CanaryConfig
        device_state: CanaryDeviceState
        endpoints_per_pool: tuple[tuple[CanaryEndpoint, ...], ...]  # one tuple per pool
        verify_plan_per_forward / write_plan_per_forward: VerifyPlan / WritePlan sized for per-forward
            capacity (= max_batch_size × max_seq_len for verify, max_batch_size for write).
        plan_input_per_forward: PlanInput with STATIC per-forward fb_* buffers (allocated once,
            mutated in place by before_forward each step).
        verify_plan_sweep_radix / write_plan_sweep: sized for sweep capacity (= total pool slots).
        step_counter: int, host-side, bumped per forward.
        last_sweep_step: int, host-side.
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        pools: Optional[list["KVCache"]] = None,
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: Optional["ReqToTokenPool"] = None,
        radix_cache: Optional["BasePrefixCache"] = None,
        per_forward_verify_capacity: int,
        per_forward_write_req_capacity: int,
        per_forward_write_entry_capacity: int,
        sweep_verify_capacity: int,
        swa_window_size: int = 0,
        buffer_groups_per_pool: Optional[list[tuple[CanaryBufferGroup, ...]]] = None,
        token_pool_allocator: Optional[object] = None,
    ) -> None:
        self.config = config
        self._device = device
        self._tp_group = tp_group
        self._req_to_token_pool = req_to_token_pool
        self._radix_cache = radix_cache
        self._swa_window_size = int(swa_window_size)

        if buffer_groups_per_pool is not None:
            if pools is not None:
                raise ValueError(
                    "kv-canary: pass either pools or buffer_groups_per_pool, not both"
                )
            self._groups_per_pool = tuple(
                tuple(groups) for groups in buffer_groups_per_pool
            )
        else:
            if pools is None:
                raise ValueError(
                    "kv-canary: either pools or buffer_groups_per_pool must be provided"
                )
            groups_per_pool: list[tuple[CanaryBufferGroup, ...]] = []
            for pool in pools:
                groups_per_pool.append(
                    attach_canary_buffers(
                        pool=pool,
                        config=config,
                        device=device,
                        allocator=token_pool_allocator,
                    )
                )
            self._groups_per_pool = tuple(groups_per_pool)

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

        self._step_counter: int = 0
        self._last_sweep_step: int = -1
        self._sweep_passes: int = 0
        self._raised: bool = False

        assert (
            self._req_to_token_pool is not None
        ), "kv-canary: req_to_token_pool must be bound at construction"

        active: set[CanaryLaunchTag] = set()
        for endpoints in self._endpoints_per_pool:
            for endpoint in endpoints:
                active.add(endpoint.kernel_kind)
        self._active_tags: tuple[CanaryLaunchTag, ...] = tuple(
            sorted(active, key=lambda tag: tag.value)
        )

        self._per_forward = PerForwardOrchestrator(
            owner=self,
            per_forward_verify_capacity=per_forward_verify_capacity,
            per_forward_write_req_capacity=per_forward_write_req_capacity,
            per_forward_write_entry_capacity=per_forward_write_entry_capacity,
        )
        self._sweep = SweepOrchestrator(
            owner=self, sweep_verify_capacity=sweep_verify_capacity
        )
        self._pump = PumpAndAllreduce(owner=self)
        self._violation = ViolationReporter(owner=self)
        self._perturb = PerturbHook(owner=self)
        self._health = HealthAndStats(owner=self)

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def before_forward(self, forward_batch: "ForwardBatch") -> None:
        """Host-side prep (perturb + plan-input fill). Caller is ``ModelRunner.forward`` — runs
        OUTSIDE the cuda-graph capture region.
        """
        self._per_forward.before_forward(forward_batch)

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        """canary_plan_step + HEAD endpoint launches. Caller is the monkey-patched
        ``model.forward`` — kernels here are captured into the cuda graph.
        """
        self._per_forward.launch_head_kernels(forward_batch)

    def launch_tail_kernels(self, forward_batch: "ForwardBatch") -> None:
        """TAIL endpoint launches. Same captured region as ``launch_head_kernels``."""
        self._per_forward.launch_tail_kernels(forward_batch)

    def end_of_step(self) -> None:
        """Sweep + async D2H pump + step bump + drain previous pump + allreduce + raise.

        Host-side, runs in ``ModelRunner.forward`` AFTER ``graph_runner.replay()`` /
        ``model.forward()`` returns.
        """
        if self.config.mode == "off":
            return

        self.maybe_run_sweep()

        any_rank_errored = self._pump.pump_and_drain()

        self.health_check_step()
        self._print_periodic_stats()

        self._perturb.undo_after_step()

        if any_rank_errored and not self._raised:
            self._raise_violation()

    def maybe_run_sweep(self) -> None:
        self._sweep.maybe_run_sweep()

    def _raise_violation(self) -> None:
        self._violation.raise_violation()

    def health_check_step(self) -> None:
        self._health.health_check_step()

    def perturb_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        self._perturb.perturb_hook(forward_batch)

    def _perturb_real_kv_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        self._perturb.perturb_real_kv_hook(forward_batch)

    def _print_periodic_stats(self) -> None:
        self._health.print_periodic_stats()

    def _invoke_plan(
        self,
        *,
        plan_input: PlanInput,
        verify_plan: VerifyPlan,
        write_plan: WritePlan,
        group: CanaryBufferGroup,
    ) -> None:
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
        out_cache_loc: Optional[torch.Tensor] = None
        input_ids: Optional[torch.Tensor] = None
        if forward_batch is not None:
            positions = forward_batch.positions
            if positions.dtype != torch.int32:
                positions = positions.to(torch.int32)
            out_cache_loc = forward_batch.out_cache_loc
            if out_cache_loc is not None and out_cache_loc.dtype != torch.int32:
                out_cache_loc = out_cache_loc.to(torch.int32)
            input_ids = forward_batch.input_ids
            if input_ids is not None and input_ids.dtype != torch.int32:
                input_ids = input_ids.to(torch.int32)

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
            num_tokens = int(positions.shape[0])
            expected_tokens_slice = self._per_forward._expected_input_tokens[
                :num_tokens
            ]
            expected_positions_slice = self._per_forward._expected_input_positions[
                :num_tokens
            ]
            endpoint.launch_per_forward(
                verify_plan=verify_plan,
                write_plan=self._per_forward._write_plan_per_forward,
                fb_input_ids=input_ids,
                fb_positions=positions,
                fb_out_cache_loc=out_cache_loc,
                input_check_mode=self.config.input_check_mode,
                expected_input_tokens=expected_tokens_slice,
                expected_input_positions=expected_positions_slice,
                violation_log=violation_log,
                real_kv_hash_mode=self.config.real_kv_hash_mode,
            )
