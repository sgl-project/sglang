"""CanaryManager: top-level canary state holder under the
SingleForwardManager design.

Holds a static list of :class:`SingleForwardManager` instances (one per
inner forward step) and the per-cycle shared facilities (sweep,
violation, health, stats, swa-divergence). Replaces the previous
``CanaryRunner`` wholesale — there is no backward-compat shim.

Dispatch model: the monkey-patched ``model.forward`` wrap calls into
``get_current_single_forward_manager()`` to fire phase 2/3 on the
currently-active SingleForwardManager. The caller marks the active
SingleForwardManager with ``with_active_single_forward_manager(i)``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.manager import PerturbManager
from sglang.srt.kv_canary.runner.health import (
    KernelRunCounterHealthChecker,
    PeriodicCanaryStatsLogger,
)
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceReport
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation_manager import ViolationManager
from sglang.srt.kv_canary.single_forward_manager.manager import (
    SingleForwardManager,
    _PreOpsMaybeInsideGraphOutput,
)
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryManager:
    """Owns canary state for one ModelRunner: a static list of SingleForwardManagers plus
    the per-cycle shared facilities (violation / health / stats / sweep /
    swa-divergence).
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        perturb_config: PerturbConfig,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        pp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: "ReqToTokenPool",
        launch_capacities: CanaryLaunchCapacities,
        swa_window_size: int = 0,
        token_oracle_manager: Optional[TokenOracleManager] = None,
        swa_allocator: Optional["SWATokenToKVPoolAllocator"] = None,
        speculative_num_steps: int = 1,
        is_eagle_draft_decode: bool = False,
    ) -> None:
        self.config = config
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._swa_allocator: Optional["SWATokenToKVPoolAllocator"] = swa_allocator
        self._outer_step_counter: int = 0
        self._active_single_forward_manager_index: Optional[int] = None

        self._buffer_groups: tuple[CanaryBufferGroup, ...] = tuple(buffer_groups)

        self._device_state = CanaryDeviceState.allocate(
            config=config, device=device, num_tags=len(CanaryLaunchTag)
        )

        self._endpoints: tuple[CanaryEndpoint, ...] = tuple(
            endpoint
            for group in self._buffer_groups
            for endpoint in build_endpoints_from_group(
                group=group, device_state=self._device_state
            )
        )
        self._active_tags: tuple[CanaryLaunchTag, ...] = tuple(
            sorted(
                {endpoint.kernel_kind for endpoint in self._endpoints},
                key=lambda tag: tag.value,
            )
        )

        self._d2h_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)

        swa_divergence_interval = (
            envs.SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL.get()
        )
        if swa_divergence_interval > 0:
            self._swa_divergence_report: Optional[SwaDivergenceReport] = (
                SwaDivergenceReport(
                    device=device,
                    d2h_stream=self._d2h_stream,
                    interval=swa_divergence_interval,
                    swa_allocator=self._swa_allocator,
                    req_to_token_pool=self._req_to_token_pool,
                )
            )
        else:
            self._swa_divergence_report = None

        self._violation_manager = ViolationManager(
            config=config,
            device_state=self._device_state,
            d2h_stream=self._d2h_stream,
            outer_step_counter_getter=self._get_outer_step_counter,
        )
        self._sweep_orchestrator = SweepOrchestrator(
            config=config,
            device_state=self._device_state,
            buffer_groups=self._buffer_groups,
            endpoints=self._endpoints,
            swa_window_size=self._swa_window_size,
            outer_step_counter_getter=self._get_outer_step_counter,
        )
        self._perturb_manager = PerturbManager(
            config=perturb_config,
            req_to_token_pool=req_to_token_pool,
            buffer_groups=self._buffer_groups,
            step_counter_getter=self._get_outer_step_counter,
            swa_window_size=self._swa_window_size,
            sweep_interval=config.sweep_interval,
        )
        self._health_checker = KernelRunCounterHealthChecker(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            outer_step_counter_getter=self._get_outer_step_counter,
            d2h_stream=self._d2h_stream,
        )
        self._stats_logger = PeriodicCanaryStatsLogger(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            outer_step_counter_getter=self._get_outer_step_counter,
            sweep_orchestrator=self._sweep_orchestrator,
            d2h_stream=self._d2h_stream,
        )

        num_sfms = max(1, speculative_num_steps - 1)
        # Perturb fires once per outer cycle (not once per SingleForwardManager). Only SingleForwardManager(0)
        # owns the perturb dispatch — its phase 1 fires pre-forward perturbs
        # and its phase 4 fires the post-forward perturb. SingleForwardManager(i>0) skips
        # perturb entirely. This matches the target case (one SingleForwardManager) which
        # already fires perturb exactly once per cycle.
        self._single_forward_managers: tuple[SingleForwardManager, ...] = tuple(
            SingleForwardManager(
                config=config,
                device=device,
                device_state=self._device_state,
                buffer_groups=self._buffer_groups,
                endpoints=self._endpoints,
                req_to_token_pool=req_to_token_pool,
                swa_window_size=self._swa_window_size,
                perturb_manager=self._perturb_manager if i == 0 else None,
                per_forward_verify_capacity=launch_capacities.per_forward_verify_capacity,
                per_forward_write_req_capacity=launch_capacities.per_forward_write_req_capacity,
                per_forward_write_entry_capacity=launch_capacities.per_forward_write_entry_capacity,
                d2h_stream=self._d2h_stream,
                token_oracle_manager=token_oracle_manager,
                swa_divergence_report=self._swa_divergence_report,
                is_eagle_draft_decode=is_eagle_draft_decode,
            )
            for i in range(num_sfms)
        )

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    def get_single_forward_manager(self, index: int) -> SingleForwardManager:
        return self._single_forward_managers[index]

    def get_current_single_forward_manager(self) -> SingleForwardManager:
        """Used by the monkey-patched ``model.forward`` wrap to find the
        active SingleForwardManager. Asserts the caller has bracketed the call with
        :meth:`with_active_single_forward_manager`."""
        assert self._active_single_forward_manager_index is not None, (
            "kv-canary: monkey-patched model.forward fired without an active "
            "SingleForwardManager index; the caller must wrap the forward in "
            "CanaryManager.with_active_single_forward_manager(i)"
        )
        return self._single_forward_managers[self._active_single_forward_manager_index]

    @contextlib.contextmanager
    def with_active_single_forward_manager(self, index: int) -> Iterator[None]:
        """Mark SingleForwardManager ``index`` as active for the duration of the with-block.
        Used by callers around each inner ``model.forward`` call."""
        assert (
            self._active_single_forward_manager_index is None
        ), "kv-canary: nested with_active_single_forward_manager is forbidden"
        self._active_single_forward_manager_index = index
        try:
            yield
        finally:
            self._active_single_forward_manager_index = None

    def pre_ops_maybe_inside_graph(
        self, forward_batch: "ForwardBatch"
    ) -> _PreOpsMaybeInsideGraphOutput:
        assert self._active_single_forward_manager_index is not None, (
            "kv-canary: pre_ops_maybe_inside_graph called without active SingleForwardManager; "
            "caller must wrap in CanaryManager.with_active_single_forward_manager(i)"
        )
        sfm = self._single_forward_managers[self._active_single_forward_manager_index]
        return sfm.pre_ops_maybe_inside_graph(forward_batch)

    def post_ops_maybe_inside_graph(
        self,
        forward_batch: "ForwardBatch",
        pre_ops_output: _PreOpsMaybeInsideGraphOutput,
    ) -> None:
        assert self._active_single_forward_manager_index is not None, (
            "kv-canary: post_ops_maybe_inside_graph called without active SingleForwardManager; "
            "caller must wrap in CanaryManager.with_active_single_forward_manager(i)"
        )
        sfm = self._single_forward_managers[self._active_single_forward_manager_index]
        sfm.post_ops_maybe_inside_graph(forward_batch, pre_ops_output)

    @contextlib.contextmanager
    def with_ops_outside_graph(
        self,
        *,
        single_forward_indices: Sequence[int],
        maybe_inaccurate_forward_batch: "ForwardBatch",
    ) -> Iterator[None]:
        self._pre_ops_outside_graph(
            single_forward_indices=single_forward_indices,
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
        )
        try:
            yield
        finally:
            self._post_ops_outside_graph(
                single_forward_indices=single_forward_indices,
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            )

    def _pre_ops_outside_graph(
        self,
        *,
        single_forward_indices: Sequence[int],
        maybe_inaccurate_forward_batch: "ForwardBatch",
    ) -> None:
        for idx in single_forward_indices:
            self._single_forward_managers[idx].pre_ops_outside_graph(
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
            )

    def _post_ops_outside_graph(
        self,
        *,
        single_forward_indices: Sequence[int],
        maybe_inaccurate_forward_batch: "ForwardBatch",
    ) -> None:
        for idx in single_forward_indices:
            self._single_forward_managers[idx].post_ops_outside_graph(
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
            )
        if self.config.mode == "off":
            return
        self._sweep_orchestrator.maybe_run_sweep()
        self._outer_step_counter += 1
        self._violation_manager.step()
        self._health_checker.step()
        self._stats_logger.step()
        if self._swa_divergence_report is not None:
            self._swa_divergence_report.step(
                outer_step_counter=self._outer_step_counter,
                forward_batch=maybe_inaccurate_forward_batch,
            )

    def mark_init_finished(self) -> None:
        """Reset every SingleForwardManager's phase tensor and enable its assert. Called
        once after warmup / cuda graph capture / piecewise compile so any
        residual phase state left by captured (init-time) kernels is
        cleared and post-init lifecycle starts from a known good IDLE."""
        for single_forward_manager in self._single_forward_managers:
            single_forward_manager.phase_checker.reset_to_idle()
            single_forward_manager.phase_checker.enable_assert()

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._sweep_orchestrator.attach_radix_cache(radix_cache)
        self._perturb_manager.attach_radix_cache(radix_cache)

    def step_shared_facilities(
        self,
        *,
        maybe_inaccurate_forward_batch: Optional["ForwardBatch"] = None,
    ) -> None:
        """Fire the per-cycle shared facilities (sweep, violation drain,
        health, stats, swa-divergence). Caller invokes this once after all
        SingleForwardManagers in the cycle have finished phase 4.

        ``maybe_inaccurate_forward_batch`` is the same instance handed to
        phase 4; the swa-divergence report reads its req_pool_indices /
        seq_lens to compute the FULL-vs-SWA index divergence. By cycle
        end the outer scheduler may already have advanced this batch,
        but the divergence metric is a coarse trend signal and tolerates
        the slight staleness.
        """
        if self.config.mode == "off":
            return

        self._sweep_orchestrator.maybe_run_sweep()
        self._outer_step_counter += 1
        self._violation_manager.step()
        self._health_checker.step()
        self._stats_logger.step()
        if self._swa_divergence_report is not None:
            self._swa_divergence_report.step(
                outer_step_counter=self._outer_step_counter,
                forward_batch=maybe_inaccurate_forward_batch,
            )

    def _get_outer_step_counter(self) -> int:
        return self._outer_step_counter
