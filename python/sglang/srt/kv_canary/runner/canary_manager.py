from __future__ import annotations

import contextlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import torch

from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag
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
from sglang.srt.kv_canary.runner.health_checker import KernelRunCounterHealthChecker
from sglang.srt.kv_canary.runner.stats_logger import PeriodicCanaryStatsLogger
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceReporter
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation_manager import ViolationManager
from sglang.srt.kv_canary.single_forward_manager.manager import (
    SingleForwardManager,
    _PreOpsMaybeInsideGraphOutput,
)
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryManager:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        perturb_config: PerturbConfig,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        device: torch.device,
        req_to_token_pool: ReqToTokenPool,
        launch_capacities: CanaryLaunchCapacities,
        swa_window_size: int = 0,
        token_oracle_manager: Optional[TokenOracleManager] = None,
        swa_allocator: Optional[SWATokenToKVPoolAllocator] = None,
        speculative_num_steps: int = 1,
        is_eagle_draft_decode: bool = False,
    ) -> None:
        self.config = config
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._swa_allocator: Optional[SWATokenToKVPoolAllocator] = swa_allocator
        self._outer_step_counter: int = 0
        self._active_single_forward_manager_index: Optional[int] = None
        self._model_forward_bracket_depth: int = 0

        self._buffer_groups: tuple[CanaryBufferGroup, ...] = tuple(buffer_groups)

        self._device_state = CanaryDeviceState.allocate(
            config=config,
            device=device,
            num_tags=len(CanaryLaunchTag),
            req_to_token_alloc_size=req_to_token_pool.req_to_token.shape[0],
            max_context_len=req_to_token_pool.max_context_len,
        )
        # Disable the chain-step position assert until warmup / cuda-graph capture finishes
        # (synthetic positions trip the +1 invariant). mark_init_finished() sets it to 1.
        self._device_state.enable_chain_position_assert.fill_(0)

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
            self._swa_divergence_report: Optional[SwaDivergenceReporter] = (
                SwaDivergenceReporter(
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
            outer_step_counter_getter=self._get_outer_step_counter,
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
        self._single_forward_managers: tuple[SingleForwardManager, ...] = tuple(
            SingleForwardManager(
                config=config,
                device=device,
                device_state=self._device_state,
                buffer_groups=self._buffer_groups,
                endpoints=self._endpoints,
                req_to_token_pool=req_to_token_pool,
                swa_window_size=self._swa_window_size,
                per_forward_verify_capacity=launch_capacities.per_forward_verify_capacity,
                per_forward_write_req_capacity=launch_capacities.per_forward_write_req_capacity,
                per_forward_write_entry_capacity=launch_capacities.per_forward_write_entry_capacity,
                d2h_stream=self._d2h_stream,
                token_oracle_manager=token_oracle_manager,
                swa_divergence_report=self._swa_divergence_report,
                is_eagle_draft_decode=is_eagle_draft_decode,
            )
            for _ in range(num_sfms)
        )

    @contextlib.contextmanager
    def with_active_single_forward_manager(self, index: int) -> Iterator[None]:
        assert (
            self._active_single_forward_manager_index is None
        ), "kv-canary: nested with_active_single_forward_manager is forbidden"
        self._active_single_forward_manager_index = index
        try:
            yield
        finally:
            assert self._active_single_forward_manager_index == index, (
                f"kv-canary: with_active_single_forward_manager({index}) exited with "
                f"_active_single_forward_manager_index="
                f"{self._active_single_forward_manager_index}; nested or mismatched bracket"
            )
            self._active_single_forward_manager_index = None

    @contextlib.contextmanager
    def model_forward_bracket_scope(self) -> Iterator[bool]:
        """Return whether this is the outermost patched ``model.forward`` call.

        Some model implementations enter another patched forward from inside the
        top-level forward (for example, a vision-language model calling its inner
        language model). Kv-canary owns one pre/post bracket per active
        SingleForwardManager; nested brackets would run a second pre-op while the
        phase checker is already in the first bracket.
        """
        self._model_forward_bracket_depth += 1
        try:
            yield self._model_forward_bracket_depth == 1
        finally:
            self._model_forward_bracket_depth -= 1

    def pre_ops_maybe_inside_graph(
        self, forward_batch: ForwardBatch
    ) -> _PreOpsMaybeInsideGraphOutput:
        assert self._active_single_forward_manager_index is not None, (
            "kv-canary: pre_ops_maybe_inside_graph called without active SingleForwardManager; "
            "caller must wrap in CanaryManager.with_active_single_forward_manager(i)"
        )
        sfm = self._single_forward_managers[self._active_single_forward_manager_index]
        return sfm.pre_ops_maybe_inside_graph(forward_batch)

    def post_ops_maybe_inside_graph(
        self,
        forward_batch: ForwardBatch,
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
        maybe_inaccurate_forward_batch: ForwardBatch,
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
        maybe_inaccurate_forward_batch: ForwardBatch,
    ) -> None:
        for idx in single_forward_indices:
            self._single_forward_managers[idx].pre_ops_outside_graph(
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
            )
        self._perturb_manager.perturb(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
        )

    def _post_ops_outside_graph(
        self,
        *,
        single_forward_indices: Sequence[int],
        maybe_inaccurate_forward_batch: ForwardBatch,
    ) -> None:
        for idx in single_forward_indices:
            self._single_forward_managers[idx].post_ops_outside_graph()
        self._perturb_manager.perturb_post_forward(
            maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
        )
        self._sweep_orchestrator.maybe_run_sweep()
        self._outer_step_counter += 1
        self._violation_manager.step()
        self._health_checker.step()
        self._stats_logger.step()
        if self._swa_divergence_report is not None:
            self._swa_divergence_report.step(
                outer_step_counter=self._outer_step_counter,
                maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
            )

    def mark_init_finished(self) -> None:
        for single_forward_manager in self._single_forward_managers:
            single_forward_manager.phase_checker.enable_assert()
        self._device_state.enable_chain_position_assert.fill_(1)

    def attach_radix_cache(self, radix_cache: BasePrefixCache) -> None:
        self._sweep_orchestrator.attach_radix_cache(radix_cache)
        self._perturb_manager.attach_radix_cache(radix_cache)

    def _get_outer_step_counter(self) -> int:
        return self._outer_step_counter


@contextmanager
def context_tuple(ctx_a, ctx_b):
    with ctx_a, ctx_b:
        yield
