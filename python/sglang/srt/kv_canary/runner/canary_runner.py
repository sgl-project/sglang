from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Iterator, Optional

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
from sglang.srt.kv_canary.runner.per_forward import PerForwardOrchestrator
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceReport
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation_manager import ViolationManager
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryRunner:
    """Owns all canary state for one ModelRunner. Constructed once during install_canary, lives
    until server shutdown. The runner itself is a thin facade; per-concern state and behavior
    live on the component classes (ViolationManager, SweepOrchestrator,
    PerturbManager, PerForwardOrchestrator, KernelRunCounterHealthChecker,
    PeriodicCanaryStatsLogger).
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
    ) -> None:
        self.config = config
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size
        self._swa_allocator: Optional["SWATokenToKVPoolAllocator"] = swa_allocator
        self._step_counter: int = 0
        self._in_forward_pass: bool = False

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
            step_counter_getter=self._get_step_counter,
        )
        self._sweep_orchestrator = SweepOrchestrator(
            config=config,
            device_state=self._device_state,
            buffer_groups=self._buffer_groups,
            endpoints=self._endpoints,
            swa_window_size=self._swa_window_size,
            step_counter_getter=self._get_step_counter,
        )
        self._perturb_manager = PerturbManager(
            config=perturb_config,
            req_to_token_pool=req_to_token_pool,
            buffer_groups=self._buffer_groups,
            step_counter_getter=self._get_step_counter,
            swa_window_size=self._swa_window_size,
            sweep_interval=config.sweep_interval,
        )
        self._per_forward_orchestrator = PerForwardOrchestrator(
            config=config,
            device=device,
            device_state=self._device_state,
            buffer_groups=self._buffer_groups,
            endpoints=self._endpoints,
            req_to_token_pool=req_to_token_pool,
            swa_window_size=self._swa_window_size,
            perturb_manager=self._perturb_manager,
            per_forward_verify_capacity=launch_capacities.per_forward_verify_capacity,
            per_forward_write_req_capacity=launch_capacities.per_forward_write_req_capacity,
            per_forward_write_entry_capacity=launch_capacities.per_forward_write_entry_capacity,
            d2h_stream=self._d2h_stream,
            token_oracle_manager=token_oracle_manager,
            swa_divergence_report=self._swa_divergence_report,
        )
        self._health_checker = KernelRunCounterHealthChecker(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            step_counter_getter=self._get_step_counter,
            d2h_stream=self._d2h_stream,
        )
        self._stats_logger = PeriodicCanaryStatsLogger(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            step_counter_getter=self._get_step_counter,
            sweep_orchestrator=self._sweep_orchestrator,
            d2h_stream=self._d2h_stream,
        )

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    def mark_init_finished(self) -> None:
        """Enable the per-forward phase-checker assert. Called once by the
        ModelRunner after all init-time work (kernel warmup, cuda graph
        capture, piecewise compile, ...) is done. Before this call the phase
        checker still launches its kernel into every captured region (so the
        graph shape stays uniform across init and post-init), but the
        device-side assert is a no-op so warmup's incomplete lifecycle does
        not raise."""
        self._per_forward_orchestrator.phase_checker.enable_assert()

    def _get_step_counter(self) -> int:
        return self._step_counter

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._sweep_orchestrator.attach_radix_cache(radix_cache)
        self._perturb_manager.attach_radix_cache(radix_cache)

    @contextlib.contextmanager
    def with_kernels_outside_cuda_graph(
        self, forward_batch: "ForwardBatch"
    ) -> Iterator[None]:
        """Launch the outside-cuda-graph kernels around a body that runs the
        cuda-graph capture / replay region.

        Fires :meth:`PerForwardOrchestrator.pre_kernels_outside_cuda_graph`
        before the body and
        :meth:`PerForwardOrchestrator.post_kernels_outside_cuda_graph` after.
        The body itself contains 1 inner forward (target case, cuda graph
        captures only ``model.forward``) or N inner forwards (EAGLE draft
        case, the body runs the multi-step draft cuda graph). Per-step
        head/tail kernel launches are dispatched from inside the captured
        region (via the monkey-patched ``model.forward``) and replay
        correctly for every inner step.

        Caller examples::

            # Target ModelRunner.forward — body runs one model.forward, captured.
            with canary_runner.with_kernels_outside_cuda_graph(forward_batch):
                output = self._forward_raw(...)

            # EAGLE draft worker — body runs the multi-step draft cuda graph.
            with draft_runner.canary_runner.with_kernels_outside_cuda_graph(
                forward_batch
            ):
                self.cuda_graph_runner.replay(forward_batch)

        Re-entry is forbidden: each cycle is bracketed exactly once. EAGLE
        disables the inner ``ModelRunner.forward`` canary_ctx via
        ``is_draft_worker`` and wraps the draft entry instead, so the outer
        cycle stays unique."""
        assert (
            not self._in_forward_pass
        ), "CanaryRunner.with_kernels_outside_cuda_graph cannot be re-entered"
        self._in_forward_pass = True
        self._pre_kernels_outside_cuda_graph(forward_batch)
        try:
            yield
        finally:
            try:
                self._post_kernels_outside_cuda_graph(forward_batch)
            finally:
                self._in_forward_pass = False

    def _pre_kernels_outside_cuda_graph(self, forward_batch: "ForwardBatch") -> None:
        self._per_forward_orchestrator.pre_kernels_outside_cuda_graph(forward_batch)

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        """Per-step PlanInput fill + plan sub-kernels + HEAD endpoint launches.
        Caller is the monkey-patched ``model.forward`` — fires once per inner
        forward, captured into the cuda graph (or replayed)."""
        self._per_forward_orchestrator.launch_head_kernels(forward_batch)

    def launch_tail_kernels(self, forward_batch: "ForwardBatch") -> None:
        """TAIL endpoint launches reusing the plan staged in
        ``launch_head_kernels``. Same captured region; fires once per inner
        forward."""
        self._per_forward_orchestrator.launch_tail_kernels(forward_batch)

    def _post_kernels_outside_cuda_graph(self, forward_batch: "ForwardBatch") -> None:
        if self.config.mode == "off":
            return

        self._per_forward_orchestrator.post_kernels_outside_cuda_graph(forward_batch)
        self._sweep_orchestrator.maybe_run_sweep()
        self._step_counter += 1
        self._violation_manager.step()
        self._health_checker.step()
        self._stats_logger.step()
        if self._swa_divergence_report is not None:
            self._swa_divergence_report.step(
                step_counter=self._step_counter,
                forward_batch=forward_batch,
            )
