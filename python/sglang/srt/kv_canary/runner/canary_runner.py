from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Iterator, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
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
from sglang.srt.kv_canary.runner.pump import ViolationSignalPump
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation import ViolationReporter
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CanaryRunner:
    """Owns all canary state for one ModelRunner. Constructed once during install_canary, lives
    until server shutdown. The runner itself is a thin facade; per-concern state and behavior
    live on the component classes (ViolationSignalPump, SweepOrchestrator, ViolationReporter,
    PerturbManager, PerForwardOrchestrator, KernelRunCounterHealthChecker,
    PeriodicCanaryStatsLogger).
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        pp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: "ReqToTokenPool",
        launch_capacities: CanaryLaunchCapacities,
        swa_window_size: int = 0,
        token_oracle_manager: Optional[TokenOracleManager] = None,
    ) -> None:
        self.config = config
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = swa_window_size

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

        self._violation_pump = ViolationSignalPump(
            config=config,
            device_state=self._device_state,
            d2h_stream=self._d2h_stream,
        )
        self._sweep_orchestrator = SweepOrchestrator(
            config=config,
            device=device,
            device_state=self._device_state,
            buffer_groups=self._buffer_groups,
            endpoints=self._endpoints,
            req_to_token_pool=req_to_token_pool,
            swa_window_size=self._swa_window_size,
            sweep_verify_capacity=launch_capacities.sweep_verify_capacity,
            violation_pump=self._violation_pump,
        )
        self._violation_reporter = ViolationReporter(
            config=config,
            device_state=self._device_state,
            violation_pump=self._violation_pump,
        )
        self._perturb_manager = PerturbManager(
            config=PerturbConfig.from_env(),
            req_to_token_pool=req_to_token_pool,
            buffer_groups=self._buffer_groups,
            violation_pump=self._violation_pump,
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
        )
        self._health_checker = KernelRunCounterHealthChecker(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            violation_pump=self._violation_pump,
            d2h_stream=self._d2h_stream,
        )
        self._stats_logger = PeriodicCanaryStatsLogger(
            config=config,
            device_state=self._device_state,
            active_tags=self._active_tags,
            violation_pump=self._violation_pump,
            sweep_orchestrator=self._sweep_orchestrator,
            d2h_stream=self._d2h_stream,
        )

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._sweep_orchestrator.attach_radix_cache(radix_cache)
        self._perturb_manager.attach_radix_cache(radix_cache)

    @contextlib.contextmanager
    def with_forward_pass(self, forward_batch: "ForwardBatch") -> Iterator[None]:
        """Bracket one forward pass: host-side prep before, host-side end-of-step after.

        Caller in ``ModelRunner.forward`` writes::

            with canary_runner.with_forward_pass(forward_batch):
                output = self._forward_raw(...)

        The body is whatever invokes ``graph_runner.replay()`` / ``model.forward()``. Cuda-graph
        capture happens inside the body; the in-graph HEAD/TAIL kernel launches are dispatched
        from the monkey-patched ``model.forward`` so they are captured (and auto-replayed) the
        same way the model itself is.
        """
        self._before_forward(forward_batch)
        try:
            yield
        finally:
            self._end_of_step()

    def _before_forward(self, forward_batch: "ForwardBatch") -> None:
        self._per_forward_orchestrator.before_forward(forward_batch)

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        """canary_plan_step + HEAD endpoint launches. Caller is the monkey-patched
        ``model.forward`` - kernels here are captured into the cuda graph.
        """
        self._per_forward_orchestrator.launch_head_kernels(forward_batch)

    def launch_tail_kernels(self, forward_batch: "ForwardBatch") -> None:
        """TAIL endpoint launches. Same captured region as ``launch_head_kernels``."""
        self._per_forward_orchestrator.launch_tail_kernels(forward_batch)

    def _end_of_step(self) -> None:
        if self.config.mode == "off":
            return

        self._per_forward_orchestrator.end_of_step()
        self._sweep_orchestrator.maybe_run_sweep()
        any_rank_errored = self._violation_pump.pump_and_drain()
        self._health_checker.step()
        self._stats_logger.step()

        if any_rank_errored and not self._violation_reporter.is_raised:
            self._violation_reporter.log_or_raise_violation()
