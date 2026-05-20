from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.endpoint import (
    CanaryEndpoint,
    build_endpoints_from_group,
)
from sglang.srt.kv_canary.mock_model.sampler import OracleSamplerHook
from sglang.srt.kv_canary.runner.health import HealthAndStats
from sglang.srt.kv_canary.runner.per_forward import PerForwardOrchestrator
from sglang.srt.kv_canary.runner.perturb import PerturbHook
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.runner.sweep import SweepOrchestrator
from sglang.srt.kv_canary.runner.violation import ViolationReporter
from sglang.srt.kv_canary.violation_state import CanaryDeviceState, CanaryHostState

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryLaunchCapacities:
    """Pre-allocation sizes for the per-forward and sweep tensors a CanaryRunner owns. Computed
    once at install_canary from ServerArgs + ModelRunner metadata; all four fields are upper
    bounds - actual per-step usage may be smaller but never larger.

    Fields:
        per_forward_verify_capacity: VerifyPlan row capacity for the per-forward HEAD/TAIL
            launches (sized to the longest sequence the per-forward path may verify in one step).
        per_forward_write_req_capacity: WritePlan row capacity for per-forward writes, also used
            to size the static fb_* PlanInput buffers (= max batch size under cuda graph).
        per_forward_write_entry_capacity: Capacity for the expected_input_* placeholder tensors,
            one entry per token written in a single forward.
        sweep_verify_capacity: VerifyPlan row capacity for the radix sweep launch, sized to the
            pool slot count bounded by the cuda grid safe upper limit.
    """

    per_forward_verify_capacity: int
    per_forward_write_req_capacity: int
    per_forward_write_entry_capacity: int
    sweep_verify_capacity: int


class CanaryRunner:
    """Owns all canary state for one ModelRunner. Constructed once during install_canary, lives
    until server shutdown. The runner itself is a thin facade; per-concern state and behavior
    live on the component classes (PumpAndAllreduce, SweepOrchestrator, ViolationReporter,
    PerturbHook, PerForwardOrchestrator, HealthAndStats).
    """

    def __init__(
        self,
        *,
        config: CanaryConfig,
        buffer_groups: tuple[CanaryBufferGroup, ...],
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: "ReqToTokenPool",
        radix_cache: Optional["BasePrefixCache"] = None,
        launch_capacities: CanaryLaunchCapacities,
        swa_window_size: int = 0,
    ) -> None:
        self.config = config
        self._device = device
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = int(swa_window_size)

        self._buffer_groups: tuple[CanaryBufferGroup, ...] = tuple(buffer_groups)

        self._device_state = CanaryDeviceState.allocate(
            config=config, device=device, num_tags=len(CanaryLaunchTag)
        )
        self._host_state = CanaryHostState.allocate(
            config=config, num_tags=len(CanaryLaunchTag)
        )

        endpoints: list[CanaryEndpoint] = []
        for group in self._buffer_groups:
            endpoints.extend(
                build_endpoints_from_group(group=group, device_state=self._device_state)
            )
        self._endpoints: tuple[CanaryEndpoint, ...] = tuple(endpoints)

        active: set[CanaryLaunchTag] = set()
        for endpoint in self._endpoints:
            active.add(endpoint.kernel_kind)
        self._active_tags: tuple[CanaryLaunchTag, ...] = tuple(
            sorted(active, key=lambda tag: tag.value)
        )

        self._pump_and_allreduce = PumpAndAllreduce(
            config=config,
            device=device,
            device_state=self._device_state,
            host_state=self._host_state,
            tp_group=tp_group,
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
            pump_and_allreduce=self._pump_and_allreduce,
        )
        self._violation_reporter = ViolationReporter(
            config=config,
            device_state=self._device_state,
            pump_and_allreduce=self._pump_and_allreduce,
        )
        self._perturb_hook = PerturbHook(
            config=config,
            req_to_token_pool=req_to_token_pool,
            buffer_groups=self._buffer_groups,
        )
        self._per_forward_orchestrator = PerForwardOrchestrator(
            config=config,
            device=device,
            device_state=self._device_state,
            buffer_groups=self._buffer_groups,
            endpoints=self._endpoints,
            req_to_token_pool=req_to_token_pool,
            swa_window_size=self._swa_window_size,
            perturb_hook=self._perturb_hook,
            per_forward_verify_capacity=launch_capacities.per_forward_verify_capacity,
            per_forward_write_req_capacity=launch_capacities.per_forward_write_req_capacity,
            per_forward_write_entry_capacity=launch_capacities.per_forward_write_entry_capacity,
        )
        self._health_and_stats = HealthAndStats(
            config=config,
            device=device,
            device_state=self._device_state,
            host_state=self._host_state,
            active_tags=self._active_tags,
            pump_and_allreduce=self._pump_and_allreduce,
            sweep_orchestrator=self._sweep_orchestrator,
        )

        if radix_cache is not None:
            self.attach_radix_cache(radix_cache)

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    @property
    def step_counter(self) -> int:
        return self._pump_and_allreduce.step_counter

    @property
    def sweep_passes(self) -> int:
        return self._sweep_orchestrator.sweep_passes

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._sweep_orchestrator.attach_radix_cache(radix_cache)
        self._perturb_hook.attach_radix_cache(radix_cache)

    def attach_oracle_sampler_hook(self, hook: OracleSamplerHook) -> None:
        """Bind the OracleSamplerHook returned by install_oracle_sampler so the per-forward
        input-check path (input_check_mode == ON) can fill expected_input_* tensors from the
        same oracle that drives sampling.
        """
        self._per_forward_orchestrator.attach_oracle_sampler_hook(hook)

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

        self._sweep_orchestrator.maybe_run_sweep()
        any_rank_errored = self._pump_and_allreduce.pump_and_drain()
        self._health_and_stats.health_check_step()
        self._health_and_stats.print_periodic_stats()
        self._perturb_hook.undo_after_step()

        if any_rank_errored and not self._violation_reporter.is_raised:
            self._violation_reporter.raise_violation()
