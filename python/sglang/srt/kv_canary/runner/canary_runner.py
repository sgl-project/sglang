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
from sglang.srt.kv_canary.pool_patch.api import attach_canary_buffers
from sglang.srt.kv_canary.runner.health import HealthAndStats
from sglang.srt.kv_canary.runner.per_forward import PerForwardOrchestrator
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
        pool: Optional["KVCache"] = None,
        device: torch.device,
        tp_group: Optional["GroupCoordinator"] = None,
        req_to_token_pool: "ReqToTokenPool",
        radix_cache: Optional["BasePrefixCache"] = None,
        launch_capacities: CanaryLaunchCapacities,
        swa_window_size: int = 0,
        buffer_groups: Optional[tuple[CanaryBufferGroup, ...]] = None,
        token_to_kv_pool_allocator: Optional[object] = None,
    ) -> None:
        self.config = config
        self._device = device
        self._req_to_token_pool = req_to_token_pool
        self._swa_window_size = int(swa_window_size)

        if buffer_groups is not None:
            if pool is not None:
                raise ValueError(
                    "kv-canary: pass either pool or buffer_groups, not both"
                )
            self._groups: tuple[CanaryBufferGroup, ...] = tuple(buffer_groups)
        else:
            if pool is None:
                raise ValueError(
                    "kv-canary: either pool or buffer_groups must be provided"
                )
            self._groups = attach_canary_buffers(
                pool=pool,
                config=config,
                device=device,
                allocator=token_to_kv_pool_allocator,
            )

        self._device_state = CanaryDeviceState.allocate(
            config=config, device=device, num_tags=len(CanaryLaunchTag)
        )

        endpoints: list[CanaryEndpoint] = []
        for group in self._groups:
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

        self._pump = PumpAndAllreduce(
            config=config,
            device=device,
            device_state=self._device_state,
            tp_group=tp_group,
        )
        self._sweep = SweepOrchestrator(
            config=config,
            device=device,
            device_state=self._device_state,
            groups=self._groups,
            endpoints=self._endpoints,
            req_to_token_pool=req_to_token_pool,
            swa_window_size=self._swa_window_size,
            sweep_verify_capacity=launch_capacities.sweep_verify_capacity,
            pump=self._pump,
        )
        self._violation = ViolationReporter(
            config=config,
            device_state=self._device_state,
            pump=self._pump,
        )
        self._perturb = PerturbHook(
            config=config,
            req_to_token_pool=req_to_token_pool,
            groups=self._groups,
        )
        self._per_forward = PerForwardOrchestrator(
            config=config,
            device=device,
            device_state=self._device_state,
            groups=self._groups,
            endpoints=self._endpoints,
            req_to_token_pool=req_to_token_pool,
            swa_window_size=self._swa_window_size,
            perturb=self._perturb,
            per_forward_verify_capacity=launch_capacities.per_forward_verify_capacity,
            per_forward_write_req_capacity=launch_capacities.per_forward_write_req_capacity,
            per_forward_write_entry_capacity=launch_capacities.per_forward_write_entry_capacity,
        )
        self._health = HealthAndStats(
            config=config,
            device=device,
            device_state=self._device_state,
            active_tags=self._active_tags,
            pump=self._pump,
            sweep=self._sweep,
        )

        if radix_cache is not None:
            self.attach_radix_cache(radix_cache)

    @property
    def active_tag_count(self) -> int:
        return len(self._active_tags)

    @property
    def step_counter(self) -> int:
        return self._pump.step_counter

    @property
    def sweep_passes(self) -> int:
        return self._sweep.sweep_passes

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._sweep.attach_radix_cache(radix_cache)
        self._perturb.attach_radix_cache(radix_cache)

    def attach_oracle_sampler_hook(self, hook: OracleSamplerHook) -> None:
        """Bind the OracleSamplerHook returned by install_oracle_sampler so the per-forward
        input-check path (input_check_mode == ON) can fill expected_input_* tensors from the
        same oracle that drives sampling.
        """
        self._per_forward.attach_oracle_sampler_hook(hook)

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
        self.before_forward(forward_batch)
        try:
            yield
        finally:
            self.end_of_step()

    def before_forward(self, forward_batch: "ForwardBatch") -> None:
        """Host-side prep (perturb + plan-input fill). Runs OUTSIDE the cuda-graph capture
        region. Prefer ``with_forward_pass`` over calling this + ``end_of_step`` directly.
        """
        self._per_forward.before_forward(forward_batch)

    def launch_head_kernels(self, forward_batch: "ForwardBatch") -> None:
        """canary_plan_step + HEAD endpoint launches. Caller is the monkey-patched
        ``model.forward`` - kernels here are captured into the cuda graph.
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

        self._sweep.maybe_run_sweep()
        any_rank_errored = self._pump.pump_and_drain()
        self._health.health_check_step()
        self._health.print_periodic_stats()
        self._perturb.undo_after_step()

        if any_rank_errored and not self._violation.is_raised:
            self._violation.raise_violation()
