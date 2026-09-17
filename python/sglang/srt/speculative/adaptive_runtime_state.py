import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    The draft and verify resources are required by adaptive workers. Algorithms
    with a draft-extend stage can also populate its optional resources.
    Switching adaptive steps swaps the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "DecodeCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "DecodeCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveSpecPolicy(Protocol):
    """Policy interface used by AdaptiveController to select runtime states."""

    @property
    def candidate_steps(self) -> list[int]: ...

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None: ...

    def get_steps_for_batch(self, batch_size: int) -> int: ...

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        num_steps: int | None = None,
    ) -> int | None: ...

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...

    def on_state_activated(self, steps: int) -> None: ...


@dataclass(frozen=True)
class SpecProfilePoint:
    """One speculative-decode cost measurement."""

    steps: int
    batch_size: int


@dataclass(frozen=True)
class SpecProfilePlan:
    """Measurements a profiling policy asks the controller to execute."""

    points: tuple[SpecProfilePoint, ...]
    seq_len: int
    n_warmup: int
    n_measure: int


@runtime_checkable
class AdaptiveProfilingPolicy(Protocol):
    """Optional profiling extension for an adaptive policy."""

    def profile_plan(
        self, worker: AdaptiveSpecWorker, *, max_running_requests: int
    ) -> SpecProfilePlan: ...

    def record_profile(self, batch_size: int, steps: int, avg_ms: float) -> None: ...

    def profile_summary(self) -> str: ...


logger = logging.getLogger(__name__)


def _broadcast_float_from_rank0(value: float) -> float:
    """Broadcast a profile measurement so every TP rank shares a cost table."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    from sglang.srt.distributed import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    if get_tensor_model_parallel_world_size() <= 1:
        return value
    tp_group = get_tp_group()
    value_tensor = torch.tensor([value], dtype=torch.float64, device=tp_group.device)
    dist.broadcast(value_tensor, src=0, group=tp_group.device_group)
    return float(value_tensor.item())


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        policy: AdaptiveSpecPolicy,
    ):
        self.worker = worker
        self.params: AdaptiveSpecPolicy = policy
        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        num_steps: int | None = None,
    ) -> None:
        """Feed verify results; switch runtime state if the policy requests it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size, num_steps
        )
        if new_step is not None:
            self._activate(new_step)

    def run_profiling(self, tree_cache, *, max_running_requests: int) -> None:
        """Run startup measurements requested by a profiling-capable policy."""
        if not isinstance(self.params, AdaptiveProfilingPolicy):
            return

        profile = self.params.profile_plan(
            self.worker, max_running_requests=max_running_requests
        )
        if not profile.points:
            log_info_on_rank0(
                logger,
                "Adaptive speculative profiling skipped: no eligible batch sizes.",
            )
            return

        from sglang.srt.speculative.spec_profiling_session import SpecProfilingSession

        original_steps = self.worker.speculative_num_steps
        log_info_on_rank0(
            logger,
            f"Adaptive speculative profiling: {len(profile.points)} points, "
            f"seq_len={profile.seq_len}, n_warmup={profile.n_warmup}, "
            f"n_measure={profile.n_measure}",
        )
        for point in profile.points:
            self._activate(point.steps)
            avg_ms = SpecProfilingSession(
                worker=self.worker,
                tree_cache=tree_cache,
                batch_size=point.batch_size,
                num_steps=point.steps,
                seq_len=profile.seq_len,
                n_warmup=profile.n_warmup,
                n_measure=profile.n_measure,
            ).measure()
            self.params.record_profile(
                point.batch_size, point.steps, _broadcast_float_from_rank0(avg_ms)
            )

        self._activate(original_steps)
        log_info_on_rank0(
            logger,
            f"Adaptive speculative profiling complete: {self.params.profile_summary()}",
        )

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
        self.params.on_state_activated(speculative_num_steps)
