import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

logger = logging.getLogger(__name__)


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

    # Device memory the build consumed, as the drop in free memory; 0.0 when
    # the builder does not measure it. Must agree across TP ranks.
    gpu_memory_gb: float = 0.0


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
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None: ...

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...

    @property
    def start_steps(self) -> list[int]: ...

    def prebuild_order(self) -> list[int]: ...

    def restrict_candidates(self, allowed: set[int]) -> list[int]: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.

    With a memory budget, init_states() always builds the states the policy can
    activate without feedback (the slots' start steps and the worker's initial
    step), builds the others in the policy's priority order until the budget is
    spent, and removes the rest from the policy's candidates for this launch.
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

    def init_states(
        self,
        cuda_graph_bs: list[int] | None = None,
        memory_budget_gb: float | None = None,
    ) -> None:
        """Build and register runtime states for the candidate steps.

        The worker registers its initial state before calling this. The slots'
        start steps are always built; *memory_budget_gb* bounds the device
        memory spent on the other states, and ``None`` builds every candidate.
        """
        self.params.set_cuda_graph_bs(cuda_graph_bs)
        start_steps = set(self.params.start_steps)

        spent_gb = 0.0
        for steps in self.params.prebuild_order():
            if steps in self._states:
                continue
            if (
                steps not in start_steps
                and memory_budget_gb is not None
                and spent_gb >= memory_budget_gb
            ):
                break
            # Cost is known only after capture, so the last optional build may
            # overshoot the budget by one state.
            state = self._build_state(steps)
            if steps not in start_steps:
                spent_gb += state.gpu_memory_gb

        dropped = self.params.restrict_candidates(set(self._states))
        if dropped:
            log_info_on_rank0(
                logger,
                f"Adaptive runtime states not built within the memory budget "
                f"({memory_budget_gb:.2f}GB, spent {spent_gb:.2f}GB on optional "
                f"states): steps={dropped}; removed from the candidates for this launch",
            )

        # Start on the initial step.
        self._activate(self.worker.speculative_num_steps)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = self.params.get_steps_for_batch(batch_size)
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results; switch runtime state if the policy requests it."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._activate(new_step)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)

    def _build_state(self, steps: int) -> SpecRuntimeState:
        state = self.worker.build_adaptive_runtime_state(
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
            cuda_graph_bs=self.params.cuda_graph_bs_for_step(steps),
        )
        self._states[steps] = state
        return state
