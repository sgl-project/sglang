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


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.

    With a memory budget, init_states() pre-builds candidate tiers in ascending
    step order until the budget is spent and defers the rest; a deferred tier is
    built the first time the policy activates it.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        policy: AdaptiveSpecPolicy,
    ):
        self.worker = worker
        self.params: AdaptiveSpecPolicy = policy
        self._states: dict[int, SpecRuntimeState] = {}
        self._deferred_steps: set[int] = set()

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

        *memory_budget_gb* bounds the device memory spent on tiers beyond the
        ones already registered; ``None`` pre-builds every tier.
        """
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        spent_gb = 0.0
        last_cost_gb = 0.0
        for steps in self.candidate_steps:
            if steps in self._states:
                continue
            if memory_budget_gb is not None and (
                spent_gb >= memory_budget_gb
                # Cost grows with steps, so the previous tier's cost is a lower
                # bound on this one; the first tier past the budget may still
                # overshoot, since cost is only known after capture.
                or spent_gb + last_cost_gb > memory_budget_gb
            ):
                self._deferred_steps.add(steps)
                continue
            last_cost_gb = self._build_state(steps).gpu_memory_gb
            spent_gb += last_cost_gb

        if self._deferred_steps:
            log_info_on_rank0(
                logger,
                "Adaptive runtime states deferred to first use: "
                f"steps={sorted(self._deferred_steps)}, "
                f"pre-built={spent_gb:.2f}GB of budget={memory_budget_gb:.2f}GB",
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
        if state is None and speculative_num_steps in self._deferred_steps:
            logger.warning(
                "Building deferred adaptive runtime state steps=%d on first use; "
                "decode stalls until capture completes",
                speculative_num_steps,
            )
            state = self._build_state(speculative_num_steps)
            self._deferred_steps.discard(speculative_num_steps)
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
