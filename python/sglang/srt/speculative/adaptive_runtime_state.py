import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController.

    Algorithm-specific formulas (the draft pool size at a given number of
    steps, and the per-tier ``speculative_num_draft_tokens``) live on the
    worker so the controller stays generic across speculative algorithms.
    """

    speculative_num_steps: int
    speculative_num_draft_tokens: int
    topk: int

    def get_draft_pool_size(self, num_steps: int) -> int:
        """Number of draft candidates produced for tier=*num_steps*.

        The controller checks this against ``speculative_num_draft_tokens``
        before activating any tier; raise / return 0 for unsupported step
        values (e.g. step=0).
        """
        ...

    def get_num_draft_tokens(self, num_steps: int) -> int:
        """``speculative_num_draft_tokens`` to use when this tier is active.

        Chain-shape workers typically tie this to ``num_steps``; tree-shape
        workers usually hold it constant across tiers and let the user pick
        the verifier graph size directly.
        """
        ...

    def build_adaptive_runtime_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → runtime state
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(num_correct_drafts_per_req)`` after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        cfg = load_adaptive_config(config_path)
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            config=cfg,
        )
        self._states: dict[int, SpecRuntimeState] = {}
        self._validate_candidate_steps()

    def _validate_candidate_steps(self) -> None:
        """Ensure the smallest candidate step yields enough draft candidates
        to satisfy that tier's ``num_draft_tokens``.

        Both formulas are algorithm-specific and live on the worker (see
        ``AdaptiveSpecWorker.get_draft_pool_size`` /
        ``get_num_draft_tokens``); the controller only enforces the generic
        invariant ``pool_size >= num_draft_tokens - 1`` for the smallest
        tier. Bigger tiers have at least as much pool as the smallest one,
        so a single check is sufficient when the per-tier budget is
        non-decreasing in ``num_steps`` (true for both EAGLE chain and tree
        policies).
        """
        min_step = min(self.params.candidate_steps)
        if min_step < 1:
            raise ValueError(
                "adaptive candidate_steps must be >= 1 "
                f"(got min={min_step}); step=0 support is tracked separately."
            )
        num_draft_tokens = self.worker.get_num_draft_tokens(min_step)
        pool_size = self.worker.get_draft_pool_size(min_step)
        if pool_size < num_draft_tokens - 1:
            raise ValueError(
                f"adaptive candidate_steps min={min_step} yields a draft "
                f"pool of only {pool_size} candidates, but the tier needs "
                f"speculative_num_draft_tokens={num_draft_tokens} "
                f"(>= {num_draft_tokens - 1}). "
                "Increase the smallest candidate step or lower "
                "speculative_num_draft_tokens."
            )

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self) -> None:
        """Build and register runtime states for all candidate steps."""
        for steps in self.params.candidate_steps:
            if steps in self._states:
                continue
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=self.worker.get_num_draft_tokens(steps),
            )
            self._states[steps] = state
        self._activate(self.params.current_steps)

    def on_verify_complete(self, num_correct_drafts_per_req: list[int]) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        if self.params.update(num_correct_drafts_per_req):
            self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
