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
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int
    speculative_num_draft_tokens: int
    topk: int

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
        self._validate_candidate_steps_against_topk()

    def _validate_candidate_steps_against_topk(self) -> None:
        """For topk>1, ensure the smallest candidate step yields enough draft
        candidates to satisfy speculative_num_draft_tokens.

        EAGLE draft_forward (eagle_worker.py) builds a score pool via
        select_top_k_tokens (spec_utils.py):
          - step i=0: appends `topk` scores
          - step i>=1: appends `topk * topk` scores per step
        organize_draft_results (eagle_utils.py) then selects
        (num_draft_tokens - 1) entries from this pool.

        Pool size at num_steps=s, topk=k (s >= 1):
            pool(s, k) = k + (s - 1) * k**2

        For topk=1 the existing server_args post-init already enforces
        num_draft_tokens = num_steps + 1, so no check is needed there.
        """
        topk = self.worker.topk
        if topk == 1:
            return
        num_draft_tokens = self.worker.speculative_num_draft_tokens
        min_step = min(self.params.candidate_steps)
        if min_step < 1:
            raise ValueError(
                "adaptive candidate_steps must be >= 1 for topk>1 "
                f"(got min={min_step}); step=0 support is tracked separately."
            )
        pool_size = topk + (min_step - 1) * topk * topk
        if pool_size < num_draft_tokens - 1:
            raise ValueError(
                f"adaptive candidate_steps min={min_step} with topk={topk} "
                f"yields a draft pool of only {pool_size} candidates, but "
                f"speculative_num_draft_tokens={num_draft_tokens} requires "
                f">= {num_draft_tokens - 1}. "
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
            # Chain (topk=1): num_draft_tokens is always num_steps + 1
            # (server_args enforces this in post-init).
            # Tree (topk>1): num_draft_tokens is the user-specified tree
            # budget; build_tree_kernel selects top-N from the score pool.
            # Hold it constant across tiers so the user retains direct
            # control over the verifier graph size.
            if self.worker.topk == 1:
                draft_tokens = steps + 1
            else:
                draft_tokens = self.worker.speculative_num_draft_tokens
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=draft_tokens,
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
