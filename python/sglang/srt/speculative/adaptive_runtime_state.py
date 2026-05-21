import bisect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import build_per_bs_params

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

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
        init_max_bs: int | None = None,
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
      2. Call ``on_verify_complete(num_correct_drafts_per_req, batch_size)`` after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        self._bs_list, self._bs_params = build_per_bs_params(config_path)
        self._states: dict[int, SpecRuntimeState] = {}
        self._cuda_graph_bs: list[int] | None = None

        logger.info(
            f"AdaptiveController initialized: bs_list={self._bs_list}, "
            f"candidate_steps={self.candidate_steps}"
        )

    @property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        all_steps: set[int] = set()
        for params in self._bs_params.values():
            all_steps.update(params.candidate_steps)
        return sorted(all_steps)

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps.

        Args:
            cuda_graph_bs: Full list of batch sizes for which CUDA graphs may be
                captured.  When provided, each step only captures graphs for the
                batch-size values that can actually reach it, saving GPU memory.
                When ``None``, all batch sizes are captured for every step
                (original behaviour).
        """
        self._cuda_graph_bs = (
            sorted(cuda_graph_bs) if cuda_graph_bs is not None else None
        )

        # All steps share the same init_max_bs so that each draft attention
        # backend allocates _sched_meta_buf with a consistent shape.
        init_max_bs = max(cuda_graph_bs) if cuda_graph_bs is not None else None

        for steps in self.candidate_steps:
            pruned_bs = (
                self._cuda_graph_bs_for_step(steps, cuda_graph_bs)
                if cuda_graph_bs is not None
                else None
            )
            if steps in self._states:
                logger.info(
                    f"init_states: step={steps}, cuda_graph_bs={pruned_bs} (pre-registered)"
                )
                continue
            logger.info(
                f"init_states: step={steps}, cuda_graph_bs={pruned_bs}, "
                f"init_max_bs={init_max_bs}"
            )
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
                init_max_bs=init_max_bs,
            )
            self._states[steps] = state

        self._activate(self._get_steps_for_batch(1))

    def activate_step_by_batch(self, batch_size: int, current_steps: int) -> None:
        """Activate the optimal step for *batch_size* if it differs from *current_steps*.

        Called before each draft round so that cross-BS-range switches
        take effect before drafting begins.
        """
        target = self._get_steps_for_batch(batch_size)
        if target != current_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """Feed verify results; switch runtime state if EMA warrants it."""
        if batch_size <= 0:
            return

        bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        params = self._bs_params[bs]
        if params.update(num_correct_drafts_per_req):
            self._activate(params.current_steps)

    def _find_closest_bs(self, target: int) -> int:
        """Find largest BS key <= target (lower-bound range match)."""
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        """Round batch_size up to the nearest CUDA graph batch size."""
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        if idx < len(self._cuda_graph_bs):
            return self._cuda_graph_bs[idx]
        return self._cuda_graph_bs[-1]

    def _get_steps_for_batch(self, batch_size: int) -> int:
        """Get the current optimal step count for a given batch size."""
        bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        return self._bs_params[bs].current_steps

    def _cuda_graph_bs_for_step(
        self, step: int, all_cuda_graph_bs: list[int]
    ) -> list[int]:
        """Return the cuda_graph_bs values that need to be *captured* for *step*.

        For each CUDA graph batch size *v*, we check which BS config slot *v*
        maps to at runtime (via ``_find_closest_bs``), and include *v* only if
        that slot's candidate_steps contains *step*.  This matches the runtime
        lookup path exactly, so we never capture graphs that can't be reached.
        """
        return [
            v
            for v in sorted(all_cuda_graph_bs)
            if step in self._bs_params[self._find_closest_bs(v)].candidate_steps
        ]

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
