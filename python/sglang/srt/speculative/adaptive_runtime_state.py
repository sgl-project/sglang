import bisect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

from sglang.srt.speculative.adaptive_spec_params import (
    DEFAULT_BS_HYSTERESIS,
    DEFAULT_BS_STEPS,
    AdaptiveSpeculativeParams,
    get_default_hysteresis,
    load_adaptive_config,
    load_bs_config,
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

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: List[int] | None = None,
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
      2. Call ``on_verify_complete(num_correct_drafts_per_req)`` after each decode verify.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        config_path: Optional[str] = None,
    ):
        self.worker = worker
        cfg = load_adaptive_config(config_path)
        bs_config = load_bs_config(cfg)

        if bs_config is None:
            bs_config = {
                bs: {"steps": steps, **DEFAULT_BS_HYSTERESIS[bs]}
                for bs, steps in DEFAULT_BS_STEPS.items()
            }

        self._init_per_bs(cfg, bs_config)
        self._states: Dict[int, SpecRuntimeState] = {}
        self._cuda_graph_bs: List[int] | None = None

        logger.info(
            f"AdaptiveController initialized: bs_list={self._bs_list}, "
            f"candidate_steps={self.candidate_steps}"
        )

    def _init_per_bs(self, cfg: dict, bs_config: Dict[int, dict]):
        """Initialize per-BS adaptive params from *bs_config*."""
        self._bs_list = sorted(bs_config.keys())
        self._bs_params: Dict[int, AdaptiveSpeculativeParams] = {}

        for bs, entry in sorted(bs_config.items()):
            steps = entry.get("steps", DEFAULT_BS_STEPS.get(bs, [1, 3, 7]))
            initial = steps[len(steps) // 2]

            defaults = get_default_hysteresis(bs)
            up_h = entry.get("up_hysteresis", defaults["up_hysteresis"])
            down_h = entry.get("down_hysteresis", defaults["down_hysteresis"])
            params_cfg = {
                **cfg,
                "candidate_steps": steps,
                "up_hysteresis": up_h,
                "down_hysteresis": down_h,
            }
            if "ceiling_coeff" in entry:
                params_cfg["ceiling_coeff"] = entry["ceiling_coeff"]
            self._bs_params[bs] = AdaptiveSpeculativeParams(
                initial_steps=initial,
                config=params_cfg,
            )

    @property
    def candidate_steps(self) -> List[int]:
        """Union of all BS slots' candidate steps."""
        all_steps: set[int] = set()
        for params in self._bs_params.values():
            all_steps.update(params.candidate_steps)
        return sorted(all_steps)

    def _find_closest_bs(self, target: int) -> int:
        """Find largest BS key <= target (lower-bound range match)."""
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        """Round batch_size up to the nearest CUDA graph batch size.

        The actual compute cost is determined by the padded CUDA graph size,
        not the raw batch size.  Config lookup should use the padded value so
        that the step ceiling matches the real cost.
        """
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        if idx < len(self._cuda_graph_bs):
            return self._cuda_graph_bs[idx]
        return self._cuda_graph_bs[-1]

    def get_steps_for_batch(self, batch_size: int) -> int:
        """Get the current optimal step count for a given batch size."""
        bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        return self._bs_params[bs].current_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def _cuda_graph_bs_for_step(
        self, step: int, all_cuda_graph_bs: List[int]
    ) -> List[int]:
        """Return the cuda_graph_bs values that need to be *captured* for *step*.

        A cuda_graph_bs entry *v* covers actual batch sizes in (prev_v, v].  We
        include *v* for *step* when that coverage interval overlaps with at least
        one BS slot whose candidate_steps contains *step*.

        """
        relevant_ranges: List[tuple] = []
        for i, slot_key in enumerate(self._bs_list):
            if step in self._bs_params[slot_key].candidate_steps:
                lo = slot_key
                hi = (
                    self._bs_list[i + 1] if i + 1 < len(self._bs_list) else float("inf")
                )
                relevant_ranges.append((lo, hi))

        if not relevant_ranges:
            return []

        result: List[int] = []
        prev = 0
        for v in sorted(all_cuda_graph_bs):
            lo_actual = prev + 1
            hi_actual = v
            for r_lo, r_hi in relevant_ranges:
                if lo_actual < r_hi and hi_actual >= r_lo:
                    result.append(v)
                    break
            prev = v

        return result

    def init_states(self, cuda_graph_bs: List[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps.

        Args:
            cuda_graph_bs: Full list of batch sizes for which CUDA graphs may be
                captured.  When provided, each step only captures graphs for the
                batch-size values that can actually reach it, saving GPU memory.
                When ``None``, all batch sizes are captured for every step
                (original behaviour).
        """
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs is not None else None

        # All steps share the same init_max_bs so that each draft attention
        # backend allocates _sched_meta_buf with a consistent shape.
        init_max_bs = max(cuda_graph_bs) if cuda_graph_bs is not None else None

        for steps in self.candidate_steps:
            if steps in self._states:
                pruned_bs = (
                    self._cuda_graph_bs_for_step(steps, cuda_graph_bs)
                    if cuda_graph_bs is not None
                    else None
                )
                logger.info(
                    f"init_states: step={steps}, cuda_graph_bs={pruned_bs} (pre-registered)"
                )
                continue
            pruned_bs = (
                self._cuda_graph_bs_for_step(steps, cuda_graph_bs)
                if cuda_graph_bs is not None
                else None
            )
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

        initial_steps = self.get_steps_for_batch(1)
        self.activate(initial_steps)

    def on_verify_complete(
        self,
        accept_lengths: list[int],
        batch_size: int = 0,
    ) -> None:
        """Feed verify results to the matching BS slot's params."""
        if batch_size <= 0:
            return

        bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        params = self._bs_params[bs]
        old_steps = params.current_steps

        changed = params.update(accept_lengths)

        if changed:
            logger.info(
                f"AdaptiveController: BS slot {bs} (actual bs={batch_size}) "
                f"steps {old_steps} -> {params.current_steps}"
            )
            self.activate(params.current_steps)

    def activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
