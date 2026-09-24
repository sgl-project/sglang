"""Per-step uniform verify width for the static DSpark verify.

Every request of a step verifies the same number of tokens ``w`` (anchor plus the
first ``w - 1`` drafts). Each extra width owns a target attention backend and a
decode graph runner built for ``w`` tokens per request; the worker swaps them onto
the target model runner around the verify forward only. The policy picks the width
that maximizes predicted committed tokens per second: the expected accept length
from running per-position acceptance, times the SPS table's steps per second at
``bs * w`` verify tokens.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable


class VerifyWidthRuntime(msgspec.Struct):
    width: int
    attn_backend: object
    graph_runner: object
    epilogue: object


@contextlib.contextmanager
def use_verify_width_runtime(model_runner, runtime: Optional[VerifyWidthRuntime]):
    """Serve the verify forward from ``runtime``'s backend and graphs. ``None``
    keeps the full-width ones the runner was initialized with."""
    if runtime is None:
        yield
        return
    saved = (model_runner.attn_backend, model_runner.decode_cuda_graph_runner)
    model_runner.attn_backend = runtime.attn_backend
    model_runner.decode_cuda_graph_runner = runtime.graph_runner
    try:
        yield
    finally:
        model_runner.attn_backend, model_runner.decode_cuda_graph_runner = saved


def verify_width_unsupported_reason(
    *, mode_value: str, target_is_mambaish: bool, simulate_acc_len: float
) -> Optional[str]:
    """Why per-step verify widths cannot run here, or None. The debug observers
    read full-width verify tensors, so they pin the full width."""
    if mode_value != "static":
        return f"needs SGLANG_RAGGED_VERIFY_MODE=static, got {mode_value!r}"
    if get_parallel().attn_dp_enabled:
        return "not supported with attention DP"
    if get_parallel().pp_size != 1:
        return "not supported with pipeline parallelism"
    if target_is_mambaish:
        return "not supported for mamba/linear-attention targets"
    if simulate_acc_len > 0:
        return "not supported with SGLANG_SIMULATE_ACC_LEN"
    if (
        envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.get()
        or envs.SGLANG_DSPARK_DEBUG_DUMP.get()
        or envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
        or envs.SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH.get()
    ):
        return "not supported with the DSpark debug observers"
    return None


def parse_verify_widths(raw: Sequence[str], *, full_width: int) -> list[int]:
    widths = sorted({int(w) for w in raw if str(w).strip()})
    for w in widths:
        if not 2 <= w < full_width:
            raise ValueError(
                f"SGLANG_DSPARK_VERIFY_WIDTHS entries must be in [2, {full_width - 1}] "
                f"(the full width {full_width} is always captured), got {w}."
            )
    return widths


class CorrectDraftHistogram:
    """Device-side counts of ``correct_len`` per width, read on the host with a
    fixed lag so every TP rank updates the policy at the same step."""

    def __init__(self, *, widths: list[int], full_width: int, device) -> None:
        self._row = {w: i for i, w in enumerate(widths)}
        self.counts = torch.zeros(
            (len(widths), full_width), dtype=torch.int64, device=device
        )
        self._host = torch.zeros(self.counts.shape, dtype=torch.int64).pin_memory()
        self._event: Optional[torch.cuda.Event] = None
        self._last = torch.zeros(self.counts.shape, dtype=torch.int64)

    def record(self, *, width: int, correct_len: torch.Tensor) -> None:
        self.counts[self._row[width]].index_add_(
            0,
            correct_len.to(torch.int64).clamp_(0, self.counts.shape[1] - 1),
            torch.ones_like(correct_len, dtype=torch.int64),
        )

    def snapshot(self) -> None:
        self._host.copy_(self.counts, non_blocking=True)
        self._event = torch.cuda.Event()
        self._event.record()

    def take_delta(self) -> Optional[torch.Tensor]:
        """Counts since the previous take, from the snapshot one interval ago."""
        if self._event is None:
            return None
        self._event.synchronize()
        current = self._host.clone()
        delta = current - self._last
        self._last = current
        return delta


class VerifyWidthPolicy:
    """Choose the verify width per step from the batch size, the SPS table and
    running per-position acceptance. Deterministic given the same inputs, so all
    TP ranks pick the same width."""

    def __init__(
        self,
        *,
        widths: list[int],
        full_width: int,
        sps_table: SpsCostTable,
        forced_width: Optional[int] = None,
        margin: float = 0.02,
        decay: float = 0.9,
        explore_every: int = 64,
    ) -> None:
        self.widths = sorted(set(widths) | {full_width})
        self.full_width = full_width
        self.sps_table = sps_table
        if forced_width is not None and forced_width not in self.widths:
            raise ValueError(
                f"SGLANG_DSPARK_FORCE_VERIFY_WIDTH={forced_width} is not a captured "
                f"width {self.widths}."
            )
        self.forced_width = forced_width
        self.margin = margin
        self.decay = decay
        self.explore_every = explore_every
        num_drafts = full_width - 1
        # Decayed counts of steps that reached draft j, and of those where it was correct.
        self._reached = [0.0] * num_drafts
        self._correct = [0.0] * num_drafts
        self._current = full_width
        self._decisions = 0

    def hazards(self) -> Optional[list[float]]:
        """Conditional acceptance of draft j given drafts 0..j-1 were accepted;
        None until every draft position has been observed."""
        if min(self._reached) <= 0:
            return None
        return [a / r for a, r in zip(self._correct, self._reached)]

    def expected_accept_len(self, width: int, hazards: list[float]) -> float:
        survival, total = 1.0, 1.0
        for j in range(width - 1):
            survival *= hazards[j]
            total += survival
        return total

    def predicted_tokens_per_sec(self, bs: int, width: int, hazards) -> float:
        return (
            bs
            * self.expected_accept_len(width, hazards)
            * self.sps_table.interp(bs * width)
        )

    def choose(self, bs: int) -> int:
        if self.forced_width is not None:
            return self.forced_width
        self._decisions += 1
        hazards = self.hazards()
        # Full width until every position is measured, and periodically after, so
        # the positions a narrow width never verifies keep being re-estimated.
        if hazards is None or self._decisions % self.explore_every == 0:
            return self.full_width
        pred = {w: self.predicted_tokens_per_sec(bs, w, hazards) for w in self.widths}
        best = max(pred, key=pred.get)
        current = self._current
        if best != current and pred[best] > pred[current] * (1.0 + self.margin):
            self._current = best
        return self._current

    def update(self, delta: torch.Tensor) -> None:
        """``delta[i, k]``: steps of width ``widths[i]`` whose request had k correct
        drafts, since the previous update."""
        num_drafts = self.full_width - 1
        reached = [0.0] * num_drafts
        correct = [0.0] * num_drafts
        for row, width in enumerate(self.widths):
            counts = delta[row].tolist()
            for j in range(width - 1):
                reached[j] += sum(counts[j:])
                correct[j] += sum(counts[j + 1 :])
        for j in range(num_drafts):
            if reached[j] > 0:
                self._reached[j] = self._reached[j] * self.decay + reached[j]
                self._correct[j] = self._correct[j] * self.decay + correct[j]


class VerifyWidthController:
    """Owns the extra-width runtimes, the width policy and its acceptance feed.
    Width decisions depend only on the step's batch size and on lagged, TP-synced
    acceptance counts, so every rank verifies at the same width."""

    def __init__(
        self,
        *,
        widths: list[int],
        full_width: int,
        sps_table: SpsCostTable,
        forced_width: Optional[int],
        device,
        update_interval: int = 32,
    ) -> None:
        self.full_width = full_width
        self.policy = VerifyWidthPolicy(
            widths=widths,
            full_width=full_width,
            sps_table=sps_table,
            forced_width=forced_width,
        )
        self.histogram = CorrectDraftHistogram(
            widths=self.policy.widths, full_width=full_width, device=device
        )
        self.extra_widths = [w for w in self.policy.widths if w != full_width]
        self.runtimes: dict[int, VerifyWidthRuntime] = {}
        self._update_interval = update_interval
        self._steps = 0

    def build_runtimes(
        self,
        *,
        model_runner,
        capture_graphs: bool,
        full_epilogue,
        make_epilogue,
    ) -> None:
        """Build a target attention backend, and decode graphs when allowed, for
        every extra width. Both read speculative_num_draft_tokens at construction,
        and the full-width epilogue's capture hook is replaced by the width's own."""
        hooks = model_runner.capture_tail_hooks
        saved_hooks = list(hooks)
        full_hook = None if full_epilogue is None else full_epilogue.capture_hook
        for width in self.extra_widths:
            epilogue = None if full_epilogue is None else make_epilogue(width)
            if full_hook is not None:
                hooks[:] = [
                    epilogue.capture_hook if h == full_hook else h for h in saved_hooks
                ]
            saved_workspace = model_runner.init_new_workspace
            try:
                with get_spec().override(speculative_num_draft_tokens=width):
                    attn_backend = model_runner._get_attention_backend(
                        init_new_workspace=True
                    )
                    graph_runner = None
                    if capture_graphs:
                        graph_runner = model_runner._decode_cuda_graph_runner_cls()(
                            model_runner,
                            attn_backend=attn_backend,
                            speculative_num_draft_tokens=width,
                        )
            finally:
                model_runner.init_new_workspace = saved_workspace
                hooks[:] = saved_hooks
            self.runtimes[width] = VerifyWidthRuntime(
                width=width,
                attn_backend=attn_backend,
                graph_runner=graph_runner,
                epilogue=epilogue,
            )

    def select(
        self, *, bs: int, eligible: bool
    ) -> tuple[int, Optional[VerifyWidthRuntime]]:
        if not eligible:
            return self.full_width, None
        width = self.policy.choose(bs)
        return width, self.runtimes.get(width)

    def observe(self, *, width: int, correct_len: torch.Tensor) -> None:
        self.histogram.record(width=width, correct_len=correct_len)
        self._steps += 1
        if self._steps % self._update_interval == 0:
            delta = self.histogram.take_delta()
            if delta is not None:
                self.policy.update(delta)
            self.histogram.snapshot()
