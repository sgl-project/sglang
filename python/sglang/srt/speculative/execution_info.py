"""Host-side dispatch observations for a completed adaptive EAGLE V2 round.

Counts are logical model forwards, not requests, tokens, rounds or CUDA graph
API calls. A multi-step draft replay covers several model forwards. A graph
dispatch returning successfully does not prove asynchronous GPU completion.
"""

from dataclasses import dataclass
from typing import Literal

Phase = Literal["draft", "target_verify", "draft_extend"]
PHASES: tuple[Phase, ...] = ("draft", "target_verify", "draft_extend")


@dataclass(frozen=True)
class PhaseExecution:
    graph_forwards: int = 0
    eager_forwards: int = 0


@dataclass(frozen=True)
class SpecRoundExecution:
    """Configuration selected for the round plus observed host dispatches.

    The first three fields are selected configuration, not measured work.
    verify_num_tokens_per_req is the actual verify input layout, not the
    graph's padded capture width or a count of accepted/proposed tokens.
    """

    num_steps: int
    draft_num_tokens_per_req: int
    num_draft_tokens: int
    verify_num_tokens_per_req: int
    draft: PhaseExecution
    target_verify: PhaseExecution
    draft_extend: PhaseExecution


class SpecExecutionRecorder:
    """Round-local builder; never retained on the worker or controller."""

    def __init__(self, num_steps: int, draft_width: int, num_draft_tokens: int):
        self.num_steps = num_steps
        self.draft_width = draft_width
        self.num_draft_tokens = num_draft_tokens
        self.verify_num_tokens_per_req: int | None = None
        self._counts = {phase: [0, 0] for phase in PHASES}

    def record(self, phase: Phase, *, graph: bool, num_forwards: int = 1):
        if num_forwards < 0:
            raise ValueError("negative model-forward count")
        self._counts[phase][0 if graph else 1] += num_forwards

    def finish(self) -> SpecRoundExecution:
        if self.verify_num_tokens_per_req is None:
            raise RuntimeError("target verify width was not observed")
        return SpecRoundExecution(
            self.num_steps,
            self.draft_width,
            self.num_draft_tokens,
            self.verify_num_tokens_per_req,
            *(PhaseExecution(*self._counts[phase]) for phase in PHASES),
        )


def observe_model_forward(recorder, phase: Phase, forward, *args, **kwargs):
    """Observe the returned dispatch verdict, not pre-dispatch eligibility.

    ModelRunnerOutput and GenerationBatchResult expose different field names.
    Callers adapt the latter explicitly at the target boundary.
    """
    output = forward(*args, **kwargs)
    if recorder is not None:
        recorder.record(phase, graph=output.can_run_graph)
    return output


class SpecExecutionWindow:
    """Bounded per-tier interval totals, consumed once at result processing.

    No batch/request references or arrival-time controller state are retained.
    The result marker survives report-window drains and prevents double count.
    """

    def __init__(self):
        self._totals = {}

    def observe(self, result):
        execution = result.spec_execution
        if execution is None or result.spec_execution_accounted:
            return
        key = (
            execution.num_steps,
            execution.draft_num_tokens_per_req,
            execution.num_draft_tokens,
            execution.verify_num_tokens_per_req,
        )
        row = self._totals.get(key)
        if row is None:
            row = self._totals[key] = dict(
                num_steps=key[0],
                draft_num_tokens_per_req=key[1],
                num_draft_tokens=key[2],
                verify_num_tokens_per_req=key[3],
                rounds=0,
                phases={
                    p: dict(graph_forwards=0, eager_forwards=0, unexecuted_rounds=0)
                    for p in PHASES
                },
            )
        row["rounds"] += 1
        for phase in PHASES:
            counts = getattr(execution, phase)
            totals = row["phases"][phase]
            totals["graph_forwards"] += counts.graph_forwards
            totals["eager_forwards"] += counts.eager_forwards
            totals["unexecuted_rounds"] += (
                counts.graph_forwards + counts.eager_forwards == 0
            )
        result.spec_execution_accounted = True

    def drain(self, *, discard: bool = False) -> list[dict] | None:
        """End the interval, optionally without materializing its output rows."""
        totals, self._totals = self._totals, {}
        if discard:
            return None
        return [totals[key] for key in sorted(totals)]
