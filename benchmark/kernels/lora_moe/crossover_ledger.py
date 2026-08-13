"""Crossover ledger — the plan §31.7 evidence record, serialized with suites.

Every gate packet from Step 2 onward carries a ledger of measured crossovers:
site | boundary | candidates | the AXIS that drives the reversal | measured
crossover location | bracketing cases | device.  Step 2 produced these as
stdout prose; Step 3 serializes them into the suite JSON so a selector can be
built from archives instead of from remembered sentences (plan §63.1 P1).

The cell-decision semantics are the ones the align-boundary study used and
gate 2 ratified: a cell is DECIDED only when one arm wins every paired sample
(unanimous sign) AND the geometric-mean margin meets ``MIN_MARGIN``; anything
else is a tie, and a crossover may only be sited between two decided cells.
``bench_align_boundary`` imports these functions rather than keeping its own
copy, so the ledger and the historical producer cannot drift.
"""

from __future__ import annotations

import msgspec

# Unlocked clocks; below this geometric-mean ratio a cell is a tie.
MIN_MARGIN = 1.05


class CellDecision(msgspec.Struct, frozen=True, kw_only=True):
    """Verdict for one cell of a two-arm comparison."""

    arm_a: str
    arm_b: str
    # None = tied (non-unanimous sign, or unanimous but under the margin).
    winner: str | None
    # Geometric mean of the per-sample a/b time ratios; > 1 means b faster.
    geo_a_over_b: float
    unanimous: bool
    min_margin: float
    # 12th review (gate-4): "charged" = both arms timed at the SAME
    # boundary; "ceiling" = the arms carry DIFFERENT boundaries (one side
    # excludes work the other includes), so the decision bounds the
    # possibility rather than adjudicating a policy; "undeclared" = a
    # legacy caller that did not state its boundaries.
    scope: str = "undeclared"

    def margin(self) -> float:
        """Winner's geometric-mean advantage (>= 1); 1.0 when tied."""
        if self.winner is None:
            return 1.0
        return self.geo_a_over_b if self.winner == self.arm_b else 1 / self.geo_a_over_b


def decide_cell(
    *,
    arm_a: str,
    samples_a: list[float],
    arm_b: str,
    samples_b: list[float],
    min_margin: float = MIN_MARGIN,
    boundary_a: str | None = None,
    boundary_b: str | None = None,
) -> CellDecision:
    """Decide one cell from paired timing samples (seconds, same order).

    Samples must be PAIRED — the i-th entries of both lists come from the
    same (seed, repeat) so clock drift cancels in the ratio.

    12th review: comparing arms timed at DIFFERENT boundaries (grouped
    route-inclusive vs sgmv prepared-input) is a CEILING statement, not a
    policy verdict — the prepared side is credited work the charged side
    pays for. Callers declare each arm's boundary; a mismatch is recorded
    and printed as scope=ceiling so adjudication cannot mistake it for a
    same-boundary win.
    """
    if len(samples_a) != len(samples_b) or not samples_a:
        raise ValueError("paired non-empty sample lists required")
    if arm_a == arm_b:
        raise ValueError("arms must be distinct")
    ratios = [a / b for a, b in zip(samples_a, samples_b)]
    geo = 1.0
    for ratio in ratios:
        geo *= ratio
    geo **= 1 / len(ratios)
    winner: str | None = None
    unanimous = all(r > 1 for r in ratios) or all(r < 1 for r in ratios)
    if all(r > 1 for r in ratios) and geo >= min_margin:
        winner = arm_b
    elif all(r < 1 for r in ratios) and 1 / geo >= min_margin:
        winner = arm_a
    if boundary_a is None or boundary_b is None:
        scope = "undeclared"
    elif boundary_a == boundary_b:
        scope = "charged"
    else:
        scope = "ceiling"
    return CellDecision(
        scope=scope,
        arm_a=arm_a,
        arm_b=arm_b,
        winner=winner,
        geo_a_over_b=geo,
        unanimous=unanimous,
        min_margin=min_margin,
    )


class CrossoverLedgerEntry(msgspec.Struct, frozen=True, kw_only=True):
    """One §31.7 ledger row.

    ``axis`` names the parameter that drives the reversal (the load-bearing
    column: a threshold without its axis cannot seed the Step-13 selector).
    ``crossover_location`` sites the flip BETWEEN two decided cells (e.g.
    ``"expected_m in (64, 96]"``); if either bracketing cell is a tie the
    crossover is not sited and this entry must say so in ``notes`` instead.
    ``bracketing_low_record_ids`` / ``bracketing_high_record_ids`` carry the
    timing records of the two bracketing cells (BOTH candidates in EACH
    cell) and ``bracketing_case_ids`` their content-addressed workloads,
    added in the same step that found the crossover.

    Do not construct directly for evidence: entries reach an archive ONLY
    through ``TimingSuite.site_crossover``, which derives ``device`` and
    ``source_revision`` from the suite, refuses record IDs the suite did
    not measure, and requires every declared candidate to have records in
    BOTH bracketing cells — a ledger row must not be able to cite evidence
    that is not in the file that carries it, nor bracket a flip with a
    one-armed cell (first and third S3 reviews).
    """

    site: str
    boundary: str
    candidates: tuple[str, ...]
    axis: str
    crossover_location: str
    bracketing_low_record_ids: tuple[str, ...]
    bracketing_high_record_ids: tuple[str, ...]
    bracketing_case_ids: tuple[str, ...]
    device: str
    source_revision: str
    cache_state: str
    notes: str = ""
