"""CUDA graph parity self-e2e: graph capture vs eager violation set byte-equal.

Phase-06 SOT pointer: testing.md §4.2 "CUDA graph 等价性 (v1 验收清单 6)". For each of the five type-a
scripted PR scenarios, run the same regression twice — once with `--disable-cuda-graph` (eager) and
once with the default cuda-graph capture path — and assert the resulting canary violation set is
byte-equal across the two runs (no ordering tolerance — atomicAdd races count as failures per the
SOT note).

Materialized as 5 top-level cases (one per PR id) so each PR scenario stays grep-able in test
reports; phase-2 may consolidate via `@pytest.mark.parametrize("pr_id", [...])` once `MockEngine`
exposes a unified scenario harness. The two TBD-name PRs (#24230 / #24401 / #22819) keep the
`pr_<id>` suffix shape and inherit the "placeholder" naming decision from their type-a siblings —
phase-2 implementers rename in lockstep when they fill in the scenario names.

This file is shipped as a phase-2 skeleton: module-level pytestmark skips until the mock_mode
subsystem lands.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_graph_capture_vs_eager_same_violation_set_pr_25015() -> None:
    pass


def test_graph_capture_vs_eager_same_violation_set_pr_24230() -> None:
    pass


def test_graph_capture_vs_eager_same_violation_set_pr_24401() -> None:
    pass


def test_graph_capture_vs_eager_same_violation_set_pr_22819() -> None:
    pass


def test_graph_capture_vs_eager_same_violation_set_pr_20711() -> None:
    pass
