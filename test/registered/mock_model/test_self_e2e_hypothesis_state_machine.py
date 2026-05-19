"""Hypothesis state-machine self-e2e (type-b): 7 cases — 3 invariants, 3 rules, 1 shrinking meta.

Phase-06 SOT pointer: testing.md §4.2 type-b. Per the SOT, this file holds a
`RuleBasedStateMachine` driving (admit / preempt / resume) rules against `MockEngine` over 1000
random steps within a 1-min wall-clock cap, with `--hypothesis-seed=0` and three invariants
(no_canary_violations / allocator_conservation / block_table_in_held) plus a shrinking meta-test
that verifies a designed-to-fail scenario shrinks to a repro under 10 steps.

This file is shipped as a phase-2 skeleton: the SOT case names and signatures are pinned here, but
the bodies wait on the `mock_mode` subsystem (`MockEngine`) and the `hypothesis` dependency. The
module-level pytestmark below skips the file; phase-2 implementers remove it once `MockEngine`
lands and `hypothesis` is present in `requirements/`.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_machine_invariant_no_canary_violations() -> None:
    pass


def test_machine_invariant_allocator_conservation() -> None:
    pass


def test_machine_invariant_block_table_in_held() -> None:
    pass


def test_machine_rule_admit_random_prompt() -> None:
    pass


def test_machine_rule_preempt_random() -> None:
    pass


def test_machine_rule_resume_random() -> None:
    pass


def test_machine_shrinking_minimal_repro_under_10_steps() -> None:
    pass
