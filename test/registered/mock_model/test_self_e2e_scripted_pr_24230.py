"""Scripted-PR regression: #24230 (scenario TBD).

Phase-06 SOT pointer: testing.md §4.2 type-a, file `test_self_e2e_scripted_pr_24230.py`. The SOT
records `TBD scenario name` for this PR. Per the dispatcher instructions we use the literal
`placeholder` token in case names so the TBD is visibly unresolved — phase-2 implementers rename
both cases (and the matching CUDA-graph-parity parametrize label) once the scenario is read off the
upstream PR description.

This file is shipped as a phase-2 skeleton: the module-level pytestmark below skips it until the
mock_mode subsystem (MockEngine + oracle + sampler hook) lands on this branch.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_pr_24230_regression_placeholder() -> None:
    pass


def test_pr_24230_with_fix_placeholder() -> None:
    pass
