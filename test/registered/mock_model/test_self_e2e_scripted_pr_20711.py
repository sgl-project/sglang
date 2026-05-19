"""Scripted-PR regression: #20711 cross-batch bitwise invariance (v1 only compares token id).

Phase-06 SOT pointer: testing.md §4.2 type-a, file `test_self_e2e_scripted_pr_20711.py`. SOT note:
"v1 只比 token id（sampler override 强制），不比 logits 数值".

This file is shipped as a phase-2 skeleton: the SOT case names and signatures are pinned here, but
the bodies wait on the `mock_mode` subsystem (`MockEngine` + sampler hook + per-batch oracle
comparison) which is not yet present on this branch. The module-level pytestmark below skips the
file in CI; phase-2 implementers remove it once `MockEngine` lands.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_cross_batch_bitwise_regression() -> None:
    pass


def test_cross_batch_bitwise_with_fix() -> None:
    pass
