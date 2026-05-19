"""Timing-jitter self-e2e (type-c): v1 out-of-scope placeholder.

Phase-06 SOT pointer: testing.md §4.2 type-c. Per the SOT, this file ships one case (`test_skipped_
v1_oos`) carrying a per-case skip with reason `phase-2; depends on timing jitter fuzzer infra`. The
file reserves the path so phase-2 implementers do not have to invent a new location once the
fuzzer ships.

Two skip layers are stacked deliberately:

1. Module-level pytestmark `phase-2; awaits mock_mode subsystem` — removable as soon as MockEngine
   lands on this branch.
2. Per-case `@pytest.mark.skip(reason="phase-2; depends on timing jitter fuzzer infra")` — survives
   the module-level removal, since the SOT mandates the per-case skip until the timing-jitter
   fuzzer itself exists. This case ships with that decorator even on the placeholder so the SOT
   contract is preserved across the eventual mock_mode landing.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


@pytest.mark.skip(reason="phase-2; depends on timing jitter fuzzer infra")
def test_skipped_v1_oos() -> None:
    pass
