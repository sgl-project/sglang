"""Contract tests for the gfx950 assembly attention segment-plan cap."""

import pytest

torch = pytest.importorskip("torch")

from sglang.kernels.ops.attention.vattn_asm_gfx950 import (  # noqa: E402
    _seg_plan_max_segments,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_CU = 256  # MI355X


@pytest.mark.parametrize(
    "num_seqs,num_kv_heads,expected",
    [
        (8, 1, 64),
        (9, 1, 56),
        (16, 1, 32),
        (17, 1, 30),
        (24, 1, 20),
        (29, 1, 16),
        (9, 2, 28),
        (14, 2, 18),
        (15, 2, 16),
    ],
)
def test_cap_is_twice_the_exact_uniform_cu_share(num_seqs, num_kv_heads, expected):
    assert _seg_plan_max_segments(NUM_CU, num_seqs, num_kv_heads) == expected


@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_cap_never_increases_with_more_sequences(num_kv_heads):
    caps = [
        _seg_plan_max_segments(NUM_CU, num_seqs, num_kv_heads)
        for num_seqs in range(1, NUM_CU + 2)
    ]
    assert caps == sorted(caps, reverse=True)
    assert all(16 <= cap <= 64 for cap in caps)
