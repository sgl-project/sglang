"""Request-owned KPool tail indices for plain and wrapped pools."""

from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.utils import get_dsa_tail_state_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_tail_indices_follow_pool_wrapper_and_compression(wrapped, compressed):
    pool = SimpleNamespace(
        use_dsa=True, kpool_use_compress=compressed, index_kpool=8, tail_extra_slots=2
    )
    if wrapped:
        pool = SimpleNamespace(use_dsa=True, full_kv_pool=pool)
    expected = [3, 6, 3, 0, 0, 10] if compressed else []
    assert get_dsa_tail_state_indices(pool, 3, 19) == expected
