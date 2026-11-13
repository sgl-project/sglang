"""Masked capacity is the smallest positive aligned bound; anything smaller
spills an expert's rows into the next expert's slab."""

import pytest

from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import masked_m_max
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("alignment", "num_tokens"),
    [(8, 0), (8, 1), (8, 8), (8, 9), (128, 128), (128, 129)],
)
def test_masked_capacity_is_minimal_positive_alignment(num_tokens, alignment):
    capacity = masked_m_max(num_tokens, alignment)
    assert capacity >= num_tokens
    assert capacity > 0
    assert capacity % alignment == 0
    assert capacity - alignment < max(1, num_tokens)
    if num_tokens and num_tokens % alignment == 0:
        assert capacity == num_tokens


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
