"""Contract tests for the DSA indexer's MQA-logits chunk budget.

On ROCm the `[num_q x num_k]` fp32 logits tensor goes to aiter's
`fp8_mqa_logits`, which only compiles below 2 GiB, so the budget that decides
chunking is a correctness bound there and not only an out-of-memory guard.

The measured memory budget is stubbed: it is the only input the limit has to
beat, and stubbing it keeps these tests on CPU.
"""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.dsa import mqa_logits_budget  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

CEILING = mqa_logits_budget.MQA_LOGITS_MAX_BYTES_ROCM
# More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    with (
        mock.patch.object(mqa_logits_budget, "_is_hip", is_hip),
        mock.patch.object(
            mqa_logits_budget,
            "mqa_logits_budget_bytes",
            return_value=mem_budget,
        ),
    ):
        return mqa_logits_budget.should_chunk_mqa_logits(num_q, num_k, 0)


def test_the_ceiling_is_the_largest_logits_aiter_still_takes():
    # 16384 x 32768 x 4 bytes is exactly 2 GiB, and aiter compares `bytes <
    # 2 GiB`, so that shape has to chunk and one KV token less must not.
    assert _decide(16_384, 32_768) == (True, CEILING)
    assert _decide(16_384, 32_767) == (False, CEILING)


def test_a_smaller_memory_budget_still_wins():
    one_gib = 2**30
    assert _decide(16_384, 32_767, mem_budget=one_gib) == (True, one_gib)


def test_off_rocm_the_budget_is_untouched():
    # Elsewhere the logits go to DeepGEMM, which has no such limit.
    assert _decide(16_384, 32_768, is_hip=False) == (False, HUGE_MEM_BUDGET)


def test_max_rows_splits_the_budget_across_the_k_axis():
    one_gib = 2**30
    # 2 GiB of columns per row: only a single row fits, never zero rows.
    assert mqa_logits_budget.mqa_logits_max_rows(one_gib, 2**29, 16_384) == 1
    # 4 KiB per row -> 262144 rows fit, but never more rows than exist.
    assert mqa_logits_budget.mqa_logits_max_rows(one_gib, 1024, 16_384) == 16_384
    assert mqa_logits_budget.mqa_logits_max_rows(one_gib, 1024, 2**19) == 262_144
    # A degenerate zero-column window still asks for the rows it has.
    assert mqa_logits_budget.mqa_logits_max_rows(one_gib, 0, 7) == 7


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
