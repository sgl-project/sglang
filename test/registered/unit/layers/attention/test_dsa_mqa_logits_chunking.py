"""Contract tests for the DSA indexer's MQA-logits chunk budget.

The indexer materialises a `[tokens x seq_len_kv]` fp32 logits tensor. On ROCm
AITER stores it with `buffer_store`, whose 32-bit byte offset caps one tensor at
2 GiB; past that AITER takes a `gl.store` path that fails to compile. So the
budget that decides chunking is a correctness bound there and not only an
out-of-memory guard, and these tests pin that.

The measured memory budget is stubbed: it is the only input the ceiling has to
beat, and stubbing it keeps these tests on CPU.
"""

import math
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.dsa import dsa_indexer  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CEILING = dsa_indexer.Indexer._MQA_LOGITS_MAX_BYTES_ROCM
BYTES_PER_ELEM = dsa_indexer.Indexer._MQA_LOGITS_BYTES_PER_ELEM
#: A cold prefill makes both logits dimensions the prompt length, so this is the
#: longest one AITER can still address.
LAST_ADDRESSABLE = math.isqrt(CEILING // BYTES_PER_ELEM)
#: More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    """`_should_chunk_mqa_logits` with the measured budget replaced.

    The instance is built without `__init__` so the real class constants are the
    ones under test.
    """
    indexer = dsa_indexer.Indexer.__new__(dsa_indexer.Indexer)
    with mock.patch.object(dsa_indexer, "_is_hip", is_hip), mock.patch.object(
        dsa_indexer.Indexer,
        "_get_mqa_logits_budget_bytes",
        return_value=mem_budget,
    ):
        return indexer._should_chunk_mqa_logits(num_q, num_k, 0)


def test_last_addressable_length_brackets_the_ceiling():
    """The length the cases below are built on, stated as the arithmetic it is."""
    assert LAST_ADDRESSABLE**2 * BYTES_PER_ELEM <= CEILING
    assert (LAST_ADDRESSABLE + 1) ** 2 * BYTES_PER_ELEM > CEILING


def test_square_logits_at_the_ceiling_are_not_chunked():
    assert _decide(LAST_ADDRESSABLE, LAST_ADDRESSABLE) == (False, CEILING)


@pytest.mark.parametrize("num_k", [23_171, 100_000, 1_000_000])
def test_chunks_stay_within_what_aiter_can_address(num_k):
    """What matters is the size of a chunk, not of the budget: the caller turns
    the budget into a row count and each chunk's logits tensor is that wide."""
    need_chunk, budget = _decide(num_k, num_k)
    assert need_chunk
    bytes_per_row = num_k * BYTES_PER_ELEM
    max_rows = max(1, budget // bytes_per_row)
    assert max_rows * bytes_per_row <= CEILING


def test_a_smaller_memory_budget_still_wins():
    """The ceiling caps the measured budget and must never raise it."""
    one_gib = 2**30
    assert _decide(LAST_ADDRESSABLE, LAST_ADDRESSABLE, mem_budget=one_gib) == (
        True,
        one_gib,
    )


def test_off_rocm_the_budget_is_untouched():
    """Elsewhere the logits go to DeepGEMM, which has no such limit."""
    assert _decide(23_171, 23_171, is_hip=False) == (False, HUGE_MEM_BUDGET)


def test_small_batches_skip_the_budget():
    assert _decide(1, 4096) == (False, 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
