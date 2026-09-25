"""Contract tests for the MQA-logits chunk decision shared by the DSA and DSV4
indexers.

On ROCm the `[num_q x num_k]` fp32 logits tensor goes to aiter's
`fp8_mqa_logits`, which only compiles below 2 GiB, so the budget that decides
chunking is a correctness bound there and not only an out-of-memory guard.

`Indexer` decides that with a budget; `IndexerKPool` splits the query rows
around its aiter call. Both are stubbed down to CPU-sized inputs.
"""

import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool  # noqa: E402
from sglang.srt.layers.attention.mqa_logits_utils import (  # noqa: E402
    MQA_LOGITS_BYTES_PER_ELEM,
    MQA_LOGITS_MAX_BYTES_ROCM,
    mqa_logits_should_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

CEILING = MQA_LOGITS_MAX_BYTES_ROCM
# More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    return mqa_logits_should_chunk(
        num_rows=num_q,
        num_cols=num_k,
        get_budget_bytes=lambda: mem_budget,
        rocm=is_hip,
    )


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


def _kpool_mqa_logits(num_q, num_k, cap_bytes):
    """Run `IndexerKPool._fp8_mqa_logits` over a stub that returns each row's
    index plus its `starts`, so a misordered reassembly or an unsliced per-row
    argument shows up in the logits."""
    rows_per_call = []

    def _kernel(q_fp8, k_fp8, k_scale, weights, starts, ends, *, clean_logits):
        rows = q_fp8.shape[0]
        assert weights.shape[0] == rows
        assert starts.shape[0] == rows
        assert ends.shape[0] == rows
        rows_per_call.append(rows)
        row_value = q_fp8[:, 0, 0] + starts
        return row_value.unsqueeze(1).expand(-1, k_fp8.shape[0]).contiguous()

    aiter_kernel_module = types.ModuleType("aiter.ops.triton.fp8_mqa_logits")
    aiter_kernel_module.fp8_mqa_logits = _kernel
    stub_modules = {
        name: types.ModuleType(name)
        for name in ("aiter", "aiter.ops", "aiter.ops.triton")
    }
    stub_modules["aiter.ops.triton.fp8_mqa_logits"] = aiter_kernel_module

    q_fp8 = torch.arange(num_q, dtype=torch.float32).view(num_q, 1, 1).expand(-1, 4, 8)
    k_fp8 = torch.zeros(num_k, 8, dtype=torch.float32)
    with (
        mock.patch.dict(sys.modules, stub_modules),
        mock.patch.object(dsa_indexer_kpool, "is_hip", lambda: True),
        mock.patch.object(dsa_indexer_kpool, "MQA_LOGITS_MAX_BYTES_ROCM", cap_bytes),
    ):
        logits = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
            q_fp8,
            k_fp8,
            torch.ones(num_k),
            torch.ones(num_q, 4),
            torch.arange(num_q, dtype=torch.float32) * 10,
            torch.full((num_q,), num_k, dtype=torch.int32),
            clean_logits=True,
        )
    return logits, rows_per_call


def test_kpool_splits_the_query_rows_to_stay_under_the_aiter_cap():
    """Above 2 GiB of logits the aiter kernel abort()s the process instead of
    raising, so the k-pool wrapper must never hand it a larger tensor."""
    num_q, num_k = 8, 4
    row_bytes = num_k * MQA_LOGITS_BYTES_PER_ELEM
    logits, rows_per_call = _kpool_mqa_logits(num_q, num_k, 3 * row_bytes + 1)

    assert rows_per_call == [3, 3, 2]
    rows = torch.arange(num_q, dtype=torch.float32)
    assert torch.equal(logits, (rows + rows * 10).unsqueeze(1).expand(-1, num_k))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
