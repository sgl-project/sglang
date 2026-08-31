import sys

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.mem_cache.kv_cache_configurator import _req_slot_capacity
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _init_decode_pool(pool):
    DecodeReqToTokenPool.__init__(
        pool,
        size=1,
        max_context_len=4,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=1,
    )
    return pool


def test_decode_pool_reports_physical_capacity():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))

    assert pool.schedulable_token_capacity(17) == 17


def test_decode_pool_supports_noop_aux_cache_contract():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))
    req_to_token = pool.req_to_token.clone()

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([1]),
        target_seq_lens_cpu=torch.tensor([3]),
    )
    pool.reset_aux_cache_allocator()

    assert torch.equal(pool.req_to_token, req_to_token)


def test_hybrid_decode_pool_initializes_aux_cache_contract():
    pool = _init_decode_pool(
        HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
    )

    assert pool.schedulable_token_capacity(17) == 17


def test_kpool_slot_capacity_covers_decode_preallocation_rows():
    pool = DecodeReqToTokenPool(
        size=32,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=64,
    )

    assert pool.req_to_token.shape[0] == 97
    assert _req_slot_capacity(pool) == 96


def test_kpool_slot_capacity_counts_padding_row_once():
    pool = ReqToTokenPool(
        size=32,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
    )

    assert pool.req_to_token.shape[0] == 33
    assert _req_slot_capacity(pool) == 32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
