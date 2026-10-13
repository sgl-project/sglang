import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


@pytest.mark.parametrize(
    "pool_cls, pool_kwargs",
    [(ReqToTokenPool, {}), (DecodeReqToTokenPool, {"pre_alloc_size": 0})],
    ids=["ReqToTokenPool", "DecodeReqToTokenPool"],
)
def test_request_pool_clear_keeps_row_generations_monotonic(pool_cls, pool_kwargs):
    """Readers that stash req_generation[row] detect row reuse by inequality with
    the live value. clear() (flush_cache) must not reset the counter: the row's
    next request would otherwise match the generation its previous request had."""
    pool = pool_cls(
        size=2,
        max_context_len=4,
        device="cpu",
        enable_memory_saver=False,
        **pool_kwargs,
    )

    def start_request() -> int:
        (row,) = pool.alloc([SimpleNamespace(kv=ReqKvInfo())])
        return row

    row = start_request()
    stored_generation = pool.req_generation[row].item()

    pool.clear()

    assert start_request() == row
    assert pool.req_generation[row].item() != stored_generation


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
