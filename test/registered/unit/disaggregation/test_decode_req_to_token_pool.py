import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import free_req_slot
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.penalties import PenaltyStore
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


def test_decode_pool_supports_noop_aux_cache_contract():
    pool = _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool))
    req_to_token = pool.req_to_token.clone()

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([1]),
        target_seq_lens_cpu=torch.tensor([3]),
    )
    pool.reset_aux_cache_allocator()

    assert torch.equal(pool.req_to_token, req_to_token)


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


@pytest.mark.parametrize(
    "make_pool",
    [
        lambda: ReqToTokenPool(
            size=1, max_context_len=4, device="cpu", enable_memory_saver=False
        ),
        lambda: _init_decode_pool(DecodeReqToTokenPool.__new__(DecodeReqToTokenPool)),
        lambda: _init_decode_pool(
            HybridMambaDecodeReqToTokenPool.__new__(HybridMambaDecodeReqToTokenPool)
        ),
    ],
    ids=["ReqToTokenPool", "DecodeReqToTokenPool", "HybridMambaDecodeReqToTokenPool"],
)
def test_request_release_frees_slot_and_penalty_row(make_pool):
    """The request's release frees its slot and its penalty row together on
    every pool; the hybrid decode pool borrows the decode __init__ but
    inherits free."""
    pool = make_pool()
    store = PenaltyStore(
        vocab_size=8, num_slots=pool.req_to_token.shape[0], device="cpu"
    )
    req = SimpleNamespace(
        kv=ReqKvInfo(),
        output_ids=[3],
        sampling_params=SamplingParams(frequency_penalty=0.5),
    )
    req.kv.req_pool_idx = slot = pool.free_slots.pop()
    store.alloc(req)
    row = req.kv.penalty_row.index

    free_req_slot(pool, req)

    assert req.kv.penalty_row is None
    assert store.free_rows == [row]
    assert slot in pool.free_slots


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
