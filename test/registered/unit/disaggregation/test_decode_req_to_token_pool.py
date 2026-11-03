import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


def _decode_mamba_spec_state_size(*, size, pre_alloc_size, replayssm_spec):
    captured = {}

    def capture(self, *, mamba_spec_state_size, **kwargs):
        captured["rows"] = mamba_spec_state_size

    with patch.object(HybridMambaDecodeReqToTokenPool, "_init_mamba_pool", capture):
        HybridMambaDecodeReqToTokenPool(
            size=size,
            max_context_len=4,
            device="cpu",
            enable_memory_saver=False,
            cache_params=None,
            mamba_layer_ids=[0],
            speculative_num_draft_tokens=6,
            enable_mamba_extra_buffer=False,
            pre_alloc_size=pre_alloc_size,
            enable_overlap_schedule=True,
            mamba_size=size + pre_alloc_size,
            enable_linear_replayssm_spec=replayssm_spec,
        )
    return captured["rows"]


@pytest.mark.parametrize(
    "replayssm_spec, pp_spec_stable_rows, rows",
    [(False, False, 8), (True, False, 8 + 16), (False, True, 8 + 16)],
    ids=["batch_rows", "replayssm_spec_ring", "pp_spec_stable_rows"],
)
def test_decode_mamba_spec_scratch_rows(replayssm_spec, pp_spec_stable_rows, rows):
    """In-transfer requests never verify, so the spec scratch skips the pre-allocated
    slots unless verify indexes it by req_pool_idx (ReplaySSM-spec, PP-spec rows)."""
    with envs.SGLANG_ENABLE_PP_SPEC.override(pp_spec_stable_rows):
        spec_rows = _decode_mamba_spec_state_size(
            size=8, pre_alloc_size=16, replayssm_spec=replayssm_spec
        )
    assert spec_rows == rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
