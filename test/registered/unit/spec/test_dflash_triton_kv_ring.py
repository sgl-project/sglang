"""Triton's DFLASH draft-window reads use the shared KV ring semantics.

The regression this guards was a target-scale ``req_to_token`` id being used
against a much smaller ring-backed draft pool. These CPU cases require no GPU:
they pin CSR lengths, per-request isolation, wrapping, and the cached-prefix
materialization floor without relying on a live engine.
"""

from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
    draft_kv_ring_unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _ring(*, window_left: int, block_size: int, dcp_size: int, req_pool_rows: int = 9):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=window_left, block_size=block_size, dcp_size=dcp_size
        ),
        dcp_size=dcp_size,
        req_pool_rows=req_pool_rows,
        device=torch.device("cpu"),
        window_left=window_left,
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 8])
def test_triton_window_uses_ring_rows_and_materialization_floor(dcp_size):
    window_left = 4
    block_size = 2
    ring = _ring(window_left=window_left, block_size=block_size, dcp_size=dcp_size)
    req_pool_indices = torch.tensor([3, 7, 1], dtype=torch.int32)
    seq_lens = torch.tensor([19, 6, 2], dtype=torch.int64)
    # These floors exceed the ordinary last-five-token starts for requests 3/7.
    ring.min_pos[3] = 17
    ring.min_pos[7] = 4
    expected_starts = [17, 4, 0]
    expected_positions = torch.tensor([17, 18, 4, 5, 0, 1])
    indptr = torch.zeros(4, dtype=torch.int64)
    indices = torch.full((6,), -1, dtype=torch.int64)

    with mock.patch(
        "sglang.srt.speculative.dflash_draft_ring.get_draft_kv_ring",
        return_value=ring,
    ):
        got_indptr, got_indices, got_lens, got_starts = update_sliding_window_buffer(
            window_kv_indptr=indptr,
            runner_ids=None,  # Deliberately invalid outside the ring branch.
            plan=None,
            req_pool_indices=req_pool_indices,
            sliding_window_size=window_left,
            seq_lens=seq_lens,
            bs=3,
            window_kv_indices=indices,
        )

    assert got_indptr.tolist() == [0, 2, 4, 6]
    assert got_lens.tolist() == [2, 2, 2]
    assert got_starts.tolist() == expected_starts
    expanded_reqs = req_pool_indices.repeat_interleave(2)
    assert got_indices.tolist() == ring.locs(expanded_reqs, expected_positions).tolist()
    assert len(set(got_indices.tolist())) == got_indices.numel()
    pool_rows = draft_kv_ring_pool_rows(
        req_pool_rows=9, rows_per_req=ring.rows_per_req, page_size=64
    )
    # Ring ids are global DCP ids (owner = loc % dcp); bound their rank-local rows.
    assert all(0 < int(loc) < pool_rows for loc in got_indices // dcp_size)


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_triton_window_wrap_matches_ring_locs(dcp_size):
    window_left = 1023
    block_size = 8
    ring = _ring(window_left=window_left, block_size=block_size, dcp_size=dcp_size)
    req_pool_indices = torch.tensor([2], dtype=torch.int32)
    seq_lens = torch.tensor([4 * ring.span + 7], dtype=torch.int64)
    starts = seq_lens - window_left
    positions = torch.arange(int(starts[0]), int(seq_lens[0]))
    indptr = torch.zeros(2, dtype=torch.int64)
    indices = torch.empty(window_left, dtype=torch.int64)

    with mock.patch(
        "sglang.srt.speculative.dflash_draft_ring.get_draft_kv_ring",
        return_value=ring,
    ):
        update_sliding_window_buffer(
            indptr,
            None,
            None,
            req_pool_indices,
            window_left,
            seq_lens,
            1,
            window_kv_indices=indices,
        )

    expected = ring.locs(req_pool_indices, positions)
    assert indices.tolist() == expected.tolist()
    assert len(set(indices.tolist())) == window_left
    # No target-scale absolute position survives as a physical ring id.
    assert int((indices // dcp_size).max()) < draft_kv_ring_pool_rows(
        req_pool_rows=9, rows_per_req=ring.rows_per_req, page_size=64
    )


def test_triton_is_an_allowed_ring_backend():
    with (
        mock.patch(
            "sglang.srt.speculative.dflash_utils.is_dflash_mla_draft_config",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.speculative.dflash_utils.get_dflash_layer_types",
            return_value=["sliding_attention"] * 5,
        ),
    ):
        assert (
            draft_kv_ring_unsupported_reason(
                object(), draft_backend="triton", draft_window_size=None
            )
            is None
        )
