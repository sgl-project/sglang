"""DFLASH draft KV ring: the rows the worker writes (DFlashDraftKVRing.locs, kept
under the pool's DCP owner rule) are exactly the rows the window kernel reads in
ring mode, the window floor holds, and an extend keeps only the tail a later
window can read."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.mla_window_indices import mla_window_sparse_indices
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
    draft_kv_ring_unsupported_reason,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

WINDOW_LEFT = 1023
BLOCK = 8


def _ring(dcp, req_pool_rows, device, dcp_rank=0):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=WINDOW_LEFT, block_size=BLOCK, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_pool_rows=req_pool_rows,
        device=device,
        window_left=WINDOW_LEFT,
        dcp_rank=dcp_rank,
    )


@pytest.mark.parametrize("dcp", [1, 8])
def test_window_reads_the_rows_the_ring_writes(dcp):
    device = "cuda"
    req_pool_rows = 9
    ring = _ring(dcp, req_pool_rows, device)
    pool_rows = draft_kv_ring_pool_rows(
        req_pool_rows=req_pool_rows, rows_per_req=ring.rows_per_req, page_size=64
    )
    req_pool_indices = torch.tensor([3, 7, 1], dtype=torch.int32, device=device)
    prefix = torch.tensor([5000, 700, 3], dtype=torch.int64, device=device)
    ring.min_pos[1] = 2  # request 2: a 2-token prefix hit
    ring.min_pos[3] = 4100
    positions = (prefix.unsqueeze(1) + torch.arange(BLOCK, device=device)).reshape(-1)
    for rank in range(dcp):
        indices, counts = mla_window_sparse_indices(
            req_to_token=torch.empty((1, 1), dtype=torch.int32, device=device),
            req_pool_indices=req_pool_indices,
            positions=positions,
            window_left=WINDOW_LEFT,
            dcp_size=dcp,
            dcp_rank=rank,
            page_size=64,
            ring_rows_per_req=ring.rows_per_req,
            min_pos=ring.min_pos,
        )
        for row, p in enumerate(positions.tolist()):
            req = int(req_pool_indices[row // BLOCK])
            lo = max(p - WINDOW_LEFT, 0, int(ring.min_pos[req]))
            keys = torch.arange(lo, p + 1, device=device)
            locs = ring.locs(torch.full_like(keys, req), keys)
            mine = locs % dcp == rank
            expected = (locs[mine] // dcp).tolist()
            got = indices[row, : int(counts[row])].tolist()
            assert got == expected, (rank, row)
            assert all(0 < r < pool_rows for r in got)


@pytest.mark.parametrize("dcp", [1, 8])
def test_window_and_block_never_share_a_row(dcp):
    ring = _ring(dcp, 4, "cpu")
    p = 123457
    # The block's rows and every row a query in that block can read.
    keys = torch.arange(p - WINDOW_LEFT, p + BLOCK)
    locs = ring.locs(torch.full_like(keys, 2), keys)
    assert locs.unique().numel() == keys.numel()
    assert ring.span >= WINDOW_LEFT + 1 + BLOCK


def test_extend_keeps_only_the_readable_tail():
    ring = _ring(8, 4, "cpu")
    lens = torch.tensor([3000, 40])
    prefix = torch.tensor([0, 100])
    positions = torch.cat(
        [torch.arange(int(a), int(a + n)) for a, n in zip(prefix, lens)]
    )
    locs = ring.extend_locs(
        req_pool_indices=torch.tensor([1, 2]),
        positions=positions,
        extend_lens=lens,
        seq_ends=prefix + lens,
    )
    first, second = locs[:3000], locs[3000:]
    kept = first != 0
    assert int(kept.sum()) == ring.span
    assert bool(kept[-ring.span :].all())
    assert first[kept].unique().numel() == ring.span
    assert bool((second != 0).all())


@pytest.mark.parametrize("seq_len", [5000, 300])
def test_pd_window_rows_pair_prefill_with_each_decode_rank(seq_len):
    """PD: a DCP-1 prefill's window rows, paired positionally with each DCP-8
    decode rank's (sink pairs dropped as the transfer does), cover every window
    position exactly once, each landing on the row that rank's window reads."""
    prefill = _ring(1, 4, "cpu")
    src = prefill.transfer_rows(2, seq_len)
    window = list(range(max(0, seq_len - WINDOW_LEFT - 1), seq_len))
    assert len(src) == len(window) and (src != 0).all()
    covered = []
    for rank in range(8):
        decode = _ring(8, 6, "cpu", dcp_rank=rank)
        dst = decode.transfer_rows(5, seq_len)
        assert len(dst) == len(src)
        for p, a, b in zip(window, src, dst):
            if a and b:
                covered.append(p)
                assert p % 8 == rank
                assert a == int(prefill.locs(torch.tensor([2]), torch.tensor([p])))
                assert b * 8 + rank == int(
                    decode.locs(torch.tensor([5]), torch.tensor([p]))
                )
    assert sorted(covered) == window


def test_decode_floor_reset_and_unsupported_drafts():
    """PD decode resets a slot's floor to its prompt end at preallocation; the
    ring is resolved off for drafts and backends it does not fit."""
    from unittest import mock

    from sglang.srt.speculative import dflash_utils

    ring = _ring(8, 4, "cpu")
    ring.min_pos[2] = 7
    ring.reset_floor(2, 900)
    assert ring.min_pos.tolist() == [0, 0, 900, 0]

    def reason(
        mla=True, layers=("sliding_attention",) * 5, backend="trtllm_mla", window=None
    ):
        with (
            mock.patch.object(
                dflash_utils, "is_dflash_mla_draft_config", return_value=mla
            ),
            mock.patch.object(
                dflash_utils, "get_dflash_layer_types", return_value=list(layers)
            ),
        ):
            return draft_kv_ring_unsupported_reason(
                object(), draft_backend=backend, draft_window_size=window
            )

    assert reason() is None
    assert reason(backend="cutedsl_mla") is None
    assert reason(mla=False) is not None
    assert reason(layers=("sliding_attention", "full_attention")) is not None
    assert reason(backend="flashinfer") is not None
    assert reason(window=4096) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
