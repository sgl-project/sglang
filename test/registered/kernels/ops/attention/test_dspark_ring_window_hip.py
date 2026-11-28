import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dspark_ring_window_hip import (
    fill_dspark_ring_window_buffer_hip,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm-only kernel")

POOL = 16


def _reference(min_pos, rows_per_req, req_pool_indices, seq_lens, window):
    """Per request: keys [start, len) with start = max(min_pos[slot], len - window, 0),
    each read from ring row 1 + slot * rows_per_req + pos % rows_per_req."""
    indptr, indices, kv_lens, starts = [0], [], [], []
    for slot, length in zip(req_pool_indices.tolist(), seq_lens.tolist()):
        start = max(int(min_pos[slot]), length - window, 0)
        kv_len = max(length - start, 0)
        indices += [
            1 + slot * rows_per_req + pos % rows_per_req
            for pos in range(start, start + kv_len)
        ]
        indptr.append(indptr[-1] + kv_len)
        kv_lens.append(kv_len)
        starts.append(start)
    return indptr, indices, kv_lens, starts


@pytest.mark.parametrize("window", [64, 600])
@pytest.mark.parametrize("rows_per_req", [0, 37])  # 0: exactly the window
def test_ring_window_indices_match_reference(window: int, rows_per_req: int) -> None:
    rows_per_req = window + rows_per_req
    # Requests: shorter than the window, wrapping the ring seam, floored by min_pos,
    # floored past their length (empty window), and long enough to need every tile.
    seq_lens = torch.tensor(
        [window // 2, window + 5, 3 * window, 10, 5 * window + 3], dtype=torch.int64
    )
    req_pool_indices = torch.tensor([3, 0, 7, 12, 15], dtype=torch.int64)
    min_pos = torch.zeros(POOL, dtype=torch.int32)
    min_pos[7] = 3 * window - window // 3
    min_pos[12] = 50
    bs = seq_lens.numel()
    indptr_buf = torch.full((bs + 4,), -7, dtype=torch.int32, device="cuda")

    indptr, indices, kv_lens, starts = fill_dspark_ring_window_buffer_hip(
        min_pos=min_pos.cuda(),
        rows_per_req=rows_per_req,
        dcp_size=1,
        window_kv_indptr=indptr_buf,
        req_pool_indices=req_pool_indices.cuda(),
        seq_lens=seq_lens.cuda(),
        bs=bs,
        sliding_window_size=window,
    )
    ref_indptr, ref_indices, ref_kv_lens, ref_starts = _reference(
        min_pos, rows_per_req, req_pool_indices, seq_lens, window
    )
    assert indptr.cpu().tolist() == ref_indptr
    assert indices.cpu().tolist() == ref_indices
    assert kv_lens.cpu().tolist() == ref_kv_lens
    assert starts.cpu().tolist() == ref_starts


def test_preallocated_graph_buffer_is_filled_in_place() -> None:
    window, rows_per_req = 64, 64
    seq_lens = torch.tensor([10, 200], dtype=torch.int64)
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    min_pos = torch.zeros(POOL, dtype=torch.int32)
    buf = torch.full((2 * window + 7,), -1, dtype=torch.int64, device="cuda")
    indptr, indices, _, _ = fill_dspark_ring_window_buffer_hip(
        min_pos=min_pos.cuda(),
        rows_per_req=rows_per_req,
        dcp_size=1,
        window_kv_indptr=torch.zeros(4, dtype=torch.int32, device="cuda"),
        req_pool_indices=req_pool_indices.cuda(),
        seq_lens=seq_lens.cuda(),
        bs=2,
        sliding_window_size=window,
        window_kv_indices=buf,
    )
    ref_indptr, ref_indices, _, _ = _reference(
        min_pos, rows_per_req, req_pool_indices, seq_lens, window
    )
    assert indices.data_ptr() == buf.data_ptr()
    assert indptr.cpu().tolist() == ref_indptr
    assert buf[: ref_indptr[-1]].cpu().tolist() == ref_indices
    assert (buf[ref_indptr[-1] :] == -1).all()


def test_empty_batch_writes_only_indptr() -> None:
    indptr, indices, kv_lens, _ = fill_dspark_ring_window_buffer_hip(
        min_pos=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
        rows_per_req=64,
        dcp_size=1,
        window_kv_indptr=torch.full((3,), 9, dtype=torch.int32, device="cuda"),
        req_pool_indices=torch.zeros(2, dtype=torch.int64, device="cuda"),
        seq_lens=torch.ones(2, dtype=torch.int64, device="cuda"),
        bs=0,
        sliding_window_size=64,
    )
    assert indptr.cpu().tolist() == [0]
    assert indices.numel() == 0 and kv_lens.numel() == 0


@pytest.mark.parametrize("window,rows_per_req", [(0, 64), (64, 63)])
def test_window_longer_than_the_ring_is_rejected(window, rows_per_req) -> None:
    """A ring shorter than the window would map distinct live keys to one row."""
    with pytest.raises(ValueError, match="at least"):
        fill_dspark_ring_window_buffer_hip(
            min_pos=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
            rows_per_req=rows_per_req,
            dcp_size=1,
            window_kv_indptr=torch.zeros(2, dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
            seq_lens=torch.ones(1, dtype=torch.int64, device="cuda"),
            bs=1,
            sliding_window_size=window,
        )


def test_short_graph_buffer_is_rejected() -> None:
    """The kernel stores without a bound check, so a buffer shorter than
    bs x window must be refused before launch."""
    with pytest.raises(ValueError, match="fewer than"):
        fill_dspark_ring_window_buffer_hip(
            min_pos=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
            rows_per_req=64,
            dcp_size=1,
            window_kv_indptr=torch.zeros(3, dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.zeros(2, dtype=torch.int64, device="cuda"),
            seq_lens=torch.ones(2, dtype=torch.int64, device="cuda"),
            bs=2,
            sliding_window_size=64,
            window_kv_indices=torch.empty(127, dtype=torch.int64, device="cuda"),
        )


def test_non_contiguous_graph_buffer_is_rejected() -> None:
    """numel() counts an expanded buffer's logical elements, not its writable slots."""
    expanded = torch.empty(1, dtype=torch.int64, device="cuda").expand(256)
    with pytest.raises(ValueError, match="contiguous 1-D int64"):
        fill_dspark_ring_window_buffer_hip(
            min_pos=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
            rows_per_req=64,
            dcp_size=1,
            window_kv_indptr=torch.zeros(3, dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.zeros(2, dtype=torch.int64, device="cuda"),
            seq_lens=torch.ones(2, dtype=torch.int64, device="cuda"),
            bs=2,
            sliding_window_size=64,
            window_kv_indices=expanded,
        )


def test_window_reads_the_rows_the_ring_wrote() -> None:
    """Reader and writer must agree across the ring seam and a partially accepted
    verify block: every key the window reads is the row the ring wrote for it."""
    from sglang.srt.speculative.dflash_draft_ring import (
        DFlashDraftKVRing,
        draft_kv_ring_rows_per_req,
    )

    window, block = 64, 6
    rows = draft_kv_ring_rows_per_req(window_left=window, block_size=block, dcp_size=1)
    ring = DFlashDraftKVRing(
        rows_per_req=rows,
        dcp_size=1,
        req_to_token_rows=POOL,
        device=torch.device("cuda"),
        window_left=window,
        dcp_rank=0,
        pool=SimpleNamespace(maybe_get_custom_mem_pool=lambda: None),
    )
    slots = torch.tensor([3, 9], dtype=torch.int64, device="cuda")
    # Request 3 has wrapped the ring twice; request 9 has not wrapped yet.
    seq_lens = torch.tensor([2 * rows + 17, 40], dtype=torch.int64, device="cuda")
    # A verify block written past each length, of which the first 2 tokens were accepted.
    block_pos = (seq_lens[:, None] + torch.arange(block, device="cuda")).reshape(-1)
    block_locs = ring.block_locs(slots, block_pos, block_size=block).view(2, block)
    seq_lens = seq_lens + 2

    indptr, indices, _, starts = fill_dspark_ring_window_buffer_hip(
        min_pos=ring.min_pos,
        rows_per_req=ring.rows_per_req,
        dcp_size=1,
        window_kv_indptr=torch.zeros(3, dtype=torch.int32, device="cuda"),
        req_pool_indices=slots,
        seq_lens=seq_lens,
        bs=2,
        sliding_window_size=window,
    )
    for b in range(2):
        read = indices[indptr[b] : indptr[b + 1]]
        positions = torch.arange(int(starts[b]), int(seq_lens[b]), device="cuda")
        written = ring.locs(slots[b].repeat(positions.numel()), positions)
        assert torch.equal(read, written)
        assert read.unique().numel() == read.numel()
        # The two accepted block tokens are read where the block wrote them.
        assert torch.equal(read[-2:], block_locs[b, :2])


def test_dcp_above_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="dcp_size == 1"):
        fill_dspark_ring_window_buffer_hip(
            min_pos=torch.zeros(POOL, dtype=torch.int32, device="cuda"),
            rows_per_req=64,
            dcp_size=2,
            window_kv_indptr=torch.zeros(2, dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
            seq_lens=torch.ones(1, dtype=torch.int64, device="cuda"),
            bs=1,
            sliding_window_size=64,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
