import sys

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
@pytest.mark.parametrize("rows_per_req", [0, 37])
def test_ring_window_indices_match_reference(window: int, rows_per_req: int) -> None:
    rows_per_req = rows_per_req or window
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
