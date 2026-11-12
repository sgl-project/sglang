import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash_ring_locs import dflash_ring_block_locs
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    draft_kv_ring_rows_per_req,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

BLOCK = 8


@pytest.mark.parametrize("dcp", [1, 8])
def test_block_locs_match_ring_locs(dcp):
    ring = DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=1023, block_size=BLOCK, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_to_token_rows=9,
        device="cuda",
    )
    req_pool_indices = torch.tensor([3, 7, 1, 8], dtype=torch.int32, device="cuda")
    prefix = torch.tensor([5000, 700, 3, 0], dtype=torch.int64, device="cuda")
    positions = (prefix.unsqueeze(1) + torch.arange(BLOCK, device="cuda")).reshape(-1)
    expected = ring.locs(req_pool_indices.repeat_interleave(BLOCK), positions)
    got = dflash_ring_block_locs(
        req_pool_indices,
        positions,
        block_size=BLOCK,
        rows_per_req=ring.rows_per_req,
        dcp_size=dcp,
    )
    assert torch.equal(got, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
