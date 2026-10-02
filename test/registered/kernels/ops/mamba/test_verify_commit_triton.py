import sys

import pytest
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_commit_track_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(accept_index, accept_lens, seq_lens, draft_token_num, track_interval):
    """Mirrors the eager branch of spec_utils._verify_commit_step_indices."""
    bs = accept_lens.shape[0]
    offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
    last = accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - offset
    if track_interval <= 0:
        return last, None
    pre = seq_lens
    post = seq_lens + accept_lens
    mask = pre // track_interval != post // track_interval
    point = post // track_interval * track_interval
    ith = torch.clamp(point - pre - 1, min=0).to(torch.int64)
    cand = accept_index[req_idx, ith] - offset
    track = torch.where(mask, cand, torch.full_like(cand, -1))
    return last, track


@pytest.mark.parametrize("bs", [1, 3, 48, 257])
@pytest.mark.parametrize("track_interval", [0, 64])
@pytest.mark.parametrize("tree_depth", [4, 3])
def test_verify_commit_steps_matches_eager(bs, track_interval, tree_depth):
    """The fused kernel must match eager on both outputs near tracking boundaries
    and when accept_index rows (max_tree_depth) are narrower than draft_token_num."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(bs + track_interval + tree_depth)
    device = "cuda"
    draft_token_num = 4
    accept_lens = torch.randint(
        1, tree_depth + 1, (bs,), device=device, dtype=torch.int32
    )
    tree_nodes = torch.argsort(torch.rand(bs, draft_token_num, device=device), dim=1)[
        :, :tree_depth
    ]
    accept_index = (
        torch.arange(bs, device=device, dtype=torch.int64).unsqueeze(1)
        * draft_token_num
        + tree_nodes
    ).to(torch.int32)
    # Cluster seq lens around tracking boundaries to exercise the crossing.
    seq_lens = torch.randint(60, 70, (bs,), device=device, dtype=torch.int64)

    exp_last, exp_track = _reference(
        accept_index, accept_lens, seq_lens, draft_token_num, track_interval
    )
    got_last, got_track = fused_commit_track_indices(
        accept_index,
        accept_lens,
        seq_lens if track_interval > 0 else None,
        draft_token_num,
        track_interval,
    )
    assert torch.equal(got_last, exp_last)
    if track_interval > 0:
        assert torch.equal(got_track, exp_track)
    else:
        assert got_track is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
