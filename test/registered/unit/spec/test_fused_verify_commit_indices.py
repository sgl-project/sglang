"""fused_verify_commit_indices == the python index math it replaces.

One Triton launch replaces ~13 small ops (mamba slot gather + last accepted
step + track interval-crossing step). Pins bitwise equality against the
original formulas across accept dtypes, a non-contiguous accept_index, and
seq_lens sitting on/off track boundaries, plus the int32 output dtype the
downstream scatters expect.
"""

import pytest
import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    fused_verify_commit_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
TRACK = 8


def _reference(accept_index, accept_lens, seq_lens, req_pool_indices, mamba_map, draft):
    bs = accept_lens.shape[0]
    offsets = torch.arange(
        0, bs * draft, step=draft, dtype=accept_lens.dtype, device=DEVICE
    )
    rows = torch.arange(bs, dtype=torch.int64, device=DEVICE)
    last = accept_index[rows, (accept_lens - 1).to(torch.int64)] - offsets
    pre, post = seq_lens, seq_lens + accept_lens
    crossed = pre // TRACK != post // TRACK
    point = post // TRACK * TRACK
    ith = torch.clamp(point - pre - 1, min=0).to(torch.int64)
    cand = accept_index[rows, ith] - offsets
    steps = torch.where(crossed, cand, torch.full_like(cand, -1))
    return mamba_map[req_pool_indices], last, steps


class TestFusedVerifyCommitIndices(CustomTestCase):
    def _case(self, bs, draft, idx_dtype, noncontig, map_dtype=torch.int32):
        gen = torch.Generator(device=DEVICE).manual_seed(bs * 100 + draft)
        accept_lens = torch.randint(
            1, draft + 1, (bs,), device=DEVICE, dtype=torch.int32, generator=gen
        )
        base = torch.arange(bs, device=DEVICE).unsqueeze(1) * draft
        accept_index = (base + torch.arange(draft, device=DEVICE)).to(idx_dtype)
        if noncontig:
            wide = torch.full((bs, draft + 2), -7, device=DEVICE, dtype=idx_dtype)
            wide[:, :draft] = accept_index
            accept_index = wide[:, :draft]
        seq_lens = torch.cat(
            [
                torch.tensor([TRACK - 1, TRACK, 1], device=DEVICE),
                torch.randint(1, 50, (bs,), device=DEVICE, generator=gen),
            ]
        )[:bs].to(torch.int64)
        req_pool_indices = torch.randperm(64, device=DEVICE)[:bs]
        mamba_map = torch.randperm(64, device=DEVICE).to(map_dtype)

        got = fused_verify_commit_indices(
            accept_index=accept_index,
            accept_lens=accept_lens,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            mamba_map=mamba_map,
            draft_token_num=draft,
            track_interval=TRACK,
        )
        ref = _reference(
            accept_index.to(torch.int64),
            accept_lens,
            seq_lens,
            req_pool_indices,
            mamba_map,
            draft,
        )
        for g, r, name in zip(got, ref, ("slots", "last_correct", "track_steps")):
            where = (
                f"{name} bs={bs} draft={draft} {idx_dtype=} {noncontig=} {map_dtype=}"
            )
            self.assertEqual(g.dtype, torch.int32, where)
            self.assertTrue(torch.equal(g.to(torch.int64), r.to(torch.int64)), where)

    def test_matches_reference(self):
        for bs in (1, 3, 7):
            for idx_dtype in (torch.int64, torch.int32):
                for noncontig in (False, True):
                    self._case(bs, 4, idx_dtype, noncontig)
        self._case(5, 6, torch.int64, False, map_dtype=torch.int64)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
