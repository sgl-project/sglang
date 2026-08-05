"""Fused verify-prep kernel == cache-locs kernel + index_select/gather pair.

eagle_prepare_for_verify used three launches per step: the uniform cache-locs
assign, then mapping[req_pool_indices] and a gather for the mamba verify
track indices. The fused kernel resolves the two-level track lookup inside
the cache-locs program. Pins bitwise equality of both outputs against the
unfused reference across ping-pong widths, batch sizes, draft widths, and
shuffled req_pool orderings.
"""

import pytest
import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs_uniform_func,
    assign_extend_cache_locs_uniform_with_track_func,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"
POOL_REQS, POOL_LEN = 64, 512


class TestFusedVerifyPrep(CustomTestCase):
    def _case(self, bs, draft, pp_size, seed):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        req_to_token = torch.randint(
            0, 1 << 20, (POOL_REQS, POOL_LEN), device=DEVICE, dtype=torch.int64
        )
        req_pool_indices = torch.randperm(
            POOL_REQS, device=DEVICE, generator=gen, dtype=torch.int64
        )[:bs]
        seq_lens = torch.randint(
            1, POOL_LEN - draft, (bs,), device=DEVICE, dtype=torch.int64, generator=gen
        )
        mapping = torch.randint(
            0, 1 << 16, (POOL_REQS, pp_size), device=DEVICE, dtype=torch.int64
        )
        track_positions = torch.randint(
            0, pp_size, (bs,), device=DEVICE, dtype=torch.int64, generator=gen
        )

        got_locs, got_track = assign_extend_cache_locs_uniform_with_track_func(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            start_offset=seq_lens,
            batch_size=bs,
            draft_token_num=draft,
            track_positions=track_positions,
            track_buffer_mapping=mapping,
            device=DEVICE,
        )
        ref_locs = assign_extend_cache_locs_uniform_func(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            start_offset=seq_lens,
            batch_size=bs,
            draft_token_num=draft,
            device=DEVICE,
        )
        ref_track = torch.gather(
            mapping[req_pool_indices], 1, track_positions.unsqueeze(1)
        ).squeeze(1)
        self.assertTrue(torch.equal(got_locs, ref_locs), f"{bs=} {draft=} {pp_size=}")
        self.assertTrue(torch.equal(got_track, ref_track), f"{bs=} {draft=} {pp_size=}")

    def test_matches_unfused_reference(self):
        for bs in (1, 3, 5):
            for pp_size in (1, 2, 4):
                self._case(bs, 4, pp_size, seed=bs * 10 + pp_size)
        self._case(2, 6, 2, seed=99)
        self._case(1, 128, 2, seed=100)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
