"""GDN ReplaySSM cursors advance alike whether they are their own arrays or sit
one per unit inside larger entries, as unified memory lays them out."""

import unittest

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_ROWS = 16
_ENTRY_BYTES = 64
_CACHE_LEN = 16
_SPEC_LEN = 3


def _entry_cursors(raw: torch.Tensor):
    """The three cursors at byte offsets 0, 4 and 8 of every entry of ``raw``."""
    ints = raw.view(torch.int32)
    write_pos = torch.as_strided(ints, (_ROWS,), (_ENTRY_BYTES // 4,), 0)
    cache_base = torch.as_strided(ints, (_ROWS,), (_ENTRY_BYTES // 4,), 1)
    is_flush = torch.as_strided(raw.view(torch.int8), (_ROWS,), (_ENTRY_BYTES,), 8)
    return write_pos, cache_base, is_flush


class TestCursorStride(CustomTestCase):
    def test_entry_cursors_advance_like_contiguous_ones(self):
        torch.manual_seed(0)
        device = "cuda"
        write_pos = torch.randint(0, 6, (_ROWS,), dtype=torch.int32, device=device)
        cache_base = torch.randint(
            0, _CACHE_LEN, (_ROWS,), dtype=torch.int32, device=device
        )
        is_flush = torch.randint(0, 2, (_ROWS,), dtype=torch.int8, device=device)
        raw = torch.full(
            (_ROWS * _ENTRY_BYTES,), 0x5A, dtype=torch.uint8, device=device
        )
        entry = _entry_cursors(raw)
        for dst, src in zip(entry, (write_pos, cache_base, is_flush)):
            dst.copy_(src)
        untouched = raw.clone()
        rows = torch.tensor([3, 0, 11, 7, 15], dtype=torch.int64, device=device)
        accepted = torch.randint(
            1, _SPEC_LEN + 2, (rows.numel(),), dtype=torch.int32, device=device
        )
        for cursors in ((write_pos, cache_base, is_flush), entry):
            commit_gdn_replayssm_spec(
                *cursors,
                num_accepted=accepted,
                replay_indices=rows,
                # Every row has a live state slot, so every cursor advances.
                state_batch_indices=torch.arange(
                    rows.numel(), dtype=torch.int64, device=device
                ),
                max_cache_len=_CACHE_LEN,
                max_spec_len=_SPEC_LEN,
                null_block_id=-1,
            )
        for got, want in zip(entry, (write_pos, cache_base, is_flush)):
            torch.testing.assert_close(got, want)
        # Bytes beyond the cursors stay as they were.
        mask = torch.ones_like(raw, dtype=torch.bool).view(_ROWS, _ENTRY_BYTES)
        mask[:, :9] = False
        self.assertTrue(
            torch.equal(
                raw.view(_ROWS, _ENTRY_BYTES)[mask],
                untouched.view(_ROWS, _ENTRY_BYTES)[mask],
            )
        )


if __name__ == "__main__":
    unittest.main()
