"""Padding-safety invariants of the KDA fused-accept verify index builder.

``_build_fused_accept_indices`` produces the slot-indexed ``[N, T]``
``ssm_state_indices`` and the per-row ``num_accepted_tokens`` gather that
flashinfer ``recurrent_kda`` consumes in fused-accept mode. The kernel's
padding contract is: a row is inactive iff its raw slot index is negative.
Padded sglang rows carry mamba slot ``-1``, so EVERY derived index
``-1 * scratch_steps + step`` must stay negative for all
``step < scratch_steps`` — an arithmetic reorder (e.g. adding the step before
the multiply) would silently activate padded rows and corrupt neighbor state.
The nat gather must clamp padded slots in-bounds (their value is never
consumed) and keep real slots' accept lengths intact.

CPU tensors only — the invariants are pure index arithmetic.
"""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
    _build_fused_accept_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")


class TestBuildFusedAcceptIndices(CustomTestCase):
    def test_real_slots_address_their_scratch_rows(self):
        for scratch_steps, draft_token_num in ((4, 4), (8, 4), (33, 33)):
            slots = torch.tensor([0, 3, 7], dtype=torch.int32)
            pool = torch.arange(10, dtype=torch.int32) + 1  # accept len = slot + 1
            indices, nat = _build_fused_accept_indices(
                slots=slots,
                scratch_steps=scratch_steps,
                draft_token_num=draft_token_num,
                accept_lens_pool=pool,
            )
            self.assertEqual(indices.shape, (3, draft_token_num))
            self.assertEqual(indices.dtype, torch.int32)
            step = torch.arange(draft_token_num, dtype=torch.int32)
            expected = slots[:, None] * scratch_steps + step[None, :]
            self.assertTrue(torch.equal(indices, expected))
            self.assertEqual(nat.dtype, torch.int32)
            self.assertTrue(torch.equal(nat, slots + 1))

    def test_padded_slot_rows_stay_fully_negative(self):
        # T == scratch_steps is the tight case: the largest step must still
        # land below zero for slot -1.
        for scratch_steps, draft_token_num in ((4, 4), (8, 8), (8, 4), (33, 33)):
            slots = torch.tensor([2, -1, 5, -1], dtype=torch.int32)
            pool = torch.full((8,), 3, dtype=torch.int32)
            indices, nat = _build_fused_accept_indices(
                slots=slots,
                scratch_steps=scratch_steps,
                draft_token_num=draft_token_num,
                accept_lens_pool=pool,
            )
            padded_rows = indices[slots < 0]
            self.assertTrue(
                (padded_rows < 0).all(),
                f"padded row leaked a non-negative index "
                f"({scratch_steps=}, {draft_token_num=}): {padded_rows.tolist()}",
            )
            real_rows = indices[slots >= 0]
            self.assertTrue((real_rows >= 0).all())
            # nat gather clamps padded slots in-bounds (value unused).
            self.assertEqual(nat.shape[0], 4)

    def test_nat_gather_reads_pool_values(self):
        slots = torch.tensor([1, 4, -1], dtype=torch.int32)
        pool = torch.tensor([9, 2, 9, 9, 5, 9], dtype=torch.int32)
        _, nat = _build_fused_accept_indices(
            slots=slots,
            scratch_steps=4,
            draft_token_num=4,
            accept_lens_pool=pool,
        )
        self.assertEqual(nat[0].item(), 2)
        self.assertEqual(nat[1].item(), 5)
        # Padded row clamps to pool row 0; the value is never consumed but the
        # gather itself must stay in-bounds.
        self.assertEqual(nat[2].item(), 9)


if __name__ == "__main__":
    unittest.main()
