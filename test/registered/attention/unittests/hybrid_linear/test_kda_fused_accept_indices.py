"""Padding-safety invariants of the KDA fused-accept verify index builder.

``build_fused_accept_indices`` produces the slot-indexed ``[N, T]``
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
    build_fused_accept_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")


class TestBuildFusedAcceptIndices(CustomTestCase):
    def test_real_slots_address_their_scratch_rows(self):
        for scratch_steps, draft_token_num in ((4, 4), (8, 4), (33, 33)):
            slots = torch.tensor([0, 3, 7], dtype=torch.int32)
            pool = torch.arange(10, dtype=torch.int32) + 1  # accept len = slot + 1
            indices, nat = build_fused_accept_indices(
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
            indices, nat = build_fused_accept_indices(
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
        _, nat = build_fused_accept_indices(
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


class TestFusedAcceptPerForwardCache(CustomTestCase):
    """The verify indices are built once per forward and shared by every KDA
    layer. That sharing is only sound while the cache dies with the forward: a
    cache that outlived it would seed the next batch from the previous batch's
    mamba slots, which is a silent wrong-state bug (no shape or index error).
    """

    @staticmethod
    def _build(slots, pool_values, draft_token_num=4, scratch_steps=4):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return build_fused_accept_indices(
            slots=torch.tensor(slots, dtype=torch.int32, device=device),
            scratch_steps=scratch_steps,
            draft_token_num=draft_token_num,
            accept_lens_pool=torch.tensor(
                pool_values, dtype=torch.int32, device=device
            ),
        )

    def test_shared_build_matches_a_per_layer_build(self):
        """What every layer reuses must equal what it would have built itself."""
        pool = [1] * 8
        pool[3], pool[5] = 2, 4
        first_idx, first_nat = self._build([3, 5], pool)
        second_idx, second_nat = self._build([3, 5], pool)
        self.assertTrue(torch.equal(first_idx, second_idx))
        self.assertTrue(torch.equal(first_nat, second_nat))

    def test_a_different_batch_builds_different_rows(self):
        """Guards the staleness mode: reusing a previous forward's tensor would
        address the previous forward's slots, and the values must differ so the
        cache-reset is observable rather than accidentally correct."""
        pool = [1] * 8
        pool[3], pool[5], pool[6] = 2, 4, 3
        idx_a, nat_a = self._build([3, 5], pool)
        idx_b, nat_b = self._build([6, 5], pool)
        self.assertFalse(torch.equal(idx_a, idx_b))
        self.assertFalse(torch.equal(nat_a, nat_b))

    def test_metadata_starts_uncached(self):
        """A forward's metadata must arrive with no indices carried over: the
        backend keys 'build once' on these being None."""
        from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

        metadata = ForwardMetadata(
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            mamba_cache_indices=torch.zeros(1, dtype=torch.int32),
        )
        self.assertIsNone(metadata.fused_accept_state_indices)
        self.assertIsNone(metadata.fused_accept_num_accepted)


if __name__ == "__main__":
    unittest.main()
