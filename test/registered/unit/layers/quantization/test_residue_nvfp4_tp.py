"""CPU tests for the residue K-extension TP shard plan.

Pure tensor math -- no GPU, no process group. The two-range rule, the
interleave-for-loader trick, and the rank-local salient rebasing are the
things that silently corrupt outputs when wrong, so they are pinned here.
"""

import pytest
import torch

from sglang.srt.layers.quantization.residue_nvfp4.tp import (
    ResidueShardPlan,
    ResidueTPError,
    interleave_extended_for_tp,
    plan_from_partition,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

KB = 4096  # K_base
S = 1024  # num_salient (ratio 0.25)


def sorted_block_indices(k_base: int, per_block: int) -> torch.Tensor:
    return torch.tensor(
        sorted(i for b in range(0, k_base, 8) for i in range(b, b + per_block)),
        dtype=torch.int64,
    )


class TestPlanFromPartition:
    def test_column_parallel_returns_none(self):
        assert (
            plan_from_partition(
                extended_dim=KB + S,
                input_size_per_partition=KB,
                input_size=KB,
                num_salient=S,
            )
            is None
        )

    def test_row_parallel_builds_plan(self):
        plan = plan_from_partition(
            extended_dim=KB + S,
            input_size_per_partition=KB // 2,
            input_size=KB,
            num_salient=S,
            tp_rank=1,
        )
        assert plan is not None
        assert plan.tp_size == 2
        assert plan.base_shard == KB // 2
        assert plan.salient_shard == S // 2
        assert plan.k_ext_shard == (KB + S) // 2

    def test_mismatched_extension_rejected(self):
        with pytest.raises(ResidueTPError, match="does not match"):
            plan_from_partition(
                extended_dim=KB + S + 8,
                input_size_per_partition=KB // 2,
                input_size=KB,
                num_salient=S,
            )

    def test_ragged_salient_rejected(self):
        with pytest.raises(ResidueTPError, match="not divisible"):
            plan_from_partition(
                extended_dim=KB + S + 1,
                input_size_per_partition=KB // 2,
                input_size=KB,
                num_salient=S + 1,
            )


class TestValidation:
    def test_boundary_off_sf_group_rejected(self):
        # base_shard = 96 is a multiple of 8 but not of 128: the shard
        # boundary would split a residue SF group.
        plan = ResidueShardPlan(k_base=192, num_salient=48, tp_size=2, tp_rank=0)
        with pytest.raises(ResidueTPError, match="residue scale-factor group"):
            plan.validate()

    def test_salient_shard_off_swizzle_grid_rejected(self):
        # salient_shard = 32 is not a multiple of 64 (SF swizzle tile).
        plan = ResidueShardPlan(k_base=2048, num_salient=64, tp_size=2, tp_rank=0)
        with pytest.raises(ResidueTPError, match="swizzled block-scale"):
            plan.validate()


class TestGatherInterleave:
    def test_gather_covers_disjoint_ranges(self):
        plan = ResidueShardPlan(k_base=KB, num_salient=S, tp_size=2, tp_rank=1)
        plan.validate()
        full = torch.arange(KB + S)
        shard = plan.gather(full.unsqueeze(0), scale=1)
        expected = torch.cat(
            [
                full[KB // 2 : KB],  # rank 1's base half
                full[KB + S // 2 : KB + S],  # rank 1's salient half
            ]
        )
        assert torch.equal(shard[0], expected)

    @pytest.mark.parametrize("scale", [1, 2, 16])
    def test_interleave_then_contiguous_narrow_equals_gather(self, scale):
        """The whole point of the permutation: after it, the stock loader's
        contiguous narrow yields exactly the two-range gather per rank."""
        tp = 2
        width = (KB + S) // scale
        full = torch.arange(2 * width).reshape(2, width).float()
        plan0 = ResidueShardPlan(KB, S, tp, 0)
        plan0.validate()
        permuted = interleave_extended_for_tp(full, plan0, scale=scale)

        per_rank = width // tp
        for r in range(tp):
            plan_r = ResidueShardPlan(KB, S, tp, r)
            contiguous = permuted.narrow(-1, r * per_rank, per_rank)
            gathered = plan_r.gather(full, scale=scale)
            assert torch.equal(contiguous, gathered), f"rank {r} mismatch"

    def test_interleave_is_rank_free(self):
        full = torch.arange(KB + S).float()
        out0 = interleave_extended_for_tp(full, ResidueShardPlan(KB, S, 2, 0), scale=1)
        out1 = interleave_extended_for_tp(full, ResidueShardPlan(KB, S, 2, 1), scale=1)
        assert torch.equal(out0, out1)


class TestLocalSalient:
    def test_rebasing(self):
        plan = ResidueShardPlan(KB, S, 2, 1)
        plan.validate()
        indices = sorted_block_indices(KB, 2)
        local = plan.local_salient_indices(indices)
        assert local.numel() == S // 2
        assert local.min() >= 0
        assert local.max() < plan.base_shard
        # Rank 1's channels are the global channels shifted down by its base.
        assert torch.equal(local, indices[S // 2 :] - KB // 2)

    def test_unsorted_indices_rejected(self):
        plan = ResidueShardPlan(KB, S, 2, 0)
        indices = sorted_block_indices(KB, 2)
        shuffled = indices[torch.randperm(indices.numel())]
        with pytest.raises(ResidueTPError):
            plan.local_salient_indices(shuffled)

    def test_non_uniform_selection_rejected(self):
        plan = ResidueShardPlan(KB, S, 2, 0)
        # All salient channels in the first half: rank 0 would own all of
        # them, rank 1 none -- the balanced-split contract is broken.
        indices = torch.arange(S, dtype=torch.int64)
        with pytest.raises(ResidueTPError, match="owns"):
            plan.local_salient_indices(indices)

    def test_local_channel_mask_slice(self):
        plan = ResidueShardPlan(KB, S, 2, 1)
        full_mask = torch.arange(KB // 8, dtype=torch.uint8)
        local = plan.local_channel_mask(full_mask)
        assert local.numel() == KB // 16
        assert torch.equal(local, full_mask[KB // 16 :])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
