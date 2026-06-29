"""Numerical correctness test for ZAYA1 MoE + MOD under TP>1.

Background: the MOD (mixture-of-depths) skip-expert residual blend must be
combined with the experts output on the correct side of the cross-rank
all-reduce. ``mod_out = hidden_states * prob`` is replicated on every TP rank,
so all-reducing it would multiply it by ``tp_size``. The model therefore masks
the *per-rank partial* experts output before the reduce and only adds the
replicated ``mod_out`` afterwards:

    sum_r(mask · partial_r) + (1 - mask) · mod_out
  = mask · experts_out_full   + (1 - mask) · mod_out

This test drives the *real* helpers used by ``ZayaBlock.forward`` --
``mod_premask_experts`` and ``mod_blend`` -- so a regression in that math is
caught. The cross-rank all-reduce is simulated by summing the per-rank partials
(the masks are replicated, so the sum is exact), which keeps the test runnable
on CPU CI without a live ``torch.distributed`` group.
"""

import unittest

import torch

from sglang.srt.models.zaya import mod_blend, mod_premask_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reference_blend(
    hidden_states: torch.Tensor,  # [T, H]
    probs: torch.Tensor,  # [T, 1]
    indices: torch.Tensor,  # [T, 1]
    experts_out_full: torch.Tensor,  # [T, H] -- already-reduced full experts output
    num_moe_experts: int,
) -> torch.Tensor:
    """Reference: apply the MOD mask to the *full* (already-reduced) experts
    output, then add the skip path. Mirrors the intended algebra directly.
    """
    mod_mask = (indices != num_moe_experts).to(experts_out_full.dtype)
    mod_out = hidden_states * probs
    return mod_mask * experts_out_full + (1.0 - mod_mask) * mod_out


def _real_tp_blend(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    indices: torch.Tensor,
    partial_experts_per_rank: list[torch.Tensor],  # one [T, H] per rank
    num_moe_experts: int,
) -> torch.Tensor:
    """Production path: ``mod_premask_experts`` per rank -> simulated all-reduce
    (sum) -> ``mod_blend``. Uses the exact helpers ``ZayaBlock.forward`` calls.
    """
    mod_out = hidden_states * probs
    reduced = None
    mod_mask = None
    for partial in partial_experts_per_rank:
        mask, masked = mod_premask_experts(partial, indices, num_moe_experts)
        mod_mask = mask
        reduced = masked if reduced is None else reduced + masked
    return mod_blend(reduced, mod_mask, mod_out)


def _buggy_old_tp_blend(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    indices: torch.Tensor,
    partial_experts_per_rank: list[torch.Tensor],
    num_moe_experts: int,
) -> torch.Tensor:
    """Old, broken sequence: all-reduce the replicated ``mod_out`` (so it gets
    scaled by ``tp_size``) then mix. Proves the test catches a regression.
    """
    tp_size = len(partial_experts_per_rank)
    mod_out_replicated = hidden_states * probs
    mod_out_after_allreduce = mod_out_replicated * tp_size  # all-reduce of replicated
    experts_out_full = torch.stack(partial_experts_per_rank, dim=0).sum(dim=0)
    mod_mask = (indices != num_moe_experts).to(experts_out_full.dtype)
    return mod_mask * experts_out_full + (1.0 - mod_mask) * mod_out_after_allreduce


class TestZayaMODUnderTP(CustomTestCase):
    def _make_partials(self, T: int, H: int, tp_size: int):
        torch.manual_seed(31)
        experts_out_full = torch.randn(T, H, dtype=torch.float32) * 0.1
        # Split into ``tp_size`` random partial tensors that sum to the full output.
        partials = []
        remaining = experts_out_full.clone()
        for _ in range(tp_size - 1):
            p = torch.randn_like(remaining) * 0.05
            partials.append(p)
            remaining = remaining - p
        partials.append(remaining)
        return experts_out_full, partials

    def _make_inputs(self, T: int, H: int, num_experts: int, frac_skip: float):
        torch.manual_seed(7)
        hidden_states = torch.randn(T, H, dtype=torch.float32)
        probs = torch.rand(T, 1, dtype=torch.float32)
        # Build indices: with probability ``frac_skip`` mark token as skip-expert.
        skip_id = num_experts  # MOD uses ``num_moe_experts`` as the skip slot
        rand = torch.rand(T, 1)
        real = torch.randint(0, num_experts, (T, 1))
        indices = torch.where(rand < frac_skip, torch.full_like(real, skip_id), real)
        return hidden_states, probs, indices

    def test_real_helpers_match_reference_for_tp(self):
        """The real ``mod_premask_experts`` / ``mod_blend`` path must equal the
        reference blend for any TP size and any skip fraction.
        """
        T, H = 8, 16
        num_experts = 4
        for tp_size in (2, 4, 8):
            for frac_skip in (0.0, 0.5, 1.0):
                hidden_states, probs, indices = self._make_inputs(
                    T, H, num_experts, frac_skip
                )
                full, partials = self._make_partials(T, H, tp_size)

                ref = _reference_blend(hidden_states, probs, indices, full, num_experts)
                real = _real_tp_blend(
                    hidden_states, probs, indices, partials, num_experts
                )

                torch.testing.assert_close(
                    real,
                    ref,
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"tp_size={tp_size} frac_skip={frac_skip}",
                )

    def test_premask_zeroes_skip_tokens(self):
        """``mod_premask_experts`` must zero the experts contribution exactly on
        skip-routed tokens and pass through real-expert tokens unchanged.
        """
        T, H = 6, 8
        num_experts = 4
        experts_out = torch.randn(T, H, dtype=torch.float32)
        # Alternate skip / real tokens.
        indices = torch.tensor(
            [[num_experts], [0], [num_experts], [1], [num_experts], [2]],
            dtype=torch.long,
        )
        mod_mask, masked = mod_premask_experts(experts_out, indices, num_experts)

        skip_rows = indices.squeeze(-1) == num_experts
        self.assertTrue(torch.all(masked[skip_rows] == 0))
        torch.testing.assert_close(masked[~skip_rows], experts_out[~skip_rows])
        # mask is 0 on skip rows, 1 elsewhere.
        self.assertTrue(torch.all(mod_mask.squeeze(-1)[skip_rows] == 0))
        self.assertTrue(torch.all(mod_mask.squeeze(-1)[~skip_rows] == 1))

    def test_old_blend_is_wrong_when_skip_used(self):
        """Sanity: confirm the old (all-reduce mod_out) formula diverges from the
        reference so a regression to that behavior would be caught.
        """
        T, H = 8, 16
        num_experts = 4
        tp_size = 4
        hidden_states, probs, indices = self._make_inputs(
            T, H, num_experts, frac_skip=0.5
        )
        full, partials = self._make_partials(T, H, tp_size)

        ref = _reference_blend(hidden_states, probs, indices, full, num_experts)
        buggy = _buggy_old_tp_blend(
            hidden_states, probs, indices, partials, num_experts
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(buggy, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
