"""Unit tests for the LoRA virtual-experts kernels under post-EP-dispatch
sentinel `topk_ids` (-1) and out-of-range expert IDs.

Covers two regression bugs that surface only with `--lora-use-virtual-experts`
+ `ep_size > 1`:

- `_fused_virtual_topk_ids` must preserve negative sentinel topk_ids. After
  EP dispatch, non-local experts arrive as `-1`; the pre-fix kernel mapped
  them onto a real virtual-expert slot belonging to another adapter and
  triggered OOB loads in downstream LoRA kernels.

- `_align_block_size_torch` / `_align_block_size_jit` (the `>= 1024`-expert
  fallback paths) must route `-1` and `>= num_experts` IDs into a sentinel
  bucket so they don't OOB-index `padded_offsets[sorted_expert_ids]` (negative
  wrap, or past-end) and don't get assigned to a real expert in the
  consumer-block table.

Both kernels run on CUDA. The fallback is gated on `virtual_num_experts >= 1024`
in production, but we exercise it directly here at smaller sizes for cheaper
iteration; one test sticks to the >1024 regime to mirror the production trigger.

Usage:
    python -m pytest test/registered/lora/test_virtual_experts_kernels.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small")

from sglang.srt.lora.triton_ops.virtual_experts import (
    _align_block_size_jit,
    _align_block_size_torch,
    _fused_virtual_topk_ids,
    fused_sanitize_expert_ids,
)


class TestFusedVirtualTopkIdsPreservesSentinels(CustomTestCase):
    """Item B regression: post-EP-dispatch -1 sentinels must NOT be remapped."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def test_negative_sentinels_preserved(self):
        # Mix of valid topk_ids in [0, num_experts), -1 sentinels (typical
        # post-EP-dispatch), and a synthetic -2 to ensure the fix doesn't
        # depend on the exact -1 value.
        topk_ids = torch.tensor(
            [
                [3, -1],
                [-1, 5],
                [0, 7],
                [-1, -1],
                [2, 9],
                [11, -2],
                [-1, 4],
                [6, -1],
            ],
            dtype=torch.int32,
            device=self.device,
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 0, 2, -1, 1, 0, 1], dtype=torch.int32, device=self.device
        )
        num_experts = 16
        max_loras = 4

        virtual_ids, _, _ = _fused_virtual_topk_ids(
            topk_ids,
            token_lora_mapping,
            num_experts,
            shared_outer=False,
            max_loras=max_loras,
        )

        # Every negative input must stay negative (and equal) in the output.
        for m in range(topk_ids.shape[0]):
            for k in range(topk_ids.shape[1]):
                base = topk_ids[m, k].item()
                if base < 0:
                    self.assertEqual(
                        virtual_ids[m, k].item(),
                        base,
                        f"negative sentinel at ({m},{k}) was remapped: "
                        f"{base} -> {virtual_ids[m, k].item()}",
                    )

    def test_positive_topk_remapped_correctly(self):
        """Sanity: valid (non-negative) IDs follow the
        `base + safe_lora * num_experts` rule."""
        topk_ids = torch.tensor(
            [[3, 1], [0, 7], [2, 9]], dtype=torch.int32, device=self.device
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 2], dtype=torch.int32, device=self.device
        )
        num_experts = 16
        max_loras = 4

        virtual_ids, _, _ = _fused_virtual_topk_ids(
            topk_ids, token_lora_mapping, num_experts, False, max_loras
        )

        for m in range(topk_ids.shape[0]):
            lora = token_lora_mapping[m].item()
            for k in range(topk_ids.shape[1]):
                base = topk_ids[m, k].item()
                expected = base + max(lora, 0) * num_experts
                self.assertEqual(virtual_ids[m, k].item(), expected)

    def test_no_lora_token_does_not_shift_base(self):
        """`token_lora_mapping[m] == -1` (no LoRA) keeps `safe_lora=0`,
        so positive bases pass through unchanged and the row mask is False."""
        topk_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=self.device)
        token_lora_mapping = torch.tensor([-1], dtype=torch.int32, device=self.device)
        num_experts = 16

        virtual_ids, mask, _ = _fused_virtual_topk_ids(
            topk_ids, token_lora_mapping, num_experts, False, max_loras=4
        )
        self.assertEqual(virtual_ids[0, 0].item(), 3)
        self.assertEqual(virtual_ids[0, 1].item(), 5)
        self.assertFalse(bool(mask[0].item()))


class _AlignBlockSizeSentinelBucketBase(CustomTestCase):
    """Shared tests for both the torch.compile and JIT align_block_size paths.

    Subclasses override ``_align`` to select the concrete implementation.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if cls is _AlignBlockSizeSentinelBucketBase:
            raise unittest.SkipTest("Base class")
        cls.device = "cuda:0"

    def _align(self, topk_ids, block_size, num_experts):
        raise NotImplementedError

    @staticmethod
    def _assigned_experts(expert_ids: torch.Tensor) -> list:
        """Return the list of real expert ids assigned to blocks (filtering
        out -1 sentinels for padding/exclusion)."""
        return expert_ids[expert_ids != -1].cpu().tolist()

    def _assert_only_real_or_sentinel(self, expert_ids: torch.Tensor, num_experts: int):
        for eid in expert_ids.cpu().tolist():
            self.assertTrue(
                eid == -1 or 0 <= eid < num_experts,
                f"expert_ids contains invalid value {eid}",
            )

    def test_all_valid_baseline(self):
        """Sanity: with no invalid IDs, every present real expert appears
        in the assignment, and no junk values leak through."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, 3], [4, 7], [1, 2], [5, 6]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = set(self._assigned_experts(expert_ids))
        self.assertEqual(assigned, set(range(num_experts)))

    def test_negative_ids_routed_to_sentinel(self):
        """`-1` tokens must not appear as real expert assignments and must
        not corrupt the assignment of real IDs."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, -1], [-1, 7], [1, -1], [-1, -1]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 1, 7):
            self.assertIn(valid_eid, assigned)

    def test_oor_ids_routed_to_sentinel(self):
        """IDs `>= num_experts` (e.g. virtual-experts remap when combined
        with non-local sentinels) must not break cumsum/searchsorted and
        must not show up as real assignments."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, 100], [50, 7], [1, 200]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 1, 7, 50):
            if valid_eid >= num_experts:
                self.assertNotIn(valid_eid, assigned)
            else:
                self.assertIn(valid_eid, assigned)

    def test_mixed_invalid_at_production_size(self):
        """Mirror the production trigger: `num_experts >= 1024` (only path
        where the large-expert fallback is invoked instead of the native
        align kernel)."""
        num_experts = 1500
        block_size = 16
        topk_ids = torch.tensor(
            [
                [-1, 500],
                [num_experts + 7, 1000],
                [num_experts * 2, 100],
                [-1, 0],
            ],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 100, 500, 1000):
            self.assertIn(valid_eid, assigned)

    def test_empty_topk_ids_does_not_crash(self):
        """Edge: empty input. Should return empty/zero outputs without
        OOB indexing on the sentinel bucket."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.empty((0, 2), dtype=torch.int32, device=self.device)

        sorted_token_ids, expert_ids, num_post_padded = self._align(
            topk_ids, block_size, num_experts
        )

        self.assertEqual(num_post_padded.item(), 0)
        self.assertEqual(self._assigned_experts(expert_ids), [])


class TestAlignBlockSizeTorchSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the pure-PyTorch torch.compile fallback path (AMD/ROCm compatible)."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_torch(topk_ids, block_size, num_experts)


class TestAlignBlockSizeJitSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the CUDA JIT kernel path (with fused_sanitize_expert_ids, as in
    production)."""

    def _align(self, topk_ids, block_size, num_experts):
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size_jit(
            topk_ids, block_size, num_experts
        )
        expert_ids = fused_sanitize_expert_ids(expert_ids, num_experts)
        return sorted_token_ids, expert_ids, num_tokens_post_padded


if __name__ == "__main__":
    unittest.main(verbosity=2)
