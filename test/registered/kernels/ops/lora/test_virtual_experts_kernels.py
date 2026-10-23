"""Sentinel handling in the LoRA virtual-experts kernels.

After EP dispatch non-local experts arrive as ``-1``. ``_fused_virtual_topk_ids``
must keep those sentinels instead of remapping them onto another adapter's
virtual-expert slot, and the ``>= 1024``-expert align paths must place every
valid (token, expert) pair exactly once in its own expert's blocks while
invalid IDs never enter a real expert's block. Either bug corrupts routing
silently.
"""

import collections
import unittest

import torch

from sglang.srt.utils import get_device, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: XPU runs the torch.compile align fallback in production and
# builds the Triton topk remap with its own backend.
register_xpu_ci(est_time=20, suite="stage-a-test-1-gpu-xpu")


def _require_accelerator_device():
    if not (is_cuda() or is_xpu()):
        raise unittest.SkipTest("CUDA or XPU required")
    return get_device(0)


from sglang.kernels.ops.lora.moe.virtual_experts import (
    _align_block_size_jit,
    _align_block_size_torch,
    _fused_virtual_topk_ids,
    fused_sanitize_expert_ids,
)


class TestFusedVirtualTopkIdsPreservesSentinels(CustomTestCase):
    """Item B regression: post-EP-dispatch -1 sentinels must NOT be remapped."""

    @classmethod
    def setUpClass(cls):
        cls.device = _require_accelerator_device()

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

    Subclasses override ``_align`` to select the concrete implementation and
    ``_invalid_ids`` with the IDs outside ``[0, num_experts)`` it accepts.
    """

    @classmethod
    def setUpClass(cls):
        if cls is _AlignBlockSizeSentinelBucketBase:
            raise unittest.SkipTest("Base class")
        cls.device = _require_accelerator_device()

    def _align(self, topk_ids, block_size, num_experts):
        raise NotImplementedError

    def _invalid_ids(self, num_experts):
        raise NotImplementedError

    def _assert_aligned(self, ids, block_size, num_experts):
        """Every valid (token, expert) pair lands exactly once in a block of
        its own expert, each expert gets ceil(pairs / block_size) blocks, a
        real-expert block holds only its pairs and padding, and a -1 block
        holds no valid pair and is no more than the invalid pairs need. The
        order inside a block is not part of the contract."""
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
        topk_ids = torch.tensor(ids, dtype=torch.int32)[order].view(-1, 2)
        sorted_ids, expert_ids, num_post_pad = self._align(
            topk_ids.to(self.device), block_size, num_experts
        )
        flat = topk_ids.view(-1).tolist()
        padding = len(flat)
        total = int(num_post_pad.reshape(-1)[0])
        self.assertEqual(total % block_size, 0, f"padded count {total}")
        self.assertLessEqual(total, sorted_ids.numel(), f"padded count {total}")
        labels = expert_ids[: total // block_size].tolist()
        blocks = sorted_ids[:total].view(-1, block_size).tolist()

        placed, blocks_per_expert = collections.Counter(), collections.Counter()
        for label, block in zip(labels, blocks):
            pairs = [j for j in block if j != padding]
            self.assertTrue(all(0 <= j < padding for j in pairs), f"block {block}")
            if label == -1:
                self.assertFalse(
                    [j for j in pairs if 0 <= flat[j] < num_experts],
                    f"valid pairs in a -1 block: {block}",
                )
                continue
            self.assertTrue(0 <= label < num_experts, f"block label {label}")
            self.assertEqual(
                {flat[j] for j in pairs}, {label}, f"block of expert {label}: {block}"
            )
            placed.update(pairs)
            blocks_per_expert[label] += 1

        valid = [j for j, e in enumerate(flat) if 0 <= e < num_experts]
        self.assertEqual(placed, collections.Counter(valid), "valid pairs")
        per_expert = collections.Counter(flat[j] for j in valid)
        self.assertEqual(
            blocks_per_expert,
            collections.Counter(
                {e: -(-n // block_size) for e, n in per_expert.items()}
            ),
        )
        num_invalid = len(flat) - len(valid)
        self.assertLessEqual(labels.count(-1), -(-num_invalid // block_size))

    def test_small_every_expert_with_invalid_ids(self):
        """Every expert populated, one spanning two blocks, invalid IDs mixed
        in."""
        num_experts = 8
        block_size = 16
        valid = list(range(num_experts)) + [3] * block_size
        ids = valid + self._invalid_ids(num_experts) * 4
        self._assert_aligned(ids, block_size, num_experts)

    def test_mixed_invalid_at_production_size(self):
        """Mirror the production trigger: `num_experts >= 1024` (only path
        where the large-expert fallback is invoked instead of the native
        align kernel). The first and last expert, lone pairs and one expert
        spanning two blocks, interleaved with invalid IDs."""
        num_experts = 1500
        block_size = 16
        valid = [0, num_experts - 1, 100, 500, 1000] + [700] * (block_size + 3)
        ids = valid + self._invalid_ids(num_experts) * 4
        self._assert_aligned(ids, block_size, num_experts)


class TestAlignBlockSizeTorchSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the pure-PyTorch torch.compile fallback path (AMD/ROCm compatible)."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_torch(topk_ids, block_size, num_experts)

    def _invalid_ids(self, num_experts):
        # Negative and >= num_experts IDs both go to its sentinel bucket.
        return [-1, num_experts + 7, 2 * num_experts]


@unittest.skipIf(
    is_xpu(),
    "_align_block_size_jit builds a CUDA JIT kernel via tvm_ffi.load_inline "
    "(requires a CUDA/nvcc install); it cannot run on XPU. The torch variant "
    "(TestAlignBlockSizeTorchSentinelBucket) covers the same alignment logic "
    "on that platform.",
)
class TestAlignBlockSizeJitSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the CUDA JIT kernel path (with fused_sanitize_expert_ids, as in
    production)."""

    def _align(self, topk_ids, block_size, num_experts):
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size_jit(
            topk_ids, block_size, num_experts
        )
        expert_ids = fused_sanitize_expert_ids(expert_ids, num_experts)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    def _invalid_ids(self, num_experts):
        # Valid inputs are -1 or an expert ID in [0, num_experts). The sort
        # uses id + 1 as an unchecked bucket index and the count it reads
        # there as a write offset, so an ID >= num_experts or < -1 can touch
        # invalid memory (seen at 1,500 experts on GB300); hardening that is
        # a production change, not this test's.
        return [-1]


if __name__ == "__main__":
    unittest.main(verbosity=2)
