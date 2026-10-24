"""Sentinel handling in the LoRA virtual-experts kernels.

After EP dispatch non-local experts arrive as ``-1``. ``_fused_virtual_topk_ids``
must keep those sentinels instead of remapping them onto another adapter's
virtual-expert slot, and the ``>= 1024``-expert align paths must place every
valid (token, expert) pair exactly once in its own expert's blocks while
invalid IDs never enter a real expert's block. Either bug corrupts routing
silently. The capacity, staged-routing and graph-replay matrices are in
test_virtual_experts_kernels_matrix.py.
"""

import collections
import unittest
from itertools import product

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
)


class TestFusedVirtualTopkIdsPreservesSentinels(CustomTestCase):
    """Item B regression: post-EP-dispatch -1 sentinels must NOT be remapped."""

    @classmethod
    def setUpClass(cls):
        cls.device = _require_accelerator_device()

    def test_negative_sentinels_preserved(self):
        """Virtual scratch IDs stay bounded without changing borrowed route metadata."""
        for dtype, max_loras in product((torch.int32, torch.int64), (1, 4)):
            with self.subTest(dtype=dtype, max_loras=max_loras):
                self._check_remap_domain(dtype, max_loras)

    def _check_remap_domain(self, dtype, max_loras):
        num_experts = 16
        wide = 2**40 if dtype == torch.int64 else 2**30
        topk_ids = torch.tensor(
            [
                [3, -1],
                [-3, 5],
                [0, 7],
                [-1, -1],
                [2, 9],
                [11, -2],
                [-1, 4],
                [6, -1],
                [16, 17],
                [wide, -wide],
                [torch.iinfo(dtype).min, torch.iinfo(dtype).max],
                [0, 15],
            ],
            dtype=dtype,
            device=self.device,
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 0, 2, -1, 1, 0, 1, 0, 1, 4, 5],
            dtype=torch.int32,
            device=self.device,
        )
        original_ids = topk_ids.clone()
        original_mapping = token_lora_mapping.clone()
        providers = {"base": _fused_virtual_topk_ids}
        if is_cuda():
            from sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts import (
                _fused_virtual_topk_ids as trtllm_virtual_ids,
            )

            providers["trtllm"] = trtllm_virtual_ids
        for (provider, remap), shared_outer, use_enabled in product(
            providers.items(), (False, True), (False, True)
        ):
            with self.subTest(
                provider=provider, shared_outer=shared_outer, enabled=use_enabled
            ):
                enabled = (
                    torch.ones(max_loras * 2, device=self.device, dtype=torch.int32)[
                        ::2
                    ]
                    if use_enabled
                    else None
                )
                if enabled is not None and max_loras > 1:
                    enabled[1] = 0

                def run():
                    return remap(
                        topk_ids,
                        token_lora_mapping,
                        num_experts,
                        shared_outer=shared_outer,
                        max_loras=max_loras,
                        adapter_enabled=enabled,
                    )

                virtual_ids, mask, virtual_experts = run()
                graph = None
                if is_cuda():
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        virtual_ids, mask, virtual_experts = run()
                for state in ("mixed", "disabled", "inactive", "recovered"):
                    topk_ids.copy_(original_ids)
                    token_lora_mapping.copy_(original_mapping)
                    if enabled is not None:
                        enabled.fill_(1)
                        if max_loras > 1:
                            enabled[1] = 0
                    if state == "disabled":
                        if enabled is not None:
                            token_lora_mapping.zero_()
                            enabled.zero_()
                        else:
                            token_lora_mapping.fill_(-1)
                    elif state == "inactive":
                        topk_ids.fill_(-1)
                    saved_ids, saved_mapping = (
                        topk_ids.clone(),
                        token_lora_mapping.clone(),
                    )
                    if graph is None:
                        virtual_ids, mask, virtual_experts = run()
                    else:
                        graph.replay()
                    expected = torch.full_like(topk_ids, -1)
                    for row, slot in enumerate(saved_mapping.cpu().tolist()):
                        active = 0 <= slot < max_loras and (
                            enabled is None or bool(enabled[slot])
                        )
                        self.assertEqual(bool(mask[row]), active)
                        for route, expert in enumerate(saved_ids[row].cpu().tolist()):
                            if (
                                active
                                and expert >= 0
                                and (shared_outer or expert < num_experts)
                            ):
                                expected[row, route] = (
                                    slot
                                    if shared_outer
                                    else expert + slot * num_experts
                                )
                    torch.testing.assert_close(virtual_ids, expected, rtol=0, atol=0)
                    self.assertTrue(
                        bool(
                            (
                                (virtual_ids == -1)
                                | ((virtual_ids >= 0) & (virtual_ids < virtual_experts))
                            ).all()
                        )
                    )
                    torch.testing.assert_close(topk_ids, saved_ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        token_lora_mapping, saved_mapping, rtol=0, atol=0
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

    def test_no_lora_token_has_no_virtual_expert(self):
        """A missing adapter never occupies a virtual expert, including slot zero."""
        topk_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=self.device)
        token_lora_mapping = torch.tensor([-1], dtype=torch.int32, device=self.device)
        num_experts = 16

        virtual_ids, mask, _ = _fused_virtual_topk_ids(
            topk_ids, token_lora_mapping, num_experts, False, max_loras=4
        )
        torch.testing.assert_close(virtual_ids, torch.full_like(virtual_ids, -1))
        torch.testing.assert_close(
            topk_ids, torch.tensor([[3, 5]], device=self.device, dtype=torch.int32)
        )
        self.assertFalse(bool(mask[0].item()))


class _AlignBlockSizeSentinelBucketBase(CustomTestCase):
    """Shared tests for both the torch.compile and JIT align_block_size paths.

    Subclasses override ``_align`` to select the concrete implementation.
    """

    @classmethod
    def setUpClass(cls):
        if cls is _AlignBlockSizeSentinelBucketBase:
            raise unittest.SkipTest("Base class")
        cls.device = _require_accelerator_device()

    def _align(self, topk_ids, block_size, num_experts):
        raise NotImplementedError

    @staticmethod
    def _invalid_ids(num_experts):
        # Both paths send negative and >= num_experts IDs to the sentinel bucket.
        return [-1, num_experts + 7, 2 * num_experts]

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
        self.assertGreaterEqual(total, 0, f"padded count {total}")
        self.assertEqual(total % block_size, 0, f"padded count {total}")
        self.assertLessEqual(total, sorted_ids.numel(), f"padded count {total}")
        self.assertLessEqual(
            total // block_size, expert_ids.numel(), f"padded count {total}"
        )
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

    def test_empty_topk_ids_does_not_crash(self):
        """Empty input pads to nothing."""
        self._assert_aligned([], 16, 8)


class TestAlignBlockSizeTorchSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the pure-PyTorch torch.compile fallback path (AMD/ROCm compatible)."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_torch(topk_ids, block_size, num_experts)


@unittest.skipIf(
    is_xpu(),
    "_align_block_size_jit builds a CUDA JIT kernel via tvm_ffi.load_inline "
    "(requires a CUDA/nvcc install); it cannot run on XPU. The torch variant "
    "(TestAlignBlockSizeTorchSentinelBucket) covers the same alignment logic "
    "on that platform.",
)
class TestAlignBlockSizeJitSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test raw-ID CUDA alignment without a post-hoc expert-ID sanitizer."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_jit(topk_ids, block_size, num_experts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
