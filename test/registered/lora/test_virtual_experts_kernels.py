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

Both kernels run on CUDA or XPU; only the CUDA-JIT align variant is CUDA-only.
The fallback is gated on `virtual_num_experts >= 1024` in production, but we
exercise it directly here at smaller sizes for cheaper iteration; one test
sticks to the >1024 regime to mirror the production trigger.

Usage:
    python -m pytest test/registered/lora/test_virtual_experts_kernels.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import get_device, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-small")
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

        original_ids = topk_ids.clone()
        providers = {"base": _fused_virtual_topk_ids}
        if is_cuda():
            from sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts import (
                _fused_virtual_topk_ids as trtllm_virtual_ids,
            )

            providers["trtllm"] = trtllm_virtual_ids
        for provider, remap in providers.items():
            for shared_outer in (False, True):
                with self.subTest(provider=provider, shared_outer=shared_outer):
                    virtual_ids, _, _ = remap(
                        topk_ids,
                        token_lora_mapping,
                        num_experts,
                        shared_outer=shared_outer,
                        max_loras=max_loras,
                    )

                    # Shared weights still follow each original route's validity,
                    # including a dropped first slot followed by a live slot.
                    for m in range(topk_ids.shape[0]):
                        slot = token_lora_mapping[m].item()
                        lora = max(slot, 0)
                        for k in range(topk_ids.shape[1]):
                            base = topk_ids[m, k].item()
                            expected = base
                            if provider == "trtllm" and slot < 0:
                                expected = -1
                            elif base >= 0:
                                expected = (
                                    lora if shared_outer else base + lora * num_experts
                                )
                            self.assertEqual(virtual_ids[m, k].item(), expected)
                    torch.testing.assert_close(topk_ids, original_ids, rtol=0, atol=0)

    @unittest.skipUnless(is_cuda(), "Merged LoRA alignment requires CUDA")
    def test_shared_outer_merged_alignment_skips_original_negative_routes(self):
        """Shared-outer blocks contain exactly live original slots after remapping."""
        from sglang.kernels.ops.lora.moe.trtllm_lora_temp.moe_lora_merged_align import (
            moe_lora_merged_align,
        )

        topk_ids = torch.tensor(
            [[-1, 2, 0], [-1, -1, -1], [1, -3, 3]],
            device=self.device,
            dtype=torch.int32,
        )
        live_ids = topk_ids.clone()
        mapping = torch.tensor([0, 1, 2], device=self.device, dtype=torch.int32)
        block_size = 16
        for fuse_scatter in (False, True):
            for state in ("mixed", "idle", "remapped"):
                with self.subTest(fuse_scatter=fuse_scatter, state=state):
                    topk_ids.copy_(live_ids)
                    mapping.copy_(
                        torch.tensor(
                            [2, 0, 1] if state == "remapped" else [0, 1, 2],
                            device=self.device,
                            dtype=torch.int32,
                        )
                    )
                    if state == "idle":
                        topk_ids.fill_(-1)
                    original_ids = topk_ids.clone()
                    original_mapping = mapping.clone()
                    sorted_ids, experts, count, _, _ = moe_lora_merged_align(
                        topk_ids,
                        mapping,
                        num_experts=4,
                        shared_outer=True,
                        max_loras=3,
                        block_size=block_size,
                        do_skip=True,
                        fuse_scatter=fuse_scatter,
                    )
                    used = int(count.item())
                    self.assertGreaterEqual(used, 0)
                    self.assertEqual(used % block_size, 0)
                    self.assertLessEqual(used, sorted_ids.numel())
                    blocks = sorted_ids[:used].view(-1, block_size)
                    self.assertTrue(
                        bool(((blocks >= 0) & (blocks <= topk_ids.numel())).all())
                    )
                    emitted = blocks[blocks < topk_ids.numel()]
                    expected = torch.nonzero(topk_ids.flatten() >= 0).flatten()
                    self.assertEqual(
                        sorted(emitted.cpu().tolist()), expected.cpu().tolist()
                    )
                    for index, block in enumerate(blocks):
                        flat_ids = block[block < topk_ids.numel()].long()
                        tokens = flat_ids // topk_ids.shape[1]
                        self.assertTrue(bool((topk_ids.flatten()[flat_ids] >= 0).all()))
                        self.assertTrue(
                            bool((original_mapping[tokens] == experts[index]).all())
                        )
                    torch.testing.assert_close(topk_ids, original_ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        mapping, original_mapping, rtol=0, atol=0
                    )

    @unittest.skipUnless(is_cuda(), "Stock virtual LoRA hook fixture requires CUDA")
    def test_gate_up_hook_uses_each_stacked_a_projection(self):
        """Independent gate/up A matrices reach their matching B and output views."""
        from sglang.kernels.ops.lora.moe import virtual_experts
        from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
        from sglang.srt.lora.lora_moe_runners import build_lora_hooks

        tokens, topk, experts, rank, hidden, intermediate = 3, 2, 2, 16, 64, 32
        dtype = torch.float16
        x = torch.zeros((tokens, hidden), device=self.device, dtype=dtype)
        x[:, 0] = torch.arange(1, tokens + 1, device=self.device, dtype=dtype)
        ids = torch.tensor(
            [[-1, 1], [0, 1], [-1, -1]], device=self.device, dtype=torch.int32
        )
        weights = torch.tensor(
            [[float("nan"), 0.5], [0.25, 0.75], [float("nan"), float("nan")]],
            device=self.device,
        )
        mapping = torch.tensor([0, 1, 0], device=self.device, dtype=torch.int32)
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }
        real_add = virtual_experts.merged_experts_fused_moe_lora_add
        # The common-A and non-gated layouts must retain one full-width call.
        for a_parts, b_parts in ((2, 2), (1, 2), (1, 1)):
            for shared_outer in (False, True):
                with self.subTest(
                    a_parts=a_parts, b_parts=b_parts, shared_outer=shared_outer
                ):
                    a = torch.zeros(
                        (2, 1 if shared_outer else experts, a_parts * rank, hidden),
                        device=self.device,
                        dtype=dtype,
                    )
                    b = torch.zeros(
                        (2, experts, b_parts * intermediate, rank),
                        device=self.device,
                        dtype=dtype,
                    )
                    for slot in range(2):
                        for expert in range(a.shape[1]):
                            a[slot, expert, 0, 0] = 2 + slot + expert
                            if a_parts == 2:
                                a[slot, expert, rank, 0] = 4 + slot + expert
                        for expert in range(experts):
                            b[slot, expert, :intermediate, 0] = 5 + slot + expert
                            if b_parts == 2:
                                b[slot, expert, intermediate:, 0] = 7 + slot + expert
                    output = torch.full(
                        (tokens, topk, b_parts * intermediate),
                        0.5,
                        device=self.device,
                        dtype=dtype,
                    )
                    safe_ids = ids.clamp_min(0)
                    a_ids = torch.zeros_like(ids) if shared_outer else safe_ids
                    selected_a = a[mapping[:, None], a_ids]
                    selected_b = b[mapping[:, None], safe_ids]
                    shrunk = torch.einsum(
                        "mh,mkrh->mkr", x.float(), selected_a.float()
                    ).to(dtype)
                    if a_parts == 2:
                        delta = torch.einsum(
                            "mkpr,mkpir->mkpi",
                            shrunk.view(tokens, topk, 2, rank).float(),
                            selected_b.view(
                                tokens, topk, 2, intermediate, rank
                            ).float(),
                        ).flatten(2)
                    else:
                        delta = torch.einsum(
                            "mkr,mkir->mki", shrunk.float(), selected_b.float()
                        )
                    delta = delta.to(dtype).masked_fill((ids < 0).unsqueeze(-1), 0)
                    expected = output + delta
                    info = SimpleNamespace(
                        max_lora_rank=rank,
                        has_active_lora=True,
                        lora_use_virtual_experts=True,
                        experts_shared_outer_loras=shared_outer,
                        gate_up_lora_a_weights=a,
                        gate_up_lora_b_weights=b,
                        token_lora_mapping=mapping,
                    )
                    calls = []

                    def observed_add(**kwargs):
                        projected = kwargs["output"]
                        self.assertEqual(projected.stride(-2), output.shape[-1])
                        self.assertEqual(
                            projected.untyped_storage().data_ptr(),
                            output.untyped_storage().data_ptr(),
                        )
                        calls.append((projected.data_ptr(), kwargs["lora_a"].shape[2]))
                        return real_add(**kwargs)

                    with (
                        override_config(config),
                        patch(
                            "sglang.srt.lora.lora_moe_runners.get_is_capture_mode",
                            return_value=False,
                        ),
                        patch.object(
                            virtual_experts,
                            "merged_experts_fused_moe_lora_add",
                            observed_add,
                        ),
                    ):
                        hooks = build_lora_hooks(x, info, ids)
                        hooks.after_gate_up(x, output, weights, ids)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self.assertEqual(len(calls), a_parts)
                    self.assertEqual(calls[0], (output.data_ptr(), rank))
                    if a_parts == 2:
                        self.assertEqual(
                            calls[1],
                            (
                                output.data_ptr()
                                + intermediate * output.element_size(),
                                rank,
                            ),
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
        if cls is _AlignBlockSizeSentinelBucketBase:
            raise unittest.SkipTest("Base class")
        cls.device = _require_accelerator_device()

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
