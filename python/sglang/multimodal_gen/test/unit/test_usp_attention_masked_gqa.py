# SPDX-License-Identifier: Apache-2.0
"""Regression tests for masked GQA in USPAttention."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import (
    USPAttention,
    _expand_kv_for_gqa,
    _same_head_count,
)

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


def _masked_forward(q, k, v, attn_mask):
    """Drive the masked non-SP branch without distributed initialization."""
    obj = USPAttention.__new__(USPAttention)
    obj.skip_sequence_parallel = True  # take the non-SP branch
    obj.softmax_scale = None
    obj.allow_cudnn_sdp = False  # plain SDPA, no kernel preference
    obj.backend = None  # not FA -> varlen fast path is skipped
    obj.attn_impl = None  # unmasked path only; must not be reached

    with (
        patch(
            f"{_LAYER}.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
    ):
        return USPAttention.forward(obj, q, k, v, attn_mask=attn_mask)


def _sdpa_reference(q, k, v, attn_mask):
    """What the masked branch should compute, written independently.

    Uses PyTorch's own ``enable_gqa`` rather than repeating the expansion under
    test, so this cannot agree with the implementation by sharing its mistake.
    """
    additive = (attn_mask.to(q.dtype)[:, None, None, :] - 1.0) * torch.finfo(
        q.dtype
    ).max
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=additive,
        enable_gqa=q.shape[2] != k.shape[2],
    )
    return out.transpose(1, 2)


class TestUSPAttentionMaskedGQA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mask = torch.tensor([[True, True, True, True, False, False]])

    def test_masked_attention_handles_gqa(self):
        q = torch.randn(1, 6, 4, 8)
        k = torch.randn(1, 6, 2, 8)
        v = torch.randn(1, 6, 2, 8)

        out = _masked_forward(q, k, v, self.mask)

        self.assertEqual(tuple(out.shape), (1, 6, 4, 8))
        torch.testing.assert_close(out, _sdpa_reference(q, k, v, self.mask))

    def test_mha_masked_output_is_unchanged(self):
        q = torch.randn(1, 6, 4, 8)
        k = torch.randn(1, 6, 4, 8)
        v = torch.randn(1, 6, 4, 8)

        out = _masked_forward(q, k, v, self.mask)
        torch.testing.assert_close(out, _sdpa_reference(q, k, v, self.mask))

        # assertIs, not assert_close: MHA must get the same objects back, so
        # nothing can perturb Z-Image's numerics.
        kt, vt = k.transpose(1, 2), v.transpose(1, 2)
        k_, v_ = _expand_kv_for_gqa(q.transpose(1, 2), kt, vt)
        self.assertIs(k_, kt)
        self.assertIs(v_, vt)

    def test_varlen_fa_gate_rejects_gqa_and_accepts_mha(self):
        """The gate the fast path needs. Reaching the real one requires CUDA +
        FlashAttention + bf16, so the predicate is tested directly."""
        gqa = (torch.empty(1, 4352, 24, 96), torch.empty(1, 4352, 8, 96))
        self.assertFalse(_same_head_count(gqa[0], gqa[1], gqa[1]))

        mha = torch.empty(1, 4352, 30, 96)
        self.assertTrue(_same_head_count(mha, mha, mha))

    def test_expand_kv_rejects_a_non_divisible_head_count(self):
        q = torch.empty(1, 24, 4, 8)
        bad = torch.empty(1, 5, 4, 8)
        with self.assertRaises(ValueError) as ctx:
            _expand_kv_for_gqa(q, bad, bad)
        self.assertIn("positive multiple", str(ctx.exception))

    def test_expanded_kv_repeats_each_group_contiguously(self):
        """Query head i must see KV head i // repeat_factor. Getting the
        interleave backwards (``repeat`` instead of ``repeat_interleave``)
        still produces the right shape and silently wrong attention."""
        q = torch.empty(1, 4, 6, 8)
        k = torch.randn(1, 2, 6, 8)
        k_, _ = _expand_kv_for_gqa(q, k, k)

        for head in range(4):
            self.assertTrue(torch.equal(k_[0, head], k[0, head // 2]))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA + FlashAttention")
class TestUSPAttentionMaskedGQAOnGPU(unittest.TestCase):
    def test_fused_varlen_path_is_still_taken_for_mha(self):
        """The only test that exercises the real gate: MHA must keep the varlen
        FA fast path, GQA must fall through to SDPA instead of reaching
        ``fused_pack_qkv``'s equal-shape assert."""
        from sglang.multimodal_gen.runtime.layers.attention.layer import (
            build_varlen_mask_meta,
        )
        from sglang.multimodal_gen.runtime.layers.attention.layer import (
            fused_pack_qkv as real_pack,
        )
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

        mask = torch.tensor([[True] * 4 + [False] * 2], device="cuda")
        meta = build_varlen_mask_meta(mask)

        def run(kv_heads):
            q = torch.randn(1, 6, 4, 8, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 6, kv_heads, 8, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 6, kv_heads, 8, device="cuda", dtype=torch.bfloat16)
            obj = USPAttention.__new__(USPAttention)
            obj.skip_sequence_parallel = True
            obj.softmax_scale = None
            obj.allow_cudnn_sdp = False
            obj.backend = AttentionBackendEnum.FA
            obj.attn_impl = None
            with (
                patch(
                    f"{_LAYER}.get_forward_context",
                    return_value=SimpleNamespace(attn_metadata=None),
                ),
                patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
                patch(f"{_LAYER}.fused_pack_qkv", side_effect=real_pack) as spy,
            ):
                out = USPAttention.forward(
                    obj, q, k, v, attn_mask=mask, attn_mask_meta=meta
                )
            return out, spy.called

        _, mha_used_fa = run(kv_heads=4)
        self.assertTrue(mha_used_fa, "MHA must keep the varlen FA fast path")

        out, gqa_used_fa = run(kv_heads=2)
        self.assertFalse(gqa_used_fa, "GQA must not reach fused_pack_qkv")
        self.assertEqual(tuple(out.shape), (1, 6, 4, 8))


if __name__ == "__main__":
    unittest.main()
