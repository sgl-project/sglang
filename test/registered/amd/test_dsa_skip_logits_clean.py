"""AMD/gfx950 tests for skipping redundant DSA logits initialization.

#28757 passes ``clean_logits=False`` to the HIP aiter ``fp8_mqa_logits`` call in
the DSA indexer, skipping the ``-inf`` pre-fill of the
``[tokens x seq_len_kv]`` MQA-logits buffer (which otherwise grows
quadratically with context). This is correctness-preserving because the
downstream topk transform re-masks the invalid positions ``[ks, ke)`` via
``lengths`` / ``row_starts`` before selecting topk -- so whatever the kernel
leaves at the invalid positions is overwritten, and the final topk selection is
identical whether or not the buffer was pre-filled with ``-inf``.

These tests verify that property end to end on the real kernels for both the
ordinary and KPool indexers: aiter ``fp8_mqa_logits`` is run with
``clean_logits=True`` and ``clean_logits=False``, each followed by the matching
masked top-k kernel, and the selected KV positions must be identical. A sanity
assertion confirms ``clean_logits=False`` genuinely leaves non-``-inf`` values
at invalid positions (otherwise the test would be vacuous).
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

# This exercises the gfx950 aiter fp8_mqa_logits + sgl-kernel topk path.
_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
        from aiter.ops.triton.utils.types import get_fp8_dtypes
        from sgl_kernel import fast_topk_v2

        from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )
    except Exception:
        _RUNNABLE = False


def _cast_kv_to_fp8(kv: torch.Tensor, e4m3: torch.dtype):
    """Per-row (per-token) symmetric fp8 cast, returning fp8 tensor + scales."""
    fp8_max = torch.finfo(e4m3).max
    amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(min=1e-4)
    scales = (amax / fp8_max).squeeze(-1)
    kv_fp8 = (kv / (amax / fp8_max)).to(e4m3)
    return kv_fp8, scales


@unittest.skipUnless(
    _RUNNABLE, "requires HIP gfx950 with aiter fp8_mqa_logits + sgl_kernel"
)
class TestDSASkipLogitsClean(CustomTestCase):
    @staticmethod
    def _assert_same_selection(topk_clean, topk_dirty):
        for i in range(topk_clean.shape[0]):
            sel_clean = sorted(x for x in topk_clean[i].tolist() if x >= 0)
            sel_dirty = sorted(x for x in topk_dirty[i].tolist() if x >= 0)
            if sel_clean != sel_dirty:
                raise AssertionError(f"top-k selection differs at row {i}")

    def _run_case(self, s_q, s_k, num_heads, head_dim, topk, seed=0):
        torch.manual_seed(seed)
        dev = "cuda"
        _, e4m3 = get_fp8_dtypes()

        q = torch.randn(s_q, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
        kv = torch.randn(s_k, head_dim, device=dev, dtype=torch.bfloat16)
        weights = torch.randn(s_q, num_heads, device=dev, dtype=torch.float32)
        q_fp8 = q.to(e4m3)
        kv_fp8, scales = _cast_kv_to_fp8(kv, e4m3)

        # Per-row valid KV range [ks, ke). ks=0 (kvcache range start); ke varies
        # so every row has invalid positions [ke, s_k) that clean_logits=True
        # pre-fills with -inf and clean_logits=False does not.
        ks = torch.zeros(s_q, dtype=torch.int32, device=dev)
        ke = torch.randint(s_k // 2, s_k + 1, (s_q,), dtype=torch.int32, device=dev)

        logits_clean = fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=True
        )
        logits_dirty = fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=False
        )
        self.assertEqual(tuple(logits_clean.shape), (s_q, s_k))
        self.assertEqual(tuple(logits_dirty.shape), (s_q, s_k))

        # Sanity: clean_logits=False must actually leave non--inf values at the
        # invalid tail (otherwise the masking is not being exercised).
        skipped_fill = False
        for i in range(s_q):
            end = int(ke[i].item())
            if end < s_k and bool((logits_dirty[i, end:] != float("-inf")).any()):
                skipped_fill = True
                break
        self.assertTrue(
            skipped_fill,
            "clean_logits=False left only -inf at invalid positions; "
            "the redundant-fill skip is not being exercised by this case",
        )

        lengths = (ke - ks).to(torch.int32)
        topk_clean = fast_topk_v2(logits_clean, lengths, topk, row_starts=ks)
        topk_dirty = fast_topk_v2(logits_dirty, lengths, topk, row_starts=ks)

        # The final topk SELECTION (set of valid indices per row; -1 == unfilled)
        # must be identical -- this is what the indexer consumes.
        self._assert_same_selection(topk_clean, topk_dirty)

    def test_skip_logits_clean_topk_equivalence(self):
        # Decode-like (small s_q) and prefill-like (larger s_q) shapes, topk=2048
        # as used by the GLM5 / DeepSeek DSA indexer.
        for s_q, s_k in [(1, 4096), (16, 4096), (128, 4096)]:
            with self.subTest(s_q=s_q, s_k=s_k):
                self._run_case(s_q=s_q, s_k=s_k, num_heads=64, head_dim=128, topk=2048)

    def test_kpool_skip_logits_clean_topk_equivalence(self):
        torch.manual_seed(41)
        dev = "cuda"
        _, e4m3 = get_fp8_dtypes()
        s_q, s_k, num_heads, head_dim = 64, 8192, 32, 128

        q_fp8 = torch.randn(
            s_q, num_heads, head_dim, device=dev, dtype=torch.bfloat16
        ).to(e4m3)
        kv_fp8, scales = _cast_kv_to_fp8(
            torch.randn(s_k, head_dim, device=dev, dtype=torch.bfloat16), e4m3
        )
        weights = torch.randn(s_q, num_heads, device=dev, dtype=torch.float32)
        ks = torch.randint(0, s_k // 4, (s_q,), dtype=torch.int32, device=dev)
        lengths = torch.randint(2049, s_k // 2, (s_q,), dtype=torch.int32, device=dev)
        ke = ks + lengths

        logits_clean = fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=True
        )
        logits_dirty = fp8_mqa_logits(
            q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits=False
        )
        topk_clean = fast_kpool_topk_transform_fused(
            logits_clean,
            lengths,
            kpool=4,
            topk=2048,
            row_starts=ks,
        )
        topk_dirty = fast_kpool_topk_transform_fused(
            logits_dirty,
            lengths,
            kpool=4,
            topk=2048,
            row_starts=ks,
        )
        self._assert_same_selection(topk_clean, topk_dirty)


if __name__ == "__main__":
    unittest.main(verbosity=3)
