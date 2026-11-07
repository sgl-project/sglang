"""Unit tests for QLEN_MASK dispatch in NGRAMWorker and FlashAttentionBackend."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.ngram_worker import _attn_backend_supports_qlen_mask
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FlashAttentionBackend:
    pass


class FlashInferAttnBackend:
    pass


def _extract_full_mask_reference(
    custom_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    offsets = torch.arange(draft_token_num).unsqueeze(0)
    cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
    cum_len = torch.nn.functional.pad(
        torch.cumsum(
            (seq_lens + draft_token_num).repeat_interleave(draft_token_num),
            dim=0,
        ),
        (1, 0),
    )[:-1]
    mask_extraction_indices = (
        cols.repeat_interleave(draft_token_num, dim=0) + cum_len[:, None]
    ).view(1, -1)
    return custom_mask[mask_extraction_indices].view(-1, draft_token_num)


def _extract_qlen_or_full_mask(
    custom_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    pool_size: int = 128,
) -> torch.Tensor:
    d = draft_token_num
    bsz = seq_lens.numel()
    if custom_mask.numel() <= pool_size * d * d:
        if custom_mask.numel() < bsz * d * d:
            custom_mask = custom_mask.as_strided((bsz * d * d,), (1,))
        return custom_mask.view(bsz * d, d)
    return _extract_full_mask_reference(custom_mask, seq_lens, draft_token_num)


class TestNgramQlenMask(CustomTestCase):
    def test_attn_backend_supports_qlen_mask_direct_and_wrapped(self):
        self.assertTrue(_attn_backend_supports_qlen_mask(FlashAttentionBackend()))
        self.assertFalse(_attn_backend_supports_qlen_mask(FlashInferAttnBackend()))
        self.assertFalse(_attn_backend_supports_qlen_mask(None))

        swa_wrapped = SimpleNamespace(full_attn_backend=FlashAttentionBackend())
        self.assertTrue(_attn_backend_supports_qlen_mask(swa_wrapped))

        hybrid_prefill_fa3 = SimpleNamespace(
            prefill_backend=FlashAttentionBackend(),
            decode_backend=FlashInferAttnBackend(),
            spec_attn_is_decode=False,
        )
        self.assertTrue(_attn_backend_supports_qlen_mask(hybrid_prefill_fa3))

        hybrid_decode_fi = SimpleNamespace(
            prefill_backend=FlashAttentionBackend(),
            decode_backend=FlashInferAttnBackend(),
            spec_attn_is_decode=True,
        )
        self.assertFalse(_attn_backend_supports_qlen_mask(hybrid_decode_fi))

    def test_qlen_mask_matches_full_mask_extraction_exact_and_padded(self):
        torch.manual_seed(42)
        d = 8
        max_bs = 16
        preallocated_tree_mask = torch.ones((max_bs * d * d,), dtype=torch.bool)

        raw_bs = 3
        seq_lens_raw = torch.tensor([512, 2048, 4096], dtype=torch.int64)
        tree_raw = torch.randint(0, 2, (raw_bs, d, d), dtype=torch.bool)
        for b in range(raw_bs):
            tree_raw[b].fill_diagonal_(True)

        preallocated_tree_mask[: raw_bs * d * d].copy_(tree_raw.flatten())
        qlen_slice = preallocated_tree_mask[: raw_bs * d * d]

        # Build reference FULL_MASK
        full_parts = []
        for i in range(raw_bs):
            ones = torch.ones((d, int(seq_lens_raw[i])), dtype=torch.bool)
            full_parts.append(torch.cat((ones, tree_raw[i]), dim=1).flatten())
        full_mask = torch.cat(full_parts, dim=0)

        # 1. Unpadded batch (bsz == raw_bs)
        ref_unpadded = _extract_full_mask_reference(full_mask, seq_lens_raw, d)
        got_unpadded = _extract_qlen_or_full_mask(qlen_slice, seq_lens_raw, d)
        self.assertTrue(torch.equal(ref_unpadded, got_unpadded))

        # 2. CUDA graph padded batch (padded_bs = 4 > raw_bs = 3)
        seq_lens_padded = torch.tensor([512, 2048, 4096, 1], dtype=torch.int64)
        got_padded = _extract_qlen_or_full_mask(qlen_slice, seq_lens_padded, d)
        self.assertEqual(got_padded.shape, (4 * d, d))
        self.assertTrue(torch.equal(got_padded[: raw_bs * d], ref_unpadded))
        self.assertTrue(torch.all(got_padded[raw_bs * d :]))


if __name__ == "__main__":
    unittest.main()
