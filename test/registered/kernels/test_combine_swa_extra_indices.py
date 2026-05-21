"""Unit tests for the ``combine_swa_extra_indices`` Triton kernel used by
the DSv4 sparse-prefill path (``flash_mla.flash_mla_sparse_fwd``).

The kernel merges per-token SWA and extra (compressed) topk indices into a
single fixed-size buffer suitable for ``flash_mla_sparse_fwd``:

    combined[i, 0, :swa_len]                    = swa_indices[i, 0, :swa_len]
    combined[i, 0, swa_len:swa_len+extra_len]   = extra_indices[i, 0, :extra_len] + s_kv_swa
    combined[i, 0, swa_len+extra_len:]          = -1  (padding)
    combined_lens[i]                            = swa_len + extra_len

Tests compare the kernel output against a straightforward PyTorch
reference implementation across a few shape/length regimes (typical
prefill, the large-q regime that motivated the kernel, full lengths, and
the all-zero-length edge case).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _reference_combine_swa_extra_indices(
    swa_indices: torch.Tensor,
    swa_topk_lens: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_topk_lens: torch.Tensor,
    s_kv_swa: int,
    alignment: int = 128,
):
    """Plain-PyTorch reference implementation for cross-checking the
    Triton kernel.

    Same contract as :func:`combine_swa_extra_indices`; uses a small Python
    loop over tokens, which is fine for unit-test scale.
    """
    num_tokens, _, topk_swa = swa_indices.shape
    _, _, topk_extra = extra_indices.shape
    raw_combined = topk_swa + topk_extra
    combined_topk = (raw_combined + alignment - 1) // alignment * alignment

    combined = torch.full(
        (num_tokens, 1, combined_topk),
        -1,
        dtype=torch.int32,
        device=swa_indices.device,
    )
    combined_lens = (swa_topk_lens + extra_topk_lens).to(torch.int32)

    for i in range(num_tokens):
        sl = int(swa_topk_lens[i].item())
        el = int(extra_topk_lens[i].item())
        if sl > 0:
            combined[i, 0, :sl] = swa_indices[i, 0, :sl]
        if el > 0:
            combined[i, 0, sl : sl + el] = extra_indices[i, 0, :el] + s_kv_swa

    return combined, combined_lens


class TestCombineSwaExtraIndices(CustomTestCase):
    """Verify ``combine_swa_extra_indices`` matches the Python reference
    across representative input shapes and length distributions."""

    def setUp(self):
        # Lazy import: the module pulls in triton which requires CUDA.
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            combine_swa_extra_indices,
        )

        self.combine = combine_swa_extra_indices

    def _make_inputs(
        self,
        *,
        num_tokens: int,
        topk_swa: int,
        topk_extra: int,
        s_kv_swa: int,
        s_kv_extra: int,
        variable_lengths: bool,
        seed: int,
        device: str = "cuda",
    ):
        gen = torch.Generator(device=device).manual_seed(seed)
        swa_indices = torch.randint(
            0,
            max(s_kv_swa, 1),
            (num_tokens, 1, topk_swa),
            generator=gen,
            dtype=torch.int32,
            device=device,
        )
        extra_indices = torch.randint(
            0,
            max(s_kv_extra, 1),
            (num_tokens, 1, topk_extra),
            generator=gen,
            dtype=torch.int32,
            device=device,
        )
        if variable_lengths:
            # Random valid prefix length in [0, topk_*]. Hits both extremes.
            swa_lens = torch.randint(
                0,
                topk_swa + 1,
                (num_tokens,),
                generator=gen,
                dtype=torch.int32,
                device=device,
            )
            extra_lens = torch.randint(
                0,
                topk_extra + 1,
                (num_tokens,),
                generator=gen,
                dtype=torch.int32,
                device=device,
            )
        else:
            swa_lens = torch.full(
                (num_tokens,), topk_swa, dtype=torch.int32, device=device
            )
            extra_lens = torch.full(
                (num_tokens,), topk_extra, dtype=torch.int32, device=device
            )

        # Stamp -1 past the valid prefix to assert the kernel doesn't peek
        # past the declared length. Use a simple mask construction so we
        # stay vectorised.
        swa_pos = torch.arange(topk_swa, device=device, dtype=torch.int32)
        extra_pos = torch.arange(topk_extra, device=device, dtype=torch.int32)
        swa_mask = swa_pos.unsqueeze(0) < swa_lens.unsqueeze(1)
        extra_mask = extra_pos.unsqueeze(0) < extra_lens.unsqueeze(1)
        swa_indices.masked_fill_(~swa_mask.unsqueeze(1), -1)
        extra_indices.masked_fill_(~extra_mask.unsqueeze(1), -1)

        return swa_indices, swa_lens, extra_indices, extra_lens

    def _run_case(
        self,
        *,
        num_tokens: int,
        topk_swa: int,
        topk_extra: int,
        s_kv_swa: int = 200_000,
        s_kv_extra: int = 100_000,
        variable_lengths: bool = True,
        seed: int = 0,
    ):
        swa, swa_lens, extra, extra_lens = self._make_inputs(
            num_tokens=num_tokens,
            topk_swa=topk_swa,
            topk_extra=topk_extra,
            s_kv_swa=s_kv_swa,
            s_kv_extra=s_kv_extra,
            variable_lengths=variable_lengths,
            seed=seed,
        )

        out_combined, out_lens = self.combine(
            swa_indices=swa,
            swa_topk_lens=swa_lens,
            extra_indices=extra,
            extra_topk_lens=extra_lens,
            s_kv_swa=s_kv_swa,
        )
        ref_combined, ref_lens = _reference_combine_swa_extra_indices(
            swa, swa_lens, extra, extra_lens, s_kv_swa
        )

        # Exact match expected -- the kernel only does loads/stores and an
        # integer add for the extra offset.
        torch.testing.assert_close(out_lens, ref_lens, rtol=0, atol=0)
        torch.testing.assert_close(out_combined, ref_combined, rtol=0, atol=0)

    def test_small_inputs(self):
        """Tiny sanity check (covers Python-side path)."""
        self._run_case(num_tokens=4, topk_swa=64, topk_extra=128, seed=0)

    def test_typical_dsv4_sizes(self):
        """Shapes representative of DSv4-Flash compress_ratio=4 prefill."""
        self._run_case(num_tokens=1024, topk_swa=512, topk_extra=2048, seed=1)

    def test_large_q_above_smem_cap(self):
        """The regime that motivated the kernel: ``b == num_tokens`` past
        the FlashMLA sparse_decode smem cap (11673), so the decode kernel
        would crash and we fall back to ``sparse_fwd``."""
        self._run_case(num_tokens=16384, topk_swa=512, topk_extra=2048, seed=2)

    def test_uniform_full_lengths(self):
        """All tokens at the max valid count -- exercises the upper-bound
        of the ``swa_len + extra_off`` write position."""
        self._run_case(
            num_tokens=128,
            topk_swa=512,
            topk_extra=512,
            variable_lengths=False,
            seed=3,
        )

    def test_zero_lengths(self):
        """All-zero-length edge case: no valid entries on either side, so
        the entire combined buffer should remain at the -1 fill and lens
        should be 0."""
        device = "cuda"
        num_tokens = 16
        topk_swa, topk_extra = 64, 64
        swa = torch.full(
            (num_tokens, 1, topk_swa), -1, dtype=torch.int32, device=device
        )
        extra = torch.full(
            (num_tokens, 1, topk_extra), -1, dtype=torch.int32, device=device
        )
        swa_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        extra_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)

        out_combined, out_lens = self.combine(
            swa_indices=swa,
            swa_topk_lens=swa_lens,
            extra_indices=extra,
            extra_topk_lens=extra_lens,
            s_kv_swa=100,
        )
        self.assertTrue(torch.all(out_lens == 0))
        self.assertTrue(torch.all(out_combined == -1))

    def test_extra_offset_is_added(self):
        """Verifies that ``extra_indices`` are actually shifted by
        ``s_kv_swa`` in the combined buffer (the kernel's one real
        arithmetic op, easy to break)."""
        device = "cuda"
        # One token, swa_len=0 so the combined buffer's prefix is entirely
        # populated from extra. extra entry 7 should land at position 0
        # of combined and have value (7 + s_kv_swa).
        swa = torch.full((1, 1, 64), -1, dtype=torch.int32, device=device)
        extra = torch.zeros((1, 1, 64), dtype=torch.int32, device=device)
        extra[0, 0, 0] = 7
        swa_lens = torch.zeros(1, dtype=torch.int32, device=device)
        extra_lens = torch.ones(1, dtype=torch.int32, device=device)

        s_kv_swa = 5_000
        out_combined, out_lens = self.combine(
            swa_indices=swa,
            swa_topk_lens=swa_lens,
            extra_indices=extra,
            extra_topk_lens=extra_lens,
            s_kv_swa=s_kv_swa,
        )
        self.assertEqual(int(out_lens[0].item()), 1)
        self.assertEqual(int(out_combined[0, 0, 0].item()), 7 + s_kv_swa)
        self.assertTrue(torch.all(out_combined[0, 0, 1:] == -1))


if __name__ == "__main__":
    unittest.main(verbosity=3)
