"""The V4 paged K-cache dequant against the torch reference, bitwise.

The kernel gathers arbitrary token slots from the paged fp8+ue8m0 cache into a
bf16 workspace; the sparse-prefill dequant dedup reuses its output across a
kv_source group, so any deviation from the reference (masking at the
tokens-per-program tail, slot-to-page math, int64 offsets, workspace-slice
strides) must fail loudly and exactly."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# (num_tokens, page_size): tails of 0/1/3 modulo the kernel's 4 tokens per
# program, both cache page sizes the pool uses, and a many-page gather.
CASES = [
    (1, 64),
    (3, 64),
    (333, 64),
    (256, 256),
    (4097, 256),
]


def _make_cache(num_pages: int, page_size: int, device) -> torch.Tensor:
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES * NOPE_ROPE_BYTES
    return torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
    "needs a CUDA device with fp8 support",
)
class TestDSV4DequantKCachePaged(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"

    def _check(self, num_tokens, page_size, *, table_dtype=torch.int32, out=None):
        num_pages = max(2, (num_tokens + page_size - 1) // page_size + 1)
        cache = _make_cache(num_pages, page_size, self.device)
        table = torch.randint(
            0,
            num_pages * page_size,
            (num_tokens,),
            dtype=table_dtype,
            device=self.device,
        )
        got = dequantize_k_cache_paged(cache, table, page_size, out=out)
        want = dequantize_k_cache_paged_ref(cache, table, page_size)
        torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)
        if out is not None:
            self.assertEqual(got.data_ptr(), out.data_ptr())

    def test_matches_reference(self):
        for num_tokens, page_size in CASES:
            with self.subTest(num_tokens=num_tokens, page_size=page_size):
                self._check(num_tokens, page_size)

    def test_int64_page_table(self):
        self._check(333, 256, table_dtype=torch.int64)

    def test_workspace_slice_output(self):
        # The sparse-prefill path hands the kernel a slice of a larger shared
        # workspace; the kernel must honor out.stride(0) instead of assuming a
        # freshly allocated destination.
        num_tokens, page_size = 777, 256
        workspace = torch.full(
            (num_tokens + 512, 1, 512),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._check(num_tokens, page_size, out=workspace[:num_tokens])
        # Rows past the slice stay untouched.
        self.assertTrue(torch.isnan(workspace[num_tokens:].float()).all())


if __name__ == "__main__":
    unittest.main(verbosity=3)
