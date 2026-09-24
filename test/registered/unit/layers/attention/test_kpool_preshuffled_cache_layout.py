"""The k-pool reader must see the index-K cache the way the non-pooled writer lays it out."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsa.index_buf_accessor import SetKAndS, _is_fp8_fnuz
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    gather_index_k_scale_prefix_into,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE_SIZE = 64


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestKpoolPreshuffledCacheLayout(CustomTestCase):
    def test_gather_reads_back_what_set_k_and_s_wrote(self):
        """A layout mismatch between the two writers never raises; it silently
        returns the wrong top-k."""
        torch.manual_seed(0)
        pool = SimpleNamespace(page_size=PAGE_SIZE)
        num_pages, seq_len = 2, PAGE_SIZE + 5
        buf = torch.zeros(
            num_pages,
            PAGE_SIZE * (INDEX_HEAD_DIM + 4),
            dtype=torch.uint8,
            device="cuda",
        )
        k = torch.randint(
            1, 127, (seq_len, INDEX_HEAD_DIM), dtype=torch.uint8, device="cuda"
        )
        scale = torch.randn(seq_len, dtype=torch.float32, device="cuda")
        fp8_dtype = torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
        SetKAndS.triton(
            pool, buf, torch.arange(seq_len, device="cuda"), k.view(fp8_dtype), scale
        )

        k_out = torch.zeros_like(k)
        scale_out = torch.zeros_like(scale)
        gather_index_k_scale_prefix_into(
            pool,
            buf,
            torch.arange(num_pages, dtype=torch.int32, device="cuda"),
            seq_len,
            k_out,
            scale_out,
        )
        torch.testing.assert_close(k_out, k, atol=0, rtol=0)
        torch.testing.assert_close(scale_out, scale, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
