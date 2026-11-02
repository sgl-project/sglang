"""MLA prefix compaction across partially filled request pages.

CPU: python -m pytest test/registered/unit/layers/attention/test_npu_mla_prefix_cache.py
NPU: run the same command with SGLANG_TEST_NPU=1 in an Ascend environment.
"""

import os
import unittest

import torch

from sglang.srt.hardware_backend.npu.attention.mla_cache import (
    build_mla_prefix_token_indices,
    gather_mla_cache_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DEVICES = ["cpu"]
if os.environ.get("SGLANG_TEST_NPU") == "1":
    import torch_npu  # noqa: F401

    DEVICES.append("npu")


class TestNpuMlaPrefixCache(unittest.TestCase):
    def check_gather(self, lengths, page_size, head_dim, dtype, device, is_nz):
        counts = [(length + page_size - 1) // page_size for length in lengths]
        total_pages = sum(counts)
        # Repeat a shared block and visit physical pages out of order.
        block_ids = [3, 1, 3, 0, 6, 2, 5, 4][:total_pages]
        pages = torch.arange(8 * page_size * head_dim, dtype=torch.float32)
        pages = pages.remainder(1009).reshape(8, page_size, 1, head_dim).to(dtype)
        expected_parts = []
        block_start = 0
        for length, count in zip(lengths, counts, strict=True):
            request_pages = block_ids[block_start : block_start + count]
            expected_parts.append(pages[request_pages].flatten(0, 1)[:length])
            block_start += count
        expected = torch.cat(expected_parts)

        if is_nz:
            storage = (
                pages.view(8, page_size, head_dim // 16, 16)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view_as(pages)
            )
        else:
            # Also cover a noncontiguous ND view, as in sliced cache buffers.
            backing = torch.empty(8, page_size, 1, head_dim * 2, dtype=dtype)
            storage = backing[..., ::2]
            storage.copy_(pages)

        indices = build_mla_prefix_token_indices(
            lengths, page_size, device=torch.device(device)
        )
        actual = gather_mla_cache_tokens(
            storage.to(device),
            torch.tensor(block_ids, device=device, dtype=torch.int32),
            indices,
            is_nz=is_nz,
        )
        self.assertEqual(actual.shape, (sum(lengths), 1, head_dim))
        self.assertEqual(indices.dtype, torch.int64)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_partial_middle_page_and_shared_blocks(self):
        for device in DEVICES:
            for is_nz in (False, True):
                for lengths in ([3, 5, 1], [4, 3, 5], [5, 4, 3]):
                    with self.subTest(device=device, is_nz=is_nz, lengths=lengths):
                        self.check_gather(lengths, 4, 32, torch.float32, device, is_nz)

    def test_zero_and_aligned_prefixes(self):
        for device in DEVICES:
            for is_nz in (False, True):
                for lengths in ([0, 1, 3], [4, 0, 8], [0, 0]):
                    with self.subTest(device=device, is_nz=is_nz, lengths=lengths):
                        self.check_gather(lengths, 4, 32, torch.float32, device, is_nz)

    def test_real_page_and_latent_rope_dimensions(self):
        for device in DEVICES:
            for is_nz in (False, True):
                for head_dim in (64, 512):
                    with self.subTest(device=device, is_nz=is_nz, head_dim=head_dim):
                        self.check_gather(
                            [128, 255, 129],
                            128,
                            head_dim,
                            torch.bfloat16,
                            device,
                            is_nz,
                        )

    def test_indices_skip_each_request_tail(self):
        indices = build_mla_prefix_token_indices(
            [3, 5, 1], 4, device=torch.device("cpu")
        )
        self.assertEqual(indices.tolist(), [0, 1, 2, 4, 5, 6, 7, 8, 12])


if __name__ == "__main__":
    unittest.main()
