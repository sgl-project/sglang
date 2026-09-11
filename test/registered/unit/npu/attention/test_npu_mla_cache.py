"""CPU checks for logical prefix reads from ND and PA-NZ MLA cache pages."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMLACachePrefixRead(CustomTestCase):
    def test_page_order_and_layout(self):
        # Both latent and RoPE buffers must preserve every token's features.
        for page_size in (16, 128):
            for head_dim in (64, 512):
                logical = torch.arange(4 * page_size * head_dim).reshape(
                    4, page_size, 1, head_dim
                )
                packed = (
                    logical.reshape(4, page_size, head_dim // 16, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape_as(logical)
                )
                for is_nz, cache in ((False, logical), (True, packed)):
                    for selected in ([3, 1, 3, 0], [], [2]):
                        with self.subTest(
                            page_size=page_size,
                            head_dim=head_dim,
                            is_nz=is_nz,
                            selected=selected,
                        ):
                            ids = torch.tensor(selected, dtype=torch.int32)
                            actual = gather_mla_cache_pages(cache, ids, is_nz=is_nz)
                            expected = logical[selected]
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertTrue(torch.equal(actual, expected))

    def test_partial_page_writes_preserve_prefix_features(self):
        # Populate NZ storage by scalar coordinates, independently of the
        # reshape/permute used by the reader. Leave unwritten slots sentinel-filled.
        page_size = 128
        slots = (128, 129, 255, 256, 383, 511)
        for head_dim in (64, 512):
            with self.subTest(head_dim=head_dim):
                tiles = head_dim // 16
                cache = torch.full((4, page_size, 1, head_dim), -1, dtype=torch.int64)
                logical = torch.full_like(cache, -1)
                flat = cache.view(-1)
                for slot in slots:
                    for dim in range(head_dim):
                        value = slot * 1000 + dim
                        logical[slot // page_size, slot % page_size, 0, dim] = value
                        offset = (
                            ((slot // page_size) * tiles + dim // 16) * page_size
                            + slot % page_size
                        ) * 16 + dim % 16
                        flat[offset] = value
                ids = torch.tensor([3, 1, 2, 1], dtype=torch.int64)
                actual = gather_mla_cache_pages(cache, ids, is_nz=True)
                self.assertTrue(torch.equal(actual, logical[[3, 1, 2, 1]]))
                # The former raw page gather must fail for this regression case.
                self.assertFalse(torch.equal(cache[ids], actual))


if __name__ == "__main__":
    unittest.main()
