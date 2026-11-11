import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSWADecodeCapacity(unittest.TestCase):
    def test_decode_capacity_preserves_both_pools(self):
        for full_pages, swa_pages in [(16, 8), (8, 16)]:
            for cross_page in [False, True]:
                with self.subTest(
                    full_pages=full_pages, swa_pages=swa_pages, cross_page=cross_page
                ):
                    page_size = 128
                    pool = SWAKVPool(
                        size=full_pages * page_size,
                        size_swa=swa_pages * page_size,
                        page_size=page_size,
                        dtype=torch.bfloat16,
                        head_num=1,
                        head_dim=8,
                        swa_attention_layer_ids=[0],
                        full_attention_layer_ids=[1],
                        device="cpu",
                    )
                    allocator = SWATokenToKVPoolAllocator(
                        size=full_pages * page_size,
                        size_swa=swa_pages * page_size,
                        page_size=page_size,
                        dtype=torch.bfloat16,
                        device="cpu",
                        kvcache=pool,
                        need_sort=False,
                    )
                    length = 8 * page_size - int(not cross_page)
                    prefix = torch.tensor([0], dtype=torch.int64)
                    seq_lens = torch.tensor([length], dtype=torch.int64)
                    held = allocator.alloc_extend(
                        prefix, prefix, seq_lens, seq_lens, prefix - 1, length
                    )
                    self.assertIsNotNone(held)
                    mapping = allocator.full_to_swa_index_mapping.clone()
                    before = (
                        allocator.full_available_size(),
                        allocator.swa_available_size(),
                    )

                    result = allocator.alloc_decode(
                        seq_lens + 1, seq_lens + 1, held[-1:]
                    )

                    self.assertEqual(
                        (
                            allocator.full_available_size(),
                            allocator.swa_available_size(),
                        ),
                        before,
                    )
                    torch.testing.assert_close(
                        allocator.full_to_swa_index_mapping[held], mapping[held]
                    )
                    if cross_page:
                        self.assertIsNone(result)
                        torch.testing.assert_close(
                            allocator.full_to_swa_index_mapping, mapping
                        )
                    else:
                        self.assertIsNotNone(result)
                        self.assertEqual(result.item(), held[-1].item() + 1)
                        self.assertGreater(
                            allocator.full_to_swa_index_mapping[result].item(), 0
                        )


if __name__ == "__main__":
    unittest.main()
