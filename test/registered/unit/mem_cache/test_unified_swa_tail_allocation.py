import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestUnifiedSWATailAllocation(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", enable_unified_memory=True), role="tokenizer"
        )

    def test_extend_binds_only_new_tail_pages(self):
        """PD tail allocation must leave new FULL-only pages unbound in SWA,
        while preserving an existing partial page and binding the trailing KV."""
        for page_size in (4, 16):
            for prefix_len, seq_len, tail_len in (
                (page_size, 5 * page_size, 0),
                (page_size, 5 * page_size, page_size),
                (page_size, 5 * page_size, 2 * page_size),
                (page_size + 2, 5 * page_size, page_size),
                (page_size + 2, 5 * page_size, 4 * page_size - 2),
                (page_size + 2, 2 * page_size - 1, page_size - 3),
                (page_size + 2, 5 * page_size - 1, page_size),
                (0, None, 1),
            ):
                with self.subTest(
                    page_size=page_size,
                    prefix_len=prefix_len,
                    seq_len=seq_len,
                    tail_len=tail_len,
                ):
                    bundle = init_unified_swa_pools(
                        device="cuda",
                        kv_cache_dtype=torch.float16,
                        head_num=1,
                        head_dim=8,
                        v_head_dim=8,
                        swa_head_num=1,
                        swa_head_dim=8,
                        swa_v_head_dim=8,
                        page_size=page_size,
                        start_layer=0,
                        end_layer=2,
                        full_attention_layer_ids=[0],
                        swa_attention_layer_ids=[1],
                        total_bytes=1 << 16,
                        enable_memory_saver=False,
                        need_sort=False,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    if seq_len is None:
                        seq_len = allocator.available_size() + page_size
                        self.assertFalse(allocator.can_reserve(seq_len, seq_len))
                    prefix_capacity = -(-prefix_len // page_size) * page_size
                    prefix = allocator.alloc(prefix_capacity)[:prefix_len]
                    prefix_swa = allocator.translate_swa_indices_for_transfer(
                        prefix
                    ).clone()
                    prefix_cpu = torch.tensor([prefix_len], dtype=torch.int64)
                    seq_cpu = torch.tensor([seq_len], dtype=torch.int64)
                    extended = allocator.alloc_extend_swa_tail(
                        prefix_lens=prefix_cpu.cuda(),
                        prefix_lens_cpu=prefix_cpu,
                        seq_lens=seq_cpu.cuda(),
                        seq_lens_cpu=seq_cpu,
                        last_loc=(
                            prefix[-1:]
                            if prefix_len
                            else torch.tensor([-1], device="cuda")
                        ),
                        extend_num_tokens=seq_len - prefix_len,
                        swa_tail_len=tail_len,
                    )
                    self.assertIsNotNone(extended)
                    self.assertEqual(extended.numel(), seq_len - prefix_len)
                    tokens = torch.cat((prefix, extended))
                    full_phys = allocator.translate_kv_indices_for_transfer(tokens)
                    swa_phys = allocator.translate_swa_indices_for_transfer(tokens)
                    self.assertTrue(bool((full_phys > 0).all()))
                    self.assertTrue(torch.equal(swa_phys[:prefix_len], prefix_swa))
                    tail_start = seq_len - tail_len
                    new_pages = torch.unique(tokens[prefix_capacity:] // page_size)
                    tail_pages = torch.unique(tokens[tail_start:] // page_size)
                    full_only_pages = new_pages[~torch.isin(new_pages, tail_pages)]
                    self.assertTrue(
                        bool(
                            (allocator.swa_v2p_page_table[full_only_pages] == -1).all()
                        )
                    )
                    if tail_len:
                        pages = allocator.swa_v2p_page_table[
                            tokens[tail_start:] // page_size
                        ]
                        self.assertTrue(bool((pages > 0).all()))
                        expected = pages * page_size + tokens[tail_start:] % page_size
                        self.assertTrue(torch.equal(swa_phys[tail_start:], expected))
                    self.assertEqual(
                        allocator.swa_attn_allocator.allocated_count(),
                        prefix_capacity
                        + torch.isin(new_pages, tail_pages).sum().item() * page_size,
                    )
                    if prefix_len < prefix_capacity:
                        reused_tokens = min(prefix_capacity, seq_len) - prefix_len
                        self.assertTrue(
                            torch.equal(
                                extended[:reused_tokens],
                                prefix[-1]
                                + torch.arange(1, reused_tokens + 1, device="cuda"),
                            )
                        )
                    allocator.free(tokens)
                    self.assertEqual(allocator.full_attn_allocator.allocated_count(), 0)
                    self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)


if __name__ == "__main__":
    unittest.main()
