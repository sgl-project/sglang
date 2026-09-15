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
            for prefix_len, tail_pages in (
                (page_size, 0),
                (page_size, 1),
                (page_size, 2),
                (page_size + 2, 1),
            ):
                with self.subTest(
                    page_size=page_size, prefix_len=prefix_len, tail_pages=tail_pages
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
                    prefix_capacity = -(-prefix_len // page_size) * page_size
                    prefix = allocator.alloc(prefix_capacity)[:prefix_len]
                    prefix_swa = allocator.translate_swa_indices_for_transfer(
                        prefix
                    ).clone()
                    seq_len = 5 * page_size
                    prefix_cpu = torch.tensor([prefix_len], dtype=torch.int64)
                    seq_cpu = torch.tensor([seq_len], dtype=torch.int64)
                    extended = allocator.alloc_extend_swa_tail(
                        prefix_lens=prefix_cpu.cuda(),
                        prefix_lens_cpu=prefix_cpu,
                        seq_lens=seq_cpu.cuda(),
                        seq_lens_cpu=seq_cpu,
                        last_loc=prefix[-1:],
                        extend_num_tokens=seq_len - prefix_len,
                        swa_tail_len=tail_pages * page_size,
                    )
                    self.assertIsNotNone(extended)
                    self.assertEqual(extended.numel(), seq_len - prefix_len)
                    tokens = torch.cat((prefix, extended))
                    full_phys = allocator.translate_kv_indices_for_transfer(tokens)
                    swa_phys = allocator.translate_swa_indices_for_transfer(tokens)
                    self.assertTrue(bool((full_phys > 0).all()))
                    self.assertTrue(torch.equal(swa_phys[:prefix_len], prefix_swa))
                    tail_start = seq_len - tail_pages * page_size
                    full_only_pages = (
                        tokens[prefix_capacity:tail_start:page_size] // page_size
                    )
                    self.assertTrue(
                        bool(
                            (allocator.swa_v2p_page_table[full_only_pages] == -1).all()
                        )
                    )
                    self.assertTrue(
                        bool((swa_phys[prefix_capacity:tail_start] == 0).all())
                    )
                    if tail_pages:
                        pages = allocator.swa_v2p_page_table[
                            tokens[tail_start::page_size] // page_size
                        ]
                        self.assertTrue(bool((pages > 0).all()))
                        expected = (
                            pages[:, None] * page_size
                            + torch.arange(page_size, device="cuda")
                        ).flatten()
                        self.assertTrue(torch.equal(swa_phys[tail_start:], expected))
                    if prefix_len < prefix_capacity:
                        self.assertTrue(
                            torch.equal(
                                extended[: prefix_capacity - prefix_len],
                                prefix[-1]
                                + torch.arange(
                                    1, prefix_capacity - prefix_len + 1, device="cuda"
                                ),
                            )
                        )


if __name__ == "__main__":
    unittest.main()
