import unittest

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
    fused_k_norm_rope_flashmla,
    fused_store_cache,
    fused_store_cache_shared,
)
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDeepSeekV4SharedOwnerWrite(CustomTestCase):

    def test_direct_store_reassembles_base_pages_at_cp8(self):
        page_size = 2
        owner_size = 8
        logical_pages = torch.arange(16, dtype=torch.int64, device="cuda")
        variants = (
            ("flashmla", 512, torch.int64, ((584 * page_size + 575) // 576) * 576),
            ("indexer", 128, torch.int32, 132 * page_size),
        )

        for name, head_dim, index_dtype, page_bytes in variants:
            with self.subTest(name=name):
                torch.manual_seed(20260721)
                values = torch.randn(
                    (logical_pages.numel(), head_dim),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                indices = (logical_pages * page_size).to(index_dtype)
                base = torch.zeros(
                    (logical_pages.numel(), page_bytes),
                    dtype=torch.uint8,
                    device="cuda",
                )
                fused_store_cache(
                    values,
                    base,
                    indices,
                    page_size=page_size,
                    type=name,
                )

                owners = []
                for owner_rank in range(owner_size):
                    cache = torch.zeros(
                        (logical_pages.numel() // owner_size, page_bytes),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    fused_store_cache_shared(
                        values,
                        cache,
                        indices,
                        page_size=page_size,
                        type=name,
                        owner_rank=owner_rank,
                        owner_size=owner_size,
                    )
                    owners.append(cache.cpu())

                base = base.cpu()
                for logical_page in logical_pages.tolist():
                    owner_rank = logical_page % owner_size
                    local_page = logical_page // owner_size
                    self.assertTrue(
                        torch.equal(base[logical_page], owners[owner_rank][local_page])
                    )

    def test_compressor_owner_write_ignores_negative_output_sentinel(self):
        page_size = 2
        compress_ratio = 4
        plan = CompressorDecodePlan.generate_legacy(
            compress_ratio,
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            torch.tensor([compress_ratio], dtype=torch.int64, device="cuda"),
        )
        loc = torch.tensor([-1], dtype=torch.int64, device="cuda")
        freqs_cis = precompute_freqs_cis(64, compress_ratio + 1, 0, 10000, 1, 32, 1).to(
            "cuda"
        )

        variants = (
            ("indexer_fp8", 128, False, 132 * page_size),
            ("indexer_fp4", 128, True, 68 * page_size),
            (
                "flashmla_fp8",
                512,
                False,
                ((584 * page_size + 575) // 576) * 576,
            ),
        )
        for name, head_dim, use_fp4, page_bytes in variants:
            with self.subTest(name=name):
                values = torch.ones((1, head_dim), dtype=torch.bfloat16, device="cuda")
                weight = torch.ones((head_dim,), dtype=torch.bfloat16, device="cuda")
                cache = torch.zeros((1, page_bytes), dtype=torch.uint8, device="cuda")

                compress_norm_rope_store(
                    values,
                    plan,
                    norm_weight=weight,
                    norm_eps=1.0e-6,
                    freq_cis=freqs_cis,
                    out_loc=loc,
                    kvcache=cache,
                    page_size=page_size,
                    use_fp4=use_fp4,
                    owner_rank=7,
                    owner_size=8,
                )

                self.assertEqual(torch.count_nonzero(cache).item(), 0)

    def test_flashmla_owner_write_maps_page_to_local_page(self):
        page_size = 2
        page_bytes = ((584 * page_size + 575) // 576) * 576
        indices = torch.tensor([0, 2, 4, 6, -1], dtype=torch.int64, device="cuda")
        values = torch.ones((indices.numel(), 512), dtype=torch.bfloat16, device="cuda")

        owner_caches = []
        for owner_rank in range(2):
            cache = torch.zeros((2, page_bytes), dtype=torch.uint8, device="cuda")
            fused_store_cache_shared(
                values,
                cache,
                indices,
                page_size=page_size,
                type="flashmla",
                owner_rank=owner_rank,
                owner_size=2,
            )
            owner_caches.append(cache.cpu())

        self.assertTrue(torch.count_nonzero(owner_caches[0]).item() > 0)
        self.assertTrue(torch.equal(owner_caches[0], owner_caches[1]))

    def test_indexer_owner_write_maps_page_to_local_page(self):
        page_size = 2
        indices = torch.tensor([0, 2, 4, 6, -1], dtype=torch.int32, device="cuda")
        values = torch.ones((indices.numel(), 128), dtype=torch.bfloat16, device="cuda")

        owner_caches = []
        for owner_rank in range(2):
            cache = torch.zeros((2, 132 * page_size), dtype=torch.uint8, device="cuda")
            fused_store_cache_shared(
                values,
                cache,
                indices,
                page_size=page_size,
                type="indexer",
                owner_rank=owner_rank,
                owner_size=2,
            )
            owner_caches.append(cache.cpu())

        self.assertTrue(torch.count_nonzero(owner_caches[0]).item() > 0)
        self.assertTrue(torch.equal(owner_caches[0], owner_caches[1]))

    def test_v2_compressor_store_matches_unsharded_indexer_cache(self):
        page_size = 2
        num_pages = 4
        num_tokens = num_pages
        compress_ratio = 4
        torch.manual_seed(20260718)
        values = torch.randn((num_tokens, 128), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((128,), dtype=torch.bfloat16, device="cuda")
        seq_lens = (
            torch.arange(1, num_tokens + 1, dtype=torch.int64, device="cuda")
            * compress_ratio
        )
        req_pool_indices = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        plan = CompressorDecodePlan.generate_legacy(
            compress_ratio, req_pool_indices, seq_lens
        )
        loc = torch.arange(
            0, num_pages * page_size, page_size, dtype=torch.int64, device="cuda"
        )
        freqs_cis = precompute_freqs_cis(
            64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
        ).to("cuda")

        base = torch.zeros(
            (num_pages, 132 * page_size), dtype=torch.uint8, device="cuda"
        )
        compress_norm_rope_store(
            values,
            plan,
            norm_weight=weight,
            norm_eps=1.0e-6,
            freq_cis=freqs_cis,
            out_loc=loc,
            kvcache=base,
            page_size=page_size,
        )

        owners = []
        for owner_rank in range(2):
            cache = torch.zeros(
                (num_pages // 2, 132 * page_size),
                dtype=torch.uint8,
                device="cuda",
            )
            compress_norm_rope_store(
                values,
                plan,
                norm_weight=weight,
                norm_eps=1.0e-6,
                freq_cis=freqs_cis,
                out_loc=loc,
                kvcache=cache,
                page_size=page_size,
                owner_rank=owner_rank,
                owner_size=2,
            )
            owners.append(cache.cpu())

        base = base.cpu()
        for logical_page in range(num_pages):
            owner = logical_page % 2
            local_page = logical_page // 2
            self.assertTrue(torch.equal(base[logical_page], owners[owner][local_page]))

    def test_v2_c4_store_reassembles_pages_one_through_eight_at_cp8(self):
        page_size = 64
        owner_size = 8
        logical_pages = torch.arange(1, 9, dtype=torch.int64, device="cuda")
        num_tokens = logical_pages.numel()
        compress_ratio = 4
        torch.manual_seed(20260720)
        values = torch.randn((num_tokens, 512), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((512,), dtype=torch.bfloat16, device="cuda")
        seq_lens = (
            torch.arange(1, num_tokens + 1, dtype=torch.int64, device="cuda")
            * compress_ratio
        )
        req_pool_indices = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        plan = CompressorDecodePlan.generate_legacy(
            compress_ratio, req_pool_indices, seq_lens
        )
        loc = logical_pages * page_size
        freqs_cis = precompute_freqs_cis(
            64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
        ).to("cuda")
        page_bytes = ((584 * page_size + 575) // 576) * 576

        base = torch.zeros((9, page_bytes), dtype=torch.uint8, device="cuda")
        compress_norm_rope_store(
            values,
            plan,
            norm_weight=weight,
            norm_eps=1.0e-6,
            freq_cis=freqs_cis,
            out_loc=loc,
            kvcache=base,
            page_size=page_size,
        )

        owners = []
        for owner_rank in range(owner_size):
            cache = torch.zeros((2, page_bytes), dtype=torch.uint8, device="cuda")
            compress_norm_rope_store(
                values,
                plan,
                norm_weight=weight,
                norm_eps=1.0e-6,
                freq_cis=freqs_cis,
                out_loc=loc,
                kvcache=cache,
                page_size=page_size,
                owner_rank=owner_rank,
                owner_size=owner_size,
            )
            owners.append(cache.cpu())

        base = base.cpu()
        for logical_page in range(1, 9):
            owner = logical_page % owner_size
            local_page = logical_page // owner_size
            self.assertTrue(torch.equal(base[logical_page], owners[owner][local_page]))

    def test_main_fused_store_matches_unsharded_swa_cache(self):
        page_size = 2
        num_pages = 4
        num_tokens = num_pages
        torch.manual_seed(20260719)
        values = torch.randn((num_tokens, 512), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((512,), dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        loc = torch.arange(
            0, num_pages * page_size, page_size, dtype=torch.int32, device="cuda"
        )
        freqs_cis = precompute_freqs_cis(64, num_tokens + 1, 0, 10000, 1, 32, 1).to(
            "cuda"
        )
        page_bytes = ((584 * page_size + 575) // 576) * 576

        base = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device="cuda")
        fused_k_norm_rope_flashmla(
            values,
            weight,
            1.0e-6,
            freqs_cis,
            positions,
            loc,
            base,
            page_size,
        )

        owners = []
        for owner_rank in range(2):
            cache = torch.zeros(
                (num_pages // 2, page_bytes), dtype=torch.uint8, device="cuda"
            )
            fused_k_norm_rope_flashmla(
                values,
                weight,
                1.0e-6,
                freqs_cis,
                positions,
                loc,
                cache,
                page_size,
                owner_rank=owner_rank,
                owner_size=2,
            )
            owners.append(cache.cpu())

        base = base.cpu()
        for logical_page in range(num_pages):
            owner = logical_page % 2
            local_page = logical_page // 2
            self.assertTrue(torch.equal(base[logical_page], owners[owner][local_page]))


if __name__ == "__main__":
    unittest.main()
