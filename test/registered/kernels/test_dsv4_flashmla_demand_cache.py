import importlib.util
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.srt.mem_cache.deepseek_v4_shared import (
    DSV4_MODEL1_DEMAND_CACHE_ROW_BYTES,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-c", runner_config="4-gpu-h100")

_FLASH_MLA_AVAILABLE = (
    importlib.util.find_spec("sgl_kernel") is not None
    and importlib.util.find_spec("sgl_kernel.flash_mla") is not None
)


def _logical_rows(page_size: int) -> torch.Tensor:
    if page_size == 2:
        return torch.arange(64, dtype=torch.int64, device="cuda")
    return torch.cat(
        tuple(
            torch.arange(
                logical_page * page_size,
                logical_page * page_size + 16,
                dtype=torch.int64,
                device="cuda",
            )
            for logical_page in range(4)
        )
    )


def _hash_alias_batches(*, page_size: int, num_sets: int) -> torch.Tensor:
    """Two concurrent queries whose physical rows hash to the same sets."""
    cp_size = 2
    first = list(range(page_size, page_size + 64))
    for second_page in range(2, 20_000):
        second = list(range(second_page * page_size, second_page * page_size + 64))
        num_pages = second[-1] // page_size + 1
        pages_per_rank = (num_pages + cp_size - 1) // cp_size

        def physical_row(logical_row: int) -> int:
            logical_page, offset = divmod(logical_row, page_size)
            physical_page = (
                logical_page % cp_size * pages_per_rank + logical_page // cp_size
            )
            return physical_page * page_size + offset

        first_sets = [
            (row ^ (row >> 13)) & (num_sets - 1) for row in map(physical_row, first)
        ]
        second_sets = [
            (row ^ (row >> 13)) & (num_sets - 1) for row in map(physical_row, second)
        ]
        if first_sets == second_sets:
            return torch.tensor((first, second), dtype=torch.int64, device="cuda")
    raise AssertionError("failed to construct a physical-row hash alias")


def _packed_model1_source(
    *, page_size: int, logical_rows: torch.Tensor, seed: int
) -> torch.Tensor:
    logical_rows = logical_rows.reshape(-1)
    num_pages = int(torch.max(logical_rows).item()) // page_size + 1
    page_bytes = ((584 * page_size + 575) // 576) * 576
    source = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    keys = torch.randn(
        (logical_rows.numel(), 512),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    fused_store_cache(
        keys,
        source,
        logical_rows,
        page_size=page_size,
        type="flashmla",
    )
    return source


def _rank_major_source(
    source: torch.Tensor, *, cp_size: int, page_size: int
) -> tuple[torch.Tensor, int]:
    num_pages = source.shape[0]
    pages_per_rank = (num_pages + cp_size - 1) // cp_size
    compact_page_bytes = 584 * page_size
    shared = torch.zeros(
        (pages_per_rank * cp_size, compact_page_bytes),
        dtype=source.dtype,
        device=source.device,
    )
    for logical_page in range(num_pages):
        physical_page = (
            logical_page % cp_size * pages_per_rank + logical_page // cp_size
        )
        shared[physical_page].copy_(source[logical_page, :compact_page_bytes])
    return shared, pages_per_rank


def _flashmla_view(source: torch.Tensor, *, page_size: int) -> torch.Tensor:
    return source[:, : page_size * 584].view(-1, page_size, 1, 584)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _FLASH_MLA_AVAILABLE, "repo-built FlashMLA is required")
class TestDSV4FlashMLADemandCache(CustomTestCase):
    def _run_base_and_demand(
        self,
        *,
        extra_page_size: int,
        cache_rows: int | None = None,
        cache_ways: int | None = None,
        shared_rank: int = 0,
        hash_alias: bool = False,
        hash_alias_batches: bool = False,
        expect_reuse: bool = True,
        expect_collision: bool = False,
        reuse_repeats: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

        is_sm100 = torch.cuda.get_device_capability()[0] == 10
        if cache_rows is None:
            cache_rows = 1 << 18 if is_sm100 else 384
        if cache_ways is None:
            cache_ways = 1 if is_sm100 else 3

        cp_size = 2
        self.assertIn(shared_rank, range(cp_size))
        swa_page_size = 128
        if hash_alias_batches or hash_alias:
            num_sets = cache_rows // cache_ways
            swa_indices = _hash_alias_batches(
                page_size=swa_page_size, num_sets=num_sets
            )
            extra_indices = _hash_alias_batches(
                page_size=extra_page_size, num_sets=num_sets
            )
            if hash_alias:
                swa_indices = swa_indices.reshape(-1)
                extra_indices = extra_indices.reshape(-1)
        else:
            swa_indices = _logical_rows(swa_page_size)
            extra_indices = _logical_rows(extra_page_size)
        swa_base = _packed_model1_source(
            page_size=swa_page_size, logical_rows=swa_indices, seed=20260808
        )
        extra_base = _packed_model1_source(
            page_size=extra_page_size, logical_rows=extra_indices, seed=20260809
        )
        swa_shared, swa_pages_per_rank = _rank_major_source(
            swa_base, cp_size=cp_size, page_size=swa_page_size
        )
        extra_shared, extra_pages_per_rank = _rank_major_source(
            extra_base, cp_size=cp_size, page_size=extra_page_size
        )
        if extra_page_size == 64:
            self.assertEqual(extra_shared.stride(0), 37_376)

        q_generator = torch.Generator(device="cuda")
        q_generator.manual_seed(20260810)
        batch_size = swa_indices.shape[0] if swa_indices.ndim == 2 else 1
        q = torch.randn(
            (batch_size, 1, 64, 512),
            dtype=torch.bfloat16,
            device="cuda",
            generator=q_generator,
        )
        indices = swa_indices.reshape(batch_size, 1, -1).to(torch.int32)
        extra_indices_3d = extra_indices.reshape(batch_size, 1, -1).to(torch.int32)
        lengths = torch.full(
            (batch_size,), indices.shape[-1], dtype=torch.int32, device="cuda"
        )

        common = dict(
            q=q,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            num_splits=None,
            softmax_scale=512**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=None,
            extra_indices_in_kvcache=extra_indices_3d,
            topk_length=lengths,
            extra_topk_length=lengths,
        )
        base_out, base_lse = flash_mla_with_kvcache(
            k_cache=_flashmla_view(swa_base, page_size=swa_page_size),
            tile_scheduler_metadata=FlashMLASchedMeta(),
            extra_k_cache=_flashmla_view(extra_base, page_size=extra_page_size),
            **common,
        )

        num_sets, remainder = divmod(cache_rows, cache_ways)
        self.assertEqual(remainder, 0)
        self.assertEqual(num_sets & (num_sets - 1), 0)
        row_cache = torch.empty(
            (cache_rows, DSV4_MODEL1_DEMAND_CACHE_ROW_BYTES),
            dtype=torch.uint8,
            device="cuda",
        )
        self.assertEqual(row_cache.stride(0) % 16, 0)
        tags = torch.zeros((num_sets, cache_ways), dtype=torch.int64, device="cuda")
        stats = torch.zeros(5, dtype=torch.int64, device="cuda")
        demand_meta = FlashMLASchedMeta()
        demand_kwargs = dict(
            shared_kv_row_cache=row_cache,
            shared_kv_cache_tags=tags,
            shared_kv_cache_stats=stats,
            shared_kv_cache_epoch=1,
            shared_kv_cache_ways=cache_ways,
            shared_kv_rank=shared_rank,
            shared_kv_size=cp_size,
            shared_swa_page_size=swa_page_size,
            shared_swa_pages_per_rank=swa_pages_per_rank,
            shared_extra_page_size=extra_page_size,
            shared_extra_pages_per_rank=extra_pages_per_rank,
        )
        demand_out, demand_lse = flash_mla_with_kvcache(
            k_cache=_flashmla_view(swa_shared, page_size=swa_page_size),
            tile_scheduler_metadata=demand_meta,
            extra_k_cache=_flashmla_view(extra_shared, page_size=extra_page_size),
            **common,
            **demand_kwargs,
        )
        torch.testing.assert_close(demand_out, base_out, atol=0, rtol=0)
        torch.testing.assert_close(demand_lse, base_lse, atol=0, rtol=0)

        torch.cuda.synchronize()
        first_stats = stats.cpu()
        self.assertGreater(first_stats[2].item(), 0, "first pass must fill remote rows")
        if expect_collision:
            self.assertGreater(
                first_stats[3].item(), 0, "adversarial rows must hit one cache set"
            )
        fills_before_reuse = first_stats[2].item()
        hits_before_reuse = first_stats[1].item()

        for _ in range(reuse_repeats):
            reuse_out, reuse_lse = flash_mla_with_kvcache(
                k_cache=_flashmla_view(swa_shared, page_size=swa_page_size),
                tile_scheduler_metadata=demand_meta,
                extra_k_cache=_flashmla_view(extra_shared, page_size=extra_page_size),
                **common,
                **demand_kwargs,
            )
            torch.testing.assert_close(reuse_out, base_out, atol=0, rtol=0)
            torch.testing.assert_close(reuse_lse, base_lse, atol=0, rtol=0)
        torch.cuda.synchronize()
        reuse_stats = stats.cpu()
        if expect_reuse:
            self.assertEqual(reuse_stats[2].item(), fills_before_reuse)
            self.assertGreater(reuse_stats[1].item(), hits_before_reuse)
        else:
            self.assertGreater(reuse_stats[2].item(), fills_before_reuse)
        return first_stats, reuse_stats

    def test_swa_and_c4_fill_then_reuse_exact_packed_rows(self):
        self._run_base_and_demand(extra_page_size=64)

    def test_swa_and_c128_fill_then_reuse_exact_packed_rows(self):
        self._run_base_and_demand(extra_page_size=2)

    def test_rank_one_local_and_remote_rows_match_base_exactly(self):
        self._run_base_and_demand(extra_page_size=64, shared_rank=1)

    def test_collision_fallback_preserves_base_result(self):
        if torch.cuda.get_device_capability()[0] == 10:
            self.skipTest("SM100 release path uses collision-free direct slots")
        first_stats, _ = self._run_base_and_demand(
            extra_page_size=64, cache_rows=1, cache_ways=1
        )
        self.assertGreater(first_stats[3].item(), 0)

    def test_one_way_hash_aliases_preserve_base_result(self):
        if torch.cuda.get_device_capability()[0] == 10:
            self.skipTest("SM100 release path uses collision-free direct slots")
        self._run_base_and_demand(
            extra_page_size=64,
            cache_rows=1 << 10,
            cache_ways=1,
            hash_alias=True,
            expect_collision=True,
        )

    def test_one_way_concurrent_hash_aliases_are_race_safe(self):
        if torch.cuda.get_device_capability()[0] == 10:
            self.skipTest("SM100 release path uses collision-free direct slots")
        self._run_base_and_demand(
            extra_page_size=64,
            cache_rows=1 << 10,
            cache_ways=1,
            hash_alias_batches=True,
            expect_collision=True,
            reuse_repeats=20,
        )

    def test_large_one_way_cache_uses_collision_free_direct_slots(self):
        first_stats, _ = self._run_base_and_demand(
            extra_page_size=64,
            cache_rows=1 << 18,
            cache_ways=1,
            hash_alias_batches=torch.cuda.get_device_capability()[0] != 10,
            reuse_repeats=20,
        )
        self.assertEqual(first_stats[3].item(), 0)


if __name__ == "__main__":
    unittest.main()
