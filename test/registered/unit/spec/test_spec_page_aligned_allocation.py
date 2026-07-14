import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_inputs(*, supports_spec_page_aligned_alloc: bool):
    allocator = SimpleNamespace(
        page_size=4,
        supports_page_aligned_alloc=True,
        supports_spec_page_aligned_alloc=supports_spec_page_aligned_alloc,
        alloc=Mock(side_effect=AssertionError("direct allocator must not be called")),
    )
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((2, 16), dtype=torch.int32)
    )
    req = SimpleNamespace(
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=4),
    )
    req_pool_indices = torch.tensor([1], dtype=torch.int64)
    cur_kv_lens = torch.tensor([4], dtype=torch.int32)
    nxt_kv_lens = torch.tensor([5], dtype=torch.int32)
    cur_kv_lens_cpu = cur_kv_lens.clone()
    nxt_kv_lens_cpu = nxt_kv_lens.clone()
    batch = SimpleNamespace(device=torch.device("cpu"))
    return SimpleNamespace(
        allocator=allocator,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        reqs=[req],
        req_pool_indices=req_pool_indices,
        cur_kv_lens=cur_kv_lens,
        cur_kv_lens_cpu=cur_kv_lens_cpu,
        nxt_kv_lens=nxt_kv_lens,
        nxt_kv_lens_cpu=nxt_kv_lens_cpu,
        batch=batch,
    )


class TestSpecPageAlignedAllocation(unittest.TestCase):
    def test_main_true_spec_false_uses_transitional_legacy_path(self) -> None:
        """Spec-disabled allocators retain the transitional legacy path until op30."""
        inputs = _make_inputs(supports_spec_page_aligned_alloc=False)
        legacy_allocator = Mock(return_value=torch.arange(4, 8, dtype=torch.int64))

        with patch.dict(
            allocation.ALLOC_EXTEND_FUNCS, {"cpu": legacy_allocator}
        ), patch.object(
            allocation,
            "get_last_loc",
            return_value=torch.tensor([-1], dtype=torch.int64),
        ), patch.object(
            allocation, "assign_req_to_token_pool_func"
        ):
            allocation.alloc_for_spec_decode(
                inputs.tree_cache,
                inputs.req_to_token_pool,
                reqs=inputs.reqs,
                req_pool_indices=inputs.req_pool_indices,
                cur_kv_lens=inputs.cur_kv_lens,
                cur_kv_lens_cpu=inputs.cur_kv_lens_cpu,
                nxt_kv_lens=inputs.nxt_kv_lens,
                nxt_kv_lens_cpu=inputs.nxt_kv_lens_cpu,
                num_needed_tokens=1,
                batch=inputs.batch,
            )

        legacy_allocator.assert_called_once()
        inputs.allocator.alloc.assert_not_called()

    def test_real_hisparse_capabilities_use_transitional_legacy_path(self) -> None:
        """Enabled HiSparse main allocators remain spec-disabled until op30."""
        for allocator_class in (
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ):
            inputs = _make_inputs(supports_spec_page_aligned_alloc=False)
            hisparse_allocator = object.__new__(allocator_class)
            hisparse_allocator.page_size = 4
            hisparse_allocator.alloc = Mock(
                side_effect=AssertionError("HiSparse spec direct alloc must not run")
            )
            inputs.tree_cache.token_to_kv_pool_allocator = hisparse_allocator
            legacy_allocator = Mock(return_value=torch.arange(4, 8, dtype=torch.int64))

            with patch.dict(
                allocation.ALLOC_EXTEND_FUNCS, {"cpu": legacy_allocator}
            ), patch.object(
                allocation,
                "get_last_loc",
                return_value=torch.tensor([-1], dtype=torch.int64),
            ), patch.object(
                allocation, "assign_req_to_token_pool_func"
            ):
                allocation.alloc_for_spec_decode(
                    inputs.tree_cache,
                    inputs.req_to_token_pool,
                    reqs=inputs.reqs,
                    req_pool_indices=inputs.req_pool_indices,
                    cur_kv_lens=inputs.cur_kv_lens,
                    cur_kv_lens_cpu=inputs.cur_kv_lens_cpu,
                    nxt_kv_lens=inputs.nxt_kv_lens,
                    nxt_kv_lens_cpu=inputs.nxt_kv_lens_cpu,
                    num_needed_tokens=1,
                    batch=inputs.batch,
                )

            legacy_allocator.assert_called_once()
            hisparse_allocator.alloc.assert_not_called()

    def test_main_true_spec_true_uses_direct_page_allocation(self) -> None:
        """Spec-enabled allocators continue to use direct page allocation."""
        inputs = _make_inputs(supports_spec_page_aligned_alloc=True)
        direct_pages = torch.arange(4, 8, dtype=torch.int64)

        with patch.object(
            allocation,
            "alloc_token_slots",
            return_value=direct_pages,
        ) as direct_allocator, patch.object(
            allocation,
            "assign_req_to_token_pool_func",
        ):
            allocation.alloc_for_spec_decode(
                inputs.tree_cache,
                inputs.req_to_token_pool,
                reqs=inputs.reqs,
                req_pool_indices=inputs.req_pool_indices,
                cur_kv_lens=inputs.cur_kv_lens,
                cur_kv_lens_cpu=inputs.cur_kv_lens_cpu,
                nxt_kv_lens=inputs.nxt_kv_lens,
                nxt_kv_lens_cpu=inputs.nxt_kv_lens_cpu,
                num_needed_tokens=1,
                batch=inputs.batch,
            )

        direct_allocator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
