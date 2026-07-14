import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import allocator_npu
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    MultiEndedAllocator,
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeChildAllocator:
    def __init__(
        self,
        *,
        available: int,
        allocation: torch.Tensor | None,
    ) -> None:
        self.available = available
        self.allocation = allocation
        self.alloc_sizes: list[int] = []
        self.freed: list[torch.Tensor] = []

    def available_size(self) -> int:
        return self.available

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.alloc_sizes.append(need_size)
        return self.allocation

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


class _FakeC4Pool:
    @staticmethod
    def translate_loc_from_full_to_compressed(
        full_indices: torch.Tensor,
    ) -> torch.Tensor:
        return full_indices[(full_indices + 1) % 4 == 0] // 4


def _make_generic_allocator(
    *,
    logical_allocation: torch.Tensor | None,
    device_allocation: torch.Tensor | None,
) -> HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 4
    allocator.device = "cpu"
    allocator.is_not_in_free_group = True
    allocator.logical_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=logical_allocation,
    )
    allocator.hisparse_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=device_allocation,
    )
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        64,
        dtype=torch.int64,
    )
    return allocator


def _make_dsv4_allocator(
    *,
    logical_allocation: torch.Tensor | None,
    device_allocation: torch.Tensor | None,
) -> DeepSeekV4HiSparseTokenToKVPoolAllocator:
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 8
    allocator.compress_ratio = 4
    allocator.hisparse_page_size = 2
    allocator.device = "cpu"
    allocator.is_not_in_free_group = True
    allocator.logical_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=logical_allocation,
    )
    allocator.hisparse_attn_allocator = _FakeChildAllocator(
        available=64,
        allocation=device_allocation,
    )
    allocator.hisparse_kvcache = _FakeC4Pool()
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        64,
        dtype=torch.int64,
    )
    return allocator


class TestHiSparseDirectAllocator(unittest.TestCase):
    def test_npu_hybrid_swa_pool_selection_uses_npu_allocator(self) -> None:
        """Ordinary NPU hybrid SWA initialization selects the NPU allocator."""
        from sglang.srt.model_executor import model_runner_kv_cache_mixin

        runner = SimpleNamespace(
            max_running_requests=2,
            server_args=SimpleNamespace(
                enable_unified_memory=False,
                disaggregation_mode="null",
                attention_backend="ascend",
                prefill_only_disable_kv_cache=False,
                max_speculative_num_draft_tokens=0,
                speculative_algorithm=None,
                enable_memory_saver=False,
                enable_page_major_kv_layout=False,
            ),
            req_to_token_pool=None,
            token_to_kv_pool=None,
            token_to_kv_pool_allocator=None,
            mambaish_config=None,
            is_hybrid_swa=True,
            is_hybrid_swa_compress=False,
            is_draft_worker=False,
            enable_hisparse=False,
            hybrid_gdn_config=None,
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            page_size=4,
            kv_cache_dtype=torch.bfloat16,
            device="cpu",
            dcp_size=1,
            post_capture_kv_active=False,
            _validate_prefill_only_disable_kv_cache_pool_family=Mock(),
            model_config=SimpleNamespace(
                context_len=128,
                head_dim=64,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                get_num_kv_heads=lambda tp_size: 2,
                hf_config=SimpleNamespace(
                    model_type="test",
                    architectures=["TestModel"],
                ),
            ),
        )
        selected_allocator = object()
        req_to_token_pool = SimpleNamespace()
        token_to_kv_pool = SimpleNamespace()

        with patch.object(
            model_runner_kv_cache_mixin,
            "_is_npu",
            True,
        ), patch.object(
            model_runner_kv_cache_mixin.current_platform,
            "is_out_of_tree",
            return_value=False,
        ), patch.object(
            model_runner_kv_cache_mixin,
            "get_req_to_token_extra_context_len",
            return_value=0,
        ), patch.object(
            model_runner_kv_cache_mixin,
            "ReqToTokenPool",
            return_value=req_to_token_pool,
        ), patch.object(
            model_runner_kv_cache_mixin,
            "SWAKVPool",
            return_value=token_to_kv_pool,
        ) as swa_pool_class, patch.object(
            model_runner_kv_cache_mixin,
            "get_attention_tp_size",
            return_value=1,
        ), patch.object(
            allocator_npu,
            "NPUSWATokenToKVPoolAllocator",
            return_value=selected_allocator,
        ) as npu_swa_class:
            model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._init_pools(runner)

        self.assertIs(runner.req_to_token_pool, req_to_token_pool)
        self.assertIs(runner.token_to_kv_pool, token_to_kv_pool)
        self.assertIs(runner.token_to_kv_pool_allocator, selected_allocator)
        swa_pool_class.assert_called_once()
        npu_swa_class.assert_called_once_with(
            64,
            32,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=runner.token_to_kv_pool,
            need_sort=False,
        )

    def test_dsv4_allocator_reuses_npu_swa_extend(self) -> None:
        """DSV4 allocation layers compressed pools over the NPU SWA parent."""
        allocator = object.__new__(DSV4NPUTokenToKVPoolAllocator)
        allocator.translate_loc_from_full_to_swa = Mock(
            return_value=torch.tensor([41], dtype=torch.int64)
        )
        bundle = object()
        allocator._alloc_c_and_state = Mock(return_value=bundle)
        prefix_lens = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([1], dtype=torch.int64)
        last_loc = torch.tensor([-1], dtype=torch.int64)

        with patch.object(
            allocator_npu.NPUSWATokenToKVPoolAllocator,
            "alloc_extend",
            return_value=torch.tensor([31], dtype=torch.int64),
        ) as npu_swa_extend:
            result = allocator.alloc_extend(
                prefix_lens,
                prefix_lens.clone(),
                seq_lens,
                seq_lens.clone(),
                last_loc,
                1,
                req_pool_indices=torch.tensor([2], dtype=torch.int64),
            )

        self.assertIs(result, bundle)
        npu_swa_extend.assert_called_once()
        allocator._alloc_c_and_state.assert_called_once()

    def test_main_and_pure_swa_capability_boundary_is_explicit(self) -> None:
        """Main SWA supports pages while PureSWA remains a page-one boundary."""
        self.assertTrue(SWATokenToKVPoolAllocator.supports_page_aligned_alloc)
        self.assertTrue(SWATokenToKVPoolAllocator.supports_spec_page_aligned_alloc)
        self.assertFalse(PureSWATokenToKVPoolAllocator.supports_page_aligned_alloc)
        self.assertFalse(PureSWATokenToKVPoolAllocator.supports_spec_page_aligned_alloc)
        pure_swa = object.__new__(PureSWATokenToKVPoolAllocator)
        pure_swa.page_size = 1
        pure_swa.swa_attn_allocator = Mock()
        pure_swa.swa_attn_allocator.alloc.return_value = torch.tensor(
            [7], dtype=torch.int64
        )

        result = pure_swa.alloc(1)

        self.assertTrue(torch.equal(result, torch.tensor([7], dtype=torch.int64)))
        pure_swa.swa_attn_allocator.alloc.assert_called_once_with(1)
        with self.assertRaisesRegex(NotImplementedError, "page_size > 1"):
            pure_swa.alloc_extend_swa_tail()

    def test_page_aligned_spec_capability_matrix_is_explicit(self) -> None:
        """Spec direct capability remains disabled only for unsupported allocators."""
        supported_classes = (
            PagedTokenToKVPoolAllocator,
            MultiEndedAllocator,
            UnifiedMambaTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
        )
        for allocator_class in supported_classes:
            self.assertTrue(allocator_class.supports_spec_page_aligned_alloc)

        unsupported_classes = (
            PureSWATokenToKVPoolAllocator,
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )
        for allocator_class in unsupported_classes:
            self.assertFalse(allocator_class.supports_spec_page_aligned_alloc)

    def test_generic_direct_alloc_publishes_complete_page_mapping(self) -> None:
        """Generic direct allocation publishes matching logical and device pages."""
        logical_indices = torch.arange(4, 8, dtype=torch.int64)
        device_indices = torch.arange(8, 12, dtype=torch.int64)
        allocator = _make_generic_allocator(
            logical_allocation=logical_indices,
            device_allocation=device_indices,
        )

        result = allocator.alloc(4)

        self.assertTrue(torch.equal(result, logical_indices))
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[logical_indices],
                device_indices,
            )
        )

    def test_generic_second_child_failure_rolls_back_logical_page(self) -> None:
        """Generic direct allocation rolls back when the device child fails."""
        logical_indices = torch.arange(4, 8, dtype=torch.int64)
        allocator = _make_generic_allocator(
            logical_allocation=logical_indices,
            device_allocation=None,
        )

        result = allocator.alloc(4)

        self.assertIsNone(result)
        self.assertEqual(len(allocator.logical_attn_allocator.freed), 1)
        self.assertTrue(
            torch.equal(
                allocator.logical_attn_allocator.freed[0],
                logical_indices,
            )
        )
        self.assertTrue(torch.all(allocator.full_to_hisparse_device_index_mapping == 0))

    def test_dsv4_direct_alloc_uses_c4_count_and_translated_keys(self) -> None:
        """DSV4 direct allocation maps translated C4 keys to C4 device slots."""
        logical_indices = torch.arange(8, 16, dtype=torch.int64)
        device_indices = torch.arange(4, 6, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            logical_allocation=logical_indices,
            device_allocation=device_indices,
        )

        result = allocator.alloc(8)
        compressed_indices = torch.tensor([2, 3], dtype=torch.int64)

        self.assertTrue(torch.equal(result, logical_indices))
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_sizes, [2])
        self.assertTrue(
            torch.equal(
                allocator.full_to_hisparse_device_index_mapping[compressed_indices],
                device_indices,
            )
        )

    def test_dsv4_second_child_failure_rolls_back_full_logical_page(self) -> None:
        """DSV4 direct allocation rolls back the full logical page on C4 OOM."""
        logical_indices = torch.arange(8, 16, dtype=torch.int64)
        allocator = _make_dsv4_allocator(
            logical_allocation=logical_indices,
            device_allocation=None,
        )

        result = allocator.alloc(8)

        self.assertIsNone(result)
        self.assertTrue(
            torch.equal(
                allocator.logical_attn_allocator.freed[0],
                logical_indices,
            )
        )
        self.assertTrue(torch.all(allocator.full_to_hisparse_device_index_mapping == 0))

    def test_logical_only_alloc_does_not_touch_device_child_or_mapping(self) -> None:
        """Logical-only allocation consumes only the logical child."""
        logical_indices = torch.arange(4, 8, dtype=torch.int64)
        allocator = _make_generic_allocator(
            logical_allocation=logical_indices,
            device_allocation=torch.arange(8, 12, dtype=torch.int64),
        )

        result = allocator.alloc_logical_only(need_size=4)

        self.assertTrue(torch.equal(result, logical_indices))
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_sizes, [])
        self.assertTrue(torch.all(allocator.full_to_hisparse_device_index_mapping == 0))

    def test_zero_direct_alloc_returns_empty_without_child_mutation(self) -> None:
        """Zero-sized direct allocation returns an empty int64 tensor."""
        allocator = _make_generic_allocator(
            logical_allocation=torch.arange(4, 8, dtype=torch.int64),
            device_allocation=torch.arange(8, 12, dtype=torch.int64),
        )

        result = allocator.alloc(0)

        self.assertEqual(result.dtype, torch.int64)
        self.assertEqual(result.numel(), 0)
        self.assertEqual(allocator.logical_attn_allocator.alloc_sizes, [])
        self.assertEqual(allocator.hisparse_attn_allocator.alloc_sizes, [])


if __name__ == "__main__":
    unittest.main()
