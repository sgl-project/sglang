import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import allocator_npu
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _OutOfTreeAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self) -> None:
        self.page_size = 4
        self.alloc = Mock(side_effect=AssertionError("allocator must not mutate"))
        self.alloc_extend = Mock(
            side_effect=AssertionError("legacy allocator must not mutate")
        )
        self.free_pages = torch.tensor([3, 5], dtype=torch.int64)
        self.release_pages = torch.tensor([7], dtype=torch.int64)

    def clear(self) -> None:
        raise AssertionError("allocator must not mutate")

    def alloc(self, need_size: int):
        raise AssertionError("allocator must not mutate")

    def free(self, free_index: torch.Tensor) -> None:
        raise AssertionError("allocator must not mutate")


def _make_inputs(
    *,
    supports_main: bool = True,
    supports_spec: bool = True,
    page_size: int = 4,
    current_len: int = 4,
    next_len: int = 5,
    allocator=None,
):
    if allocator is None:
        allocator = SimpleNamespace(
            page_size=page_size,
            supports_page_aligned_alloc=supports_main,
            supports_spec_page_aligned_alloc=supports_spec,
            alloc=Mock(side_effect=AssertionError("allocator must not mutate")),
            alloc_extend=Mock(
                side_effect=AssertionError("legacy allocator must not mutate")
            ),
            free_pages=torch.tensor([3, 5], dtype=torch.int64),
            release_pages=torch.tensor([7], dtype=torch.int64),
        )
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(32, dtype=torch.int32).reshape(2, 16),
    )
    req = SimpleNamespace(
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=current_len),
    )
    cur_kv_lens = torch.tensor([current_len], dtype=torch.int32)
    nxt_kv_lens = torch.tensor([next_len], dtype=torch.int32)
    return SimpleNamespace(
        allocator=allocator,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        reqs=[req],
        req_pool_indices=torch.tensor([1], dtype=torch.int64),
        cur_kv_lens=cur_kv_lens,
        cur_kv_lens_cpu=cur_kv_lens.clone(),
        nxt_kv_lens=nxt_kv_lens,
        nxt_kv_lens_cpu=nxt_kv_lens.clone(),
        batch=SimpleNamespace(device=torch.device("cpu")),
    )


def _call_spec(inputs) -> None:
    allocation.alloc_for_spec_decode(
        inputs.tree_cache,
        inputs.req_to_token_pool,
        reqs=inputs.reqs,
        req_pool_indices=inputs.req_pool_indices,
        cur_kv_lens=inputs.cur_kv_lens,
        cur_kv_lens_cpu=inputs.cur_kv_lens_cpu,
        nxt_kv_lens=inputs.nxt_kv_lens,
        nxt_kv_lens_cpu=inputs.nxt_kv_lens_cpu,
        num_needed_tokens=inputs.nxt_kv_lens_cpu.item() - inputs.cur_kv_lens_cpu.item(),
        batch=inputs.batch,
    )


class TestSpecPageAlignedAllocation(unittest.TestCase):
    def _assert_failure_without_mutation(
        self,
        inputs,
        *,
        exception_type: type[BaseException],
        is_cuda_backend: bool = True,
        is_hip_backend: bool = False,
        platform_enum: PlatformEnum = PlatformEnum.CUDA,
    ) -> str:
        before_row = inputs.req_to_token_pool.req_to_token.clone()
        before_watermark = inputs.reqs[0].kv.kv_allocated_len
        before_free_pages = inputs.allocator.free_pages.clone()
        before_release_pages = inputs.allocator.release_pages.clone()

        with patch.object(allocation, "_is_npu", False), patch.object(
            allocation, "_is_cuda", is_cuda_backend
        ), patch.object(allocation, "_is_hip", is_hip_backend), patch.object(
            allocation.platforms,
            "_current_platform",
            SimpleNamespace(_enum=platform_enum),
        ), patch.object(
            allocation,
            "evict_from_tree_cache",
            side_effect=AssertionError("eviction must not run"),
        ), patch.object(
            allocation,
            "alloc_token_slots",
            side_effect=AssertionError("allocation must not run"),
        ), patch.object(
            allocation,
            "assign_req_to_token_pool_func",
            side_effect=AssertionError("writer must not run"),
        ), patch.object(
            allocation,
            "maybe_write_dsv4_extend",
            side_effect=AssertionError("DSV4 hook must not run"),
        ), patch.object(
            allocator_npu,
            "alloc_for_spec_decode_npu",
            side_effect=AssertionError("NPU entry must not run"),
        ):
            with self.assertRaises(exception_type) as error:
                _call_spec(inputs)

        inputs.allocator.alloc.assert_not_called()
        inputs.allocator.alloc_extend.assert_not_called()
        self.assertEqual(inputs.reqs[0].kv.kv_allocated_len, before_watermark)
        self.assertTrue(torch.equal(inputs.req_to_token_pool.req_to_token, before_row))
        self.assertTrue(torch.equal(inputs.allocator.free_pages, before_free_pages))
        self.assertTrue(
            torch.equal(inputs.allocator.release_pages, before_release_pages)
        )
        return str(error.exception)

    def test_capability_false_matrix_fails_before_mutation(self) -> None:
        """Every unsupported page-allocation capability quadrant fails atomically."""
        for supports_main, supports_spec in (
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(
                supports_main=supports_main,
                supports_spec=supports_spec,
            ):
                inputs = _make_inputs(
                    supports_main=supports_main,
                    supports_spec=supports_spec,
                )
                message = self._assert_failure_without_mutation(
                    inputs,
                    exception_type=NotImplementedError,
                )
                self.assertIn("phase=spec decode", message)
                self.assertIn("page_size=4", message)
                self.assertIn(f"supports_page_aligned_alloc={supports_main}", message)
                self.assertIn(
                    f"supports_spec_page_aligned_alloc={supports_spec}", message
                )

    def test_real_hisparse_spec_capability_fails_before_mutation(self) -> None:
        """Generic and DSV4 HiSparse allocators fail at the defensive boundary."""
        for allocator_class in (
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ):
            with self.subTest(allocator_class=allocator_class.__name__):
                hisparse_allocator = object.__new__(allocator_class)
                hisparse_allocator.page_size = 4
                hisparse_allocator.alloc = Mock(
                    side_effect=AssertionError("HiSparse allocator must not mutate")
                )
                hisparse_allocator.alloc_extend = Mock(
                    side_effect=AssertionError(
                        "HiSparse legacy allocator must not mutate"
                    )
                )
                hisparse_allocator.free_pages = torch.tensor([3, 5], dtype=torch.int64)
                hisparse_allocator.release_pages = torch.tensor([7], dtype=torch.int64)
                inputs = _make_inputs(allocator=hisparse_allocator)
                message = self._assert_failure_without_mutation(
                    inputs,
                    exception_type=NotImplementedError,
                )
                self.assertIn(allocator_class.__name__, message)
                self.assertIn("supports_page_aligned_alloc=True", message)
                self.assertIn("supports_spec_page_aligned_alloc=False", message)

    def test_out_of_tree_default_capabilities_fail_before_mutation(self) -> None:
        """Out-of-tree allocators inherit the base class default-false contract."""
        inputs = _make_inputs(allocator=_OutOfTreeAllocator())

        message = self._assert_failure_without_mutation(
            inputs,
            exception_type=NotImplementedError,
            platform_enum=PlatformEnum.OOT,
        )

        self.assertIn("_OutOfTreeAllocator", message)
        self.assertIn("supports_page_aligned_alloc=False", message)

    def test_capability_error_precedes_alignment_error(self) -> None:
        """Capability rejection precedes direct-contract alignment validation."""
        inputs = _make_inputs(
            supports_main=False,
            supports_spec=False,
            current_len=3,
            next_len=4,
        )

        message = self._assert_failure_without_mutation(
            inputs,
            exception_type=NotImplementedError,
        )

        self.assertIn("supports_page_aligned_alloc=False", message)

    def test_true_capabilities_reject_nonaligned_state_before_mutation(self) -> None:
        """Supported direct allocation still rejects a nonaligned watermark atomically."""
        inputs = _make_inputs(current_len=3, next_len=4)

        message = self._assert_failure_without_mutation(
            inputs,
            exception_type=AssertionError,
            is_cuda_backend=False,
            platform_enum=PlatformEnum.XPU,
        )

        self.assertIn("current watermark is not page-aligned", message)

    def test_unknown_backend_fails_without_registry_fallback(self) -> None:
        """Unknown page-allocation backends fail instead of using a registry fallback."""
        inputs = _make_inputs()

        message = self._assert_failure_without_mutation(
            inputs,
            exception_type=NotImplementedError,
            is_cuda_backend=False,
            is_hip_backend=False,
            platform_enum=PlatformEnum.UNSPECIFIED,
        )

        self.assertIn("unsupported_backend", message)

    def test_true_capabilities_use_direct_page_allocation(self) -> None:
        """XPU and MUSA trust true/true allocator capabilities for direct allocation."""
        for platform_enum in (PlatformEnum.XPU, PlatformEnum.MUSA):
            with self.subTest(platform_enum=platform_enum):
                inputs = _make_inputs()
                direct_pages = torch.arange(4, 8, dtype=torch.int64)

                with patch.object(allocation, "_is_npu", False), patch.object(
                    allocation, "_is_cuda", False
                ), patch.object(allocation, "_is_hip", False), patch.object(
                    allocation.platforms,
                    "_current_platform",
                    SimpleNamespace(_enum=platform_enum),
                ), patch.object(
                    allocation,
                    "alloc_token_slots",
                    return_value=direct_pages,
                ) as direct_allocator, patch.object(
                    allocation,
                    "assign_req_to_token_pool_func",
                ):
                    _call_spec(inputs)

                direct_allocator.assert_called_once_with(
                    tree_cache=inputs.tree_cache,
                    num_tokens=4,
                    phase="Spec decode",
                )
                self.assertEqual(inputs.reqs[0].kv.kv_allocated_len, 8)

    def test_page_one_ignores_page_capabilities(self) -> None:
        """Page-one allocation remains direct regardless of page capabilities."""
        inputs = _make_inputs(
            supports_main=False,
            supports_spec=False,
            page_size=1,
        )

        with patch.object(allocation, "_is_npu", False), patch.object(
            allocation,
            "alloc_token_slots",
            return_value=torch.tensor([7], dtype=torch.int64),
        ) as direct_allocator, patch.object(
            allocation,
            "assign_req_to_token_pool_func",
        ):
            _call_spec(inputs)

        direct_allocator.assert_called_once()
        self.assertEqual(inputs.reqs[0].kv.kv_allocated_len, 5)

    def test_npu_uses_explicit_entry_without_capability_gate(self) -> None:
        """NPU allocation bypasses non-NPU capabilities through its explicit entry."""
        inputs = _make_inputs(supports_main=False, supports_spec=False)
        inputs.batch.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="test")
        )

        with patch.object(allocation, "_is_npu", True), patch.object(
            allocator_npu,
            "alloc_for_spec_decode_npu",
            return_value=torch.tensor([7], dtype=torch.int64),
        ) as npu_entry, patch.object(
            allocation,
            "assign_req_to_token_pool_func",
        ):
            _call_spec(inputs)

        npu_entry.assert_called_once()
        self.assertEqual(inputs.reqs[0].kv.kv_allocated_len, 5)

    def test_legacy_registry_symbol_is_removed(self) -> None:
        """The generic allocation registry is absent from the main module."""
        self.assertFalse(hasattr(allocation, "ALLOC_EXTEND_FUNCS"))


if __name__ == "__main__":
    unittest.main()
