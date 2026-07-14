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


class TestNPUAllocationAuthority(unittest.TestCase):
    def test_get_last_loc_handles_fresh_rows_and_preserves_length_dtype(self) -> None:
        """NPU endpoint lookup returns fresh sentinels and table slots in lens dtype."""
        req_to_token = torch.tensor(
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            dtype=torch.int32,
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32)

        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                prefix_lens = torch.tensor([0, 3], dtype=dtype)

                result = allocator_npu.get_last_loc(
                    req_to_token,
                    req_pool_indices,
                    prefix_lens,
                )

                self.assertEqual(result.dtype, dtype)
                self.assertTrue(
                    torch.equal(result, torch.tensor([-1, 23], dtype=dtype))
                )

    def test_extend_entry_uses_ragged_prefix_endpoints(self) -> None:
        """NPU extend derives anchors only from ragged prefix tensors."""
        tree_cache = SimpleNamespace()
        batch = SimpleNamespace(device=torch.device("cpu"))
        prefix_tensors = [
            torch.tensor([101, 102], dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
        ]
        prefix_lens = torch.tensor([2, 0], dtype=torch.int64)
        seq_lens = torch.tensor([4, 3], dtype=torch.int64)
        output = torch.tensor([201, 202, 203, 204, 205], dtype=torch.int64)

        with patch.object(
            allocator_npu,
            "_alloc_paged_token_slots_extend_npu",
            return_value=output,
        ) as paged_entry:
            result = allocator_npu.alloc_for_extend_npu(
                tree_cache,
                prefix_tensors=prefix_tensors,
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens.clone(),
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.clone(),
                extend_num_tokens=5,
                req_pool_indices=torch.tensor([4, 9], dtype=torch.int64),
                dsv4_state_lens=None,
                batch=batch,
            )

        self.assertIs(result, output)
        self.assertTrue(
            torch.equal(
                paged_entry.call_args.kwargs["last_loc"],
                torch.tensor([102, -1], dtype=torch.int64),
            )
        )

    def test_decode_entry_uses_combined_endpoint_anchor_and_next_lens(self) -> None:
        """NPU decode anchors at the current combined endpoint before allocation."""
        req_to_token = torch.arange(64, dtype=torch.int32).reshape(2, 32)
        batch = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            tree_cache=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(),
            reqs=[],
        )
        next_combined_lens = torch.tensor([8, 13], dtype=torch.int64)
        output = torch.tensor([301, 302], dtype=torch.int64)

        with patch.object(
            allocator_npu,
            "_alloc_paged_token_slots_decode_npu",
            return_value=output,
        ) as paged_entry:
            result = allocator_npu.alloc_for_decode_npu(
                batch,
                next_combined_lens=next_combined_lens,
                next_combined_lens_cpu=next_combined_lens.clone(),
                token_per_req=2,
            )

        self.assertIs(result, output)
        call_kwargs = paged_entry.call_args.kwargs
        self.assertTrue(torch.equal(call_kwargs["last_loc"], torch.tensor([5, 42])))
        self.assertTrue(torch.equal(call_kwargs["seq_lens"], next_combined_lens))
        self.assertTrue(torch.equal(call_kwargs["seq_lens_cpu"], next_combined_lens))

    def test_decode_caller_passes_encoder_offset_combined_endpoints(self) -> None:
        """Encoder-decoder routing passes combined KV endpoints to the NPU entry."""
        req_to_token = torch.zeros((2, 32), dtype=torch.int32)

        def write(indices, values) -> None:
            req_to_token[indices] = values

        batch = SimpleNamespace(
            seq_lens=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int64),
            encoder_lens=torch.tensor([7, 11], dtype=torch.int64),
            encoder_lens_cpu=[7, 11],
            model_config=SimpleNamespace(is_encoder_decoder=True),
            tree_cache=SimpleNamespace(
                token_to_kv_pool_allocator=SimpleNamespace(page_size=4)
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=req_to_token,
                write=write,
            ),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=10)),
                SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=16)),
            ],
            device=torch.device("cpu"),
            maybe_evict_swa=Mock(),
        )
        output = torch.tensor([401, 402], dtype=torch.int64)

        with patch.object(allocation, "_is_npu", True), patch.object(
            allocation,
            "_alloc_page_size",
            return_value=4,
        ), patch.object(
            allocator_npu,
            "alloc_for_decode_npu",
            return_value=output,
        ) as npu_entry, patch.object(
            allocation,
            "maybe_write_dsv4_decode",
        ):
            allocation.alloc_for_decode(batch, token_per_req=1)

        npu_entry.assert_called_once()
        call_args = npu_entry.call_args
        self.assertIs(call_args.args[0], batch)
        self.assertTrue(
            torch.equal(
                call_args.kwargs["next_combined_lens"],
                torch.tensor([11, 17], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                call_args.kwargs["next_combined_lens_cpu"],
                torch.tensor([11, 17], dtype=torch.int64),
            )
        )
        self.assertEqual(call_args.kwargs["token_per_req"], 1)

    def test_spec_entry_uses_current_endpoints_and_forwards_lens(self) -> None:
        """NPU spec allocation derives anchors from current endpoints and forwards lenses."""
        req_to_token = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]],
            dtype=torch.int32,
        )
        tree_cache = SimpleNamespace()
        req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        current_lens = torch.tensor([0, 3], dtype=torch.int64)
        next_lens = torch.tensor([2, 4], dtype=torch.int64)
        batch = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test"))
        )
        output = torch.tensor([501, 502, 503], dtype=torch.int64)

        with patch.object(
            allocator_npu,
            "_alloc_paged_token_slots_extend_npu",
            return_value=output,
        ) as paged_entry:
            result = allocator_npu.alloc_for_spec_decode_npu(
                tree_cache,
                req_to_token_pool,
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                cur_kv_lens=current_lens,
                cur_kv_lens_cpu=current_lens.clone(),
                nxt_kv_lens=next_lens,
                nxt_kv_lens_cpu=next_lens.clone(),
                num_needed_tokens=3,
                batch=batch,
            )

        self.assertIs(result, output)
        call_kwargs = paged_entry.call_args.kwargs
        self.assertTrue(torch.equal(call_kwargs["last_loc"], torch.tensor([-1, 22])))
        self.assertTrue(torch.equal(call_kwargs["prefix_lens"], current_lens))
        self.assertTrue(torch.equal(call_kwargs["prefix_lens_cpu"], current_lens))
        self.assertTrue(torch.equal(call_kwargs["seq_lens"], next_lens))
        self.assertTrue(torch.equal(call_kwargs["seq_lens_cpu"], next_lens))
        self.assertEqual(call_kwargs["extend_num_tokens"], 3)


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
