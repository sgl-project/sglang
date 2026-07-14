import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation import decode
from sglang.srt.mem_cache import allocation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, page_size: int = 4) -> None:
        self.page_size = page_size
        self.device = torch.device("cpu")
        self.calls: list[dict[str, object]] = []

    def alloc_extend(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["extend_num_tokens"]), dtype=torch.int64)

    def alloc(self, need_size: int) -> torch.Tensor:
        self.calls.append({"need_size": need_size})
        return torch.arange(need_size, dtype=torch.int64)

    def alloc_logical_only(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["need_size"]), dtype=torch.int64)

    def alloc_extend_swa_tail(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.arange(int(kwargs["extend_num_tokens"]), dtype=torch.int64)


class TestDecodePreallocPageAligned(unittest.TestCase):
    def test_decode_prealloc_rounds_physical_endpoint(self) -> None:
        """PD preallocation rounds physical capacity without changing fill_len."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=4,
                prefix_len=4,
                prefix_indices=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
                fill_len=5,
                delta_len=1,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 4)
        self.assertEqual(allocator.calls[0]["need_size"], 4)

    def test_decode_prealloc_rejects_misalignment_before_watermark_write(self) -> None:
        """PD preallocation rejects a partial prefix before mutating request state."""
        allocator = _Allocator()
        req = SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=4, swa_evicted_seqlen=0)
        )

        with (
            mock.patch.object(decode, "_is_npu", False),
            mock.patch.object(allocation, "_is_npu", False),
            self.assertRaisesRegex(AssertionError, "prefix lens"),
        ):
            decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=3,
                prefix_len=3,
                prefix_indices=torch.tensor([5, 6, 7], dtype=torch.int64),
                fill_len=5,
                delta_len=2,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 4)
        self.assertEqual(allocator.calls, [])

    def test_hisparse_prealloc_rounds_logical_capacity(self) -> None:
        """HiSparse preallocation publishes slots through the aligned endpoint."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc_hisparse(
                req=req,
                allocator=allocator,
                fill_len=5,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 8)
        self.assertEqual(allocator.calls[0]["need_size"], 8)

    def test_hisparse_host_allocation_uses_real_fill_length(self) -> None:
        """HiSparse host allocation excludes page-alignment padding."""
        allocator = _Allocator()
        allocator.available_size = mock.Mock(return_value=8)
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(5)),
            output_ids=[],
            kv=None,
            set_extend_range=mock.Mock(),
        )

        def alloc_reqs(reqs: list[SimpleNamespace]) -> torch.Tensor:
            reqs[0].req_pool_idx = 0
            return torch.tensor([0], dtype=torch.int64)

        host_pool = SimpleNamespace(
            alloc_paged_token_slots=mock.Mock(
                return_value=torch.arange(5, dtype=torch.int64)
            )
        )
        coordinator = SimpleNamespace(
            mem_pool_host=host_pool,
            req_to_host_pool=mock.Mock(),
            req_to_host_pool_allocated_len=mock.Mock(),
            host_token_len=mock.Mock(side_effect=lambda length: length),
        )
        queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
        queue.req_to_token_pool = SimpleNamespace(
            alloc=mock.Mock(side_effect=alloc_reqs),
            write=mock.Mock(),
        )
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace()
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )
        queue._pre_alloc_fill_len = mock.Mock(return_value=5)
        queue._uses_swa_tail_prealloc = mock.Mock(return_value=False)
        queue._swa_tail_len = mock.Mock(return_value=0)

        host_indices = queue._pre_alloc(req)

        self.assertEqual(req.kv.kv_allocated_len, 8)
        coordinator.host_token_len.assert_called_once_with(5)
        self.assertEqual(
            host_pool.alloc_paged_token_slots.call_args.args[-1],
            5,
        )
        self.assertEqual(host_indices.tolist(), [0, 1, 2, 3, 4])

    def test_decode_prealloc_aligns_swa_tail_around_real_endpoint(self) -> None:
        """SWA-tail preallocation separates padded capacity from the real endpoint."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with mock.patch.object(decode, "_is_npu", False):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=0,
                prefix_len=0,
                prefix_indices=torch.empty((0,), dtype=torch.int64),
                fill_len=6,
                delta_len=6,
                uses_swa_tail=True,
                swa_tail_len=2,
            )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(locations.numel(), 8)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [8])
        self.assertEqual(allocator.calls[0]["swa_tail_end"], 6)

    def test_npu_prealloc_keeps_continuation_length(self) -> None:
        """NPU preallocation retains its existing real-length continuation."""
        allocator = _Allocator()
        req = SimpleNamespace(kv=None)

        with (
            mock.patch.object(decode, "_is_npu", True),
            mock.patch.object(allocation, "_is_npu", True),
        ):
            locations = decode.alloc_for_decode_prealloc(
                req=req,
                allocator=allocator,
                total_prefix_len=3,
                prefix_len=3,
                prefix_indices=torch.tensor([5, 6, 7], dtype=torch.int64),
                fill_len=5,
                delta_len=2,
                uses_swa_tail=False,
                swa_tail_len=0,
            )

        self.assertEqual(req.kv.kv_allocated_len, 5)
        self.assertEqual(locations.numel(), 2)
        self.assertEqual(allocator.calls[0]["seq_lens_cpu"].tolist(), [5])

    def test_transferred_prefix_bypasses_zero_based_swa_tail_helper(self) -> None:
        """A transferred prefix routes through ordinary extend ownership."""
        fill_len = 510
        transferred_prefix_len = 64
        allocator = _Allocator(page_size=64)
        allocator.available_size = mock.Mock(return_value=512)
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(fill_len)),
            output_ids=[],
            kv=None,
            set_extend_range=mock.Mock(),
        )

        def alloc_reqs(reqs: list[SimpleNamespace]) -> torch.Tensor:
            reqs[0].req_pool_idx = 0
            return torch.tensor([0], dtype=torch.int64)

        req_to_token_pool = SimpleNamespace(
            alloc=mock.Mock(side_effect=alloc_reqs),
            write=mock.Mock(),
        )
        queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
        queue.req_to_token_pool = req_to_token_pool
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = SimpleNamespace()
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )
        queue._uses_swa_tail_prealloc = mock.Mock(return_value=True)
        queue._swa_tail_len = mock.Mock(return_value=126)

        queue._pre_alloc(
            req,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            prefix_len=0,
            total_prefix_len=transferred_prefix_len,
        )

        self.assertEqual(len(allocator.calls), 1)
        self.assertNotIn("swa_tail_end", allocator.calls[0])
        self.assertEqual(
            allocator.calls[0]["prefix_lens_cpu"].tolist(),
            [transferred_prefix_len],
        )
        self.assertEqual(
            allocator.calls[0]["extend_num_tokens"],
            512 - transferred_prefix_len,
        )


if __name__ == "__main__":
    unittest.main()
