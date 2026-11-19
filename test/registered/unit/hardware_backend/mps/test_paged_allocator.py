"""MPS pages must survive partial extension, exhaustion and grouped reuse."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMPSPagedAllocator(CustomTestCase):
    def test_debug_mode_rejects_misaligned_and_duplicate_allocations(self):
        """Debug checks must also run when decode needs no new physical page."""
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.mps.allocator import (
            MPSPagedTokenToKVPoolAllocator,
        )

        with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
            allocator = MPSPagedTokenToKVPoolAllocator(
                size=64,
                page_size=16,
                dtype=torch.float32,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
        for decode in (False, True):
            for duplicate in (False, True):
                with self.subTest(decode=decode, duplicate=duplicate):
                    count = 2 if duplicate else 1
                    allocator.free_pages = torch.tensor([1, 1, 2, 3])
                    lengths = torch.ones(count, dtype=torch.int64)
                    last = torch.full((count,), -1 if duplicate else 0)
                    with self.assertRaises(AssertionError):
                        if decode:
                            allocator.alloc_decode(lengths, lengths, last)
                        else:
                            prefix = torch.zeros_like(lengths)
                            allocator.alloc_extend(
                                prefix, prefix, lengths, lengths, last, count
                            )
        allocator.free_pages = torch.empty(0, dtype=torch.int64)
        with self.assertRaises(AssertionError):
            allocator.alloc_decode(
                torch.tensor([2, 2]), torch.tensor([2, 2]), torch.tensor([16, 16])
            )

    @unittest.skipUnless(torch.backends.mps.is_available(), "Requires Torch MPS")
    def test_kv_buffers_use_separate_mps_allocations(self):
        """Aliased K/V backing storage makes exported MLX decode copy whole pools."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.runtime_context import get_context

        for backend in ("eager", "mlx-compiled"):
            with (
                self.subTest(backend=backend),
                get_context().override_server_args(
                    device="mps",
                    mps_execution_backend=backend,
                    disable_overlap_schedule=True,
                ),
            ):
                pool = MHATokenToKVPool(
                    size=64,
                    page_size=16,
                    dtype=torch.bfloat16,
                    head_num=2,
                    head_dim=64,
                    layer_num=2,
                    device="mps",
                    enable_memory_saver=False,
                )
                for key, value in zip(pool.k_buffer, pool.v_buffer):
                    self.assertEqual(
                        key.untyped_storage().data_ptr()
                        == value.untyped_storage().data_ptr(),
                        backend == "eager",
                    )

    def test_partial_pages_exhaustion_and_reuse(self):
        devices = ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device):
                self._check_partial_pages(device)

    def _check_partial_pages(self, device):
        from sglang.srt.hardware_backend.mps.allocator import (
            MPSPagedTokenToKVPoolAllocator,
        )
        from sglang.srt.platforms.mps import MpsSRTPlatform
        from sglang.srt.runtime_context import get_context

        self.enterContext(
            get_context().override_server_args(
                device=device,
                mps_execution_backend="mlx-compiled" if device == "mps" else "eager",
                disable_overlap_schedule=True,
            )
        )
        allocator_cls = (
            MpsSRTPlatform().get_paged_allocator_cls()
            if device == "mps"
            else MPSPagedTokenToKVPoolAllocator
        )
        for page_size in (16, 32, 64):
            for need_sort in (False, True):
                with self.subTest(page_size=page_size, need_sort=need_sort):
                    p = page_size
                    allocator = allocator_cls(
                        size=4 * p,
                        page_size=p,
                        dtype=torch.float32,
                        device=device,
                        kvcache=None,
                        need_sort=need_sort,
                    )
                    allocator.free_pages = torch.tensor([3, 1, 4, 2], device=device)
                    prefix = torch.tensor([0, 0])
                    lengths = torch.tensor([p - 1, p + 1])
                    slots = allocator.alloc_extend(
                        prefix.to(device),
                        prefix,
                        lengths.to(device),
                        lengths,
                        torch.tensor([-1, -1], device=device),
                        2 * p,
                    )
                    self.assertEqual(
                        slots.cpu().tolist(),
                        [*range(3 * p, 4 * p - 1), *range(p, 2 * p), 4 * p],
                    )
                    self.assertEqual(allocator.available_size(), p)
                    last = slots[torch.tensor([p - 2, 2 * p - 1], device=device)]
                    lengths = lengths + 1
                    inside = allocator.alloc_decode(
                        lengths.to(device), lengths, last.int()
                    )
                    self.assertEqual(inside.dtype, torch.int64)
                    self.assertEqual(inside.cpu().tolist(), [4 * p - 1, 4 * p + 1])
                    self.assertEqual(allocator.available_size(), p)
                    lengths = lengths + 1
                    boundary = allocator.alloc_decode(
                        lengths.to(device), lengths, inside.int()
                    )
                    self.assertEqual(boundary.dtype, torch.int64)
                    self.assertEqual(boundary.cpu().tolist(), [2 * p, 4 * p + 2])
                    self.assertEqual(allocator.available_size(), 0)
                    prefix = lengths
                    lengths = lengths + torch.tensor([3, 0])
                    partial = allocator.alloc_extend(
                        prefix.to(device),
                        prefix,
                        lengths.to(device),
                        lengths,
                        boundary,
                        3,
                    )
                    self.assertEqual(
                        partial.cpu().tolist(), list(range(2 * p + 1, 2 * p + 4))
                    )
                    full = torch.tensor([2 * p + 1, p + 3])
                    self.assertIsNone(
                        allocator.alloc_extend(
                            lengths.to(device),
                            lengths,
                            full.to(device),
                            full,
                            torch.stack((partial[-1], boundary[-1])),
                            int((full - lengths).sum()),
                        )
                    )
                    self.assertIsNone(
                        allocator.alloc_decode(full.to(device), full, boundary)
                    )
                    self.assertEqual(allocator.available_size(), 0)
                    allocator.free_group_begin()
                    for freed in (slots, inside, boundary, partial):
                        allocator.free(freed)
                    allocator.free_group_end()
                    self.assertEqual(allocator.available_size(), 4 * p)
                    reused = allocator.alloc(4 * p)
                    self.assertEqual(
                        reused.cpu().sort().values.tolist(), list(range(p, 5 * p))
                    )
                    allocator.free_segment(reused[: 2 * p], start_pos=0)
                    self.assertEqual(allocator.available_size(), 2 * p)
                    again = allocator.alloc(2 * p)
                    self.assertEqual(
                        again.cpu().sort().values.tolist(),
                        reused[: 2 * p].cpu().sort().values.tolist(),
                    )


if __name__ == "__main__":
    unittest.main()
