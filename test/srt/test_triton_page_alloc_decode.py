import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.allocator import alloc_decode_kernel
from sglang.srt.utils.common import next_power_of_2

# Test configurations
BATCH_SIZES = [1, 32, 64]
SEQ_LENS = [666]
PAGE_SIZE = 64
NUM_PAGES = 100000


@triton.jit
def alloc_decode_kernel_v1(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    """Old version of alloc_decode_kernel that depends on bs_upper as constexpr."""
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_v2_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_v2_pages = tl.sum(num_v2_pages)
    new_page_start_loc = sum_num_v2_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


def setup_decode_test(bs, seq_len, device="cuda"):
    """Setup test tensors for decode allocation test.

    In decode phase, each request extends by exactly 1 token.
    seq_len is the length AFTER adding the new token (i.e., after decode).

    The constraint from allocator.py:
        (last_loc + 2) % page_size == seq_len % page_size

    This means: last_loc % page_size == (seq_len - 2) % page_size
    We place last_loc in page 1 (offset PAGE_SIZE) for simplicity.
    """
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)

    # last_loc should satisfy: (last_loc + 2) % page_size == seq_len % page_size
    # => last_loc % page_size == (seq_len - 2) % page_size
    # We place last_loc in page 1 for simplicity
    last_loc_offset = (seq_len - 2) % PAGE_SIZE
    last_loc = torch.full(
        (bs,), PAGE_SIZE + last_loc_offset, dtype=torch.int64, device=device
    )

    free_pages = torch.arange(1, NUM_PAGES + 1, dtype=torch.int64, device=device)
    out_indices = torch.empty((bs,), dtype=torch.int64, device=device)

    return seq_lens, last_loc, free_pages, out_indices


class TestAllocDecode(unittest.TestCase):
    """Test alloc_decode_kernel functions."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if CUDA is not available."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    # ==================== Compilation Time Tests ====================

    def _test_alloc_decode_compile_time_v1(self, bs, seq_len):
        """Test compilation time for alloc_decode_kernel_v1."""
        seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
            bs, seq_len, "cuda"
        )
        bs_pow2 = next_power_of_2(bs)

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            alloc_decode_kernel_v1[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"Old - bs={bs}, seq_len={seq_len}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def _test_alloc_decode_compile_time_v2(self, bs, seq_len):
        """Test compilation time for alloc_decode_kernel (new version)."""
        step_size_bs = 512
        seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
            bs, seq_len, "cuda"
        )

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            alloc_decode_kernel[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(step_size_bs),
                PAGE_SIZE,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"New - bs={bs}, seq_len={seq_len}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def test_alloc_decode_all_compile_v1(self):
        """Test compilation time for old kernel with various batch sizes."""
        print()
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_decode_compile_time_v1(bs, seq_len)
                    torch.cuda.empty_cache()

    def test_alloc_decode_all_compile_v2(self):
        """Test compilation time for new kernel with various batch sizes."""
        print()
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_decode_compile_time_v2(bs, seq_len)
                    torch.cuda.empty_cache()

    # ==================== Runtime Performance Tests ====================

    def _test_alloc_decode_performance_v1(self, bs, seq_len):
        """Test runtime performance for alloc_decode_kernel_v1."""
        # Warmup
        for _ in range(10):
            seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
                bs, seq_len, "cuda"
            )
            bs_pow2 = next_power_of_2(bs)
            alloc_decode_kernel_v1[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
                bs, seq_len, "cuda"
            )
            bs_pow2 = next_power_of_2(bs)
            alloc_decode_kernel_v1[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(f"Old - bs={bs}, seq_len={seq_len}: avg={avg_runtime:.3f}ms")
        return avg_runtime

    def _test_alloc_decode_performance_v2(self, bs, seq_len):
        """Test runtime performance for alloc_decode_kernel (new version)."""
        step_size_bs = 512

        # Warmup
        for _ in range(10):
            seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
                bs, seq_len, "cuda"
            )
            alloc_decode_kernel[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(step_size_bs),
                PAGE_SIZE,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            seq_lens, last_loc, free_pages, out_indices = setup_decode_test(
                bs, seq_len, "cuda"
            )
            alloc_decode_kernel[(bs,)](
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(step_size_bs),
                PAGE_SIZE,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(f"New - bs={bs}, seq_len={seq_len}: avg={avg_runtime:.3f}ms")
        return avg_runtime

    def test_alloc_decode_all_performance_v1(self):
        """Test runtime performance for old kernel with various configurations."""
        print()
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_decode_performance_v1(bs, seq_len)
                    torch.cuda.empty_cache()

    def test_alloc_decode_all_performance_v2(self):
        """Test runtime performance for new kernel with various configurations."""
        print()
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_decode_performance_v2(bs, seq_len)
                    torch.cuda.empty_cache()

    # ==================== Correctness Tests ====================

    def _test_alloc_decode_correctness(self, bs, seq_len):
        """Test consistency between alloc_decode_kernel_v1 and alloc_decode_kernel (new)."""
        step_size_bs = 512

        # Setup for old version
        seq_lens, last_loc, free_pages_v1, out_v1 = setup_decode_test(
            bs, seq_len, "cuda"
        )

        # Setup for new version (need separate free_pages and output)
        free_pages_v2 = torch.arange(1, NUM_PAGES + 1, dtype=torch.int64, device="cuda")
        out_v2 = torch.empty((bs,), dtype=torch.int64, device="cuda")

        # Run old version
        bs_pow2 = next_power_of_2(bs)
        alloc_decode_kernel_v1[(bs,)](
            seq_lens,
            last_loc,
            free_pages_v1,
            out_v1,
            bs_pow2,
            PAGE_SIZE,
        )
        torch.cuda.synchronize()

        # Run new version
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            free_pages_v2,
            out_v2,
            next_power_of_2(step_size_bs),
            PAGE_SIZE,
        )
        torch.cuda.synchronize()

        # Check bounds
        max_valid_index = NUM_PAGES * PAGE_SIZE
        self.assertTrue(
            torch.all(out_v1 >= 0) and torch.all(out_v1 < max_valid_index),
            "Old version output out of bounds",
        )
        self.assertTrue(
            torch.all(out_v2 >= 0) and torch.all(out_v2 < max_valid_index),
            "New version output out of bounds",
        )

        # Check consistency
        self.assertTrue(
            torch.equal(out_v1, out_v2),
            f"Output mismatch for bs={bs}, seq_len={seq_len}\n"
            f"Old: {out_v1[:10]}...\nNew: {out_v2[:10]}...",
        )

    def test_alloc_decode_correctness(self):
        """Test correctness with various test cases."""
        print()
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_decode_correctness(bs, seq_len)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
