import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.allocator import alloc_extend_kernel
from sglang.srt.utils import next_power_of_2

# Test configurations
BATCH_SIZES = [1, 32, 64]
SEQ_LENS = [1, 1000, 32000, 64000]
PAGE_SIZE = 64
NUM_PAGES = 100000


@triton.jit
def alloc_extend_kernel_old(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


def setup_alloc_test(bs, seq_len, prefix_len=0, device="cuda"):
    """Setup test tensors for allocation test."""
    prefix_lens = torch.full((bs,), prefix_len, dtype=torch.int64, device=device)
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)
    last_loc = torch.full((bs,), prefix_len - 1, dtype=torch.int64, device=device)
    free_pages = torch.arange(1, NUM_PAGES + 1, dtype=torch.int64, device=device)

    extend_num_tokens = (seq_lens - prefix_lens).sum().item()
    out_indices = torch.empty((extend_num_tokens,), dtype=torch.int64, device=device)

    return prefix_lens, seq_lens, last_loc, free_pages, out_indices, extend_num_tokens


class TestAllocExtend(unittest.TestCase):
    """Test alloc_extend_kernel and alloc_extend_kernel_v2 functions."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if CUDA is not available."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    # ==================== Compilation Time Tests ====================

    def _test_alloc_extend_compile_time_v1(self, bs, seq_len):
        """Test compilation time for alloc_extend_kernel."""
        prefix_len = 0
        prefix_lens, seq_lens, last_loc, free_pages, out_indices, extend_num_tokens = (
            setup_alloc_test(bs, seq_len, prefix_len, "cuda")
        )

        # Compute required parameters
        max_extend_tokens_pow2 = next_power_of_2(extend_num_tokens)
        bs_pow2 = next_power_of_2(bs)

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            alloc_extend_kernel_old[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
                max_extend_tokens_pow2,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"V1 - bs={bs}, seq_len={seq_len}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def _test_alloc_extend_compile_time_v2(self, bs, seq_len):
        """Test compilation time for alloc_extend_kernel_v2."""
        prefix_len = 0
        step = 512
        max_bs = 128

        prefix_lens, seq_lens, last_loc, free_pages, out_indices, extend_num_tokens = (
            setup_alloc_test(bs, seq_len, prefix_len, "cuda")
        )

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(max_bs),
                PAGE_SIZE,
                next_power_of_2(step),
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000

        print(
            f"V2 - bs={bs}, seq_len={seq_len}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    # ==================== Runtime Performance Tests ====================

    def _test_alloc_extend_performance_v1(self, bs, seq_len):
        """Test runtime performance for alloc_extend_kernel."""
        prefix_len = 0

        # Warmup
        for _ in range(10):
            (
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                extend_num_tokens,
            ) = setup_alloc_test(bs, seq_len, prefix_len, "cuda")
            max_extend_tokens_pow2 = next_power_of_2(extend_num_tokens)
            bs_pow2 = next_power_of_2(bs)

            alloc_extend_kernel_old[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
                max_extend_tokens_pow2,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            (
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                extend_num_tokens,
            ) = setup_alloc_test(bs, seq_len, prefix_len, "cuda")
            max_extend_tokens_pow2 = next_power_of_2(extend_num_tokens)
            bs_pow2 = next_power_of_2(bs)

            alloc_extend_kernel_old[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                bs_pow2,
                PAGE_SIZE,
                max_extend_tokens_pow2,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100

        print(f"V1 - bs={bs}, seq_len={seq_len} " f"avg={avg_runtime:.3f}ms")

    def _test_alloc_extend_performance_v2(self, bs, seq_len):
        """Test runtime performance for alloc_extend_kernel_v2."""
        prefix_len = 0
        step = 512
        max_bs = 128

        # Warmup
        for _ in range(10):
            (
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                extend_num_tokens,
            ) = setup_alloc_test(bs, seq_len, prefix_len, "cuda")

            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(max_bs),
                PAGE_SIZE,
                next_power_of_2(step),
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            (
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                extend_num_tokens,
            ) = setup_alloc_test(bs, seq_len, prefix_len, "cuda")

            alloc_extend_kernel[(bs,)](
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                next_power_of_2(max_bs),
                PAGE_SIZE,
                next_power_of_2(step),
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100

        print(f"V2 - bs={bs}, seq_len={seq_len} " f"avg={avg_runtime:.3f}ms")

    # ==================== Correctness Tests ====================

    def _test_alloc_extend_correctness(self, bs, seq_len, prefix_len):
        """Test consistency between alloc_extend_kernel and alloc_extend_kernel_v2."""
        step = 512
        max_bs = 128

        # Setup for v1
        prefix_lens, seq_lens, last_loc, free_pages_v1, out_v1, extend_num_tokens = (
            setup_alloc_test(bs, seq_len, prefix_len, "cuda")
        )

        # Setup for v2 (need separate free_pages)
        free_pages_v2 = torch.arange(1, NUM_PAGES + 1, dtype=torch.int64, device="cuda")
        out_v2 = torch.empty((extend_num_tokens,), dtype=torch.int64, device="cuda")

        # Run v1
        max_extend_tokens_pow2 = next_power_of_2(extend_num_tokens)
        bs_pow2 = next_power_of_2(bs)

        alloc_extend_kernel_old[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages_v1,
            out_v1,
            bs_pow2,
            PAGE_SIZE,
            max_extend_tokens_pow2,
        )
        torch.cuda.synchronize()

        # Run v2
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages_v2,
            out_v2,
            next_power_of_2(max_bs),
            PAGE_SIZE,
            next_power_of_2(step),
        )
        torch.cuda.synchronize()

        # Check bounds
        max_valid_index = NUM_PAGES * PAGE_SIZE
        self.assertTrue(
            torch.all(out_v1 >= 0) and torch.all(out_v1 < max_valid_index),
            "V1 output out of bounds",
        )
        self.assertTrue(
            torch.all(out_v2 >= 0) and torch.all(out_v2 < max_valid_index),
            "V2 output out of bounds",
        )

        # Check consistency
        self.assertTrue(
            torch.equal(out_v1, out_v2),
            f"Output mismatch for bs={bs}, seq_len={seq_len}, prefix_len={prefix_len}",
        )

    # ==================== Test Generation Methods ====================

    def test_all_compile_time_v1(self):
        """Test compilation time for alloc_extend_kernel with various configurations."""
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                if bs >= 32 and seq_len >= 32000:  # skip v1 compile which is too slow
                    continue
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_extend_compile_time_v1(bs, seq_len)
                    torch.cuda.empty_cache()

    def test_all_compile_time_v2(self):
        """Test compilation time for alloc_extend_kernel_v2 with various configurations."""
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_extend_compile_time_v2(bs, seq_len)

    def test_all_performance_v1(self):
        """Test runtime performance for alloc_extend_kernel with various configurations."""
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                if bs >= 32 and seq_len >= 32000:  # skip v1 compile which is too slow
                    continue
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_extend_performance_v1(bs, seq_len)

    def test_all_performance_v2(self):
        """Test runtime performance for alloc_extend_kernel_v2 with various configurations."""
        for bs in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                with self.subTest(bs=bs, seq_len=seq_len):
                    self._test_alloc_extend_performance_v2(bs, seq_len)

    def test_all_correctness(self):
        """Test correctness with various test cases."""
        test_cases = [
            (1, 100, 0),
            (4, 500, 100),
            (8, 1000, 200),
            (16, 2000, 500),
            (32, 5000, 1000),
            (1, 63, 0),
            (1, 64, 0),
            (1, 65, 0),
            (1, 511, 0),
            (1, 513, 0),
            (1, 1024, 0),
            (4, 30, 10),
            (4, 70, 30),
        ]

        for bs, seq_len, prefix_len in test_cases:
            with self.subTest(bs=bs, seq_len=seq_len, prefix_len=prefix_len):
                self._test_alloc_extend_correctness(bs, seq_len, prefix_len)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
