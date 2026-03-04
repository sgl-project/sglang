import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.eagle_info_v2 import fill_accepted_out_cache_loc
from sglang.srt.utils.common import next_power_of_2

# Test configurations
BATCH_SIZES = [1, 32, 64, 128, 256, 512]
ACCEPT_RATES = [0.3, 0.5, 0.8, 1.0]  # Fraction of valid entries
STEP_SIZE = 512
MAX_NUM_LOOPS = 8


@triton.jit
def fill_accepted_out_cache_loc_v1(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    """V1 version of fill_accepted_out_cache_loc that depends on size_upper as constexpr."""
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


def setup_fill_accepted_test(size, accept_rate=0.5, device="cuda"):
    """Setup test tensors for fill_accepted_out_cache_loc test.

    Args:
        size: Total number of entries in accept_index
        accept_rate: Fraction of entries that are valid (not -1)
        device: Device to create tensors on

    Returns:
        accept_index: Tensor with some valid indices (>=0) and some invalid (-1)
        out_cache_loc: Source cache locations
        accepted_out_cache_loc: Output tensor for accepted locations
        num_valid: Number of valid entries
    """
    # Create accept_index with some valid and some invalid entries
    accept_index = torch.full((size,), -1, dtype=torch.int32, device=device)
    num_valid = int(size * accept_rate)

    # Set first num_valid entries to valid indices
    if num_valid > 0:
        valid_indices = torch.arange(num_valid, dtype=torch.int32, device=device)
        accept_index[:num_valid] = valid_indices

    # Shuffle to distribute valid entries randomly
    perm = torch.randperm(size, device=device)
    accept_index = accept_index[perm]

    # Create out_cache_loc with unique values
    out_cache_loc = torch.arange(size * 2, dtype=torch.int64, device=device)

    # Output tensor
    accepted_out_cache_loc = torch.empty((num_valid,), dtype=torch.int64, device=device)

    return accept_index, out_cache_loc, accepted_out_cache_loc, num_valid


class TestFillAcceptedOutCacheLoc(unittest.TestCase):
    """Test fill_accepted_out_cache_loc kernel functions."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if CUDA is not available."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    # ==================== Compilation Time Tests ====================

    def _test_compile_time_v1(self, size):
        """Test compilation time for fill_accepted_out_cache_loc_v1."""
        accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
            setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
        )
        size_pow2 = next_power_of_2(size)

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            fill_accepted_out_cache_loc_v1[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                size_pow2,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"V1 - size={size}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def _test_compile_time_v2(self, size):
        """Test compilation time for fill_accepted_out_cache_loc (v2 version)."""
        accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
            setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
        )

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            fill_accepted_out_cache_loc[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                STEP_SIZE,
                MAX_NUM_LOOPS,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"V2 - size={size}: max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def test_fill_accepted_all_compile_v1(self):
        """Test compilation time for v1 kernel with various sizes."""
        print()
        for size in BATCH_SIZES:
            with self.subTest(size=size):
                self._test_compile_time_v1(size)
                torch.cuda.empty_cache()

    def test_fill_accepted_all_compile_v2(self):
        """Test compilation time for v2 kernel with various sizes."""
        print()
        for size in BATCH_SIZES:
            with self.subTest(size=size):
                self._test_compile_time_v2(size)

    # ==================== Runtime Performance Tests ====================

    def _test_performance_v1(self, size):
        """Test runtime performance for fill_accepted_out_cache_loc_v1."""
        # Warmup
        for _ in range(10):
            accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
                setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
            )
            size_pow2 = next_power_of_2(size)
            fill_accepted_out_cache_loc_v1[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                size_pow2,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
                setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
            )
            size_pow2 = next_power_of_2(size)
            fill_accepted_out_cache_loc_v1[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                size_pow2,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(f"V1 - size={size}: avg={avg_runtime:.3f}ms")
        return avg_runtime

    def _test_performance_v2(self, size):
        """Test runtime performance for fill_accepted_out_cache_loc (v2 version)."""
        # Warmup
        for _ in range(10):
            accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
                setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
            )
            fill_accepted_out_cache_loc[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                STEP_SIZE,
                MAX_NUM_LOOPS,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            accept_index, out_cache_loc, accepted_out_cache_loc, _ = (
                setup_fill_accepted_test(size, accept_rate=0.5, device="cuda")
            )
            fill_accepted_out_cache_loc[(size,)](
                accept_index,
                out_cache_loc,
                accepted_out_cache_loc,
                STEP_SIZE,
                MAX_NUM_LOOPS,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(f"V2 - size={size}: avg={avg_runtime:.3f}ms")
        return avg_runtime

    def test_fill_accepted_all_performance_v1(self):
        """Test runtime performance for v1 kernel with various sizes."""
        print()
        for size in [64, 128, 256]:
            with self.subTest(size=size):
                self._test_performance_v1(size)

    def test_fill_accepted_all_performance_v2(self):
        """Test runtime performance for v2 kernel with various sizes."""
        print()
        for size in [64, 128, 256, 512]:
            with self.subTest(size=size):
                self._test_performance_v2(size)

    # ==================== Correctness Tests ====================

    def _test_correctness(self, size, accept_rate):
        """Test consistency between v1 and v2 versions of fill_accepted_out_cache_loc."""
        # Use fixed seed for reproducibility
        torch.manual_seed(42)

        # Setup for v1
        accept_index, out_cache_loc, out_v1, num_valid = setup_fill_accepted_test(
            size, accept_rate, device="cuda"
        )

        # Setup for v2 (need separate output)
        out_v2 = torch.empty((num_valid,), dtype=torch.int64, device="cuda")

        if num_valid == 0:
            # Skip if no valid entries
            return

        # Run v1 version
        size_pow2 = next_power_of_2(size)
        fill_accepted_out_cache_loc_v1[(size,)](
            accept_index,
            out_cache_loc,
            out_v1,
            size_pow2,
        )
        torch.cuda.synchronize()

        # Run v2 version
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            out_cache_loc,
            out_v2,
            STEP_SIZE,
            MAX_NUM_LOOPS,
        )
        torch.cuda.synchronize()

        # Check consistency
        if not torch.equal(out_v1, out_v2):
            diff_mask = out_v1 != out_v2
            diff_indices = torch.where(diff_mask)[0]
            first_diff_idx = diff_indices[0].item()
            self.fail(
                f"Output mismatch for size={size}, accept_rate={accept_rate}\n"
                f"First diff at index {first_diff_idx}\n"
                f"V1[{first_diff_idx}]={out_v1[first_diff_idx].item()}, "
                f"V2[{first_diff_idx}]={out_v2[first_diff_idx].item()}\n"
                f"Total diffs: {diff_mask.sum().item()}"
            )

    def test_fill_accepted_correctness(self):
        """Test correctness by comparing v1 and v2 kernel outputs."""
        print()
        for size in BATCH_SIZES:
            for accept_rate in ACCEPT_RATES:
                with self.subTest(size=size, accept_rate=accept_rate):
                    self._test_correctness(size, accept_rate)
                    torch.cuda.empty_cache()

    def test_fill_accepted_large_batch_v2(self):
        """Test correctness with large sizes (> step_size)."""
        print()
        test_cases = [
            # (size, accept_rate) - sizes > STEP_SIZE to test loop
            (513, 0.5),
            (768, 0.5),
            (1024, 0.5),
        ]

        for size, accept_rate in test_cases:
            with self.subTest(size=size, accept_rate=accept_rate):
                self._test_correctness(size, accept_rate)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
