import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs
from sglang.srt.utils.common import next_power_of_2

# Test configurations
BATCH_SIZES = [1, 32, 64]
DRAFT_TOKEN_NUMS = [5, 10, 20]
POOL_LEN = 8192


@triton.jit
def assign_extend_cache_locs_v1(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """Old version of assign_extend_cache_locs that depends on bs_upper as constexpr."""
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def setup_assign_extend_test(bs, draft_token_num, seq_len_base=100, device="cuda"):
    """Setup test tensors for assign_extend_cache_locs test.

    Simulates the extend phase where each request extends by draft_token_num tokens.
    - req_pool_indices: indices into req_to_token pool
    - req_to_token: 2D tensor mapping (req_idx, token_idx) -> cache_loc
    - start_offset: starting kv length for each request
    - end_offset: ending kv length for each request (start + draft_token_num)
    """
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)

    # Create req_to_token with unique cache locations
    # Each request has POOL_LEN slots, filled with sequential values
    req_to_token = torch.zeros((bs, POOL_LEN), dtype=torch.int64, device=device)
    for i in range(bs):
        # Fill with unique values: req_i gets values [i*POOL_LEN, (i+1)*POOL_LEN)
        req_to_token[i] = torch.arange(
            i * POOL_LEN, (i + 1) * POOL_LEN, dtype=torch.int64, device=device
        )

    # start_offset and end_offset define the range to copy
    start_offset = torch.full((bs,), seq_len_base, dtype=torch.int64, device=device)
    end_offset = start_offset + draft_token_num

    total_tokens = bs * draft_token_num
    out_cache_loc = torch.empty((total_tokens,), dtype=torch.int64, device=device)

    return req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc


class TestAssignExtendCacheLocs(unittest.TestCase):
    """Test assign_extend_cache_locs kernel functions."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if CUDA is not available."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    # ==================== Compilation Time Tests ====================

    def _test_compile_time_v1(self, bs, draft_token_num):
        """Test compilation time for assign_extend_cache_locs_v1."""
        req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
            setup_assign_extend_test(bs, draft_token_num, device="cuda")
        )
        bs_pow2 = next_power_of_2(bs)

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            assign_extend_cache_locs_v1[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                bs_pow2,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"Old - bs={bs}, draft_token_num={draft_token_num}: "
            f"max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def _test_compile_time_v2(self, bs, draft_token_num):
        """Test compilation time for assign_extend_cache_locs (new version)."""
        step_size_bs = 512
        max_num_loops_bs = 8  # Support up to 4096 batch size
        req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
            setup_assign_extend_test(bs, draft_token_num, device="cuda")
        )

        # Measure compilation time
        torch.cuda.synchronize()
        compile_times = []
        for _ in range(3):
            start = time.perf_counter()
            assign_extend_cache_locs[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                next_power_of_2(step_size_bs),
                max_num_loops_bs,
            )
            torch.cuda.synchronize()
            compile_times.append(time.perf_counter() - start)

        max_compile_time = max(compile_times) * 1000
        avg_compile_time = sum(compile_times) / len(compile_times) * 1000
        print(
            f"New - bs={bs}, draft_token_num={draft_token_num}: "
            f"max={max_compile_time:.2f}ms, avg={avg_compile_time:.2f}ms"
        )

    def test_assign_extend_all_compile_v1(self):
        """Test compilation time for old kernel with various batch sizes."""
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                with self.subTest(bs=bs, draft_token_num=draft_token_num):
                    self._test_compile_time_v1(bs, draft_token_num)
                    torch.cuda.empty_cache()

    def test_assign_extend_all_compile_v2(self):
        """Test compilation time for new kernel with various batch sizes."""
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                with self.subTest(bs=bs, draft_token_num=draft_token_num):
                    self._test_compile_time_v2(bs, draft_token_num)
                    torch.cuda.empty_cache()

    # ==================== Runtime Performance Tests ====================

    def _test_performance_v1(self, bs, draft_token_num):
        """Test runtime performance for assign_extend_cache_locs_v1."""
        # Warmup
        for _ in range(10):
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
                setup_assign_extend_test(bs, draft_token_num, device="cuda")
            )
            bs_pow2 = next_power_of_2(bs)
            assign_extend_cache_locs_v1[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                bs_pow2,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
                setup_assign_extend_test(bs, draft_token_num, device="cuda")
            )
            bs_pow2 = next_power_of_2(bs)
            assign_extend_cache_locs_v1[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                bs_pow2,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(
            f"Old - bs={bs}, draft_token_num={draft_token_num}: avg={avg_runtime:.3f}ms"
        )
        return avg_runtime

    def _test_performance_v2(self, bs, draft_token_num):
        """Test runtime performance for assign_extend_cache_locs (new version)."""
        step_size_bs = 512
        max_num_loops_bs = 8

        # Warmup
        for _ in range(10):
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
                setup_assign_extend_test(bs, draft_token_num, device="cuda")
            )
            assign_extend_cache_locs[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                next_power_of_2(step_size_bs),
                max_num_loops_bs,
            )
            torch.cuda.synchronize()

        # Performance measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for _ in range(100):
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
                setup_assign_extend_test(bs, draft_token_num, device="cuda")
            )
            assign_extend_cache_locs[(bs,)](
                req_pool_indices,
                req_to_token,
                start_offset,
                end_offset,
                out_cache_loc,
                POOL_LEN,
                next_power_of_2(step_size_bs),
                max_num_loops_bs,
            )

        end_event.record()
        torch.cuda.synchronize()

        total_runtime = start_event.elapsed_time(end_event)
        avg_runtime = total_runtime / 100
        print(
            f"New - bs={bs}, draft_token_num={draft_token_num}: avg={avg_runtime:.3f}ms"
        )
        return avg_runtime

    def test_assign_extend_all_performance_v1(self):
        """Test runtime performance for old kernel with various configurations."""
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                with self.subTest(bs=bs, draft_token_num=draft_token_num):
                    self._test_performance_v1(bs, draft_token_num)
                    torch.cuda.empty_cache()

    def test_assign_extend_all_performance_v2(self):
        """Test runtime performance for new kernel with various configurations."""
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                with self.subTest(bs=bs, draft_token_num=draft_token_num):
                    self._test_performance_v2(bs, draft_token_num)
                    torch.cuda.empty_cache()

    # ==================== Correctness Tests ====================

    def _test_correctness(self, bs, draft_token_num, seq_len_base):
        """Test consistency between old and new versions of assign_extend_cache_locs."""
        step_size_bs = 512

        # Setup for old version
        req_pool_indices, req_to_token, start_offset, end_offset, out_v1 = (
            setup_assign_extend_test(bs, draft_token_num, seq_len_base, device="cuda")
        )

        # Setup for new version (need separate output)
        out_v2 = torch.empty((bs * draft_token_num,), dtype=torch.int64, device="cuda")

        # Run old version
        bs_pow2 = next_power_of_2(bs)
        assign_extend_cache_locs_v1[(bs,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_v1,
            POOL_LEN,
            bs_pow2,
        )
        torch.cuda.synchronize()

        # Run new version
        max_num_loops_bs = 8
        assign_extend_cache_locs[(bs,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_v2,
            POOL_LEN,
            next_power_of_2(step_size_bs),
            max_num_loops_bs,
        )
        torch.cuda.synchronize()

        # Check bounds
        max_valid_index = bs * POOL_LEN
        self.assertTrue(
            torch.all(out_v1 >= 0) and torch.all(out_v1 < max_valid_index),
            "Old version output out of bounds",
        )
        self.assertTrue(
            torch.all(out_v2 >= 0) and torch.all(out_v2 < max_valid_index),
            "New version output out of bounds",
        )

        # Check consistency
        if not torch.equal(out_v1, out_v2):
            diff_mask = out_v1 != out_v2
            diff_indices = torch.where(diff_mask)[0]
            first_diff_idx = diff_indices[0].item()
            # Find which request this belongs to
            req_idx = first_diff_idx // draft_token_num
            self.fail(
                f"Output mismatch for bs={bs}, draft_token_num={draft_token_num}, "
                f"seq_len_base={seq_len_base}\n"
                f"First diff at index {first_diff_idx} (request {req_idx})\n"
                f"Old[{first_diff_idx}]={out_v1[first_diff_idx].item()}, "
                f"New[{first_diff_idx}]={out_v2[first_diff_idx].item()}\n"
                f"Total diffs: {diff_mask.sum().item()}"
            )

    def _test_correctness_values(self, bs, draft_token_num, seq_len_base):
        """Test that output values are correct (not just consistent between versions)."""
        step_size_bs = 512
        max_num_loops_bs = 8

        req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc = (
            setup_assign_extend_test(bs, draft_token_num, seq_len_base, device="cuda")
        )

        assign_extend_cache_locs[(bs,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            POOL_LEN,
            next_power_of_2(step_size_bs),
            max_num_loops_bs,
        )
        torch.cuda.synchronize()

        # Verify output values
        # For each request i, the output should contain:
        # req_to_token[i, start_offset[i]:end_offset[i]]
        # which equals [i*POOL_LEN + seq_len_base, i*POOL_LEN + seq_len_base + draft_token_num)
        expected = []
        for i in range(bs):
            for j in range(draft_token_num):
                expected.append(i * POOL_LEN + seq_len_base + j)
        expected = torch.tensor(expected, dtype=torch.int64, device="cuda")

        if not torch.equal(out_cache_loc, expected):
            diff_mask = out_cache_loc != expected
            diff_indices = torch.where(diff_mask)[0]
            first_diff_idx = diff_indices[0].item()
            req_idx = first_diff_idx // draft_token_num
            token_idx = first_diff_idx % draft_token_num
            self.fail(
                f"Output values mismatch for bs={bs}, draft_token_num={draft_token_num}\n"
                f"First diff at index {first_diff_idx} (request {req_idx}, token {token_idx})\n"
                f"Expected: {expected[first_diff_idx].item()}, Got: {out_cache_loc[first_diff_idx].item()}\n"
                f"Total diffs: {diff_mask.sum().item()}"
            )

    def test_assign_extend_correctness(self):
        """Test correctness by comparing old and new kernel outputs.

        Note: Old kernel uses bs_upper as constexpr which may have issues
        with very large batch sizes. For bs > 512, use test_assign_extend_correctness_values
        to verify new kernel correctness independently.
        """
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                for seq_len_base in [100, 200, 300, 400, 500]:
                    with self.subTest(
                        bs=bs,
                        draft_token_num=draft_token_num,
                        seq_len_base=seq_len_base,
                    ):
                        self._test_correctness(bs, draft_token_num, seq_len_base)
                        torch.cuda.empty_cache()

    def test_assign_extend_correctness_values(self):
        """Test that output values are correct (independent of old kernel)."""
        print()
        for bs in BATCH_SIZES:
            for draft_token_num in DRAFT_TOKEN_NUMS:
                for seq_len_base in [100, 200, 300, 400, 500]:
                    with self.subTest(
                        bs=bs,
                        draft_token_num=draft_token_num,
                        seq_len_base=seq_len_base,
                    ):
                        self._test_correctness_values(bs, draft_token_num, seq_len_base)
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
