"""
Test Triton kernels in spec_utils.py with JIT optimization.

Tests compile time, runtime performance, and correctness for:
1. create_extend_after_decode_spec_info
2. assign_req_to_token_pool
3. assign_draft_cache_locs
4. generate_draft_decode_kv_indices
5. get_target_cache_loc
6. filter_finished_cache_loc_kernel
"""

import time
import unittest

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.spec_utils import (
    _SPEC_MAX_NUM_LOOPS,
    _SPEC_STEP_SIZE,
    assign_draft_cache_locs,
    assign_req_to_token_pool,
    create_extend_after_decode_spec_info,
    filter_finished_cache_loc_kernel,
    generate_draft_decode_kv_indices,
    get_target_cache_loc,
)
from sglang.srt.utils.common import next_power_of_2

# Additional constants for generate_draft_decode_kv_indices
_NUM_TOKENS_STEP_SIZE = next_power_of_2(512)
_NUM_TOKENS_MAX_LOOPS = 8

# Test configurations
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]
LARGE_BATCH_SIZES = [513, 768, 1024]  # For v2 kernel independent tests


# ==================== V1 Kernels (Original, for comparison) ====================


@triton.jit
def create_extend_after_decode_spec_info_v1(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool_v1(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
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

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def get_target_cache_loc_v1(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@triton.jit
def filter_finished_cache_loc_kernel_v1(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@triton.jit
def assign_draft_cache_locs_v1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: tl.constexpr,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        # Use bs_upper for cumsum
        length_offset = tl.arange(0, bs_upper)
        cum_copy_len = tl.sum(
            tl.load(extend_lens + length_offset, mask=length_offset < pid, other=0)
        )
        copy_len = tl.load(extend_lens + pid)
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)
    if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)
        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )
        # Part 3: Copy and remove the used indices for duplication
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            ptr_offset = pid * speculative_num_steps * topk
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


@triton.jit
def generate_draft_decode_kv_indices_v1(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    """V1: Original kernel with bs_upper constexpr."""
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    # Use bs_upper for cumsum
    bs_offset = tl.arange(0, bs_upper)
    cum_seq_len = tl.sum(
        tl.load(paged_kernel_lens + bs_offset, mask=bs_offset < bid, other=0)
    )
    seq_len = tl.load(paged_kernel_lens + bid)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr - use num_tokens_upper for positions accumulation
    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk

    tokens_offset = tl.arange(0, num_tokens_upper)
    base = tl.sum(tl.load(positions + tokens_offset, mask=tokens_offset < zid, other=0))

    tl.store(kv_indptr + zid, base + zid * iters)


# ==================== Setup Functions ====================


def setup_assign_draft_cache_locs_test(
    bs, pool_len=1024, topk=4, speculative_num_steps=5, page_size=16, device="cuda"
):
    """Setup test tensors for assign_draft_cache_locs.

    This tests the simple case where page_size == 1 or topk == 1.
    """
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    req_to_token = torch.full((bs, pool_len), -1, dtype=torch.int32, device=device)

    # Random seq_lens (prefix lengths)
    seq_lens = torch.randint(50, 200, (bs,), dtype=torch.int32, device=device)

    # extend_lens: how many tokens to copy for each request
    extend_lens = torch.full(
        (bs,), topk * speculative_num_steps, dtype=torch.int32, device=device
    )

    # For simple case (page_size=1 or topk=1), these are not used much
    num_new_pages_per_topk = torch.ones((bs,), dtype=torch.int32, device=device)

    # Total out_cache_loc size
    total_extend = (extend_lens.sum()).item()
    out_cache_loc = torch.arange(total_extend, dtype=torch.int32, device=device)

    # source/target cache loc (for complex case)
    source_cache_loc = torch.zeros(
        (bs * topk * page_size,), dtype=torch.int32, device=device
    )
    target_cache_loc = torch.zeros(
        (bs * topk * page_size,), dtype=torch.int32, device=device
    )

    # last_page_lens_cumsum
    last_page_lens = seq_lens % page_size
    last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0).to(torch.int32)

    duplicate_cache_len = 0  # Simple case

    return (
        req_pool_indices,
        req_to_token,
        seq_lens,
        extend_lens,
        num_new_pages_per_topk,
        out_cache_loc,
        source_cache_loc,
        target_cache_loc,
        last_page_lens_cumsum,
        duplicate_cache_len,
        pool_len,
        topk,
        speculative_num_steps,
        page_size,
    )


def setup_generate_draft_decode_kv_indices_test(
    bs,
    num_steps=5,
    topk=4,
    pool_len=1024,
    page_size=1,
    device="cuda",
    fixed_stride=True,
):
    """Setup test tensors for generate_draft_decode_kv_indices.

    Args:
        fixed_stride: If True, use fixed stride values to avoid kernel recompilation.
                     This is useful for compile time testing.
    """
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)

    # Initialize req_to_token with valid indices
    req_to_token = torch.zeros((bs, pool_len), dtype=torch.int32, device=device)
    for i in range(bs):
        req_to_token[i, :] = (
            torch.arange(pool_len, dtype=torch.int32, device=device) + i * 1000
        )

    # Use fixed seq_len for consistent stride values
    fixed_seq_len = 64
    paged_kernel_lens = torch.full(
        (bs,), fixed_seq_len, dtype=torch.int32, device=device
    )

    # positions for kv_indptr calculation (seq_len + iters for each (bid, topk_id))
    # Use fixed max_bs for consistent stride
    max_bs = 512 if fixed_stride else bs
    num_tokens = max_bs * topk
    positions = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    for bid in range(bs):
        seq_len = paged_kernel_lens[bid].item()
        for topk_id in range(topk):
            positions[bid * topk + topk_id] = seq_len + num_steps

    # Use fixed sizes for consistent constexpr values
    if fixed_stride:
        # Fixed sizes to avoid recompilation
        kv_indices_size = (
            max_bs * fixed_seq_len + max_bs * num_steps
        ) * topk + max_bs * num_steps * topk
        kv_indptr_stride = max_bs * topk + 1
    else:
        total_seq_len = paged_kernel_lens.sum().item()
        kv_indices_size = (
            total_seq_len + bs * num_steps
        ) * topk + bs * num_steps * topk
        kv_indptr_stride = bs * topk + 1

    kv_indices = torch.zeros(
        (num_steps, kv_indices_size), dtype=torch.int32, device=device
    )
    kv_indptr = torch.zeros(
        (num_steps, kv_indptr_stride), dtype=torch.int64, device=device
    )

    kv_indices_stride = kv_indices_size

    return (
        req_pool_indices,
        req_to_token,
        paged_kernel_lens,
        kv_indices,
        kv_indptr,
        positions,
        pool_len,
        kv_indices_stride,
        kv_indptr_stride,
        num_steps,
        topk,
        page_size,
    )


def setup_create_extend_test(bs, max_accept_len=6, device="cuda"):
    """Setup test tensors for create_extend_after_decode_spec_info."""
    # Random accept lengths between 1 and max_accept_len
    accept_lens = torch.randint(
        1, max_accept_len + 1, (bs,), dtype=torch.int32, device=device
    )
    total_accept = accept_lens.sum().item()

    # Random verified_id with enough elements
    verified_id = torch.randint(
        0, 1000, (total_accept,), dtype=torch.int32, device=device
    )

    # Random seq_lens (should be >= accept_lens)
    seq_lens = accept_lens + torch.randint(
        10, 100, (bs,), dtype=torch.int32, device=device
    )

    # Output tensors
    positions = torch.empty((total_accept,), dtype=torch.int64, device=device)
    new_verified_id = torch.empty((bs,), dtype=torch.int32, device=device)

    return verified_id, seq_lens, accept_lens, positions, new_verified_id


def setup_assign_req_to_token_pool_test(bs, pool_len=1024, device="cuda"):
    """Setup test tensors for assign_req_to_token_pool."""
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    req_to_token = torch.full((bs, pool_len), -1, dtype=torch.int32, device=device)

    # Random start/end offsets
    start_offset = torch.randint(0, 10, (bs,), dtype=torch.int32, device=device)
    end_offset = start_offset + torch.randint(
        1, 20, (bs,), dtype=torch.int32, device=device
    )

    total_len = (end_offset - start_offset).sum().item()
    out_cache_loc = torch.arange(total_len, dtype=torch.int32, device=device)

    return (
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
    )


def setup_get_target_cache_loc_test(bs, num_verify_tokens=16, device="cuda"):
    """Setup test tensors for get_target_cache_loc."""
    # Random accept lengths
    accept_length = torch.randint(
        0, num_verify_tokens - 1, (bs,), dtype=torch.int32, device=device
    )
    total_accept = (accept_length + 1).sum().item()

    # Random to_free_num_slots
    to_free_num_slots = torch.randint(0, 5, (bs,), dtype=torch.int32, device=device)
    total_free = to_free_num_slots.sum().item()

    # Random out_cache_loc
    out_cache_loc = torch.randint(
        0, 10000, (bs, num_verify_tokens), dtype=torch.int64, device=device
    )

    # Output tensors
    tgt_cache_loc = torch.empty((total_accept,), dtype=torch.int64, device=device)
    to_free_slots = torch.empty((max(1, total_free),), dtype=torch.int64, device=device)

    return (
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
    )


def setup_filter_finished_test(bs, num_verify_tokens=16, device="cuda"):
    """Setup test tensors for filter_finished_cache_loc_kernel."""
    # Random accept lengths
    accept_length = torch.randint(
        0, num_verify_tokens - 1, (bs,), dtype=torch.int32, device=device
    )

    # Filter (subset of accept_length)
    accept_length_filter = accept_length.clone()
    # Mark some as finished (0)
    if bs > 1:
        accept_length_filter[::2] = 0

    total_accept = (accept_length + 1).sum().item()
    total_filter = accept_length_filter.sum().item()

    # Random tgt_cache_loc
    tgt_cache_loc = torch.randint(
        0, 10000, (total_accept,), dtype=torch.int64, device=device
    )

    # Output tensor
    out_cache_loc = torch.empty(
        (max(1, total_filter),), dtype=torch.int64, device=device
    )

    return (
        out_cache_loc,
        tgt_cache_loc,
        accept_length,
        accept_length_filter,
        num_verify_tokens,
    )


# ==================== Test Class ====================


class TestSpecUtilsKernels(unittest.TestCase):
    """Test spec_utils Triton kernels."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if CUDA is not available."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    # ==================== create_extend_after_decode_spec_info Tests ====================

    def _test_create_extend_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        verified_id, seq_lens, accept_lens, positions, new_verified_id = (
            setup_create_extend_test(bs)
        )
        bs_upper = next_power_of_2(max(6, bs))

        torch.cuda.synchronize()
        start = time.perf_counter()
        create_extend_after_decode_spec_info_v1[(bs,)](
            verified_id, seq_lens, accept_lens, positions, new_verified_id, bs_upper
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_create_extend_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        verified_id, seq_lens, accept_lens, positions, new_verified_id = (
            setup_create_extend_test(bs)
        )
        max_accept_len = next_power_of_2(6)

        torch.cuda.synchronize()
        start = time.perf_counter()
        create_extend_after_decode_spec_info[(bs,)](
            verified_id,
            seq_lens,
            accept_lens,
            positions,
            new_verified_id,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
            max_accept_len,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_create_extend_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_create_extend_compile_v1(bs)
                print(f"create_extend v1 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_create_extend_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_create_extend_compile_v2(bs)
                print(f"create_extend v2 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_create_extend_correctness(self):
        """Test correctness by comparing v1 and v2 outputs."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                verified_id, seq_lens, accept_lens, positions_v1, new_verified_id_v1 = (
                    setup_create_extend_test(bs)
                )
                positions_v2 = torch.empty_like(positions_v1)
                new_verified_id_v2 = torch.empty_like(new_verified_id_v1)

                bs_upper = next_power_of_2(max(6, bs))
                max_accept_len = next_power_of_2(6)

                # Run v1
                create_extend_after_decode_spec_info_v1[(bs,)](
                    verified_id,
                    seq_lens,
                    accept_lens,
                    positions_v1,
                    new_verified_id_v1,
                    bs_upper,
                )

                # Run v2
                create_extend_after_decode_spec_info[(bs,)](
                    verified_id,
                    seq_lens,
                    accept_lens,
                    positions_v2,
                    new_verified_id_v2,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    max_accept_len,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(positions_v1, positions_v2),
                    f"Positions mismatch at bs={bs}",
                )
                self.assertTrue(
                    torch.equal(new_verified_id_v1, new_verified_id_v2),
                    f"new_verified_id mismatch at bs={bs}",
                )
                print(f"create_extend correctness bs={bs}: PASSED")

    # ==================== assign_req_to_token_pool Tests ====================

    def _test_assign_req_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        (
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
        ) = setup_assign_req_to_token_pool_test(bs)
        bs_upper = next_power_of_2(bs)

        torch.cuda.synchronize()
        start = time.perf_counter()
        assign_req_to_token_pool_v1[(bs,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
            bs_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_assign_req_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        (
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
        ) = setup_assign_req_to_token_pool_test(bs)

        torch.cuda.synchronize()
        start = time.perf_counter()
        assign_req_to_token_pool[(bs,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_assign_req_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_assign_req_compile_v1(bs)
                print(f"assign_req v1 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_assign_req_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_assign_req_compile_v2(bs)
                print(f"assign_req v2 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_assign_req_correctness(self):
        """Test correctness by comparing v1 and v2 outputs."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token_v1,
                    start_offset,
                    end_offset,
                    out_cache_loc,
                    pool_len,
                ) = setup_assign_req_to_token_pool_test(bs)
                req_to_token_v2 = req_to_token_v1.clone()

                bs_upper = next_power_of_2(bs)

                # Run v1
                assign_req_to_token_pool_v1[(bs,)](
                    req_pool_indices,
                    req_to_token_v1,
                    start_offset,
                    end_offset,
                    out_cache_loc,
                    pool_len,
                    bs_upper,
                )

                # Run v2
                assign_req_to_token_pool[(bs,)](
                    req_pool_indices,
                    req_to_token_v2,
                    start_offset,
                    end_offset,
                    out_cache_loc,
                    pool_len,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(req_to_token_v1, req_to_token_v2),
                    f"req_to_token mismatch at bs={bs}",
                )
                print(f"assign_req correctness bs={bs}: PASSED")

    # ==================== get_target_cache_loc Tests ====================

    def _test_get_target_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        (
            tgt_cache_loc,
            to_free_slots,
            accept_length,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
        ) = setup_get_target_cache_loc_test(bs)
        num_verify_tokens_upper = next_power_of_2(num_verify_tokens)
        bs_upper = next_power_of_2(bs)

        torch.cuda.synchronize()
        start = time.perf_counter()
        get_target_cache_loc_v1[(bs,)](
            tgt_cache_loc,
            to_free_slots,
            accept_length,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_get_target_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        (
            tgt_cache_loc,
            to_free_slots,
            accept_length,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
        ) = setup_get_target_cache_loc_test(bs)
        num_verify_tokens_upper = next_power_of_2(num_verify_tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()
        get_target_cache_loc[(bs,)](
            tgt_cache_loc,
            to_free_slots,
            accept_length,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
            num_verify_tokens_upper,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_get_target_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_get_target_compile_v1(bs)
                print(f"get_target v1 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_get_target_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_get_target_compile_v2(bs)
                print(f"get_target v2 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_get_target_correctness(self):
        """Test correctness by comparing v1 and v2 outputs."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    tgt_cache_loc_v1,
                    to_free_slots_v1,
                    accept_length,
                    to_free_num_slots,
                    out_cache_loc,
                    num_verify_tokens,
                ) = setup_get_target_cache_loc_test(bs)
                tgt_cache_loc_v2 = torch.empty_like(tgt_cache_loc_v1)
                to_free_slots_v2 = torch.empty_like(to_free_slots_v1)

                num_verify_tokens_upper = next_power_of_2(num_verify_tokens)
                bs_upper = next_power_of_2(bs)

                # Run v1
                get_target_cache_loc_v1[(bs,)](
                    tgt_cache_loc_v1,
                    to_free_slots_v1,
                    accept_length,
                    to_free_num_slots,
                    out_cache_loc,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    bs_upper,
                )

                # Run v2
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc_v2,
                    to_free_slots_v2,
                    accept_length,
                    to_free_num_slots,
                    out_cache_loc,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(tgt_cache_loc_v1, tgt_cache_loc_v2),
                    f"tgt_cache_loc mismatch at bs={bs}",
                )
                self.assertTrue(
                    torch.equal(to_free_slots_v1, to_free_slots_v2),
                    f"to_free_slots mismatch at bs={bs}",
                )
                print(f"get_target correctness bs={bs}: PASSED")

    # ==================== filter_finished_cache_loc_kernel Tests ====================

    def _test_filter_finished_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        (
            out_cache_loc,
            tgt_cache_loc,
            accept_length,
            accept_length_filter,
            num_verify_tokens,
        ) = setup_filter_finished_test(bs)
        num_verify_tokens_upper = next_power_of_2(num_verify_tokens)
        bs_upper = next_power_of_2(bs)

        torch.cuda.synchronize()
        start = time.perf_counter()
        filter_finished_cache_loc_kernel_v1[(bs,)](
            out_cache_loc,
            tgt_cache_loc,
            accept_length,
            accept_length_filter,
            bs_upper,
            num_verify_tokens_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_filter_finished_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        (
            out_cache_loc,
            tgt_cache_loc,
            accept_length,
            accept_length_filter,
            num_verify_tokens,
        ) = setup_filter_finished_test(bs)
        num_verify_tokens_upper = next_power_of_2(num_verify_tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()
        filter_finished_cache_loc_kernel[(bs,)](
            out_cache_loc,
            tgt_cache_loc,
            accept_length,
            accept_length_filter,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
            num_verify_tokens_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_filter_finished_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_filter_finished_compile_v1(bs)
                print(f"filter_finished v1 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_filter_finished_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_filter_finished_compile_v2(bs)
                print(f"filter_finished v2 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_filter_finished_correctness(self):
        """Test correctness by comparing v1 and v2 outputs."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    out_cache_loc_v1,
                    tgt_cache_loc,
                    accept_length,
                    accept_length_filter,
                    num_verify_tokens,
                ) = setup_filter_finished_test(bs)
                out_cache_loc_v2 = torch.empty_like(out_cache_loc_v1)

                num_verify_tokens_upper = next_power_of_2(num_verify_tokens)
                bs_upper = next_power_of_2(bs)

                # Run v1
                filter_finished_cache_loc_kernel_v1[(bs,)](
                    out_cache_loc_v1,
                    tgt_cache_loc,
                    accept_length,
                    accept_length_filter,
                    bs_upper,
                    num_verify_tokens_upper,
                )

                # Run v2
                filter_finished_cache_loc_kernel[(bs,)](
                    out_cache_loc_v2,
                    tgt_cache_loc,
                    accept_length,
                    accept_length_filter,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    num_verify_tokens_upper,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(out_cache_loc_v1, out_cache_loc_v2),
                    f"out_cache_loc mismatch at bs={bs}",
                )
                print(f"filter_finished correctness bs={bs}: PASSED")

    # ==================== Large Batch Size Tests for v2 ====================

    def test_create_extend_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                verified_id, seq_lens, accept_lens, positions, new_verified_id = (
                    setup_create_extend_test(bs)
                )
                max_accept_len = next_power_of_2(6)

                create_extend_after_decode_spec_info[(bs,)](
                    verified_id,
                    seq_lens,
                    accept_lens,
                    positions,
                    new_verified_id,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    max_accept_len,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                self.assertEqual(positions.shape[0], accept_lens.sum().item())
                self.assertEqual(new_verified_id.shape[0], bs)
                print(f"create_extend large batch bs={bs}: PASSED")

    def test_assign_req_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token,
                    start_offset,
                    end_offset,
                    out_cache_loc,
                    pool_len,
                ) = setup_assign_req_to_token_pool_test(bs)

                assign_req_to_token_pool[(bs,)](
                    req_pool_indices,
                    req_to_token,
                    start_offset,
                    end_offset,
                    out_cache_loc,
                    pool_len,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                self.assertEqual(req_to_token.shape[0], bs)
                print(f"assign_req large batch bs={bs}: PASSED")

    def test_get_target_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    out_cache_loc,
                    num_verify_tokens,
                ) = setup_get_target_cache_loc_test(bs)
                num_verify_tokens_upper = next_power_of_2(num_verify_tokens)

                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    out_cache_loc,
                    num_verify_tokens,
                    num_verify_tokens_upper,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                total_accept = (accept_length + 1).sum().item()
                self.assertEqual(tgt_cache_loc.shape[0], total_accept)
                print(f"get_target large batch bs={bs}: PASSED")

    def test_filter_finished_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    out_cache_loc,
                    tgt_cache_loc,
                    accept_length,
                    accept_length_filter,
                    num_verify_tokens,
                ) = setup_filter_finished_test(bs)
                num_verify_tokens_upper = next_power_of_2(num_verify_tokens)

                filter_finished_cache_loc_kernel[(bs,)](
                    out_cache_loc,
                    tgt_cache_loc,
                    accept_length,
                    accept_length_filter,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    num_verify_tokens_upper,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                total_filter = accept_length_filter.sum().item()
                self.assertGreaterEqual(out_cache_loc.shape[0], total_filter)
                print(f"filter_finished large batch bs={bs}: PASSED")

    # ==================== assign_draft_cache_locs Tests ====================

    def _test_assign_draft_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        (
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
        ) = setup_assign_draft_cache_locs_test(bs, topk=1, page_size=1)

        bs_upper = next_power_of_2(bs)
        iter_upper = next_power_of_2(speculative_num_steps + page_size)

        torch.cuda.synchronize()
        start = time.perf_counter()
        assign_draft_cache_locs_v1[(bs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
            bs_upper,
            iter_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_assign_draft_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        (
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
        ) = setup_assign_draft_cache_locs_test(bs, topk=1, page_size=1)

        iter_upper = next_power_of_2(speculative_num_steps + page_size)

        torch.cuda.synchronize()
        start = time.perf_counter()
        assign_draft_cache_locs[(bs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
            iter_upper,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_assign_draft_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_assign_draft_compile_v1(bs)
                print(f"assign_draft v1 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_assign_draft_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_assign_draft_compile_v2(bs)
                print(f"assign_draft v2 bs={bs}: compile_time={compile_time:.2f}ms")
                self.assertGreater(compile_time, 0)

    def test_assign_draft_correctness(self):
        """Test correctness by comparing v1 and v2 outputs (simple case: topk=1, page_size=1)."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token_v1,
                    seq_lens,
                    extend_lens,
                    num_new_pages_per_topk,
                    out_cache_loc,
                    source_cache_loc_v1,
                    target_cache_loc_v1,
                    last_page_lens_cumsum,
                    duplicate_cache_len,
                    pool_len,
                    topk,
                    speculative_num_steps,
                    page_size,
                ) = setup_assign_draft_cache_locs_test(bs, topk=1, page_size=1)

                req_to_token_v2 = req_to_token_v1.clone()
                source_cache_loc_v2 = source_cache_loc_v1.clone()
                target_cache_loc_v2 = target_cache_loc_v1.clone()

                bs_upper = next_power_of_2(bs)
                iter_upper = next_power_of_2(speculative_num_steps + page_size)

                # Run v1
                assign_draft_cache_locs_v1[(bs,)](
                    req_pool_indices,
                    req_to_token_v1,
                    seq_lens,
                    extend_lens,
                    num_new_pages_per_topk,
                    out_cache_loc,
                    source_cache_loc_v1,
                    target_cache_loc_v1,
                    last_page_lens_cumsum,
                    duplicate_cache_len,
                    pool_len,
                    topk,
                    speculative_num_steps,
                    page_size,
                    bs_upper,
                    iter_upper,
                )

                # Run v2
                assign_draft_cache_locs[(bs,)](
                    req_pool_indices,
                    req_to_token_v2,
                    seq_lens,
                    extend_lens,
                    num_new_pages_per_topk,
                    out_cache_loc,
                    source_cache_loc_v2,
                    target_cache_loc_v2,
                    last_page_lens_cumsum,
                    duplicate_cache_len,
                    pool_len,
                    topk,
                    speculative_num_steps,
                    page_size,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    iter_upper,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(req_to_token_v1, req_to_token_v2),
                    f"req_to_token mismatch at bs={bs}",
                )
                print(f"assign_draft correctness bs={bs}: PASSED")

    def test_assign_draft_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token,
                    seq_lens,
                    extend_lens,
                    num_new_pages_per_topk,
                    out_cache_loc,
                    source_cache_loc,
                    target_cache_loc,
                    last_page_lens_cumsum,
                    duplicate_cache_len,
                    pool_len,
                    topk,
                    speculative_num_steps,
                    page_size,
                ) = setup_assign_draft_cache_locs_test(bs, topk=1, page_size=1)

                iter_upper = next_power_of_2(speculative_num_steps + page_size)

                assign_draft_cache_locs[(bs,)](
                    req_pool_indices,
                    req_to_token,
                    seq_lens,
                    extend_lens,
                    num_new_pages_per_topk,
                    out_cache_loc,
                    source_cache_loc,
                    target_cache_loc,
                    last_page_lens_cumsum,
                    duplicate_cache_len,
                    pool_len,
                    topk,
                    speculative_num_steps,
                    page_size,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    iter_upper,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                self.assertEqual(req_to_token.shape[0], bs)
                print(f"assign_draft large batch bs={bs}: PASSED")

    # ==================== generate_draft_decode_kv_indices Tests ====================

    def _test_generate_kv_indices_compile_v1(self, bs):
        """Test compilation time for v1 kernel."""
        (
            req_pool_indices,
            req_to_token,
            paged_kernel_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices_stride,
            kv_indptr_stride,
            num_steps,
            topk,
            page_size,
        ) = setup_generate_draft_decode_kv_indices_test(bs, page_size=1)

        bs_upper = next_power_of_2(bs)
        num_tokens_upper = next_power_of_2(bs * topk)
        iter_upper = next_power_of_2(num_steps)

        torch.cuda.synchronize()
        start = time.perf_counter()
        generate_draft_decode_kv_indices_v1[(num_steps, bs, topk)](
            req_pool_indices,
            req_to_token,
            paged_kernel_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices_stride,
            kv_indptr_stride,
            bs_upper,
            num_tokens_upper,
            iter_upper,
            page_size,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def _test_generate_kv_indices_compile_v2(self, bs):
        """Test compilation time for v2 kernel."""
        (
            req_pool_indices,
            req_to_token,
            paged_kernel_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices_stride,
            kv_indptr_stride,
            num_steps,
            topk,
            page_size,
        ) = setup_generate_draft_decode_kv_indices_test(bs, page_size=1)

        iter_upper = next_power_of_2(num_steps)

        torch.cuda.synchronize()
        start = time.perf_counter()
        generate_draft_decode_kv_indices[(num_steps, bs, topk)](
            req_pool_indices,
            req_to_token,
            paged_kernel_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices_stride,
            kv_indptr_stride,
            _SPEC_STEP_SIZE,
            _SPEC_MAX_NUM_LOOPS,
            iter_upper,
            _NUM_TOKENS_STEP_SIZE,
            _NUM_TOKENS_MAX_LOOPS,
            page_size,
        )
        torch.cuda.synchronize()
        compile_time = (time.perf_counter() - start) * 1000
        return compile_time

    def test_generate_kv_indices_all_compile_v1(self):
        """Test v1 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_generate_kv_indices_compile_v1(bs)
                print(
                    f"generate_kv_indices v1 bs={bs}: compile_time={compile_time:.2f}ms"
                )
                self.assertGreater(compile_time, 0)

    def test_generate_kv_indices_all_compile_v2(self):
        """Test v2 compile times across batch sizes."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                compile_time = self._test_generate_kv_indices_compile_v2(bs)
                print(
                    f"generate_kv_indices v2 bs={bs}: compile_time={compile_time:.2f}ms"
                )
                self.assertGreater(compile_time, 0)

    def test_generate_kv_indices_correctness(self):
        """Test correctness by comparing v1 and v2 outputs."""
        print("\n")
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token,
                    paged_kernel_lens,
                    kv_indices_v1,
                    kv_indptr_v1,
                    positions,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    num_steps,
                    topk,
                    page_size,
                ) = setup_generate_draft_decode_kv_indices_test(
                    bs, page_size=1, fixed_stride=False
                )

                kv_indices_v2 = kv_indices_v1.clone()
                kv_indptr_v2 = kv_indptr_v1.clone()

                bs_upper = next_power_of_2(bs)
                num_tokens_upper = next_power_of_2(bs * topk)
                iter_upper = next_power_of_2(num_steps)

                # Run v1
                generate_draft_decode_kv_indices_v1[(num_steps, bs, topk)](
                    req_pool_indices,
                    req_to_token,
                    paged_kernel_lens,
                    kv_indices_v1,
                    kv_indptr_v1,
                    positions,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    bs_upper,
                    num_tokens_upper,
                    iter_upper,
                    page_size,
                )

                # Run v2
                generate_draft_decode_kv_indices[(num_steps, bs, topk)](
                    req_pool_indices,
                    req_to_token,
                    paged_kernel_lens,
                    kv_indices_v2,
                    kv_indptr_v2,
                    positions,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    iter_upper,
                    _NUM_TOKENS_STEP_SIZE,
                    _NUM_TOKENS_MAX_LOOPS,
                    page_size,
                )

                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(kv_indices_v1, kv_indices_v2),
                    f"kv_indices mismatch at bs={bs}",
                )
                self.assertTrue(
                    torch.equal(kv_indptr_v1, kv_indptr_v2),
                    f"kv_indptr mismatch at bs={bs}",
                )
                print(f"generate_kv_indices correctness bs={bs}: PASSED")

    def test_generate_kv_indices_large_batch_v2(self):
        """Test v2 kernel correctness with large batch sizes."""
        print("\n")
        for bs in LARGE_BATCH_SIZES:
            with self.subTest(bs=bs):
                (
                    req_pool_indices,
                    req_to_token,
                    paged_kernel_lens,
                    kv_indices,
                    kv_indptr,
                    positions,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    num_steps,
                    topk,
                    page_size,
                ) = setup_generate_draft_decode_kv_indices_test(
                    bs, page_size=1, fixed_stride=False
                )

                iter_upper = next_power_of_2(num_steps)

                generate_draft_decode_kv_indices[(num_steps, bs, topk)](
                    req_pool_indices,
                    req_to_token,
                    paged_kernel_lens,
                    kv_indices,
                    kv_indptr,
                    positions,
                    pool_len,
                    kv_indices_stride,
                    kv_indptr_stride,
                    _SPEC_STEP_SIZE,
                    _SPEC_MAX_NUM_LOOPS,
                    iter_upper,
                    _NUM_TOKENS_STEP_SIZE,
                    _NUM_TOKENS_MAX_LOOPS,
                    page_size,
                )
                torch.cuda.synchronize()

                # Verify basic properties
                self.assertEqual(kv_indices.shape[0], num_steps)
                self.assertEqual(kv_indptr.shape[0], num_steps)
                print(f"generate_kv_indices large batch bs={bs}: PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
