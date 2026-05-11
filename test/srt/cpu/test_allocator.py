"""
Unit tests and benchmarks for alloc_extend_kernel_cpu and alloc_decode_kernel_cpu.
"""

import time
import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _ceil_div(a, b):
    """Integer ceiling division. Works for both Python ints and torch Tensors."""
    return (a + b - 1) // b


def alloc_extend_kernel_pytorch(
    pre_lens, seq_lens, last_loc, free_pages, out_indices, page_size
):
    """Reference PyTorch implementation."""
    bs = pre_lens.shape[0]
    extend_lens = seq_lens - pre_lens
    extend_cumsum = torch.cumsum(extend_lens, dim=0)
    output_start_locs = extend_cumsum - extend_lens

    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    num_new_pages = num_pages_after - num_pages_before
    new_pages_cumsum = torch.cumsum(num_new_pages, dim=0)
    new_page_start_locs = new_pages_cumsum - num_new_pages

    for i in range(bs):
        pre_len = pre_lens[i].item()
        seq_len = seq_lens[i].item()
        extend_len = seq_len - pre_len
        if extend_len == 0:
            continue

        output_start = output_start_locs[i].item()
        new_page_start = new_page_start_locs[i].item()
        num_new_pages_self = num_new_pages[i].item()
        last_loc_i = last_loc[i].item()

        page_boundary = _ceil_div(pre_len, page_size) * page_size
        num_part1 = min(seq_len, page_boundary) - pre_len
        if num_part1 > 0:
            offsets = torch.arange(num_part1, device=out_indices.device)
            out_indices[output_start : output_start + num_part1] = (
                last_loc_i + 1 + offsets
            )

        if pre_len + num_part1 == seq_len:
            continue

        full_page_end = (seq_len // page_size) * page_size
        full_page_start = _ceil_div(pre_len, page_size) * page_size
        num_part2 = full_page_end - full_page_start
        if num_part2 > 0:
            offsets = torch.arange(num_part2, device=out_indices.device)
            page_indices = offsets // page_size
            in_page_offsets = offsets % page_size
            pages = free_pages[new_page_start + page_indices]
            out_indices[
                output_start + num_part1 : output_start + num_part1 + num_part2
            ] = (pages * page_size + in_page_offsets)

        if pre_len + num_part1 + num_part2 == seq_len:
            continue

        num_part3 = seq_len - (seq_len // page_size) * page_size
        if num_part3 > 0:
            start_page = free_pages[new_page_start + num_new_pages_self - 1].item()
            offsets = torch.arange(num_part3, device=out_indices.device)
            out_indices[
                output_start
                + num_part1
                + num_part2 : output_start
                + num_part1
                + num_part2
                + num_part3
            ] = (start_page * page_size + offsets)


def alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out_indices, page_size):
    """Reference PyTorch implementation."""
    pre_lens = seq_lens - 1
    num_pages_after = _ceil_div(seq_lens, page_size)
    num_pages_before = _ceil_div(pre_lens, page_size)
    num_new_pages = num_pages_after - num_pages_before

    new_pages_cumsum = torch.cumsum(num_new_pages, dim=0)
    new_page_start_locs = new_pages_cumsum - num_new_pages

    no_new_page_mask = num_new_pages == 0
    out_indices[no_new_page_mask] = last_loc[no_new_page_mask].to(out_indices.dtype) + 1

    new_page_mask = num_new_pages > 0
    if new_page_mask.any():
        new_page_offsets = new_page_start_locs[new_page_mask]
        pages = free_pages[new_page_offsets]
        out_indices[new_page_mask] = pages.to(out_indices.dtype) * page_size


def _gen_extend_test_data(bs, page_size, max_pre_len=128, max_extend_len=64):
    """Generate consistent test data for alloc_extend_kernel."""
    pre_lens = torch.randint(0, max_pre_len + 1, (bs,), dtype=torch.int64)
    extend_lens = torch.randint(1, max_extend_len + 1, (bs,), dtype=torch.int64)
    seq_lens = pre_lens + extend_lens

    # Compute last_loc: for each request, simulate the physical location of
    # the last token in the prefix. last_loc must satisfy:
    #   (last_loc + 1) % page_size == pre_len % page_size
    # We simulate by placing prefix pages sequentially starting from some base page.
    last_loc = torch.zeros(bs, dtype=torch.int64)
    page_counter = 0
    for i in range(bs):
        pre_len = pre_lens[i].item()
        if pre_len == 0:
            last_loc[i] = 0
        else:
            num_pages = (pre_len + page_size - 1) // page_size
            # last page of prefix
            last_page = page_counter + num_pages - 1
            in_page_offset = (pre_len - 1) % page_size
            last_loc[i] = last_page * page_size + in_page_offset
            page_counter += num_pages

    # Generate free pages (enough for all requests)
    total_new_pages = 0
    for i in range(bs):
        pages_after = (seq_lens[i].item() + page_size - 1) // page_size
        pages_before = (pre_lens[i].item() + page_size - 1) // page_size
        total_new_pages += pages_after - pages_before
    free_pages = torch.arange(
        page_counter, page_counter + total_new_pages + 100, dtype=torch.int64
    )

    extend_num_tokens = extend_lens.sum().item()
    return pre_lens, seq_lens, last_loc, free_pages, extend_num_tokens


def _gen_decode_test_data(bs, page_size, max_seq_len=256):
    """Generate consistent test data for alloc_decode_kernel."""
    # seq_lens are the lengths AFTER the decode step (already incremented by 1)
    seq_lens = torch.randint(2, max_seq_len + 1, (bs,), dtype=torch.int64)

    # last_loc: simulate physical location of the last allocated token
    last_loc = torch.zeros(bs, dtype=torch.int64)
    page_counter = 0
    for i in range(bs):
        pre_len = seq_lens[i].item() - 1  # length before decode
        if pre_len == 0:
            last_loc[i] = 0
        else:
            num_pages = (pre_len + page_size - 1) // page_size
            last_page = page_counter + num_pages - 1
            in_page_offset = (pre_len - 1) % page_size
            last_loc[i] = last_page * page_size + in_page_offset
            page_counter += num_pages

    # Free pages
    total_new_pages = 0
    for i in range(bs):
        pages_after = (seq_lens[i].item() + page_size - 1) // page_size
        pages_before = ((seq_lens[i].item() - 1) + page_size - 1) // page_size
        total_new_pages += pages_after - pages_before
    free_pages = torch.arange(
        page_counter, page_counter + total_new_pages + 100, dtype=torch.int64
    )

    return seq_lens, last_loc, free_pages


class TestAllocExtendKernel(CustomTestCase):
    def _run_test(self, bs, page_size, max_pre_len=128, max_extend_len=64):
        pre_lens, seq_lens, last_loc, free_pages, extend_num_tokens = (
            _gen_extend_test_data(bs, page_size, max_pre_len, max_extend_len)
        )

        # Reference
        out_ref = torch.empty(extend_num_tokens, dtype=torch.int64)
        alloc_extend_kernel_pytorch(
            pre_lens, seq_lens, last_loc, free_pages, out_ref, page_size
        )

        # CPU kernel
        out_cpu = torch.empty(extend_num_tokens, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
            pre_lens, seq_lens, last_loc, free_pages, out_cpu, page_size
        )

        torch.testing.assert_close(out_cpu, out_ref)

    def test_basic_page_size_1(self):
        self._run_test(bs=4, page_size=1)

    def test_basic_page_size_16(self):
        self._run_test(bs=8, page_size=16)

    def test_basic_page_size_128(self):
        self._run_test(bs=8, page_size=128)

    def test_single_request(self):
        self._run_test(bs=1, page_size=16)

    def test_large_batch(self):
        self._run_test(bs=256, page_size=16, max_pre_len=512, max_extend_len=256)

    def test_all_full_pages(self):
        """All pre_lens and seq_lens are multiples of page_size."""
        page_size = 16
        bs = 4
        pre_lens = torch.tensor([0, 16, 32, 48], dtype=torch.int64)
        seq_lens = torch.tensor([16, 32, 64, 64], dtype=torch.int64)
        extend_lens = seq_lens - pre_lens
        extend_num_tokens = extend_lens.sum().item()

        last_loc = torch.zeros(bs, dtype=torch.int64)
        page_counter = 0
        for i in range(bs):
            pre_len = pre_lens[i].item()
            if pre_len == 0:
                last_loc[i] = 0
            else:
                num_pages = pre_len // page_size
                last_page = page_counter + num_pages - 1
                last_loc[i] = last_page * page_size + page_size - 1
                page_counter += num_pages
        total_new_pages = sum(
            (seq_lens[i].item() + page_size - 1) // page_size
            - (pre_lens[i].item() + page_size - 1) // page_size
            for i in range(bs)
        )
        free_pages = torch.arange(
            page_counter, page_counter + total_new_pages + 10, dtype=torch.int64
        )

        out_ref = torch.empty(extend_num_tokens, dtype=torch.int64)
        alloc_extend_kernel_pytorch(
            pre_lens, seq_lens, last_loc, free_pages, out_ref, page_size
        )
        out_cpu = torch.empty(extend_num_tokens, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
            pre_lens, seq_lens, last_loc, free_pages, out_cpu, page_size
        )
        torch.testing.assert_close(out_cpu, out_ref)

    def test_zero_extend(self):
        """Some requests have zero extend length."""
        page_size = 16
        bs = 4
        pre_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        seq_lens = torch.tensor([10, 25, 30, 50], dtype=torch.int64)
        extend_lens = seq_lens - pre_lens
        extend_num_tokens = extend_lens.sum().item()

        last_loc = torch.zeros(bs, dtype=torch.int64)
        page_counter = 0
        for i in range(bs):
            pre_len = pre_lens[i].item()
            if pre_len == 0:
                last_loc[i] = 0
            else:
                num_pages = (pre_len + page_size - 1) // page_size
                last_page = page_counter + num_pages - 1
                in_page_offset = (pre_len - 1) % page_size
                last_loc[i] = last_page * page_size + in_page_offset
                page_counter += num_pages

        total_new_pages = sum(
            (seq_lens[i].item() + page_size - 1) // page_size
            - (pre_lens[i].item() + page_size - 1) // page_size
            for i in range(bs)
        )
        free_pages = torch.arange(
            page_counter, page_counter + total_new_pages + 10, dtype=torch.int64
        )

        out_ref = torch.empty(extend_num_tokens, dtype=torch.int64)
        alloc_extend_kernel_pytorch(
            pre_lens, seq_lens, last_loc, free_pages, out_ref, page_size
        )
        out_cpu = torch.empty(extend_num_tokens, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
            pre_lens, seq_lens, last_loc, free_pages, out_cpu, page_size
        )
        torch.testing.assert_close(out_cpu, out_ref)

    def _run_test_int32(self, bs, page_size, max_pre_len=128, max_extend_len=64):
        pre_lens, seq_lens, last_loc, free_pages, extend_num_tokens = (
            _gen_extend_test_data(bs, page_size, max_pre_len, max_extend_len)
        )
        # Convert to int32
        pre_lens = pre_lens.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        last_loc = last_loc.to(torch.int32)
        free_pages = free_pages.to(torch.int32)

        # Reference (use int64 for reference)
        out_ref = torch.empty(extend_num_tokens, dtype=torch.int64)
        alloc_extend_kernel_pytorch(
            pre_lens.to(torch.int64),
            seq_lens.to(torch.int64),
            last_loc.to(torch.int64),
            free_pages.to(torch.int64),
            out_ref,
            page_size,
        )

        # CPU kernel with int32
        out_cpu = torch.empty(extend_num_tokens, dtype=torch.int32)
        torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
            pre_lens, seq_lens, last_loc, free_pages, out_cpu, page_size
        )

        torch.testing.assert_close(out_cpu.to(torch.int64), out_ref)

    def test_int32_basic(self):
        self._run_test_int32(bs=8, page_size=16)

    def test_int32_large_batch(self):
        self._run_test_int32(bs=64, page_size=16)

    def _run_test(self, bs, page_size, max_seq_len=256):
        seq_lens, last_loc, free_pages = _gen_decode_test_data(
            bs, page_size, max_seq_len
        )

        # Reference
        out_ref = torch.empty(bs, dtype=torch.int64)
        alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out_ref, page_size)

        # CPU kernel
        out_cpu = torch.empty(bs, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
            seq_lens, last_loc, free_pages, out_cpu, page_size
        )

        torch.testing.assert_close(out_cpu, out_ref)

    def test_basic_page_size_1(self):
        self._run_test(bs=4, page_size=1)

    def test_basic_page_size_16(self):
        self._run_test(bs=8, page_size=16)

    def test_basic_page_size_128(self):
        self._run_test(bs=8, page_size=128)

    def test_single_request(self):
        self._run_test(bs=1, page_size=16)

    def test_large_batch(self):
        self._run_test(bs=1024, page_size=16)

    def test_all_need_new_page(self):
        """All requests need a new page (seq_len is at page boundary)."""
        page_size = 16
        bs = 4
        # seq_lens at multiples of page_size + 1 => need new page
        seq_lens = torch.tensor([17, 33, 49, 65], dtype=torch.int64)
        last_loc = torch.zeros(bs, dtype=torch.int64)
        page_counter = 0
        for i in range(bs):
            pre_len = seq_lens[i].item() - 1
            num_pages = (pre_len + page_size - 1) // page_size
            last_page = page_counter + num_pages - 1
            in_page_offset = (pre_len - 1) % page_size
            last_loc[i] = last_page * page_size + in_page_offset
            page_counter += num_pages

        free_pages = torch.arange(
            page_counter, page_counter + bs + 10, dtype=torch.int64
        )

        out_ref = torch.empty(bs, dtype=torch.int64)
        alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out_ref, page_size)
        out_cpu = torch.empty(bs, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
            seq_lens, last_loc, free_pages, out_cpu, page_size
        )
        torch.testing.assert_close(out_cpu, out_ref)

    def test_none_need_new_page(self):
        """No requests need a new page."""
        page_size = 16
        bs = 4
        seq_lens = torch.tensor([2, 3, 10, 15], dtype=torch.int64)
        last_loc = torch.zeros(bs, dtype=torch.int64)
        page_counter = 0
        for i in range(bs):
            pre_len = seq_lens[i].item() - 1
            if pre_len == 0:
                last_loc[i] = 0
            else:
                num_pages = (pre_len + page_size - 1) // page_size
                last_page = page_counter + num_pages - 1
                in_page_offset = (pre_len - 1) % page_size
                last_loc[i] = last_page * page_size + in_page_offset
                page_counter += num_pages

        free_pages = torch.arange(page_counter, page_counter + 10, dtype=torch.int64)

        out_ref = torch.empty(bs, dtype=torch.int64)
        alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out_ref, page_size)
        out_cpu = torch.empty(bs, dtype=torch.int64)
        torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
            seq_lens, last_loc, free_pages, out_cpu, page_size
        )
        torch.testing.assert_close(out_cpu, out_ref)

    def _run_test_int32(self, bs, page_size, max_seq_len=256):
        seq_lens, last_loc, free_pages = _gen_decode_test_data(
            bs, page_size, max_seq_len
        )
        seq_lens = seq_lens.to(torch.int32)
        last_loc = last_loc.to(torch.int32)
        free_pages = free_pages.to(torch.int32)

        # Reference
        out_ref = torch.empty(bs, dtype=torch.int64)
        alloc_decode_kernel_pytorch(
            seq_lens.to(torch.int64),
            last_loc.to(torch.int64),
            free_pages.to(torch.int64),
            out_ref,
            page_size,
        )

        # CPU kernel with int32
        out_cpu = torch.empty(bs, dtype=torch.int32)
        torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
            seq_lens, last_loc, free_pages, out_cpu, page_size
        )

        torch.testing.assert_close(out_cpu.to(torch.int64), out_ref)

    def test_int32_basic(self):
        self._run_test_int32(bs=8, page_size=16)

    def test_int32_large_batch(self):
        self._run_test_int32(bs=256, page_size=16)

    @staticmethod
    def bench_alloc_extend(
        bs, page_size, max_pre_len=512, max_extend_len=256, warmup=10, iters=100
    ):
        pre_lens, seq_lens, last_loc, free_pages, extend_num_tokens = (
            _gen_extend_test_data(bs, page_size, max_pre_len, max_extend_len)
        )

        # Benchmark reference (PyTorch)
        for _ in range(warmup):
            out = torch.empty(extend_num_tokens, dtype=torch.int64)
            alloc_extend_kernel_pytorch(
                pre_lens, seq_lens, last_loc, free_pages, out, page_size
            )
        t0 = time.perf_counter()
        for _ in range(iters):
            out = torch.empty(extend_num_tokens, dtype=torch.int64)
            alloc_extend_kernel_pytorch(
                pre_lens, seq_lens, last_loc, free_pages, out, page_size
            )
        ref_time = (time.perf_counter() - t0) / iters * 1e6  # us

        # Benchmark CPU kernel
        for _ in range(warmup):
            out = torch.empty(extend_num_tokens, dtype=torch.int64)
            torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
                pre_lens, seq_lens, last_loc, free_pages, out, page_size
            )
        t0 = time.perf_counter()
        for _ in range(iters):
            out = torch.empty(extend_num_tokens, dtype=torch.int64)
            torch.ops.sgl_kernel.alloc_extend_kernel_cpu(
                pre_lens, seq_lens, last_loc, free_pages, out, page_size
            )
        cpu_time = (time.perf_counter() - t0) / iters * 1e6  # us

        print(
            f"alloc_extend | bs={bs:4d} page_size={page_size:3d} extend_tokens={extend_num_tokens:6d} | "
            f"PyTorch: {ref_time:8.1f} us | C++: {cpu_time:8.1f} us | "
            f"Speedup: {ref_time / cpu_time:5.2f}x"
        )

    @staticmethod
    def bench_alloc_decode(bs, page_size, max_seq_len=256, warmup=10, iters=100):
        seq_lens, last_loc, free_pages = _gen_decode_test_data(
            bs, page_size, max_seq_len
        )

        # Benchmark reference (PyTorch)
        for _ in range(warmup):
            out = torch.empty(bs, dtype=torch.int64)
            alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out, page_size)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = torch.empty(bs, dtype=torch.int64)
            alloc_decode_kernel_pytorch(seq_lens, last_loc, free_pages, out, page_size)
        ref_time = (time.perf_counter() - t0) / iters * 1e6  # us

        # Benchmark CPU kernel
        for _ in range(warmup):
            out = torch.empty(bs, dtype=torch.int64)
            torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
                seq_lens, last_loc, free_pages, out, page_size
            )
        t0 = time.perf_counter()
        for _ in range(iters):
            out = torch.empty(bs, dtype=torch.int64)
            torch.ops.sgl_kernel.alloc_decode_kernel_cpu(
                seq_lens, last_loc, free_pages, out, page_size
            )
        cpu_time = (time.perf_counter() - t0) / iters * 1e6  # us

        print(
            f"alloc_decode | bs={bs:4d} page_size={page_size:3d} | "
            f"PyTorch: {ref_time:8.1f} us | C++: {cpu_time:8.1f} us | "
            f"Speedup: {ref_time / cpu_time:5.2f}x"
        )


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        sys.argv.remove("--bench")
        print("=" * 80)
        print("Benchmark: alloc_extend_kernel")
        print("=" * 80)
        for bs in [1, 4, 16, 64, 256]:
            for page_size in [1, 16, 128]:
                BenchmarkScheduler.bench_alloc_extend(bs, page_size)

        print()
        print("=" * 80)
        print("Benchmark: alloc_decode_kernel")
        print("=" * 80)
        for bs in [1, 4, 16, 64, 256, 1024]:
            for page_size in [1, 16, 128]:
                BenchmarkScheduler.bench_alloc_decode(bs, page_size)
    else:
        unittest.main()
