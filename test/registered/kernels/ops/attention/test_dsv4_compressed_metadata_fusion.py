"""Correctness tests for fused DSV4 compressed-attention metadata setup."""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_c4_sparse_metadata,
    init_compression_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_INDEX_ALIGNMENT = 64


def _align_page_index_width(width: int) -> int:
    return (
        (width + PAGE_INDEX_ALIGNMENT - 1)
        // PAGE_INDEX_ALIGNMENT
        * PAGE_INDEX_ALIGNMENT
    )


def _expected_c128_page_indices(
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    c128_page_size = page_size // 128
    logical_width = c128_page_size * page_table.shape[1]
    storage_width = _align_page_index_width(logical_width)
    expected = torch.full(
        (seq_lens.shape[0], storage_width),
        -1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    offsets = torch.arange(logical_width, device=seq_lens.device)
    page_indices = offsets // c128_page_size
    offsets_in_page = offsets % c128_page_size
    mapped = page_table[:, page_indices] * c128_page_size + offsets_in_page
    valid = offsets.unsqueeze(0) < (seq_lens // 128).unsqueeze(1)
    expected[:, :logical_width] = torch.where(valid, mapped, -1)
    return expected


class TestDSV4CompressedMetadataFusion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda")

    def test_c128_producer_returns_final_aligned_storage(self):
        batch_size = 4
        page_size = 256
        for max_pages in (32, 33, 4097):
            logical_width = (page_size // 128) * max_pages
            storage_width = _align_page_index_width(logical_width)
            capacity = page_size * max_pages
            seq_lens = torch.tensor(
                [0, 129, min(capacity, 511), capacity],
                dtype=torch.int32,
                device=self.device,
            )
            positions = seq_lens - 1
            raw_out_loc = (
                torch.arange(1, batch_size + 1, device=self.device, dtype=torch.int64)
                * 512
            )
            page_table = (
                torch.arange(
                    batch_size * max_pages,
                    device=self.device,
                    dtype=torch.int32,
                ).view(batch_size, max_pages)
                + 100
            )
            expected = _expected_c128_page_indices(seq_lens, page_table, page_size)

            for live_prefix_only in (False, True):
                with self.subTest(
                    max_pages=max_pages,
                    live_prefix_only=live_prefix_only,
                ):
                    outputs = init_compression_metadata(
                        seq_lens,
                        positions,
                        raw_out_loc,
                        page_table,
                        page_size,
                        compute_page_indices=True,
                        live_prefix_only=live_prefix_only,
                    )
                    actual = outputs[-1]
                    self.assertIsNotNone(actual)
                    assert actual is not None
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertEqual(actual.shape, (batch_size, storage_width))
                    self.assertEqual(actual.stride(), (storage_width, 1))
                    if not live_prefix_only:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(
                            torch.all(actual[:, logical_width:] == -1).item()
                        )
                        continue

                    for row, raw_len in enumerate((seq_lens // 128).tolist()):
                        live_width = min(max(raw_len, 1), logical_width)
                        torch.testing.assert_close(
                            actual[row, :live_width],
                            expected[row, :live_width],
                            rtol=0,
                            atol=0,
                        )
                    # The logical suffix and alignment padding are intentionally
                    # undefined in live-prefix mode.

    def test_compression_metadata_without_page_indices(self):
        seq_lens = torch.tensor(
            [0, 127, 128, 513], dtype=torch.int32, device=self.device
        )
        positions = seq_lens - 1
        raw_out_loc = torch.tensor(
            [0, 128, 256, 1024], dtype=torch.int64, device=self.device
        )
        outputs = init_compression_metadata(
            seq_lens,
            positions,
            raw_out_loc,
            compute_page_indices=False,
            live_prefix_only=True,
        )
        self.assertIsNone(outputs[-1])
        torch.testing.assert_close(outputs[2], seq_lens // 4, rtol=0, atol=0)
        torch.testing.assert_close(outputs[3], torch.clamp_min(seq_lens // 4, 1))
        torch.testing.assert_close(outputs[6], seq_lens // 128, rtol=0, atol=0)
        torch.testing.assert_close(outputs[7], torch.clamp_min(seq_lens // 128, 1))

    def test_c4_sparse_metadata_parity_and_fallback(self):
        for topk in (512, 1024):
            source = torch.full((10,), -123, dtype=torch.int32, device=self.device)
            lengths = source[::2]
            lengths.copy_(
                torch.tensor(
                    [1, topk - 1, topk, topk + 1, 100_000],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            sparse_lengths, sparse_page_indices = init_c4_sparse_metadata(lengths, topk)
            self.assertEqual(sparse_lengths.dtype, torch.int32)
            self.assertEqual(sparse_page_indices.dtype, torch.int32)
            self.assertEqual(sparse_page_indices.shape, (lengths.numel(), topk))
            torch.testing.assert_close(
                sparse_lengths, torch.clamp(lengths, max=topk), rtol=0, atol=0
            )
            self.assertTrue(torch.all(sparse_page_indices == -1).item())

            empty_lengths = torch.empty(0, dtype=torch.int32, device=self.device)
            empty_sparse_lengths, empty_sparse_page_indices = init_c4_sparse_metadata(
                empty_lengths, topk
            )
            self.assertEqual(empty_sparse_lengths.shape, (0,))
            self.assertEqual(empty_sparse_page_indices.shape, (0, topk))

            cpu_source = torch.tensor(
                [1, -9, topk, -9, topk + 1, -9], dtype=torch.int32
            )
            cpu_lengths = cpu_source[::2]
            cpu_sparse_lengths, cpu_sparse_page_indices = init_c4_sparse_metadata(
                cpu_lengths, topk
            )
            torch.testing.assert_close(
                cpu_sparse_lengths,
                torch.clamp(cpu_lengths, max=topk),
                rtol=0,
                atol=0,
            )
            self.assertEqual(cpu_sparse_lengths.dtype, torch.int32)
            self.assertEqual(cpu_sparse_page_indices.shape, (3, topk))
            self.assertTrue(torch.all(cpu_sparse_page_indices == -1).item())

    def test_unaligned_c128_and_c4_cuda_graph_replay(self):
        batch_size = 4
        page_size = 256
        max_pages = 33
        logical_width = (page_size // 128) * max_pages
        storage_width = _align_page_index_width(logical_width)
        capacity = page_size * max_pages

        for live_prefix_only in (False, True):
            seq_lens = torch.tensor(
                [0, 129, 511, capacity], dtype=torch.int32, device=self.device
            )
            positions = seq_lens - 1
            raw_out_loc = seq_lens.to(torch.int64).clone()
            page_table = torch.arange(
                batch_size * max_pages,
                dtype=torch.int32,
                device=self.device,
            ).view(batch_size, max_pages)

            init_compression_metadata(
                seq_lens,
                positions,
                raw_out_loc,
                page_table,
                page_size,
                compute_page_indices=True,
                live_prefix_only=live_prefix_only,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = init_compression_metadata(
                    seq_lens,
                    positions,
                    raw_out_loc,
                    page_table,
                    page_size,
                    compute_page_indices=True,
                    live_prefix_only=live_prefix_only,
                )

            c128_page_indices = outputs[-1]
            self.assertIsNotNone(c128_page_indices)
            assert c128_page_indices is not None
            output_ptrs = tuple(
                output.data_ptr() for output in outputs if output is not None
            )
            for replay_index, replay_seq_lens in enumerate(
                (
                    [0, 129, 511, capacity],
                    [capacity, 1, 256, 0],
                    [0, 129, 511, capacity],
                )
            ):
                seq_lens.copy_(
                    torch.tensor(replay_seq_lens, dtype=torch.int32, device=self.device)
                )
                positions.copy_(seq_lens - 1)
                raw_out_loc.copy_(seq_lens.to(torch.int64))
                page_table.copy_(
                    torch.arange(
                        batch_size * max_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ).view(batch_size, max_pages)
                    + replay_index * 10_000
                )
                c128_page_indices.fill_(123)
                graph.replay()
                torch.cuda.synchronize()

                expected = _expected_c128_page_indices(seq_lens, page_table, page_size)
                self.assertEqual(c128_page_indices.shape, (batch_size, storage_width))
                self.assertEqual(c128_page_indices.stride(), (storage_width, 1))
                self.assertEqual(
                    tuple(
                        output.data_ptr() for output in outputs if output is not None
                    ),
                    output_ptrs,
                )
                if not live_prefix_only:
                    torch.testing.assert_close(
                        c128_page_indices, expected, rtol=0, atol=0
                    )
                    self.assertTrue(
                        torch.all(c128_page_indices[:, logical_width:] == -1).item()
                    )
                    continue
                for row, raw_len in enumerate((seq_lens // 128).tolist()):
                    live_width = min(max(raw_len, 1), logical_width)
                    torch.testing.assert_close(
                        c128_page_indices[row, :live_width],
                        expected[row, :live_width],
                        rtol=0,
                        atol=0,
                    )

        for topk in (512, 1024):
            length_storage = torch.zeros(
                batch_size * 2, dtype=torch.int32, device=self.device
            )
            lengths = length_storage[::2]
            lengths.copy_(
                torch.tensor([1, topk - 1, topk, topk + 1], device=self.device)
            )
            init_c4_sparse_metadata(lengths, topk)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sparse_lengths, sparse_page_indices = init_c4_sparse_metadata(
                    lengths, topk
                )
            output_ptrs = (
                sparse_lengths.data_ptr(),
                sparse_page_indices.data_ptr(),
            )
            for values in (
                [1, topk - 1, topk, topk + 1],
                [topk + 99, 3, 1, 100_000],
            ):
                lengths.copy_(
                    torch.tensor(values, dtype=torch.int32, device=self.device)
                )
                sparse_page_indices.fill_(123)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    sparse_lengths,
                    torch.clamp(lengths, max=topk),
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(torch.all(sparse_page_indices == -1).item())
                self.assertEqual(
                    (
                        sparse_lengths.data_ptr(),
                        sparse_page_indices.data_ptr(),
                    ),
                    output_ptrs,
                )


if __name__ == "__main__":
    unittest.main()
