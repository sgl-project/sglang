"""Prefix-index copies preserve ragged row order, tails and output bounds."""

import unittest
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.attention.kv_indices import (
    create_chunked_prefix_cache_kv_indices,
    create_chunked_prefix_cache_kv_indices_2d,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

BLOCK_SIZE = 1024
GUARD = 32
SENTINEL = 0x7EADBEEF
LENGTHS = [
    [0],
    [0, 0, 0],
    [1],
    [17],
    [0, 1, 511, 512, 513, 1023, 1024, 1025],
    [16384],
    [17, 16381, 0],
    [49152],
    [114688],
    [131072],
    [17, 131069, 0],
    [31] * 64,
]


def _inputs(lengths, strided=False, request_dtype=torch.int64):
    batch = len(lengths)
    starts = [3 + 11 * i for i in range(batch)]
    pools = list(reversed(range(batch)))
    width = max(s + n for s, n in zip(starts, lengths)) + 19
    backing = torch.randint(
        0,
        1_000_000,
        (batch + 1, width + (37 if strided else 0)),
        dtype=torch.int32,
        device="cuda",
    )
    table = backing[:, :width]
    reqs = torch.tensor(pools, dtype=request_dtype, device="cuda")
    begin = torch.tensor(starts, dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda")
    return table, reqs, begin, lens, indptr, starts, pools


def _reference(table, lengths, starts, pools):
    return torch.cat([table[p, s : s + n] for p, s, n in zip(pools, starts, lengths)])


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
class TestChunkedPrefixKVIndices(CustomTestCase):
    def test_matches_reference_and_preserves_bounds(self):
        torch.manual_seed(0)
        for lengths in LENGTHS:
            for strided in (False, True):
                for request_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        lengths=lengths, strided=strided, dtype=request_dtype
                    ):
                        table, reqs, begin, lens, indptr, starts, pools = _inputs(
                            lengths, strided, request_dtype
                        )
                        expected = _reference(table, lengths, starts, pools)
                        original_table = table.clone()
                        for parallel in (False, True):
                            storage = torch.full(
                                (sum(lengths) + 2 * GUARD,),
                                SENTINEL,
                                dtype=torch.int32,
                                device="cuda",
                            )
                            out = storage[GUARD : GUARD + sum(lengths)]
                            args = (
                                table,
                                reqs,
                                begin,
                                lens,
                                indptr,
                                out,
                                table.stride(0),
                            )
                            if parallel:
                                # Extra blocks test the inactive-program guard.
                                blocks = (
                                    triton.cdiv(max(max(lengths), 1), BLOCK_SIZE) + 2
                                )
                                create_chunked_prefix_cache_kv_indices_2d[
                                    (len(lengths), blocks)
                                ](*args, BLOCK_SIZE=BLOCK_SIZE)
                            else:
                                create_chunked_prefix_cache_kv_indices[(len(lengths),)](
                                    *args
                                )
                            self.assertTrue(torch.equal(out, expected))
                            self.assertTrue(bool((storage[:GUARD] == SENTINEL).all()))
                            self.assertTrue(
                                bool(
                                    (storage[GUARD + sum(lengths) :] == SENTINEL).all()
                                )
                            )
                        self.assertTrue(torch.equal(table, original_table))

    def test_cuda_graph_replay_with_shorter_and_empty_rows(self):
        lengths = [2051, 17, 0]
        table, reqs, begin, lens, indptr, starts, pools = _inputs(lengths)
        storage = torch.full(
            (sum(lengths) + 2 * GUARD,), SENTINEL, dtype=torch.int32, device="cuda"
        )
        out = storage[GUARD : GUARD + sum(lengths)]
        args = (table, reqs, begin, lens, indptr, out, table.stride(0))
        grid = (len(lengths), triton.cdiv(max(lengths), BLOCK_SIZE))

        def launch():
            create_chunked_prefix_cache_kv_indices_2d[grid](
                *args, BLOCK_SIZE=BLOCK_SIZE
            )

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for current in (lengths, [1023, 0, 0], [0, 0, 0]):
            with self.subTest(lengths=current):
                storage.fill_(SENTINEL)
                lens.copy_(torch.tensor(current, dtype=torch.int32, device="cuda"))
                indptr.copy_(
                    torch.tensor(
                        [0, *accumulate(current)], dtype=torch.int32, device="cuda"
                    )
                )
                graph.replay()
                expected = _reference(table, current, starts, pools)
                self.assertTrue(
                    torch.equal(storage[GUARD : GUARD + sum(current)], expected)
                )
                self.assertTrue(bool((storage[:GUARD] == SENTINEL).all()))
                self.assertTrue(
                    bool((storage[GUARD + sum(current) :] == SENTINEL).all())
                )

    def test_chunked_prefix_caller_and_dcp_filter(self):
        from sglang.srt.layers.dcp import layout
        from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        table = torch.arange(4 * 4096, dtype=torch.int32, device="cuda").view(4, 4096)
        lengths = [[1025, 17, 1], [1023, 0, 0]]
        starts = [[3, 11, 7], [1028, 28, 8]]
        pools = [2, 0, 1]

        for dcp_enabled in (False, True):
            with self.subTest(dcp_enabled=dcp_enabled):
                batch = mix.ForwardBatchDeepSeekMHAMixin()
                batch.batch_size = len(pools)
                batch.num_prefix_chunks = len(lengths)
                batch.req_pool_indices = torch.tensor(
                    pools, dtype=torch.int64, device="cuda"
                )
                batch.prefix_chunk_num_tokens = [sum(n) for n in lengths]
                batch.prefix_chunk_max_seq_lens = list(map(max, lengths))
                batch.prefix_chunk_starts_cpu = [torch.tensor(s) for s in starts]
                batch.prefix_chunk_seq_lens_cpu = [torch.tensor(n) for n in lengths]
                batch.prefix_chunk_starts = torch.tensor(
                    starts, dtype=torch.int32, device="cuda"
                )
                batch.prefix_chunk_seq_lens = torch.tensor(
                    lengths, dtype=torch.int32, device="cuda"
                )
                batch.prefix_chunk_cu_seq_lens = torch.tensor(
                    [[0, *accumulate(n)] for n in lengths],
                    dtype=torch.int32,
                    device="cuda",
                )
                runner_ids = SimpleNamespace(
                    reads_are_translated=False,
                    translate_dcp_read_ids=lambda indices: indices,
                )
                backend = SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(req_to_token=table),
                    runner_ids=runner_ids,
                )
                grids = []

                class ObservedKernel:
                    def __getitem__(self, grid):
                        grids.append(grid)
                        return create_chunked_prefix_cache_kv_indices_2d[grid]

                parallel = SimpleNamespace(
                    dcp_enabled=dcp_enabled, dcp_size=2, dcp_rank=1
                )
                with (
                    forward_context(ForwardContext(attn_backend=backend)),
                    patch.object(layout, "get_parallel", return_value=parallel),
                    patch.object(
                        mix,
                        "create_chunked_prefix_cache_kv_indices_2d",
                        ObservedKernel(),
                    ),
                ):
                    batch.prepare_chunked_kv_indices(
                        torch.device("cuda"), pack_all_prefix_chunks=True
                    )

                expected_chunks = []
                for n, s in zip(lengths, starts):
                    pieces = [
                        table[p, start : start + length]
                        for p, start, length in zip(pools, s, n)
                    ]
                    if dcp_enabled:
                        pieces = [
                            piece[(1 - start) % 2 :: 2]
                            for piece, start in zip(pieces, s)
                        ]
                    expected_chunks.append(torch.cat(pieces))
                for actual, expected in zip(
                    batch.prefix_chunk_kv_indices, expected_chunks
                ):
                    self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    torch.equal(batch.prefix_all_kv_indices, torch.cat(expected_chunks))
                )
                self.assertEqual(grids, [(3, 2), (3, 1)])
                dense_size = max(chunk.numel() for chunk in expected_chunks)
                self.assertTrue(
                    torch.equal(
                        batch.prefix_dense_kv_indices,
                        torch.arange(dense_size, dtype=torch.int32, device="cuda"),
                    )
                )
                self.assertTrue(bool((batch.mha_empty_kv_indptr == 0).all()))


if __name__ == "__main__":
    unittest.main()
