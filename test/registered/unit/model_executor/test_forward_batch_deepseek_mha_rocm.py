"""ROCm packed prefix-chunk layout: no chunk exceeds SGLANG_MAX_KV_CHUNK_CAPACITY keys,
each prefix token is covered once and in order, and only HIP takes this layout."""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import create_autospec

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor import forward_batch_deepseek_mha_mixin as mix
from sglang.srt.model_executor.forward_batch_deepseek_mha_rocm import (
    get_num_packed_prefix_chunks,
    get_packed_prefix_chunk_seq_lens,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CASES = [
    # (prefix_lens, chunk_capacity)
    ([64, 64, 64], 48),
    ([16, 32, 48, 64], 64),
    ([1, 64], 31),
    ([256, 0, 1536], 1024),
    ([0, 0, 5], 4),
    ([7], 7),
    ([3, 5, 0, 9, 1], 6),
]


def _prepare(prefix_lens, capacity):
    bs = len(prefix_lens)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.zeros(bs, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        seq_lens=torch.tensor(prefix_lens, dtype=torch.int64) + 1,
        out_cache_loc=torch.zeros(bs, dtype=torch.int64),
        seq_lens_sum=sum(prefix_lens) + bs,
    )
    fb.extend_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32)
    fb.extend_prefix_lens_cpu = list(prefix_lens)
    fb.iteration_ids = object()

    def pack(plan, *, req_pool_indices, seq_lens, indptr, out, kv_start_idx):
        out.zero_()
        return True

    runner_ids = create_autospec(RunnerIds, instance=True)
    runner_ids.reads_are_translated = True
    runner_ids.pack_read_stream = pack
    pool = create_autospec(MLATokenToKVPool, instance=True)
    pool.start_layer, pool.layer_num = 0, 1
    pool.layer_router = PlainPoolRouter(pool, "full")
    backend = SimpleNamespace(
        runner_ids=runner_ids,
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((bs, 1))),
    )
    with (
        envs.SGLANG_MAX_KV_CHUNK_CAPACITY.override(capacity),
        forward_context(ForwardContext(attn_backend=backend)),
    ):
        fb.prepare_chunked_prefix_cache_info(torch.device("cpu"))
    return fb


class TestPackedPrefixChunkLayout(CustomTestCase):
    def _assert_valid_layout(self, prefix_lens, capacity, starts, seq_lens):
        self.assertEqual(starts.shape, seq_lens.shape)
        self.assertEqual(starts.shape[1], len(prefix_lens))
        self.assertEqual(starts.dtype, torch.int32)
        self.assertEqual(seq_lens.dtype, torch.int32)

        # Each request's pieces tile [0, prefix_len) in chunk order.
        for req, prefix_len in enumerate(prefix_lens):
            covered = []
            for chunk in range(starts.shape[0]):
                start, length = int(starts[chunk, req]), int(seq_lens[chunk, req])
                self.assertGreaterEqual(length, 0)
                covered.extend(range(start, start + length))
            self.assertEqual(covered, list(range(prefix_len)))

        # Every chunk but the last is full and the last is not empty,
        # so no layout needs more than ceil(sum / capacity) chunks.
        totals = seq_lens.sum(dim=1).tolist()
        self.assertTrue(all(total == capacity for total in totals[:-1]))
        self.assertTrue(0 < totals[-1] <= capacity)

    def test_packed_layout_covers_each_prefix_token_once_within_capacity(self):
        for prefix_lens, capacity in _CASES:
            with self.subTest(prefix_lens=prefix_lens, capacity=capacity):
                starts, seq_lens = get_packed_prefix_chunk_seq_lens(
                    prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
                    num_prefix_chunks=get_num_packed_prefix_chunks(
                        prefix_lens_cpu=prefix_lens, chunk_capacity=capacity
                    ),
                    prefix_chunk_len=capacity,
                )
                self._assert_valid_layout(prefix_lens, capacity, starts, seq_lens)

    def test_only_hip_packs_long_and_short_prefixes_together(self):
        # Request-major chunks give each request capacity // batch_size keys,
        # so the longest prefix alone needs 22 of them here.
        prefix_lens, capacity = [350000, 15000, 7495, 0], 65536

        with mock.patch.object(mix, "_is_hip", False):
            fb = _prepare(prefix_lens, capacity)
        self.assertEqual(fb.num_prefix_chunks, 22)
        self.assertEqual(fb.prefix_chunk_len, capacity // len(prefix_lens))
        expected_starts = [[i * fb.prefix_chunk_len] * 4 for i in range(22)]
        self.assertEqual(fb.prefix_chunk_starts.tolist(), expected_starts)

        with mock.patch.object(mix, "_is_hip", True):
            fb = _prepare(prefix_lens, capacity)
        self.assertEqual(fb.num_prefix_chunks, 6)
        self.assertEqual(fb.prefix_chunk_len, capacity)
        self._assert_valid_layout(
            prefix_lens, capacity, fb.prefix_chunk_starts, fb.prefix_chunk_seq_lens
        )
        self.assertTrue(torch.equal(fb.prefix_chunk_starts_cpu, fb.prefix_chunk_starts))
        self.assertTrue(
            torch.equal(fb.prefix_chunk_seq_lens_cpu, fb.prefix_chunk_seq_lens)
        )
        self.assertEqual(
            [len(indices) for indices in fb.prefix_chunk_kv_indices],
            fb.prefix_chunk_seq_lens.sum(dim=1).tolist(),
        )


if __name__ == "__main__":
    unittest.main()
