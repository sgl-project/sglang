"""Checkpoint-free checks of decoder replay dependencies and cached windows."""

import unittest
from array import array
from itertools import product

import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    decoder_replay_tail_len,
    late_layer_tail_layout,
    late_layer_tail_lens,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import _req_radix_key, release_kv_cache
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_swa_cache,
    setup_allocator_context,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _attention_stack(x, positions, window, layers):
    memory = torch.cos(torch.arange(24, dtype=x.dtype).view(3, 8))
    distance = positions[:, None] - positions[None, :]
    visible = (distance >= 0) & (distance < window)
    visible = torch.cat(
        [visible, torch.ones(len(x), len(memory), dtype=torch.bool)], dim=1
    )
    for _ in range(layers):
        kv = torch.cat([x, memory])
        scores = (x @ kv.T / 8**0.5).masked_fill(~visible, -torch.inf)
        x = x + scores.softmax(-1) @ kv
    return x


def _inputs(length):
    return 0.1 * torch.randn(
        length, 8, generator=torch.Generator().manual_seed(123), dtype=torch.float64
    )


class TestDecoderReplayDependencies(CustomTestCase):
    def test_replay_preserves_the_final_window(self):
        for window, layers in ((2, 2), (4, 4), (8, 3)):
            with self.subTest(window=window, layers=layers):
                tail_len = decoder_replay_tail_len(window, layers + 2, 2)
                length = tail_len + 19
                x = _inputs(length)
                reference = _attention_stack(x, torch.arange(length), window, layers)
                rows, lens, floor = late_layer_tail_layout(
                    extend_lens_cpu=[length],
                    seq_lens_cpu=[length],
                    tail_len=tail_len,
                    device=torch.device("cpu"),
                )
                self.assertEqual(lens, [tail_len])
                self.assertTrue(torch.all(floor == length - tail_len))
                replay = _attention_stack(x[rows], rows, window, layers)
                torch.testing.assert_close(
                    replay[-window:], reference[-window:], rtol=1e-12, atol=1e-12
                )
                old = _attention_stack(
                    x[-window:], torch.arange(length - window, length), window, layers
                )
                self.assertFalse(
                    torch.allclose(old[-1], reference[-1], rtol=1e-4, atol=1e-4)
                )

    def test_flash_bound_and_prompt_readers(self):
        tail_len = decoder_replay_tail_len(128, 40, 20, page_size=256)
        self.assertEqual(tail_len, 3052)
        self.assertEqual(decoder_replay_tail_len(128, 40, 20), 2669)
        self.assertEqual(
            late_layer_tail_lens([79, 512, 8192], tail_len), [79, 512, 3052]
        )
        self.assertEqual(
            late_layer_tail_lens([79, 512, 8192], tail_len, [False, False, True]),
            [79, 512, 8192],
        )


class TestDecoderReplayCache(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)

    def test_reuses_only_complete_windows(self):
        for backend, (page_size, window), finished, branch in product(
            ("python", "rust"),
            ((1, 4), (4, 4), (16, 4), (4, 6)),
            (False, True),
            (False, True),
        ):
            for residue in range(page_size):
                with (
                    self.subTest(
                        backend=backend,
                        page_size=page_size,
                        window=window,
                        finished=finished,
                        branch=branch,
                        residue=residue,
                    ),
                    envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
                    envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(
                        False
                    ),
                ):
                    self._check_window(page_size, window, finished, branch, residue)

    def _check_window(self, page_size, window, finished, branch, residue):
        layers = 3
        tail_len = decoder_replay_tail_len(window, layers + 2, 2, page_size=page_size)
        length = 80 + residue
        allocator, cache = build_swa_cache(
            occupancy=0, page_size=page_size, window=window
        )
        self.assertEqual(
            cache._tree_core_backend,
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(),
        )
        cache.decoder_swa_bounded_replay = True
        x = _inputs(length)
        reference = _attention_stack(x, torch.arange(length), window, layers)
        rows = torch.arange(max(0, length - tail_len), length)
        replay = _attention_stack(x[rows], rows, window, layers)
        values = torch.full_like(x, torch.nan)
        values[rows] = replay

        req = Req("prefill", "", array("q", range(1, length + 1)), SamplingParams())
        pool = cache.req_to_token_pool
        self.assertIsNotNone(pool.alloc([req]))
        slots = allocator.alloc((length + page_size - 1) // page_size * page_size)
        self.assertIsNotNone(slots)
        pool.write((req.kv.req_pool_idx, slice(0, length)), slots[:length])
        req.kv.kv_committed_len = req.kv.kv_allocated_len = length
        req.lock = cache.lock(cache.root_node_handle())
        if branch:
            req.swa_branching_seqlen = 32
        if finished:
            req.finished_reason = FINISH_LENGTH(length=0)
            release_kv_cache(req, cache, checkpoint=True)
        else:
            req.refresh_fill_ids()
            cache.checkpoint(req, up_to=length)
            release_kv_cache(req, cache, checkpoint=False)

        for query_length in range(1, length + 2):
            query = array("q", range(1, query_length + 1))
            next_req = Req("reuse", "", query, SamplingParams())
            for limit in (None, query_length - 1):
                hit = cache.match_prefix(
                    MatchPrefixParams(key=_req_radix_key(cache, next_req, query, limit))
                )
                end = hit.device_prefix_len
                if query_length > length or (query_length == length and limit is None):
                    self.assertEqual(end, length // page_size * page_size)
                if end:
                    start = max(0, end - window)
                    torch.testing.assert_close(
                        values[start:end], reference[start:end], rtol=1e-12, atol=1e-12
                    )
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
