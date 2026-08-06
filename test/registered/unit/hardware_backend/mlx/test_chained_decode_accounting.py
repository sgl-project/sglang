"""Regression tests for MLX chained-decode accounting (#30093).

Before this fix, `event_loop_overlap_mlx`'s chained-decode path
(`_launch_chained` in `hardware_backend/mlx/scheduler_mixin.py`) never called
`ScheduleBatch.prepare_for_decode()`. Chained tokens therefore never went
through `alloc_for_decode()`: no real KV-pool slot was reserved, no entry was
written into `req_to_token`, and `seq_lens` / `kv_committed_len` never
advanced. Two observable consequences:

* Corruption risk: the MLX decode-KV pool sync (`_sync_decode_kv_to_pool`)
  derives its copy range from the per-request MLX cache offset, which keeps
  advancing every chained step. Reading `req_to_token` past the real
  scheduler-committed bound returns either the reserved padding slot (fresh
  row) or another request's slot ids (reused row).
* Accounting loss: `kv_committed_len` — the radix-insertion boundary read by
  `cache_finished_req` (see `RadixCache.cache_finished_req`,
  `mem_cache/radix_cache.py`) — permanently undercounts chained output, so
  those tokens are never prefix-cacheable.

These tests exercise the real scheduling primitives `ScheduleBatch`, `Req`,
`ReqToTokenPool`, `TokenToKVPoolAllocator`, and `RadixCache` directly (no
mocked scheduler internals) to prove that after the fix:

1. every chained step gets a real, freshly allocated, non-zero, sequential
   KV-pool slot recorded in `req_to_token`;
2. `kv_committed_len` / `seq_lens` track the number of generated tokens
   exactly, not just the pre-chain prefix;
3. those chained positions are visible to a subsequent radix prefix match;
4. two concurrently chaining requests never collide on the same slots;
5. the MLX overlap loop (`event_loop_overlap_mlx`) declines to extend a
   chain instead of raising when the KV pool is exhausted, several chained
   steps compound correctly, and accounting is continuous across a
   chain-break back into an ordinary scheduled decode step.

This module intentionally does not re-test the decode-KV pool *sync*
mechanism (`_sync_decode_kv_to_pool`, `_req_synced_offset`) or the
release-boundary flush for a freed-and-reused `req_to_token` row: those are
a separate invariant (a request finishing while holding unflushed decode KV,
addressed by the open PR #30147's `flush_request_decode_kv` /
`prepare_for_kv_cache_release` changes) that this fix does not touch.
"""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"


class _DummyKVCache(KVCache):
    """Scheduler-facing KV cache that allocates no real GPU/Metal memory.

    Mirrors `hardware_backend/mlx/model_runner_stub.py`'s `_DummyKVCache`:
    the MLX backend manages real attention KV itself, so the core scheduler
    only needs this to size `TokenToKVPoolAllocator`'s slot free-list.
    """

    def __init__(self, size, dtype, device):
        self.size = size
        self.page_size = 1
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def get_key_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no key buffer")

    def get_value_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no value buffer")

    def get_kv_buffer(self, layer_id):
        raise RuntimeError("_DummyKVCache has no kv buffer")

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        raise RuntimeError("_DummyKVCache cannot set kv buffer")

    def get_kv_size_bytes(self):
        return 0, 0


def _install_server_args(test_case, **fields):
    """Publish a throwaway ServerArgs for the duration of one test.

    prepare_for_decode() reads get_server_args()/get_exec() internally;
    real ScheduleBatch/Req plumbing requires the runtime context to be
    published, same as `test_chunked_prefix_cache_gate.py`.
    """
    override = get_context().override_server_args(**fields)
    override.install()
    test_case.addCleanup(override.restore)


def _make_pools(pool_size=64, req_slots=8, max_context_len=128, disable_radix=False):
    rtp = ReqToTokenPool(
        size=req_slots,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )
    dummy_kv = _DummyKVCache(pool_size, torch.float32, "cpu")
    allocator = TokenToKVPoolAllocator(
        size=pool_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=dummy_kv,
        need_sort=False,
    )
    tree_cache = RadixCache(
        CacheInitParams(
            disable=disable_radix,
            req_to_token_pool=rtp,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return rtp, allocator, tree_cache


def _make_req(rid, prompt_len, max_new_tokens=64):
    sp = SamplingParams()
    sp.max_new_tokens = max_new_tokens
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(prompt_len)),
        sampling_params=sp,
    )
    return req


def _admit_prefill(req, rtp, allocator, tree_cache):
    """Simulate the prefill admission that would have run before decode:
    allocate the row + real prefix slots and write them into req_to_token,
    exactly like `alloc_for_extend` does. Returns req_pool_idx.
    """
    prompt_len = len(req.origin_input_ids)
    req_pool_idx = rtp.alloc([req])[0]
    prefix_slots = allocator.alloc(prompt_len)
    assert prefix_slots is not None
    rtp.write((req_pool_idx, slice(0, prompt_len)), prefix_slots.to(torch.int32))
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = prompt_len
    req.kv = ReqKvInfo(kv_allocated_len=prompt_len, swa_evicted_seqlen=0)
    req.decode_batch_idx = 0
    return req_pool_idx


def _make_decode_batch(reqs, req_pool_indices, rtp, allocator, tree_cache):
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    model_config = SimpleNamespace(is_encoder_decoder=False, vocab_size=100)
    batch = ScheduleBatch(
        reqs=reqs,
        req_to_token_pool=rtp,
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    seq_lens = [r.kv_committed_len for r in reqs]
    batch.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int64)
    batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    return batch


class TestChainedPrepareForDecodeAccounting(CustomTestCase):
    """Direct calls to the real `ScheduleBatch.prepare_for_decode()`.

    This is exactly what `_launch_chained` now calls once per chained
    step (`hardware_backend/mlx/scheduler_mixin.py`), so these tests pin the
    core mechanism the MLX fix depends on, with zero MLX/mocking involved.
    """

    def setUp(self):
        _install_server_args(self)

    def test_fresh_row_chained_steps_get_real_sequential_nonzero_slots(self):
        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=5)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        # Simulate 4 chained decode steps: each is exactly what
        # _launch_chained does for one chained token.
        for _ in range(4):
            batch.prepare_for_decode()

        row = rtp.req_to_token[req_pool_idx, :9].tolist()
        # 5 prefix + 4 chained positions: real, non-zero, strictly
        # sequential, no duplicates. Before the fix, positions 5..8 would
        # never have been written at all (they would still read the
        # zero-initialized padding value).
        self.assertEqual(row, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertNotIn(0, row)
        self.assertEqual(len(set(row)), len(row))

    def test_kv_committed_len_and_seq_lens_track_every_chained_token(self):
        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        n_chained = 7
        for _ in range(n_chained):
            batch.prepare_for_decode()

        self.assertEqual(req.kv_committed_len, 3 + n_chained)
        self.assertEqual(int(batch.seq_lens.item()), 3 + n_chained)
        self.assertEqual(int(batch.seq_lens_cpu.item()), 3 + n_chained)
        self.assertEqual(req.decode_batch_idx, n_chained)

    def test_chained_positions_become_prefix_cacheable(self):
        # Reproduces the issue's exact accounting-loss scenario: a single
        # request's decode output must be fully radix-visible, not
        # truncated at the pre-chain prefix.
        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=4)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        n_chained = 5
        for _ in range(n_chained):
            batch.prepare_for_decode()

        total_len = 4 + n_chained
        req.output_ids = array("q", range(100, 100 + n_chained))
        req.to_finish = None
        release_kv_cache(req, tree_cache, is_insert=True)

        full_ids = req.origin_input_ids + req.output_ids
        match = tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(full_ids, None)))
        self.assertEqual(len(match.device_indices), total_len)

    def test_two_concurrently_chaining_requests_never_collide(self):
        rtp, allocator, tree_cache = _make_pools(pool_size=64, req_slots=8)
        req_a = _make_req("a", prompt_len=3)
        req_b = _make_req("b", prompt_len=3)
        idx_a = _admit_prefill(req_a, rtp, allocator, tree_cache)
        idx_b = _admit_prefill(req_b, rtp, allocator, tree_cache)
        batch = _make_decode_batch(
            [req_a, req_b], [idx_a, idx_b], rtp, allocator, tree_cache
        )

        for _ in range(4):
            batch.prepare_for_decode()

        slots_a = set(rtp.req_to_token[idx_a, :7].tolist())
        slots_b = set(rtp.req_to_token[idx_b, :7].tolist())
        self.assertEqual(len(slots_a & slots_b), 0)
        self.assertNotIn(0, slots_a)
        self.assertNotIn(0, slots_b)

    def test_allocation_exhaustion_raises_instead_of_silently_corrupting(self):
        # Pins why the MLX loop must check check_decode_mem() before
        # extending the chain: an unguarded prepare_for_decode() call on an
        # exhausted pool fails loud rather than silently reusing/aliasing a
        # slot.
        rtp, allocator, tree_cache = _make_pools(pool_size=4, req_slots=4)
        req = _make_req("r1", prompt_len=4)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        self.assertFalse(batch.check_decode_mem())
        with self.assertRaises(RuntimeError):
            batch.prepare_for_decode()


class _StopLoop(Exception):
    """Sentinel to break out of event_loop_overlap_mlx's `while True`."""


def _run_loop(scheduler):
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
        SchedulerMlxOverlapMixin,
    )

    try:
        SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)
    except _StopLoop:
        pass


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestEventLoopOverlapMlxRealAccounting(CustomTestCase):
    """Drives the real `event_loop_overlap_mlx` loop with a real
    `ScheduleBatch`/`Req`/pool stack, mocking only the scheduler wrapper and
    the MLX forward pass itself (the model), per the MLX test conventions
    (`test/registered/unit/README.md`): fake the model, not the scheduling
    logic under test.
    """

    def setUp(self):
        _install_server_args(self)

    def _make_scheduler(self, batch, *, recv_side_effect):
        from collections import deque
        from unittest.mock import MagicMock

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler.running_batch = batch
        scheduler.last_batch = None
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.request_receiver.recv_requests.side_effect = recv_side_effect
        scheduler._prepare_mlx_launch.side_effect = lambda b: (
            SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, b)
        )
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_several_chained_steps_advance_real_req_to_token(self):
        from unittest.mock import MagicMock

        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        # recv_requests: empty every iteration; enough iterations for one
        # fresh launch + several chained continuations.
        n_iters = 6
        scheduler = self._make_scheduler(
            batch, recv_side_effect=[[] for _ in range(n_iters)] + [_StopLoop()]
        )
        plan = SimpleNamespace(batch_to_run=batch, running_batch=batch)
        scheduler.get_next_batch_to_run.return_value = plan

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            pending_decode,
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )

        _run_loop(scheduler)

        n_chained_calls = scheduler.tp_worker.async_chained_decode_mlx.call_count
        self.assertGreater(n_chained_calls, 0)
        expected_committed = 3 + n_chained_calls
        self.assertEqual(req.kv_committed_len, expected_committed)
        row = rtp.req_to_token[req_pool_idx, :expected_committed].tolist()
        self.assertNotIn(0, row)
        self.assertEqual(len(set(row)), len(row))

    def test_check_decode_mem_declines_chain_when_pool_exhausted(self):
        from unittest.mock import MagicMock

        # Pool has exactly 2 spare tokens beyond the 3-token prefix: one
        # fresh decode step (consumes 1) leaves exactly 1 more available,
        # so the chain can extend once and must then decline rather than
        # raising out of alloc_for_decode().
        rtp, allocator, tree_cache = _make_pools(pool_size=5, req_slots=4)
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        n_iters = 6
        scheduler = self._make_scheduler(
            batch, recv_side_effect=[[] for _ in range(n_iters)] + [_StopLoop()]
        )

        def get_next_batch_to_run(**_kwargs):
            # Emulates update_running_batch(): real accounting call, same
            # object identity, exactly what the standard scheduler path does
            # for a fresh/post-break decode step.
            if batch.check_decode_mem():
                batch.prepare_for_decode()
            return SimpleNamespace(batch_to_run=batch, running_batch=batch)

        scheduler.get_next_batch_to_run.side_effect = get_next_batch_to_run

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            pending_decode,
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )

        _run_loop(scheduler)

        # Never raised (the loop must have declined to chain rather than
        # calling alloc_for_decode() on an exhausted pool), and committed
        # length must never exceed the pool's real capacity.
        self.assertLessEqual(req.kv_committed_len, 3 + 2)
        row = rtp.req_to_token[req_pool_idx, : req.kv_committed_len].tolist()
        self.assertNotIn(0, row)
        self.assertEqual(len(set(row)), len(row))

    def test_chain_extends_correctly_with_radix_cache_disabled(self):
        # disable_radix_cache only turns off the MLX-side pool sync
        # (model_runner.py, untouched by this patch); the core scheduler
        # accounting this patch adds to _launch_chained does not depend on
        # it and must behave identically.
        from unittest.mock import MagicMock

        rtp, allocator, tree_cache = _make_pools(disable_radix=True)
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        n_iters = 5
        scheduler = self._make_scheduler(
            batch, recv_side_effect=[[] for _ in range(n_iters)] + [_StopLoop()]
        )
        plan = SimpleNamespace(batch_to_run=batch, running_batch=batch)
        scheduler.get_next_batch_to_run.return_value = plan

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            pending_decode,
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )

        _run_loop(scheduler)

        n_chained_calls = scheduler.tp_worker.async_chained_decode_mlx.call_count
        self.assertGreater(n_chained_calls, 0)
        self.assertEqual(req.kv_committed_len, 3 + n_chained_calls)

    def test_chain_break_then_ordinary_decode_continues_accounting(self):
        # A new prefill arriving mid-chain forces a break: pending_next is
        # finalized and the loop falls back to get_next_batch_to_run(),
        # which (in the real scheduler) calls update_running_batch() ->
        # prepare_for_decode() again on the SAME schedule_batch identity.
        # The break must not skip a position or double count one.
        from unittest.mock import MagicMock

        rtp, allocator, tree_cache = _make_pools()
        req = _make_req("r1", prompt_len=3)
        req_pool_idx = _admit_prefill(req, rtp, allocator, tree_cache)
        batch = _make_decode_batch([req], [req_pool_idx], rtp, allocator, tree_cache)

        n_iters = 8
        scheduler = self._make_scheduler(batch, recv_side_effect=[])

        # After 3 iterations, a new prefill shows up in the waiting queue,
        # forcing exactly one chain break; get_next_batch_to_run() then
        # clears it (in the real scheduler this would be consumed by the
        # prefill branch -- here we only care about the decode
        # continuation, so just clear it back out) and continues decoding
        # via a real prepare_for_decode() call, exactly like
        # update_running_batch() would.
        def get_next_batch_to_run(**_kwargs):
            scheduler.waiting_queue = []
            batch.prepare_for_decode()
            return SimpleNamespace(batch_to_run=batch, running_batch=batch)

        scheduler.get_next_batch_to_run.side_effect = get_next_batch_to_run

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = (
            MagicMock(),
            [],
            [],
            pending_decode,
            "decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            MagicMock(),
            [],
            [],
            MagicMock(),
            "decode",
        )

        recv_count = {"n": 0}

        def recv_requests():
            recv_count["n"] += 1
            if recv_count["n"] == 4:
                # Simulate a new prefill request landing mid-chain.
                scheduler.waiting_queue = [MagicMock()]
            if recv_count["n"] > n_iters:
                raise _StopLoop()
            return []

        scheduler.request_receiver.recv_requests.side_effect = recv_requests

        _run_loop(scheduler)

        n_chained_calls = scheduler.tp_worker.async_chained_decode_mlx.call_count
        n_fresh_calls = scheduler.get_next_batch_to_run.call_count
        total_decode_steps = n_chained_calls + n_fresh_calls
        self.assertGreater(n_chained_calls, 0)
        self.assertGreater(n_fresh_calls, 1)
        # No gap, no double count across the break: every decode step
        # (chained or fresh) advanced kv_committed_len by exactly 1.
        self.assertEqual(req.kv_committed_len, 3 + total_decode_steps)
        row = rtp.req_to_token[req_pool_idx, : req.kv_committed_len].tolist()
        self.assertNotIn(0, row)
        self.assertEqual(len(set(row)), len(row))
        self.assertEqual(row, sorted(row))


if __name__ == "__main__":
    unittest.main()
