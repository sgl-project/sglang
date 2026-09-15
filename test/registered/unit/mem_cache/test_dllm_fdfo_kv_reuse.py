"""Tests dLLM FDFO KV slot reuse in alloc_for_extend."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    DecLockRefParams,
    EvictParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, base=1000, page_size=1):
        self.base = base
        self.page_size = page_size
        self.alloc_calls = []
        self.extend_calls = []

    def available_size(self):
        return 1 << 30

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        return torch.arange(self.base, self.base + need_size, dtype=torch.int64)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        **kwargs,
    ):
        self.extend_calls.append(
            {
                "extend_num_tokens": extend_num_tokens,
                "seq_lens_cpu": seq_lens_cpu.tolist(),
            }
        )
        return torch.arange(self.base, self.base + extend_num_tokens, dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self, allocator):
        self.page_size = allocator.page_size
        self.token_to_kv_pool_allocator = allocator

    def is_chunk_cache(self):
        return True


class _SchedulerHarness:
    process_dllm_staging_reqs = SchedulerDllmMixin.process_dllm_staging_reqs
    _cleanup_dllm_req = SchedulerDllmMixin._cleanup_dllm_req
    _abort_dllm_req_exact = SchedulerDllmMixin._abort_dllm_req_exact
    _retract_dllm_req = SchedulerDllmMixin._retract_dllm_req
    _retract_or_abort_dllm_req = SchedulerDllmMixin._retract_or_abort_dllm_req
    # Use the real teardown so the cache-handle contract is exercised.
    _release_aborted_request = Scheduler._release_aborted_request


def _make_req(rid, prefix, block_size, *, req_pool_idx=None, reuse=False):
    return SimpleNamespace(
        rid=rid,
        cache_request_handle=CacheRequestHandle(rid=rid, attempt_id=0),
        prefix_indices=torch.tensor(prefix, dtype=torch.int32),
        # Admitted at some point, which is what stashes a request as locked.
        dllm_phase=DllmReqPhase.STAGING_DECODE,
        dllm_incomplete_ids=array("q", range(block_size)) if reuse else array("q"),
        inflight_middle_chunks=1 if req_pool_idx is not None else 0,
        last_node=None,
        lock_receipt=DecLockRefParams(),
        # `_make_abort_req` reads these to build the weight-version spans that
        # every abort output carries.
        output_ids=array("q"),
        weight_version_events=[],
        # The real record, so the scheduler reads the same fields and defaults
        # it does in production.
        kv=ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=len(prefix) if req_pool_idx is not None else 0,
            kv_allocated_len=(
                len(prefix) + block_size if req_pool_idx is not None else 0
            ),
        ),
    )


def _remove_allocated_req_slots(pool, *reqs):
    for req in reqs:
        if req.kv.req_pool_idx in pool.free_slots:
            pool.free_slots.remove(req.kv.req_pool_idx)


def _make_batch(pool, allocator, reqs, extend_lens):
    seq_lens_cpu = torch.tensor(
        [
            len(req.prefix_indices) + extend_len
            for req, extend_len in zip(reqs, extend_lens)
        ],
        dtype=torch.int64,
    )
    return SimpleNamespace(
        device="cpu",
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=_FakeTreeCache(allocator),
        prefix_lens=[len(req.prefix_indices) for req in reqs],
        extend_lens=extend_lens,
        seq_lens=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        extend_num_tokens=sum(extend_lens),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: True,
    )


def _seed_retained_block(pool, req, values):
    prefix_len = len(req.prefix_indices)
    pool.req_to_token[req.kv.req_pool_idx, :prefix_len] = req.prefix_indices
    pool.req_to_token[req.kv.req_pool_idx, prefix_len : prefix_len + len(values)] = (
        torch.tensor(values, dtype=torch.int32)
    )


class TestDllmFdfoKvReuse(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        override = get_context().override_server_args(
            attention_backend="torch_native", dcp_size=1
        )
        override.install()
        self.addCleanup(override.restore)

    def test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows(self):
        allocator = _FakeAllocator(base=200)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [4])
        # Allocation order is not semantically meaningful (ReqToTokenPool.alloc
        # picks whichever free slot is cheapest to pop), so only pin the
        # reused row's index and that the fresh row got a different, real slot.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        fresh_idx = req_pool_indices_cpu[1].item()
        self.assertNotEqual(fresh_idx, 1)
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 200, 201, 202, 203])
        self.assertEqual(self.pool.req_to_token[1, 4:8].tolist(), [100, 101, 102, 103])
        self.assertEqual(
            self.pool.req_to_token[fresh_idx, 4:8].tolist(), [200, 201, 202, 203]
        )
        self.assertEqual(reused.kv.kv_allocated_len, 8)
        self.assertEqual(fresh.kv.kv_allocated_len, 8)

    def test_alloc_for_extend_all_reuse_allocates_nothing(self):
        allocator = _FakeAllocator(base=900)
        req0 = _make_req(
            "r0", [1, 2, 3, 4], self.block_size, req_pool_idx=1, reuse=True
        )
        req1 = _make_req(
            "r1", [5, 6, 7, 8], self.block_size, req_pool_idx=2, reuse=True
        )
        _remove_allocated_req_slots(self.pool, req0, req1)
        _seed_retained_block(self.pool, req0, [300, 301, 302, 303])
        _seed_retained_block(self.pool, req1, [400, 401, 402, 403])

        batch = _make_batch(self.pool, allocator, [req0, req1], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        self.assertEqual(allocator.alloc_calls, [])
        self.assertEqual(req_pool_indices_cpu.tolist(), [1, 2])
        self.assertEqual(out.tolist(), [300, 301, 302, 303, 400, 401, 402, 403])

    def test_alloc_for_extend_paged_mixed_reuse_skips_reused_rows(self):
        allocator = _FakeAllocator(base=500, page_size=4)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        fresh = _make_req("fresh", [20, 21, 22, 23], self.block_size)
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused, fresh], [4, 4])
        out, _, req_pool_indices_cpu = alloc_for_extend(batch)

        # See test_alloc_for_extend_mixed_reuse_allocates_only_fresh_and_writes_rows:
        # allocation order is not semantically meaningful.
        self.assertEqual(req_pool_indices_cpu[0].item(), 1)
        self.assertNotEqual(req_pool_indices_cpu[1].item(), 1)
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 500, 501, 502, 503])
        self.assertEqual(
            allocator.extend_calls,
            [{"extend_num_tokens": 4, "seq_lens_cpu": [4, 8]}],
        )

    def test_alloc_for_extend_rejects_partial_retained_block_reuse(self):
        allocator = _FakeAllocator(base=700)
        reused = _make_req(
            "reuse", [10, 11, 12, 13], self.block_size, req_pool_idx=1, reuse=True
        )
        _remove_allocated_req_slots(self.pool, reused)
        _seed_retained_block(self.pool, reused, [100, 101, 102, 103])

        batch = _make_batch(self.pool, allocator, [reused], [2])
        with self.assertRaisesRegex(RuntimeError, "full block"):
            alloc_for_extend(batch)

    def test_dllm_manager_pop_aborted_reqs_removes_waiting_and_staging(self):
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        waiting = _make_req("abort-waiting", [1], self.block_size)
        staging = _make_req("abort-staging", [2], self.block_size)
        keep = _make_req("keep", [3], self.block_size)
        manager.waiting_queue = [waiting, keep]
        manager.staging_queue = [staging, waiting]

        aborted = manager.pop_aborted_reqs(False, "abort")

        self.assertEqual(
            [req.rid for req in aborted], ["abort-waiting", "abort-staging"]
        )
        self.assertEqual(manager.waiting_queue, [keep])
        self.assertEqual(manager.staging_queue, [])

    def test_staging_no_token_retracts_instead_of_aborting(self):
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        victim = _make_req("job_1", [1], self.block_size)
        keep = _make_req("job_10", [2], self.block_size)
        for req in (victim, keep):
            req.dllm_phase = DllmReqPhase.STAGING_DECODE
            req.is_retracted = False
            req.origin_input_ids = [1]
            req.output_ids = [7, 8]
            req.dllm_config = SimpleNamespace(block_size=self.block_size)
            req.dllm_algo_state = object()
            req.dllm_block_offset = self.block_size
            req.reset_for_retract = lambda r=req: setattr(r, "is_retracted", True)
            req.time_stats = SimpleNamespace(set_retract_time=lambda: None)
            req.reset_dllm_for_retract = lambda r=req: (
                ReqDllmMixin.reset_dllm_for_retract(r)
            )
        manager.waiting_queue = [victim, keep]
        manager.staging_queue = []

        outputs = []
        freed = []
        scheduler = _SchedulerHarness()
        scheduler.dllm_manager = manager
        # A victim with no req_pool_idx still holds uncached prefix slots, which
        # _cleanup_dllm_req releases straight through the allocator.
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: freed.append(indices)
        )
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda msg, req: outputs.append((msg, req))
            )
        )
        adder = SimpleNamespace(
            can_run_list=[],
            running_batch=SimpleNamespace(is_empty=lambda: True),
            add_dllm_staging_req=lambda req: AddReqResult.NO_TOKEN,
        )

        result = scheduler.process_dllm_staging_reqs(adder, [victim, keep])
        self.assertEqual(result, AddReqResult.NO_TOKEN)

        scheduler._retract_or_abort_dllm_req(
            SimpleNamespace(is_empty=lambda: True),
        )

        # Retraction reclaims KV without telling the client, and both requests
        # stay managed so the freed pages can serve them next round.
        self.assertEqual(outputs, [])
        self.assertEqual(manager.waiting_queue, [victim, keep])
        self.assertIsNone(victim.kv.req_pool_idx)
        self.assertEqual(victim.kv.kv_allocated_len, 0)
        # Exactly the victim's uncached prefix slots go back to the allocator.
        self.assertEqual([t.tolist() for t in freed], [[1]])
        self.assertEqual(victim.dllm_phase, DllmReqPhase.INCOMING_DECODE)
        self.assertEqual(len(victim.dllm_incomplete_ids), 0)
        self.assertEqual(victim.output_ids, [7, 8])
        # Only the first blocker in manager order is given up.
        self.assertEqual(keep.dllm_phase, DllmReqPhase.STAGING_DECODE)

    def test_exact_abort_cleans_stashed_fdfo_before_pop_and_response(self):
        events = []
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        req = _make_req("job_1", [10, 11, 12, 13], self.block_size)
        keep = _make_req("job_10", [20], self.block_size)
        req.kv.cache_protected_len = 2
        req.kv.kv_allocated_len = 4
        req.last_node = 42
        req.lock_receipt = DecLockRefParams(
            node_id=req.last_node,
            swa_uuid_for_lock=7,
            skipped_lock_components=(ComponentType.MAMBA,),
        )
        manager.waiting_queue = [req, keep]
        manager.staging_queue = [req]

        original_pop = manager.pop_aborted_reqs

        def tracked_pop(abort_all, rid, *, exact=False):
            events.append("pop")
            return original_pop(abort_all, rid, exact=exact)

        manager.pop_aborted_reqs = tracked_pop

        freed = []

        def free(indices):
            events.append("free_kv")
            freed.append(indices.clone())

        unlocked = []

        def dec_lock_ref(node, params):
            self.assertIs(params, req.lock_receipt)
            events.append("unlock")
            unlocked.append(node)

        outputs = []

        def send_output(msg, output_req):
            events.append("send")
            outputs.append((msg, output_req))

        scheduler = _SchedulerHarness()
        scheduler.dllm_manager = manager
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=free)

        def finish(handle, outcome):
            self.assertEqual(handle, req.cache_request_handle)
            self.assertEqual(outcome, CacheRequestOutcome.ABORT)
            events.append("finish")

        scheduler.tree_cache = SimpleNamespace(dec_lock_ref=dec_lock_ref, finish=finish)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=send_output)
        )

        last_node = req.last_node
        scheduler._abort_dllm_req_exact(req)

        self.assertEqual(events, ["finish", "free_kv", "unlock", "pop", "send"])
        self.assertEqual(len(freed), 1)
        self.assertEqual(freed[0].tolist(), [12, 13])
        self.assertEqual(unlocked, [last_node])
        self.assertIsNone(req.last_node)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertEqual(req.kv.kv_allocated_len, 0)
        self.assertEqual(manager.waiting_queue, [keep])
        self.assertEqual(manager.staging_queue, [])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0][0].rid, "job_1")
        self.assertIs(outputs[0][1], req)
        # Built through `_make_abort_req` like every other abort path, so the
        # tokenizer manager sees the same payload it does elsewhere.
        self.assertIsNotNone(outputs[0][0].weight_versions)

    def _make_stashed_unified_req(self, phase):
        # These CPU tests exercise slot ownership; no KV tensor data is read.
        allocator = TokenToKVPoolAllocator(
            size=8, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                disable=False,
                tree_components=(ComponentType.FULL,),
            )
        )
        req = Req(
            rid="stashed",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
            dllm_config=SimpleNamespace(block_size=self.block_size),
        )
        req.output_ids = array("q", [5, 6, 7, 8])
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_range(0, 8)
        req.dllm_phase = phase
        req.dllm_initialized = True
        self.pool.alloc([req])
        self.pool.write((req.kv.req_pool_idx, slice(0, 8)), allocator.alloc(8))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 8
        req.last_node = cache.root_node_handle()
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()

        # Follow the FDFO stash path: caching takes a new lock and stores its
        # receipt; freeing the request slot leaves that lock and KV alive.
        cache.cache_unfinished_req(req, chunked=True)
        self.pool.free(req)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(req.lock_receipt.node_id, req.last_node)
        self.assertEqual(cache.protected_size(), 8)
        self.assertEqual(allocator.available_size(), 0)
        cache.dec_lock_ref = Mock(wraps=cache.dec_lock_ref)

        scheduler = _SchedulerHarness()
        scheduler.tree_cache = cache
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.dllm_manager = DllmManager(SimpleNamespace(max_running_requests=4))
        scheduler.dllm_manager.waiting_queue = [req]
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )
        return scheduler, req

    def test_stashed_unified_req_retracts_under_kv_pressure(self):
        for phase in (DllmReqPhase.STAGING_PREFILL, DllmReqPhase.STAGING_DECODE):
            with self.subTest(phase=phase):
                scheduler, req = self._make_stashed_unified_req(phase)
                cache = scheduler.tree_cache
                node, receipt = req.last_node, req.lock_receipt

                scheduler._retract_or_abort_dllm_req(
                    SimpleNamespace(is_empty=lambda: True)
                )

                cache.dec_lock_ref.assert_called_once_with(node, receipt)
                self.assertIs(cache.dec_lock_ref.call_args.args[1], receipt)
                self.assertEqual(scheduler.dllm_manager.waiting_queue, [req])
                scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
                self.assertTrue(req.is_retracted)
                self.assertEqual(req.retraction_count, 1)
                self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)
                self.assertEqual(list(req.output_ids), [5, 6, 7, 8])
                self.assertIsNone(req.last_node)
                self.assertIsNone(req.lock_receipt.node_id)
                self.assertTrue(req.kv.is_kv_released)
                self.assertEqual(cache.protected_size(), 0)
                self.assertEqual(cache.evictable_size(), 8)
                cache.sanity_check()
                cache.evict(EvictParams(num_tokens=8))
                self.assertEqual(
                    scheduler.token_to_kv_pool_allocator.available_size(), 8
                )

    def test_stashed_unified_req_aborts_and_releases_kv(self):
        for phase in (DllmReqPhase.STAGING_PREFILL, DllmReqPhase.STAGING_DECODE):
            with self.subTest(phase=phase):
                scheduler, req = self._make_stashed_unified_req(phase)
                cache = scheduler.tree_cache
                node, receipt = req.last_node, req.lock_receipt
                scheduler.dllm_manager.staging_queue = [req]

                scheduler._abort_dllm_req_exact(req)

                cache.dec_lock_ref.assert_called_once_with(node, receipt)
                self.assertIs(cache.dec_lock_ref.call_args.args[1], receipt)
                self.assertEqual(scheduler.dllm_manager.waiting_queue, [])
                self.assertEqual(scheduler.dllm_manager.staging_queue, [])
                send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
                send_output.assert_called_once()
                self.assertEqual(send_output.call_args.args[0].rid, req.rid)
                self.assertIs(send_output.call_args.args[1], req)
                self.assertIsNone(req.last_node)
                self.assertTrue(req.kv.is_kv_released)
                self.assertEqual(cache.protected_size(), 0)
                self.assertEqual(cache.evictable_size(), 8)
                cache.sanity_check()
                cache.evict(EvictParams(num_tokens=8))
                self.assertEqual(
                    scheduler.token_to_kv_pool_allocator.available_size(), 8
                )

    def test_abort_of_never_admitted_req_keeps_shared_prefix_locked(self):
        """An INCOMING request only ran match_prefix, which does not lock.

        Releasing its `last_node` would drop a ref it never took, and that node
        is shared with whichever live request actually put the prefix there.
        """
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import (
            CacheInitParams,
            InsertParams,
            RadixCache,
            RadixKey,
        )

        freed = []
        allocator = SimpleNamespace(
            device="cpu",
            page_size=1,
            available_size=lambda: 1 << 30,
            free=lambda indices: freed.append(indices.tolist()),
        )
        pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        cache = RadixCache(
            CacheInitParams(
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                disable=False,
            )
        )

        prefix = [1, 2, 3, 4]
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", prefix)),
                value=torch.arange(len(prefix), dtype=torch.int64),
            )
        )

        # A live request holding the prefix, as cache_unfinished_req would.
        live = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        cache.inc_lock_ref(live.last_device_node)
        self.assertEqual(cache.protected_size(), len(prefix))

        # The victim: matched the same node this round, never admitted, so
        # init_next_round_input left cache_protected_len == len(prefix_indices).
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        victim = _make_req("never-admitted", prefix, self.block_size)
        victim.dllm_phase = DllmReqPhase.INCOMING_PREFILL
        victim.prefix_indices = match.device_indices
        victim.last_node = match.last_device_node
        victim.kv = ReqKvInfo(cache_protected_len=len(match.device_indices))
        self.assertIs(victim.last_node, live.last_device_node)

        scheduler = _SchedulerHarness()
        scheduler.tree_cache = cache
        scheduler.token_to_kv_pool_allocator = allocator

        scheduler._cleanup_dllm_req(victim, is_abort=True)

        # The live request's prefix stays protected, and nothing is handed back.
        self.assertEqual(live.last_device_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), len(prefix))
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(freed, [])

    def test_retract_keeps_the_cache_attempt_state_that_abort_drops(self):
        """Retraction preserves the cache attempt; abort finishes its handle."""
        released = []
        manager = DllmManager(SimpleNamespace(max_running_requests=4))
        retracted = _make_req("job_1", [1], self.block_size)
        aborted = _make_req("job_2", [2], self.block_size)
        for req in (retracted, aborted):
            req.origin_input_ids = [1]
            req.output_ids = [7]
            req.dllm_config = SimpleNamespace(block_size=self.block_size)
            req.reset_for_retract = lambda: None
            req.reset_dllm_for_retract = lambda: None
            req.time_stats = SimpleNamespace(set_retract_time=lambda: None)
            # A STAGING request stashed by cache_unfinished_req: the req slot is
            # gone but kv_allocated_len still records the old length until the
            # teardown calls mark_kv_released(). Non-zero so that call is pinned.
            req.kv.kv_allocated_len = self.block_size
        manager.waiting_queue = [retracted, aborted]

        scheduler = _SchedulerHarness()
        scheduler.dllm_manager = manager
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda indices: None
        )
        scheduler.tree_cache = SimpleNamespace(
            finish=lambda handle, outcome: released.append((handle, outcome)),
            dec_lock_ref=lambda node, params: None,
        )
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=lambda msg, req: None)
        )

        scheduler._retract_dllm_req(retracted)
        self.assertEqual(released, [])
        # Retraction still gives the KV back, just not the cache state.
        self.assertEqual(retracted.kv.kv_allocated_len, 0)

        scheduler._abort_dllm_req_exact(aborted)
        self.assertEqual(
            released, [(aborted.cache_request_handle, CacheRequestOutcome.ABORT)]
        )


if __name__ == "__main__":
    unittest.main()
