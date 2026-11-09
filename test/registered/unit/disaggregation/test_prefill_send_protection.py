"""Keep chunked KV transfers alive across radix-cache checkpoints."""

import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingAllocator(SWATokenToKVPoolAllocator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freed = set()

    def free_full(self, indices):
        self.freed.update(indices.tolist())
        return super().free_full(indices)

    def free_segment(self, indices, *, start_pos):
        self.freed.update(indices.tolist())
        return super().free_segment(indices, start_pos=start_pos)

    def free_full_segment(self, indices, *, start_pos):
        self.freed.update(indices.tolist())
        return super().free_full_segment(indices, start_pos=start_pos)


class _Sender:
    """Transfers remain in flight until the test explicitly retires them."""

    # No transfer engine: a page move waits for nothing of these sends.
    kv_mgr = None
    bootstrap_room = 0

    def __init__(self):
        self.in_flight = set()
        self.sends = []

    def get_max_transfer_tokens(self):
        return None

    def should_send_kv_chunk(self, num_pages, last_chunk):
        return num_pages > 0 or last_chunk

    def send(self, page_indices, state_indices=None, num_kv_tokens=None):
        pages = np.asarray(page_indices).reshape(-1).tolist()
        self.in_flight.update(pages)
        self.sends.append(pages)


def _scheduler(cache, allocator, pool):
    scheduler = SchedulerDisaggregationPrefillMixin()
    scheduler.tree_cache = cache
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.req_to_token_pool = pool
    scheduler.enable_overlap = False
    scheduler.enable_staging = False
    scheduler.kv_checksum_computer = None
    scheduler.disagg_metadata_buffers = SimpleNamespace(set_buf=lambda req: None)
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
    )
    scheduler.disagg_prefill_pending_chunk_rids = set()
    return scheduler


def _drive(backend, chunk_ends, *, sibling, swa_size=128):
    pool = ReqToTokenPool(
        size=4, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv = SWAKVPool(
        size=256,
        size_swa=swa_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    allocator = _RecordingAllocator(
        size=256,
        size_swa=swa_size,
        page_size=1,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=False,
        req_to_token_pool=pool,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            sliding_window_size=4,
            tree_core_backend=backend,
            tree_slots=tree_slots_of((ComponentType.FULL, ComponentType.SWA), window=4),
        )
    )

    def insert(tokens):
        indices = allocator.alloc(len(tokens))
        result = cache.insert(
            InsertParams(key=RadixKey(array("q", tokens)), value=indices)
        )
        allocator.free(indices[: result.prefix_len])

    prefix = list(range(1, 5))
    old_suffix = list(range(100, 116))
    insert(prefix)
    insert(prefix + old_suffix)
    base = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", prefix)))
    ).last_device_node
    held = cache.lock(base)
    cache.evict(EvictParams(window_num_tokens={"swa": len(old_suffix)}))
    cache.unlock(held)

    tokens = prefix + old_suffix[:8] + list(range(200, 208))
    tokens += list(range(300, 300 + chunk_ends[-1] - len(tokens)))
    req = Req("branched", None, array("q", tokens), SamplingParams(max_new_tokens=1))
    pool.alloc([req])
    req.init_next_round_input(tree_cache=cache)
    assert req.swa_branching_seqlen == 12
    assert not req.pending_bootstrap
    pool.write(
        (req.kv.req_pool_idx, slice(0, len(prefix))),
        cache.path_device_indices(req.last_node),
    )
    req.extend_end = len(prefix)
    req.lock = cache.lock(req.last_node)
    req.disagg_kv_sender = sender = _Sender()
    scheduler = _scheduler(cache, allocator, pool)
    scheduler.chunked_req = req
    running = SimpleNamespace(batch_is_full=True)
    row = pool.req_to_token[req.kv.req_pool_idx]
    steps = []

    for i, end in enumerate(chunk_ends):
        if i == 1 and sibling:
            insert(tokens[:20])
        need = end - req.extend_end
        available = allocator.swa_available_size()
        assert available + cache.window_evictable_size("swa") >= need, (
            f"prefill stalled at {req.extend_end}, protected={req.kv.cache_protected_len}"
        )
        if available < need:
            cache.evict(EvictParams(window_num_tokens={"swa": need - available}))
        sent_before = req.start_send_idx
        row_before = row[:sent_before].clone()
        in_flight = set(sender.in_flight)
        allocator.freed.clear()
        pool.write(
            (req.kv.req_pool_idx, slice(req.extend_end, end)), allocator.alloc(need)
        )
        req.kv.set_row_len(end)
        req.extend_end = end
        if end < len(tokens):
            scheduler.process_prefill_chunk(last_batch=None, running_batch=running)
        else:
            scheduler.checkpoint_disagg_prefill(req)
            scheduler.send_kv_chunk(req, last_chunk=True)
        steps.append(
            SimpleNamespace(
                freed=set(allocator.freed),
                freed_in_flight=allocator.freed & in_flight,
                remapped=not torch.equal(row[:sent_before], row_before),
                protected=req.kv.cache_protected_len,
                sent=req.start_send_idx,
                swa_evicted=req.kv.get_evicted_seqlen("swa"),
            )
        )

    sender.in_flight.clear()
    release_kv_cache(req, cache, checkpoint=False)
    return steps


class TestPrefillSendProtection(CustomTestCase):
    def test_checkpoint_dedup_preserves_in_flight_pages(self):
        for backend in ("python", "rust"):
            with (
                self.subTest(backend=backend),
                get_context().override_server_args(page_size=1),
            ):
                first, second = _drive(backend, (20, 28), sibling=True)
                self.assertFalse(
                    second.freed_in_flight, "checkpoint freed pages still being sent"
                )
                self.assertFalse(second.remapped)
                self.assertEqual((first.protected, first.sent), (12, 12))
                self.assertEqual(len(second.freed), 8)
                self.assertEqual((second.protected, second.sent), (28, 28))

    def test_long_branch_keeps_checkpointing_and_releasing_swa(self):
        ends = tuple(range(20, 125, 8))
        for backend in ("python", "rust"):
            with (
                self.subTest(backend=backend),
                get_context().override_server_args(page_size=1),
                envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True),
            ):
                steps = _drive(backend, ends, sibling=False, swa_size=40)
                self.assertEqual([s.sent for s in steps], [12, *ends[1:]])
                self.assertEqual([s.protected for s in steps], [12, *ends[1:]])
                self.assertTrue(
                    all(not s.freed_in_flight and not s.remapped for s in steps)
                )
                evicted = [s.swa_evicted for s in steps]
                self.assertEqual(evicted, sorted(evicted))
                self.assertGreater(evicted[-1], evicted[0])

    def test_send_boundaries(self):
        # end, final, sharing, skip_insert, protected, start, staging, expected
        cases = [
            (None, False, True, False, 11, 0, False, 8),
            (20, False, True, False, 11, 0, False, 8),
            (8, False, True, False, 20, 0, False, 8),
            (None, False, True, False, 24, 0, False, 24),
            (None, False, True, False, 0, 0, False, 0),
            (None, False, True, False, 4, 12, False, 12),
            (None, False, False, False, 0, 0, False, 24),
            (None, False, True, True, 0, 0, False, 24),
            (31, True, True, False, 0, 0, False, 31),
            (None, False, True, False, 19, 4, True, 12),
            (None, False, True, False, 7, 4, True, 4),
            (31, True, True, False, 0, 4, True, 31),
        ]
        for end, final, sharing, skip, protected, start, staging, expected in cases:
            with (
                self.subTest(
                    case=(end, final, sharing, skip, protected, start, staging)
                ),
                get_context().override_server_args(page_size=4, chunked_prefill_size=8),
            ):
                req = Req("boundary", None, array("q", range(31)), SamplingParams())
                req.kv.req_pool_idx = 0
                req.kv.protect_prefix(protected)
                req.skip_radix_cache_insert = skip
                req.extend_end = 24
                req.start_send_idx = req.disagg_decode_prefix_len = start
                req.disagg_kv_sender = sender = _Sender()
                scheduler = _scheduler(
                    SimpleNamespace(supports_prefix_sharing=lambda: sharing),
                    SimpleNamespace(
                        page_size=4,
                        translate_kv_indices_for_transfer=lambda x: x,
                        # A pool whose rows never move records no transfer.
                        storage=SimpleNamespace(
                            record_network_transfer=lambda *args, **kwargs: None
                        ),
                    ),
                    SimpleNamespace(req_to_token=torch.arange(32).reshape(1, -1)),
                )
                scheduler.enable_staging = staging
                scheduler.send_kv_chunk(req, last_chunk=final, end_idx=end)
                self.assertEqual(req.start_send_idx, expected)
                pages = [p for send in sender.sends for p in send]
                self.assertEqual(pages, list(range(start // 4, (expected + 3) // 4)))


if __name__ == "__main__":
    unittest.main()
