"""PD admission and ownership while UnifiedRadixCache restores a host prefix."""

import tempfile
import unittest
from array import array
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import poll_and_all_reduce
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

# The TP case spawns two gloo workers that each re-import torch and sglang.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done


class _CPUHostPool(TokenToKVPoolAllocator):
    """Host slots with the anchor-entry view the tree's backup path reads: no
    shared allocation domain, so the tree evicts host space itself."""

    anchor_entry = SimpleNamespace(
        host_pool=SimpleNamespace(shared_allocation_domain=None), host_evict_fn=None
    )


class _CPUController:
    """Replace DMA submission/completion; tree allocation and locks stay real."""

    def __init__(self, allocator):
        self.allocator = allocator
        self.write_policy = "write_through"
        self.mem_pool_host = _CPUHostPool(
            size=1024, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        self.layer_done_counter = SimpleNamespace(
            producer_index=-1,
            num_counters=3,
            events=[SimpleNamespace(finish_event=_Event()) for _ in range(3)],
        )
        self.load_queue = []
        self.ack_load_queue = []
        self.ack_write_queue = []

    def write(self, device_indices, node_id, extra_pools, flush=True):
        self.ack_write_queue.append(HiCacheAck(_Event(), _Event(), [node_id]))
        return self.mem_pool_host.alloc(len(device_indices))

    def load(self, host_indices, node_id, extra_pools):
        indices = self.allocator.alloc(len(host_indices))
        if indices is not None:
            self.load_queue.append(node_id)
        return indices

    def start_loading(self):
        if not self.load_queue:
            return -1
        counter = self.layer_done_counter
        counter.producer_index = (counter.producer_index + 1) % counter.num_counters
        event = counter.events[counter.producer_index].finish_event
        assert event.done
        event.done = False
        self.ack_load_queue.append(HiCacheAck(_Event(), event, self.load_queue[:]))
        self.load_queue.clear()
        return counter.producer_index


def _match_state(req):
    return (
        req.prefix_len,
        req.kv.cache_protected_len,
        req.host_hit_length,
        req.last_node,
    )


class _Queue(DecodeHiCachePreallocMixin, DecodeHiCacheTransferMixin):
    def __init__(self, cache):
        self.tree_cache = cache
        self.scheduler = SimpleNamespace(enable_decode_hicache=True)


class TestDecodeHiCacheL2(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.allocator = TokenToKVPoolAllocator(
            size=1024, dtype=torch.float32, device="cpu", kvcache=None, need_sort=False
        )
        self.pool = ReqToTokenPool(
            size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                # _backup builds actions by hand, which the Rust core does not expose.
                tree_core_backend="python",
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=128,
                tree_slots=tree_slots_of((ComponentType.FULL,)),
            )
        )
        self.controller = _CPUController(self.allocator)
        self.cache.cache_controller = self.controller
        self.cache.components[
            ComponentType.FULL
        ]._full_kv_pool_host = self.controller.mem_pool_host
        self.cache.tree_core.set_hicache_enabled()
        self.cache.load_back_threshold = 10
        self.queue = _Queue(self.cache)
        storage_probe = patch.object(
            self.cache,
            "query_storage_hit_length",
            side_effect=AssertionError("L2-only cache queried L3"),
        )
        storage_probe.start()
        self.addCleanup(storage_probe.stop)
        self.tokens = array("q", range(384))

    def _insert(self, length):
        key = RadixKey(self.tokens[:length])
        slots = self.allocator.alloc(length)
        result = self.cache.insert(InsertParams(key=key, value=slots))
        self.allocator.free(slots[: result.prefix_len])
        return self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node

    def _backup(self, node):
        action = self.cache.tree_core.build_backup_kv_action(node)
        self.cache._execute_and_commit_kv_backup(action)
        self.cache.writing_check()

    def _request(self, index, length=384):
        req = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=self.tokens[:length],
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        req.kv.req_pool_idx = index
        result = match_kv_cache(req, self.cache, req.origin_input_ids)
        req.lock = self.cache.lock(result.last_device_node)
        pm = self.queue._build_decode_prefix_match(req, result)
        self.pool.write((index, slice(0, pm.l1_prefix_len)), pm.prefix_indices)
        return DecodeRequest(
            req=req,
            kv_receiver=SimpleNamespace(poll=lambda: KVPoll.Success),
            prefix_match=pm,
        )

    def _host_prefix(self):
        l1 = self._insert(128)
        holder = self.cache.lock(l1)
        l2 = self._insert(256)
        self._backup(l2)
        self.cache.evict(EvictParams(num_tokens=128))
        self.addCleanup(self.cache.unlock, holder)
        return l1

    def _complete(self):
        for ack in self.controller.ack_load_queue:
            ack.finish_event.done = True
        self.cache.loading_check()

    def test_l2_miss_and_host_hit_do_not_query_storage(self):
        dr = self._request(0)
        self.assertEqual(dr.prefix_match.decode_prefix_len, 0)
        self.cache.unlock(dr.req.lock)
        self._host_prefix()
        dr = self._request(1)
        self.assertEqual(dr.prefix_match.l1_prefix_len, 128)
        self.assertEqual(dr.prefix_match.l2_host_hit_length, 128)
        self.assertEqual(dr.prefix_match.l3_storage_hit_length, 0)
        self.cache.unlock(dr.req.lock)

    def test_concurrent_shared_prefix_and_l1_follower_wait_for_dma(self):
        self._host_prefix()
        requests = [self._request(0), self._request(1)]
        # The rematch must not overwrite the admission-time match on the request.
        admitted = [_match_state(dr.req) for dr in requests]
        self.queue._process_hicache_local_restores(requests)
        self.assertEqual(len(self.controller.ack_load_queue), 1)
        follower = self._request(2)
        self.assertFalse(follower.prefix_match.needs_local_restore)
        requests.append(follower)
        admitted.append(_match_state(follower.req))
        self.queue._process_hicache_local_restores(requests)
        for dr in requests:
            self.assertEqual(
                HiCacheRestoreGatedKVReceiver(dr).poll(), KVPoll.Transferring
            )
            self.assertEqual(dr.req.last_node, dr.prefix_match.last_device_node)
        self._complete()
        self.queue._process_hicache_local_restores(requests)
        self.assertEqual([_match_state(dr.req) for dr in requests], admitted)
        for dr in requests:
            self.assertEqual(HiCacheRestoreGatedKVReceiver(dr).poll(), KVPoll.Success)
            self.queue._commit_hicache_local_restore_to_req(dr)
            self.assertEqual(dr.req.prefix_len, 256)
            torch.testing.assert_close(
                self.pool.req_to_token[dr.req.kv.req_pool_idx, :256].long(),
                self.cache.prefix_device_indices(dr.req).long(),
            )
            self.cache.unlock(dr.req.lock)
        self.cache.sanity_check()

    def test_abort_and_failed_restore_release_only_their_own_locks(self):
        l1 = self._host_prefix()
        loader, dr = self._request(0), self._request(1)
        self.queue._process_hicache_local_restores([loader, dr])
        for pending in (dr, loader):
            self.queue._clean_hicache_prefetch_resources(pending)
            self.cache.unlock(pending.req.lock)
        # Aborted mid-DMA: the in-flight load still pins the restored node.
        self.assertEqual(
            self.cache.evict(EvictParams(num_tokens=1024)).num_tokens_evicted, 0
        )
        self._complete()
        self.assertEqual(
            self.cache.evict(EvictParams(num_tokens=1024)).num_tokens_evicted, 128
        )

        failed = self._request(2)
        held = self.allocator.alloc(self.allocator.available_size())
        with self.assertLogs(
            "sglang.srt.disaggregation.decode_hicache_mixin", level="WARNING"
        ):
            self.queue._process_hicache_local_restores([failed])
        self.assertEqual(failed.hicache_restore_status, HiCacheRestoreResult.FAILED)
        self.assertEqual(failed.req.last_node, l1)
        self.queue._clean_hicache_prefetch_resources(failed)
        self.cache.unlock(failed.req.lock)
        self.allocator.free(held)
        # The holder's lock on the L1 prefix survives the failed request's cleanup.
        self.assertEqual(
            self.cache.evict(EvictParams(num_tokens=1024)).num_tokens_evicted, 0
        )
        self.cache.sanity_check()

    def test_rematch_does_not_extend_into_the_pd_transfer_suffix(self):
        self._host_prefix()
        dr = self._request(0)
        # Another completed request inserts more of the same prompt.
        self._insert(384)
        self.queue._process_hicache_local_restores([dr])
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(len(dr.hicache_restored_kv_indices), 128)
        self.pool.req_to_token[0, 256:384] = 999
        self.queue._commit_hicache_local_restore_to_req(dr)
        self.assertTrue(torch.all(self.pool.req_to_token[0, 256:384] == 999))
        # An unlimited rematch would lock the 384-token node instead.
        self.assertEqual(len(self.cache.path_device_indices(dr.req.last_node)), 256)
        self.cache.unlock(dr.req.lock)
        self.cache.sanity_check()

    def test_completed_event_does_not_free_slot_before_ack_drain(self):
        self.controller.layer_done_counter.num_counters = 1
        self._host_prefix()
        loader = self._request(0)
        self.queue._process_hicache_local_restores([loader])
        follower = self._request(1)
        self.controller.ack_load_queue[0].finish_event.done = True
        self.assertFalse(self.cache.has_free_load_back_slot())
        self.queue._process_hicache_local_restores([loader, follower])
        self.assertEqual(loader.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(follower.hicache_load_consumer_index, -1)
        self.assertEqual(follower.hicache_restore_status, HiCacheRestoreResult.PENDING)
        self.cache.loading_check()
        self.assertTrue(self.cache.has_free_load_back_slot())
        self.queue._process_hicache_local_restores([loader, follower])
        self.assertEqual(follower.hicache_load_consumer_index, 0)
        self.assertEqual(follower.hicache_restore_status, HiCacheRestoreResult.READY)
        for dr in (loader, follower):
            self.queue._commit_hicache_local_restore_to_req(dr)
            self.cache.unlock(dr.req.lock)
        self.cache.sanity_check()

    def test_tp_restore_readiness_is_synchronized_by_transfer_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_tp_restore_worker, args=(directory + "/store",), nprocs=2)


def _tp_restore_worker(rank, store):
    dist.init_process_group(
        "gloo",
        init_method="file://" + store,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    case = TestDecodeHiCacheL2()
    try:
        case.setUp()
        case.cache.tp_group = dist.group.WORLD
        case.cache.tp_world_size = 2
        case._host_prefix()
        loader = case._request(0)
        requests = [loader]
        case.queue._process_hicache_local_restores(requests)
        case.controller.ack_load_queue[0].finish_event.done = rank == 0
        # Followers arrive while one rank has already completed its DMA.
        for index in (1, 2):
            requests.append(case._request(index))
            case.queue._process_hicache_local_restores(requests)
            case.assertEqual(len(case.controller.ack_load_queue), 1)
            case.assertEqual(
                loader.hicache_restore_status,
                (
                    HiCacheRestoreResult.READY
                    if rank == 0
                    else HiCacheRestoreResult.PENDING
                ),
            )
            polls = poll_and_all_reduce(
                [HiCacheRestoreGatedKVReceiver(dr) for dr in requests], dist.group.WORLD
            )
            case.assertTrue(all(p == KVPoll.Transferring for p in polls))
        case._complete()
        case.queue._process_hicache_local_restores(requests)
        polls = poll_and_all_reduce(
            [HiCacheRestoreGatedKVReceiver(dr) for dr in requests], dist.group.WORLD
        )
        case.assertTrue(all(p == KVPoll.Success for p in polls))
        for dr in requests:
            case.queue._commit_hicache_local_restore_to_req(dr)
            case.cache.unlock(dr.req.lock)
        case.cache.sanity_check()
    finally:
        case.doCleanups()
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
