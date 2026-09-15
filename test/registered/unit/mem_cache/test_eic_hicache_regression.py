import os
import sys
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.eic_cache_controller import (
    EICCacheController,
    EICCacheOperation,
    get_content_hash,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.eic_chunk_cache import EICChunkCache
from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache
from sglang.srt.mem_cache.eic_pp_reconcile import EICPPReconciler
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class FakeTreeNode:
    # Minimal radix node for the finalize last_node walk: len(key), value (None ==
    # evicted), parent. Depth bookkeeping uses the sum of key lengths on the path.
    def __init__(self, key_len: int, resident: bool, parent):
        self.key = list(range(key_len))
        self.value = [0] if resident else None
        self.parent = parent


class FakeHostPool:
    def alloc(self, size: int):
        return torch.arange(size, dtype=torch.int32)


class FakeDeviceAllocator:
    def alloc(self, size: int):
        return torch.arange(100, 100 + size, dtype=torch.int32)


class TestEICHiCacheRegression(unittest.TestCase):
    def test_match_from_remote_skips_zero_committed_tokens(self):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = {}
        cache.tp_size = 1
        cache.pp_size = 1
        cache.pp_group = None
        cache.cache_controller = mock.Mock()
        cache._match_for_remote_fetch = mock.Mock(
            side_effect=AssertionError("empty remote key should be skipped")
        )

        req = mock.Mock(
            rid="health-rid",
            origin_input_ids=[0],
            output_ids=[],
            extra_key=None,
        )

        cache.match_from_remote([req])

        cache._match_for_remote_fetch.assert_not_called()
        cache.cache_controller.batch_find_longest_prefix_in_eic.assert_not_called()
        self.assertEqual(cache.match_req_set, {})

    def test_match_from_remote_gates_and_indexes_by_queue(self):
        # match_from_remote first MIN-reduces the queue length (num_ready gate),
        # then reduces a vector indexed by queue position -- both PP-invariant
        # lengths. Here we capture the CP/TP reduce (PP uses p2p, tested
        # separately); the shapes are what keep the reduces in lockstep.
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = {}
        cache.tp_size = 2
        cache.tp_group = object()
        cache.pp_size = 1
        cache.pp_group = None
        cache.page_size = 1
        cache.eic_check_max_num = 0
        cache.root_node = object()
        cache._insert_remote_node = mock.Mock()
        node = SimpleNamespace(id=1, content_hash=None)

        # Fetch only the middle req; the others' prefix already covers the key.
        def match(root, key):
            match.i += 1
            return (0, node) if match.i == 2 else (len(key), node)

        match.i = 0
        cache._match_for_remote_fetch = match
        cache.cache_controller = mock.Mock()
        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [8]
        reqs = [
            mock.Mock(
                rid=r, origin_input_ids=list(range(8)), output_ids=[9], extra_key=None
            )
            for r in ("a", "b", "c")
        ]

        captured = []

        def fake_all_reduce(t, op=None, group=None):
            if group is cache.tp_group:
                captured.append(t.tolist())

        with mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
            return_value=False,
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=fake_all_reduce
        ):
            EICPagedHiRadixCache.match_from_remote(cache, reqs)

        # First the num_ready gate (scalar == queue length), then the per-slot
        # vector (length == queue, only the fetched middle slot carries the hit).
        self.assertEqual(captured, [3, [0, 8, 0]])

    def test_eic_controller_write_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_host = FakeHostPool()
        controller.write_queue = Queue()

        device_indices = torch.tensor([7, 8, 9], dtype=torch.int32)
        host_indices = controller.write(device_indices, priority=-2, node_id=11)

        self.assertEqual(host_indices.tolist(), [0, 1, 2])
        operation = controller.write_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertEqual(operation.host_indices.tolist(), [0, 1, 2])
        self.assertIs(operation.device_indices, device_indices)
        self.assertEqual(operation.node_id, 11)
        self.assertEqual(operation.node_ids, [11])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -2)

    def test_eic_controller_load_uses_queue_api(self):
        controller = object.__new__(EICCacheController)
        controller.mem_pool_device_allocator = FakeDeviceAllocator()
        controller.load_queue = Queue()

        host_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        with mock.patch(
            "sglang.srt.managers.eic_cache_controller.torch.cuda.current_stream"
        ) as current_stream:
            stream = mock.Mock()
            current_stream.return_value = stream
            device_indices = controller.load(host_indices, priority=-3, node_id=12)

        stream.synchronize.assert_called_once()
        self.assertEqual(device_indices.tolist(), [100, 101, 102])
        operation = controller.load_queue.get_nowait()
        self.assertIsInstance(operation, EICCacheOperation)
        self.assertIs(operation.host_indices, host_indices)
        self.assertEqual(operation.device_indices.tolist(), [100, 101, 102])
        self.assertEqual(operation.node_id, 12)
        self.assertEqual(operation.node_ids, [12])
        self.assertIsNone(operation.content_hash)
        self.assertEqual(operation.priority, -3)

    def test_unified_tree_node_exposes_storage_hash_helpers(self):
        root = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        child.parent = root
        root.hash_value = ["root-hash"]
        child.hash_value = ["child-hash-0", "child-hash-1"]

        self.assertEqual(root.get_last_hash_value(), "root-hash")
        self.assertEqual(child.get_last_hash_value(), "child-hash-1")
        self.assertEqual(child.get_prefix_hash_values(child.parent), ["root-hash"])
        self.assertFalse(
            hasattr(root, "hicache_storage_pass_prefix_keys"),
            "storage pass-prefix config belongs to the cache, not tree nodes",
        )

    def test_batch_exists_impl_failed_batch_keeps_cardinality(self):
        # A failed top-level mexist must yield exactly one False per input key.
        # The EIC outcome object may still carry per-object status codes; if the
        # loop below falls through to them we would return 2*N results, which
        # later overflows component_keys indexing in _batch_io_v2. See the
        # `continue` guard in EICStorage._batch_exists_impl.
        fake_eic = SimpleNamespace(
            StatusCode=SimpleNamespace(SUCCESS=0, FAILED=1),
            StringVector=list,
            ExistOption=SimpleNamespace,
        )
        with mock.patch.dict(sys.modules, {"eic": fake_eic}):
            from sglang.srt.mem_cache.storage.eic.eic_storage import EICStorage

            storage = object.__new__(EICStorage)
            storage.eic_namespace = "poc"
            storage._get_eic_key = lambda keys: list(keys)

            outcome = SimpleNamespace(status_codes=[fake_eic.StatusCode.SUCCESS] * 3)
            storage.connection = mock.Mock()
            storage.connection.mexist.return_value = (
                fake_eic.StatusCode.FAILED,
                outcome,
            )

            keys = [f"k{i}" for i in range(3)]
            result = storage._batch_exists_impl(keys)

        self.assertEqual(result, [False, False, False])
        self.assertEqual(len(result), len(keys))

    def test_eic_chunk_cache_passes_tp_group_to_controller(self):
        cache = object.__new__(EICChunkCache)
        cache.page_size = 16
        cache.load_cache_event = object()
        cache.token_to_kv_pool_host = object()
        params = SimpleNamespace(
            token_to_kv_pool_allocator=object(), tp_cache_group=object()
        )
        server_args = SimpleNamespace()

        with mock.patch(
            "sglang.srt.mem_cache.eic_chunk_cache.EICCacheController"
        ) as controller_cls:
            EICChunkCache._init_cache_controller(cache, params, server_args)

        controller_cls.assert_called_once_with(
            params.token_to_kv_pool_allocator,
            cache.token_to_kv_pool_host,
            cache.page_size,
            tp_group=params.tp_cache_group,
            load_cache_event=cache.load_cache_event,
            write_policy="write_through",
            server_args=server_args,
        )

    def test_free_swa_skips_reserved_page_and_dedups_aliases(self):
        # free_swa must drop reserved-page sentinels + dedup aliases under EIC.
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

        alloc = object.__new__(SWATokenToKVPoolAllocator)
        alloc.page_size = 256
        alloc.swa_attn_allocator = mock.Mock()
        alloc.full_to_swa_index_mapping = torch.tensor(
            [6, 8, 10, 21, 512, 512, 512, 768], dtype=torch.int64
        )
        alloc._expand_to_full_pages = lambda idx: idx
        alloc.dedup_aliased_swa = True

        SWATokenToKVPoolAllocator.free_swa(alloc, torch.arange(8))

        freed = alloc.swa_attn_allocator.free.call_args[0][0]
        self.assertEqual(sorted(freed.tolist()), [512, 768])

    def test_pp_bcast_from_first_is_nonblocking_isend(self):
        # PP consensus must be a non-blocking isend down the pipeline: the stages
        # run out of phase, so a collective or a blocking send/recv (which makes
        # one stage wait for another) deadlocks. First stage isends its value
        # downstream (queued in work_list) and keeps it; a later stage overwrites
        # with the upstream (PP0's) value via recv. No all_reduce anywhere.
        from sglang.srt.mem_cache.eic_hiradix_cache import EICPagedHiRadixCache

        # first stage (rank 0 of 2): isend downstream, never recv
        c0 = object.__new__(EICPagedHiRadixCache)
        c0.pp_size, c0.pp_rank, c0.pp_group, c0.work_list = 2, 0, object(), []
        sent = []

        def fake_isend(t, group_dst=None, group=None, tag=None):
            sent.append((group_dst, t.item()))
            return "work"

        with mock.patch.object(
            torch.distributed, "isend", side_effect=fake_isend
        ), mock.patch.object(
            torch.distributed, "recv", side_effect=AssertionError("source never recvs")
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("no collective")
        ):
            t = torch.tensor(5, dtype=torch.int64)
            EICPagedHiRadixCache._pp_bcast_from_first(c0, t)
        self.assertEqual(sent, [(1, 5)])
        self.assertEqual(c0.work_list, ["work"])  # drained next round
        self.assertEqual(t.item(), 5)  # authoritative source keeps its value

        # last stage (rank 1 of 2): recv overwrites with PP0's value, no isend
        c1 = object.__new__(EICPagedHiRadixCache)
        c1.pp_size, c1.pp_rank, c1.pp_group, c1.work_list = 2, 1, object(), []

        def fake_recv(t, group_src=None, group=None, tag=None):
            t.fill_(3)  # PP0's authoritative value

        with mock.patch.object(
            torch.distributed, "recv", side_effect=fake_recv
        ), mock.patch.object(
            torch.distributed,
            "isend",
            side_effect=AssertionError("last rank sends nothing"),
        ), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("no collective")
        ):
            t = torch.tensor(8, dtype=torch.int64)
            EICPagedHiRadixCache._pp_bcast_from_first(c1, t)
        self.assertEqual(t.item(), 3)

    def test_drain_acks_single_stage_resolves_locally(self):
        # With one PP stage the local ack IS the verdict: d + complete.
        cache = object.__new__(EICPagedHiRadixCache)
        cache.pp_size, cache.pp_group, cache.tp_size = 1, None, 1
        cache._admit_verdict = {}
        cache._loadback_rid = {55: "r0"}
        cache.ongoing_load_admit = {
            "r0": {
                "h": 10,
                "d": 4,
                "load_node": SimpleNamespace(id=55),
                "complete": None,
            }
        }
        cache.ongoing_load_back = {}
        q = Queue()
        q.put((55, 8))
        cache.cache_controller = SimpleNamespace(ack_load_queue=q)
        EICPagedHiRadixCache._drain_local_acks(cache)
        self.assertEqual(cache._admit_verdict, {10: 12})
        self.assertEqual(cache.ongoing_load_admit["r0"]["complete"], 8)

    def test_finalize_cross_pp_clamp_keeps_spanning_resident_last_node(self):
        # Device-only warm req, cross-PP MIN 10 lands MID-node (below this stage's
        # device match 12). last_node must stay the resident node SPANNING depth 10
        # (bottom 12 >= 10) so inc_lock_ref protects every slot the req reads.
        cache = object.__new__(EICPagedHiRadixCache)
        root = FakeTreeNode(0, True, None)
        cache.root_node = root
        cache.dec_lock_ref = lambda node: None
        cache._h_rid = {7: "r0"}
        node_a = FakeTreeNode(8, True, root)  # covers (0, 8]
        node_b = FakeTreeNode(4, True, node_a)  # covers (8, 12], spans clamp 10
        req = mock.Mock()
        req.rid = "r0"
        req.prefix_indices = torch.arange(12, dtype=torch.int64)
        req.last_node = node_b
        cache.ongoing_load_admit = {
            "r0": {
                "h": 7,
                "d": 12,
                "alloc": 0,
                "load_node": None,
                "new_indices": None,
                "deferred_lock": node_b,
                "complete": None,
            }
        }
        cache._admit_verdict = {7: 10}
        ok = EICPagedHiRadixCache._finalize_load_admit(cache, req)
        self.assertTrue(ok)
        self.assertEqual(len(req.prefix_indices), 10)
        self.assertIs(req.last_node, node_b)  # spanning resident node kept
        req.set_extend_range.assert_not_called()
        self.assertEqual(req.storage_hit_length, 0)
        self.assertEqual(req.host_hit_length, 0)
        self.assertEqual(cache.ongoing_load_admit, {})
        self.assertEqual(cache._admit_verdict, {})
        self.assertEqual(cache._h_rid, {})

    def test_finalize_partial_load_lands_on_resident_not_evicted_tail(self):
        # Partial load-back (loaded 10 < span alloc 16): the failed tail node is
        # evicted. last_node must land on the deepest RESIDENT node at depth 14
        # (device 4 + loaded 10), NOT the evicted tail the walk skips over; the
        # whole [14, 20) tail is freed at the verdict boundary.
        cache = object.__new__(EICPagedHiRadixCache)
        root = FakeTreeNode(0, True, None)
        cache.root_node = root
        cache.dec_lock_ref = lambda node: None
        cache._h_rid = {9: "r1"}
        node_a = FakeTreeNode(4, True, root)  # device match, (0, 4]
        node_c = FakeTreeNode(10, True, node_a)  # loaded host, (4, 14]
        node_d = FakeTreeNode(6, False, node_c)  # failed tail, evicted, (14, 20]
        node_d.id = 55
        req = mock.Mock()
        req.rid = "r1"
        req.prefix_indices = torch.arange(4, dtype=torch.int64)
        req.last_node = node_d
        cache.ongoing_load_admit = {
            "r1": {
                "h": 9,
                "d": 4,
                "alloc": 16,
                "load_node": node_d,
                "new_indices": torch.arange(100, 116, dtype=torch.int64),
                "deferred_lock": node_a,
                "complete": 10,
            }
        }
        cache._admit_verdict = {9: 14}  # FINAL == this stage's loaded length
        freed = []
        cache._free_failed_loadback = lambda nid, c: freed.append((nid, c))
        ok = EICPagedHiRadixCache._finalize_load_admit(cache, req)
        self.assertTrue(ok)
        self.assertEqual(freed, [(55, 10)])  # free [14, 20) at the verdict boundary
        self.assertEqual(len(req.prefix_indices), 14)  # 4 device + 10 loaded
        self.assertEqual(req.prefix_indices[4:].tolist(), list(range(100, 110)))
        self.assertIs(req.last_node, node_c)  # resident node at depth 14, not node_d
        req.set_extend_range.assert_not_called()
        self.assertEqual(req.storage_hit_length, 10)
        self.assertEqual(req.host_hit_length, 10)

    def _make_pool_cache(self, total, page):
        free_ids = set(range(1, total + 1))

        def alloc(n):
            ids = sorted(free_ids)[:n]
            free_ids.difference_update(ids)
            return torch.tensor(ids, dtype=torch.int64)

        def free(t):
            ids = set(t.tolist())
            self.assertFalse(ids & free_ids, "double free")
            free_ids.update(ids)

        def evict_device(dev, host):
            free(dev)
            return len(dev)

        c = object.__new__(EICPagedHiRadixCache)
        c.disable, c.is_eagle, c.page_size, c.device = False, False, page, "cpu"
        c.sliding_window_size, c.pp_size = None, 1
        c.evictable_size_ = c.protected_size_ = 0
        c.evictable_leaves = set()
        c.write_through_threshold = 10**9
        c.load_back_threshold, c.load_back_check = 0, False
        c.ongoing_load_back = {}
        c.calculate_hash_fn = get_content_hash
        c.cache_controller = SimpleNamespace(
            write_policy="write_through",
            mem_pool_device_allocator=SimpleNamespace(free=free),
            evict_device=evict_device,
            load_page=lambda host_indices, node_id, content_hash: alloc(
                len(host_indices)
            ),
        )
        c.token_to_kv_pool_allocator = SimpleNamespace(free=free)
        root = TreeNode()
        root.key, root.value, root.lock_ref = RadixKey([], None), [], 1
        c.root_node = root
        return c, alloc, free, free_ids

    def test_failed_load_under_same_prefix_req_keeps_pool_invariant(self):
        # Chat cc32 crash: req A's load of a shared-prefix tail T failed while req
        # B, admitted mid-load, had matched T's in-flight slots as a GPU hit. The
        # failed-tail free returned them to the allocator and B's insert re-linked
        # them into T: available + evictable exceeded total by exactly T.
        total = 64
        c, alloc, free, free_ids = self._make_pool_cache(total, page=4)
        key = RadixKey(list(range(32)), None)
        c.insert(InsertParams(key=key, value=alloc(32)))
        tail = c.root_node.children[key.child_key(4)]
        c._split_node(tail.key, tail, 16)
        tail.host_value = torch.arange(16)
        c._evict_backuped(tail)  # tail [16, 32) lives only in EIC

        a = c.match_prefix(MatchPrefixParams(key=key))
        c.load_back(a.best_match_node)  # A kicks the load; DMA in flight
        b = c.match_prefix(MatchPrefixParams(key=key))
        self.assertTrue(c.prefix_loading(b.last_device_node))  # B defers
        c._free_failed_loadback(tail.id, 0)  # the whole tail's mget failed

        b = c.match_prefix(MatchPrefixParams(key=key))  # B retries after settle
        self.assertFalse(c.prefix_loading(b.last_device_node))
        b_kv = torch.cat([b.device_indices, alloc(32 - len(b.device_indices))])
        c.inc_lock_ref(b.last_device_node)
        prefix_len = c.insert(InsertParams(key=key, value=b_kv)).prefix_len
        free(b_kv[len(b.device_indices) : prefix_len])
        c.dec_lock_ref(b.last_device_node)

        cached = set()
        stack = list(c.root_node.children.values())
        while stack:
            n = stack.pop()
            if n.value is not None:
                cached.update(n.value.tolist())
            stack.extend(n.children.values())
        self.assertFalse(cached & free_ids)
        self.assertEqual(len(free_ids) + c.evictable_size_ + c.protected_size_, total)

    def test_chunk_insert_through_inflight_load_keeps_pool_invariant(self):
        # A running chunked req recomputing the same prefix reaches the in-flight
        # tail T at its next chunk boundary. Inserting then would free its own KV
        # and adopt T's still-landing slots; after T's load fails and the req
        # finishes, those slots would be both free and cached.
        total = 64
        c, alloc, free, free_ids = self._make_pool_cache(total, page=4)
        c.cache_controller.write_page = lambda **kw: None
        key = RadixKey(list(range(32)), None)
        c.insert(InsertParams(key=key, value=alloc(32)))
        head = c.root_node.children[key.child_key(4)]
        tail = head
        head = c._split_node(tail.key, tail, 16)
        tail.host_value = torch.arange(16)
        c._evict_backuped(tail)

        # The chunked req holds the shared head and its own recomputed tail.
        own = torch.cat([head.value, alloc(16)])
        r2t = own.view(1, -1).clone()
        c.req_to_token_pool = SimpleNamespace(
            req_to_token=r2t, write=lambda idx, v: r2t.__setitem__(idx, v)
        )
        req = SimpleNamespace(
            fill_ids=list(range(32)),
            extra_key=None,
            req_pool_idx=0,
            cache_protected_len=16,
            last_node=head,
            prefix_indices=own[:16],
        )
        c.inc_lock_ref(head)

        c.load_back(c.match_prefix(MatchPrefixParams(key=key)).best_match_node)
        c.cache_unfinished_req(req, chunked=True)  # chunk boundary mid-load
        c._free_failed_loadback(tail.id, 0)

        kv = c.req_to_token_pool.req_to_token[0, :32].to(torch.int64)
        prefix_len = c.insert(InsertParams(key=key, value=kv)).prefix_len
        free(kv[req.cache_protected_len : prefix_len])
        c.dec_lock_ref(req.last_node)

        cached = set()
        stack = list(c.root_node.children.values())
        while stack:
            n = stack.pop()
            if n.value is not None:
                cached.update(n.value.tolist())
            stack.extend(n.children.values())
        self.assertFalse(cached & free_ids)
        self.assertEqual(len(free_ids) + c.evictable_size_ + c.protected_size_, total)

    def test_partial_mget_refetches_only_failed_keys(self):
        # A partial mget (per-key RPC timeouts) used to cut the load at the first
        # failed key, discarding every later page that did arrive. The failed keys
        # are re-got once into their own buffers; the device copy then covers the
        # whole batch, and a key that fails again still cuts the prefix there.
        from sglang.srt.mem_cache import eic_memory_pool as pool_mod

        S = SimpleNamespace(SUCCESS=0, FAILED=1, PARTIAL_FAILED=2)

        class Buffers(list):
            def append(self, ptr, size, registered):
                super().append(ptr)

        fake_eic = SimpleNamespace(
            StatusCode=S,
            StringVector=list,
            IOBuffers=Buffers,
            GetOption=lambda: SimpleNamespace(),
        )
        objs = [torch.zeros(2) for _ in range(4)]
        client = object.__new__(pool_mod.EICKVClient)
        client.eic_namespace = "ns"
        client.allocate_eic_read_buffer = lambda n: (objs, None, list(range(n)), True)
        client.kv_cache_read_mem_pool = SimpleNamespace(free_to_mempool=lambda p: None)
        calls = []

        def mget(keys, option, vals):
            calls.append((list(keys), list(vals)))
            if len(calls) == 1:
                codes = [S.SUCCESS, S.FAILED, S.SUCCESS, S.FAILED]
                return S.PARTIAL_FAILED, vals, SimpleNamespace(status_codes=codes)
            return S.SUCCESS, vals, SimpleNamespace(status_codes=[S.SUCCESS] * 2)

        client.connection = SimpleNamespace(mget=mget)
        copied = []
        with mock.patch.object(pool_mod, "eic", fake_eic):
            _, mask = client.batch_get(
                ["k0", "k1", "k2", "k3"],
                torch.arange(4),
                copy_func=lambda dev, pool, idx: copied.append(list(idx)),
            )
        self.assertEqual(
            calls[1], (["k1", "k3"], [objs[1].data_ptr(), objs[3].data_ptr()])
        )
        self.assertEqual(mask, [True] * 4)
        self.assertEqual(copied, [[0, 1, 2, 3]])

        calls.clear()
        copied.clear()

        def mget_k3_fails_again(keys, option, vals):
            calls.append(list(keys))
            if len(calls) == 1:
                codes = [S.SUCCESS, S.FAILED, S.SUCCESS, S.FAILED]
            else:
                codes = [S.SUCCESS, S.FAILED]
            return S.PARTIAL_FAILED, vals, SimpleNamespace(status_codes=codes)

        client.connection = SimpleNamespace(mget=mget_k3_fails_again)
        with mock.patch.object(pool_mod, "eic", fake_eic):
            _, mask = client.batch_get(
                ["k0", "k1", "k2", "k3"],
                torch.arange(4),
                copy_func=lambda dev, pool, idx: copied.append(list(idx)),
            )
        self.assertEqual(mask, [True, True, True, False])
        self.assertEqual(copied, [[0, 1, 2]])

        calls.clear()

        def mget_mostly_down(keys, option, vals):
            calls.append(list(keys))
            codes = [S.SUCCESS, S.FAILED, S.FAILED, S.FAILED]
            return S.PARTIAL_FAILED, vals, SimpleNamespace(status_codes=codes)

        client.connection = SimpleNamespace(mget=mget_mostly_down)
        with mock.patch.object(pool_mod, "eic", fake_eic):
            _, mask = client.batch_get(["k0", "k1", "k2", "k3"])
        self.assertEqual(len(calls), 1)  # backend down: no retry round
        self.assertEqual(mask, [True, False, False, False])

    def test_match_stops_at_resident_node_under_evicted_gap(self):
        # A failed load-back frees a chain node whose child was inserted while
        # the load was in flight, leaving resident KV under an evicted gap. The
        # match used to append that child's slots after the prefix above the gap,
        # splicing KV from non-adjacent positions into one prefix.
        c, alloc, free, free_ids = self._make_pool_cache(64, page=4)
        key = RadixKey(list(range(24)), None)
        c.insert(InsertParams(key=key, value=alloc(24)))
        low = c.root_node.children[key.child_key(4)]
        mid = c._split_node(low.key, low, 16)
        top = c._split_node(mid.key, mid, 8)
        mid.host_value = torch.arange(8)
        free(mid.value)
        c.evictable_size_ -= len(mid.value)
        mid.value = None  # the gap [8, 16); `low` [16, 24) stays resident

        m = c.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(m.device_indices.tolist(), top.value.tolist())
        self.assertIs(m.last_device_node, top)
        self.assertEqual(m.host_hit_length, 8)

    def test_prefix_loading_covers_split_chain_until_settled(self):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.pp_size = 1
        root = TreeNode()
        cache.root_node = root
        a = TreeNode()  # resident ancestor the load hangs from
        a.parent = root
        t = TreeNode()  # load chain bottom
        t.parent = a
        child = TreeNode()
        child.parent = t
        cache.ongoing_load_back = {t.id: (a, t, 256)}
        self.assertFalse(cache.prefix_loading(a))
        self.assertTrue(cache.prefix_loading(t))
        self.assertTrue(cache.prefix_loading(child))
        upper = TreeNode()  # a match split T: upper half sits between a and t
        upper.parent, t.parent = a, upper
        self.assertTrue(cache.prefix_loading(upper))
        cache.ongoing_load_back.pop(t.id)  # _free_failed_loadback settled it
        self.assertFalse(cache.prefix_loading(child))

    # ---- two-stage lockstep protocol tests --------------------------------

    class FakeStore:
        def __init__(self):
            self.kv = {}

        def set(self, key, value):
            self.kv[key] = value

        def get(self, key):
            return self.kv[key]

        def check(self, keys):
            return all(k in self.kv for k in keys)

        def delete_key(self, key):
            return self.kv.pop(key, None) is not None

    def _make_stage(self, pp_rank, store, dag_box, pp_size=2):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.pp_size, cache.pp_rank = pp_size, pp_rank
        cache.pp_group = object() if pp_size > 1 else None
        cache.tp_size, cache.rank, cache.tp_group = 1, 0, None
        cache.load_back_threshold = 10
        cache.load_back_reserve = 16384
        cache.evictable_size_ = 0
        cache.root_node = FakeTreeNode(0, True, None)
        cache.ongoing_load_admit = {}
        cache.ongoing_load_back = {}
        cache._admit_verdict = {}
        cache._h_rid = {}
        cache._rid_epoch = {}
        cache._loadback_rid = {}
        cache._report_outbox = {}
        cache._span_reports = {}
        cache._load_reports = {}
        cache._await_load = {}
        cache._verdict_outbox = []
        cache._tombstone = {}
        cache._pub_seq = 0
        cache._next_seq = {}
        cache._round = 0
        cache._store_handle = store
        cache.locks = []
        cache.inc_lock_ref = lambda node: cache.locks.append(node)
        cache.dec_lock_ref = lambda node: cache.locks.remove(node)
        cache.freed = []
        cache._free_failed_loadback = lambda nid, c: cache.freed.append((nid, c))
        cache._clip_host_chain = lambda node, excess, quota: node
        if pp_rank == 0:
            # rank0 isends: record the packed verdicts for this round.
            cache._pp_bcast_from_first = lambda buf, tag=None: dag_box.__setitem__(
                "buf", buf.clone()
            )
        else:
            # rank>0 blocking-recv: the k-th recv pairs with the k-th send.
            cache._pp_bcast_from_first = lambda buf, tag=None: buf.copy_(dag_box["buf"])
        q = Queue()
        cache.cache_controller = SimpleNamespace(
            ack_load_queue=q,
            mem_pool_device_allocator=SimpleNamespace(available_size=lambda: 10**9),
        )
        return cache

    def _make_req(self, rid, d, hh, load_node_id):
        node = mock.Mock()
        node.evicted = False
        node.value = [1]
        node.key = list(range(max(hh, 1)))
        node.parent = None
        node.id = load_node_id
        req = mock.Mock()
        req.rid = rid
        req.prefix_indices = torch.arange(d, dtype=torch.int64)
        req.host_hit_length = hh
        req.needs_host_load_back = lambda: hh > 0
        req.best_match_node = node
        req.last_node = node
        return req

    def test_two_stage_load_back_reconciles_span_and_final(self):
        # Host metadata diverges (hh 16 vs 12): SPAN = min(4+16, 4+12) = 16, both
        # stages allocate quota 12 in the SAME round; loads land 12 vs 8: FINAL =
        # min(16, 4+12, 4+8) = 12; both admit 12 in the SAME round and free the
        # [12, 16) tail with identical arguments.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        for s in (s0, s1):
            s.load_back = lambda node, allow_evict=None: torch.arange(
                12, dtype=torch.int64
            )
        r0 = self._make_req("req-a", 4, 16, load_node_id=100)
        r1 = self._make_req("req-a", 4, 12, load_node_id=200)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        # round 1: s1 publishes its span report; s0 sees only its own.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-a"]["load_node"], None)
        # round 2: s0 drains the store, forms SPAN=16, both stages kick alloc 12.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-a"]["alloc"], 12)
        self.assertEqual(s1.ongoing_load_admit["req-a"]["alloc"], 12)
        # acks land: stage0 loads 12/12, stage1 only 8/12.
        s0.cache_controller.ack_load_queue.put((100, 12))
        s1.cache_controller.ack_load_queue.put((200, 8))
        # round 3: acks drain; s1 publishes its LOADED report.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        # round 4: s0 drains it, forms FINAL=12, both apply in the same round.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 12)
        self.assertEqual(len(r1.prefix_indices), 12)
        r0.set_extend_range.assert_not_called()
        r1.set_extend_range.assert_not_called()
        # identical uniform-boundary frees: [FINAL, SPAN) == complete arg 8.
        self.assertEqual(s0.freed, [(100, 8)])
        self.assertEqual(s1.freed, [(200, 8)])
        # no leaked state, no leaked locks.
        for s in (s0, s1):
            self.assertEqual(s.ongoing_load_admit, {})
            self.assertEqual(s._admit_verdict, {})
            self.assertEqual(s._h_rid, {})
            self.assertEqual(s.locks, [])
        self.assertEqual(store.kv, {})  # every store batch consumed and deleted

    def test_two_stage_cold_req_single_round_trip(self):
        # No stage has host data: SPAN decision short-circuits to FINAL=min(d)
        # (one round trip), both admit device-only in the same round.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        r0 = self._make_req("req-b", 8, 0, load_node_id=0)
        r1 = self._make_req("req-b", 8, 0, load_node_id=0)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 8)
        self.assertEqual(len(r1.prefix_indices), 8)
        for s in (s0, s1):
            self.assertEqual(s.locks, [])

    def test_release_tombstones_and_drops_straggler_verdicts(self):
        # Release after the span reports are in flight: the late verdict and any
        # straggler reports must die on arrival (epoch/tombstone), leaving no
        # state and no locks behind.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        r0 = self._make_req("req-c", 4, 16, load_node_id=100)
        r1 = self._make_req("req-c", 4, 16, load_node_id=200)
        EICPagedHiRadixCache.check_load_back_progress(s0, r0)
        EICPagedHiRadixCache.check_load_back_progress(s1, r1)
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)  # s1's span report is now in the store
        EICPagedHiRadixCache.release_load_admit(s0, "req-c")
        EICPagedHiRadixCache.release_load_admit(s1, "req-c")
        h = s0._rid_hash("req-c")
        self.assertIn(h, s0._tombstone)
        # rank0 drains the straggler AFTER the release: tombstone drops it.
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0._span_reports, {})
        self.assertEqual(s0._verdict_outbox, [])
        for s in (s0, s1):
            self.assertEqual(s.ongoing_load_admit, {})
            self.assertEqual(s.locks, [])

    def test_rid_reuse_after_release_still_reconciles(self):
        # A client retry reuses the rid within the tombstone TTL. The tombstone
        # must kill only the RELEASED incarnation's stragglers; the retry's
        # higher-epoch reports must pass or the retry wedges forever.
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        EICPagedHiRadixCache.check_load_back_progress(
            s0, self._make_req("req-d", 8, 0, load_node_id=0)
        )
        EICPagedHiRadixCache.check_load_back_progress(
            s1, self._make_req("req-d", 8, 0, load_node_id=0)
        )
        EICPagedHiRadixCache.loading_check(s0)
        EICPagedHiRadixCache.loading_check(s1)  # epoch-1 span report published
        EICPagedHiRadixCache.release_load_admit(s0, "req-d")
        EICPagedHiRadixCache.release_load_admit(s1, "req-d")
        # retry: same rid re-enters the gate (epoch 2) while tombstoned.
        r0 = self._make_req("req-d", 8, 0, load_node_id=0)
        r1 = self._make_req("req-d", 8, 0, load_node_id=0)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        for _ in range(3):
            EICPagedHiRadixCache.loading_check(s0)
            EICPagedHiRadixCache.loading_check(s1)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s0, r0))
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s1, r1))
        self.assertEqual(len(r0.prefix_indices), 8)

    def test_stale_ack_after_release_is_orphaned_not_attributed(self):
        # Release with the load still in flight, then the same rid re-gates.
        # The OLD incarnation's ack must be orphan-freed (whole span), never
        # attributed to the new incarnation (whose node_id differs).
        store, box = self.FakeStore(), {}
        s0 = self._make_stage(0, store, box)
        s1 = self._make_stage(1, store, box)
        for s in (s0, s1):
            s.load_back = lambda node, allow_evict=None, _s=s: (
                _s.ongoing_load_back.__setitem__(node.id, (node, node, 12)),
                torch.arange(12, dtype=torch.int64),
            )[1]
        r0 = self._make_req("req-e", 4, 16, load_node_id=100)
        r1 = self._make_req("req-e", 4, 16, load_node_id=200)
        EICPagedHiRadixCache.check_load_back_progress(s0, r0)
        EICPagedHiRadixCache.check_load_back_progress(s1, r1)
        for _ in range(2):  # span verdict forms and applies -> loads kicked
            EICPagedHiRadixCache.loading_check(s0)
            EICPagedHiRadixCache.loading_check(s1)
        self.assertEqual(s0.ongoing_load_admit["req-e"]["load_node"].id, 100)
        EICPagedHiRadixCache.release_load_admit(s0, "req-e")
        EICPagedHiRadixCache.release_load_admit(s1, "req-e")
        # same rid re-gates (epoch 2, no load kicked yet: node_id None)
        EICPagedHiRadixCache.check_load_back_progress(
            s0, self._make_req("req-e", 4, 16, load_node_id=101)
        )
        # the old incarnation's ack lands now: must orphan-free the WHOLE span.
        s0.cache_controller.ack_load_queue.put((100, 12))
        EICPagedHiRadixCache.loading_check(s0)
        self.assertEqual(s0.freed, [(100, 0)])
        self.assertIsNone(s0.ongoing_load_admit["req-e"]["complete"])

    def test_pp1_cold_req_admits_with_zero_deferral(self):
        # pp<=1 keeps the old behavior: a cold/device-only req admits on its
        # FIRST gate pass, no verdict round trip.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        req = self._make_req("req-f", 8, 0, load_node_id=0)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(len(req.prefix_indices), 8)
        self.assertEqual(s.locks, [])
        self.assertEqual(s.ongoing_load_admit, {})

    def test_pp1_warm_req_kicks_at_gate_and_admits_on_ack(self):
        # pp<=1 warm path: load kicks at gate entry, req defers, the local ack
        # IS the verdict (d + complete), failed tail freed at that boundary.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        s.load_back = lambda node, allow_evict=None: torch.arange(16, dtype=torch.int64)
        req = self._make_req("req-g", 4, 16, load_node_id=100)
        self.assertFalse(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(s.ongoing_load_admit["req-g"]["load_node"].id, 100)
        s.cache_controller.ack_load_queue.put((100, 10))
        EICPagedHiRadixCache.loading_check(s)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(len(req.prefix_indices), 14)  # 4 device + 10 loaded
        self.assertEqual(s.freed, [(100, 10)])
        self.assertEqual(s.locks, [])

    def test_gate_refuses_load_back_without_pool_headroom(self):
        # The agent-workload OOM: the async gate allocates outside add_one_req's
        # budget, so N queued reqs each pinned a load-back until the pool was full
        # with 0 running reqs and 0 evictable. Below the reserve the gate must
        # admit device-only instead of kicking another load.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        kicked = []
        s.load_back = lambda node, allow_evict=None: kicked.append(
            node
        ) or torch.arange(16, dtype=torch.int64)
        s.cache_controller.mem_pool_device_allocator.available_size = lambda: 6144
        req = self._make_req("req-oom", 4, 16, load_node_id=101)
        self.assertTrue(EICPagedHiRadixCache.check_load_back_progress(s, req))
        self.assertEqual(kicked, [])
        self.assertEqual(len(req.prefix_indices), 4)  # device-only admit
        self.assertEqual(s.ongoing_load_admit, {})

    def test_device_indexed_host_pages_ignore_hicache_ratio(self):
        # EIC shared-page mode is device-indexed, so pages beyond device_pages are
        # unreachable: device_indexed=True must clamp to device_pages+1 regardless of
        # --hicache-ratio, while the non-EIC path keeps honoring the ratio.
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _deepseek_v4_num_host_pages,
        )

        page_size = swa_page_size = 256
        kv = SimpleNamespace(size=339968, swa_size=271872, swa_page_size=swa_page_size)
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(size_full=339968)
        )
        args = SimpleNamespace(hicache_size=0, hicache_ratio=2.0)
        kwargs = dict(
            params=params,
            server_args=args,
            kvcache=kv,
            page_size=page_size,
            swa_page_size=swa_page_size,
        )
        device_full = 339968 // page_size
        device_swa = 271872 // swa_page_size

        self.assertEqual(
            _deepseek_v4_num_host_pages(**kwargs, device_indexed=True),
            (device_full + 1, device_swa + 1),
        )
        # Non-EIC host cache genuinely uses mem_pool_host.alloc(), so ratio must hold.
        self.assertEqual(
            _deepseek_v4_num_host_pages(**kwargs, device_indexed=False),
            (device_full * 2, device_swa * 2),
        )

    def test_eic_calls_the_assembler_with_its_current_signature(self):
        # The port left six kwargs (page_size, tp_group, attn_cp_group,
        # attn_tp_group, pp_rank, pp_size) on the call that ep_main had folded
        # into `params`, so --enable-eic-cache died with TypeError at startup on
        # the first DSV4 launch. Bind the call against the real signature.
        import ast
        import inspect

        from sglang.srt.mem_cache import eic_memory_pool
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        sig = inspect.signature(hybrid_pool_assembler.build_deepseek_v4_hicache_stack)
        calls = [
            n
            for n in ast.walk(ast.parse(inspect.getsource(eic_memory_pool)))
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "build_deepseek_v4_hicache_stack"
        ]
        self.assertTrue(calls)
        for call in calls:
            sig.bind_partial(**{kw.arg: None for kw in call.keywords})

    def test_swa_evict_release_prefix_reaches_free_swa(self):
        # EIC's swa_evict_release_prefix property is only honored if _evict_swa
        # forwards it; without the kwarg it defaults False, the loaded-prefix
        # branch never fires, and out-of-window prefix SWA leaks.
        import ast
        import inspect

        from sglang.srt.managers import schedule_batch

        src = inspect.getsource(schedule_batch.ScheduleBatch._evict_swa)
        call = next(
            n
            for n in ast.walk(ast.parse(src.strip()))
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "free_swa_out_of_window_slots"
        )
        self.assertIn("release_cache_protected_prefix", [k.arg for k in call.keywords])

    def test_chunk_cache_does_not_back_up_every_finished_req(self):
        # Dropping the is_decode term during the port flipped save_cache from
        # always-False to always-True, so a prefill-only chunk-cache request
        # would write its whole KV to EIC on finish.
        seen = []
        cache = EICChunkCache.__new__(EICChunkCache)
        cache.save_decode_cache = False
        cache.write_backup = lambda req, save_decode_cache: seen.append(
            save_decode_cache
        )
        cache.req_to_token_pool = SimpleNamespace(free=lambda idx: None)
        cache.cache_finished_req(
            SimpleNamespace(req_pool_idx=0), is_insert=True, kv_len_to_handle=0
        )
        self.assertEqual(seen, [False])

    def test_finalize_asserts_on_verdict_before_ack(self):
        # I5 enforcement: a FINAL verdict may never land before the local ack
        # when a load was kicked -- silent free of in-flight DMA otherwise.
        s = self._make_stage(0, self.FakeStore(), {}, pp_size=1)
        s.dec_lock_ref = lambda n: None
        node = self._make_req("x", 4, 16, load_node_id=100).best_match_node
        req = self._make_req("req-h", 4, 16, load_node_id=100)
        s.ongoing_load_admit = {
            "req-h": {
                "h": 5,
                "d": 4,
                "alloc": 12,
                "load_node": node,
                "new_indices": torch.arange(12, dtype=torch.int64),
                "deferred_lock": node,
                "complete": None,
            }
        }
        s._h_rid = {5: "req-h"}
        s._admit_verdict = {5: 10}
        with self.assertRaises(AssertionError):
            EICPagedHiRadixCache._finalize_load_admit(s, req)

    def _make_load_controller(self, page_data):
        cc = object.__new__(EICCacheController)
        cc.tp_world_size = 1
        cc.page_size = 4
        cc.ack_load_queue = Queue()
        cc.mem_pool_host = SimpleNamespace(
            get_page_data=mock.Mock(side_effect=page_data)
        )
        return cc

    def _load_op(self):
        op = object.__new__(EICCacheOperation)
        op.node_id = 7
        op.content_hash = ["h0", "h1", "h2"]
        op.host_indices = torch.arange(12, dtype=torch.int64)
        op.device_indices = torch.arange(12, dtype=torch.int64)
        return op

    def test_load_acks_zero_when_backend_raises(self):
        # A missing ack strands the request holding its KV until abort, and
        # under PP it also blocks the verdict for every other stage.
        cc = self._make_load_controller(RuntimeError("eic client blew up"))
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 0))

    def test_short_all_true_mask_is_not_full_success(self):
        # The backend bails out early on failure, returning fewer mask entries
        # than pages; all(mask) would read that as a complete hit.
        cc = self._make_load_controller([[True]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 4))

        cc = self._make_load_controller([[]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 0))

    def test_full_mask_acks_every_token(self):
        cc = self._make_load_controller([[True, True, True]])
        EICCacheController.load_operation_shared(cc, self._load_op())
        self.assertEqual(cc.ack_load_queue.get_nowait(), (7, 12))

    def _make_reconciler(self, pp_rank, store, pp_size=2):
        r = EICPPReconciler(
            prefix="hipf", pp_rank=pp_rank, pp_size=pp_size, pp_group=object(), rank=0
        )
        r.eic = True
        r._store_handle = store
        return r

    def test_reconciler_min_across_stages(self):
        store = self.FakeStore()
        s0, s1 = self._make_reconciler(0, store), self._make_reconciler(1, store)
        h = EICPPReconciler.rid_hash("r")
        s0.report(h, 1, 512)
        s1.report(h, 1, 256)
        self.assertEqual(s1.collect(), [])
        self.assertEqual(s0.collect(), [(h, 1, 256)])

    def test_reconciler_waits_for_every_stage(self):
        s0 = self._make_reconciler(0, self.FakeStore())
        s0.report(EICPPReconciler.rid_hash("r"), 1, 512)
        self.assertEqual(s0.collect(), [])

    def test_reconciler_tombstone_is_epoch_aware(self):
        store = self.FakeStore()
        s0, s1 = self._make_reconciler(0, store), self._make_reconciler(1, store)
        h = EICPPReconciler.rid_hash("r")
        s0.release(h, 1)
        s1.report(h, 1, 256)
        s1.report(h, 2, 128)
        s1.collect()
        s0._drain_peers()
        self.assertNotIn((h, 1), s0._reports)
        self.assertIn((h, 2), s0._reports)

    def test_reconciler_epoch_never_reused(self):
        r = self._make_reconciler(0, self.FakeStore())
        self.assertEqual([r.bump_epoch("r"), r.bump_epoch("r")], [1, 2])

    def _make_writing_check_cache(self, write_policy):
        # Minimal EICPagedHiRadixCache for exercising writing_check's dec_lock_ref
        # gating. One acked node in ongoing_write_through, TP=1 (no all_reduce).
        cache = object.__new__(EICPagedHiRadixCache)
        cache.tp_group = None
        node = SimpleNamespace(id=7, host_value=object())
        cache.ongoing_write_through = {7: node}
        ackq = Queue()
        ackq.put((7, True))
        cache.cache_controller = SimpleNamespace(
            write_policy=write_policy, ack_write_queue=ackq
        )
        cache.dec_lock_ref = mock.Mock()
        return cache

    def test_writing_check_skips_dec_lock_ref_under_write_back_policy(self):
        # The device-pool double-count crash: under write_back policy nodes are never
        # inc_lock_ref'd, so writing_check() (evict throttle / check_hicache_events,
        # which pass no flag) must NOT dec_lock_ref -- doing so drains a running req's
        # protected pages into evictable ("pool memory leak detected").
        import torch as _torch

        cache = self._make_writing_check_cache("write_back")
        with mock.patch.object(_torch.distributed, "get_world_size", return_value=1):
            EICPagedHiRadixCache.writing_check(cache)  # no explicit flag
        cache.dec_lock_ref.assert_not_called()
        self.assertEqual(cache.ongoing_write_through, {})

    def test_writing_check_dec_lock_ref_under_write_through_policy(self):
        # write_through nodes ARE inc_lock_ref'd, so the release must still fire.
        import torch as _torch

        cache = self._make_writing_check_cache("write_through")
        with mock.patch.object(_torch.distributed, "get_world_size", return_value=1):
            EICPagedHiRadixCache.writing_check(cache)
        cache.dec_lock_ref.assert_called_once()
        self.assertEqual(cache.ongoing_write_through, {})

    def test_writing_check_no_barrier_when_not_blocking(self):
        # The forward-loop hazard: under write_back the per-loop call sites must NOT
        # spin waiting for every ack (a hung EIC write thread would freeze forward_ct
        # until the watchdog SIGQUITs the server). Assert on the spin itself -- the
        # barrier's own 30s timeout means a missing `and blocking` still returns, so
        # only the sleep count distinguishes "never entered" from "entered and bailed".
        import torch as _torch

        cache = self._make_writing_check_cache("write_back")
        cache.ongoing_write_through = {7: SimpleNamespace(id=7, host_value=object())}
        cache.cache_controller.ack_write_queue = Queue()  # qsize(0) != ongoing(1)
        with mock.patch.object(
            _torch.distributed, "get_world_size", return_value=1
        ), mock.patch("sglang.srt.mem_cache.eic_hiradix_cache.time.sleep") as slept:
            EICPagedHiRadixCache.writing_check(cache)  # blocking defaults False
        slept.assert_not_called()
        self.assertIn(7, cache.ongoing_write_through)

    def test_writing_check_barrier_times_out(self):
        # evict's blocking barrier must be bounded: if acks never arrive it breaks
        # after the timeout instead of hanging the scheduler indefinitely. Without
        # the deadline the loop spins forever, so a bounded sleep count is the claim.
        import torch as _torch

        cache = self._make_writing_check_cache("write_back")
        cache.ongoing_write_through = {7: SimpleNamespace(id=7, host_value=object())}
        cache.cache_controller.ack_write_queue = Queue()  # never satisfies barrier
        with mock.patch.object(
            _torch.distributed, "get_world_size", return_value=1
        ), mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache.time.perf_counter",
            # start, one spin (<30s), past 30s -> break, final cost_time read
            side_effect=[0.0, 5.0, 40.0, 40.1],
        ), mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache.time.sleep"
        ) as slept:
            EICPagedHiRadixCache.writing_check(cache, write_back=True, blocking=True)
        self.assertEqual(slept.call_count, 1)

    def test_async_batch_set_drops_when_write_pool_exhausted(self):
        # Backend-failure OOM guard: when the async write pool is exhausted (the
        # consumer thread is behind a slow/failing EIC backend), async_batch_set must
        # DROP the batch, not allocate an unbounded host tensor per call. The old
        # fallback accumulated 1000+ mallocs in seconds and OOM-killed the rank.
        from sglang.srt.mem_cache.eic_memory_pool import EICKVClient

        c = object.__new__(EICKVClient)
        c.kv_cache_write_mem_pool = SimpleNamespace(left_count=lambda: 0)
        c.write_queue = Queue()
        c._write_drop_ct = 0
        copied = []
        ret = EICKVClient.async_batch_set(
            c,
            keys=["a", "b", "c"],
            obj_inputs=None,
            device_indices=torch.arange(3),
            copy_func=lambda idx, objs: copied.append(idx),
        )
        self.assertFalse(ret)  # dropped
        self.assertTrue(c.write_queue.empty())  # nothing enqueued
        self.assertEqual(copied, [])  # no device->host copy / malloc

    def test_async_batch_set_enqueues_when_pool_has_room(self):
        # Normal path unchanged: with pool headroom the batch allocates registered
        # slots and is enqueued for the write thread.
        from sglang.srt.mem_cache.eic_memory_pool import EICKVClient

        c = object.__new__(EICKVClient)
        slots = [torch.zeros(2) for _ in range(3)]
        c.kv_cache_write_mem_pool = SimpleNamespace(
            left_count=lambda: 8,
            try_allocate_kv_cache=lambda shape, dtype, count: (slots[:count], None),
        )
        c.kv_cache_shape = (2,)
        c.kv_cache_dtype = torch.float32
        c.write_queue = Queue()
        copied = []
        ret = EICKVClient.async_batch_set(
            c,
            keys=["a", "b", "c"],
            obj_inputs=None,
            device_indices=torch.arange(3),
            copy_func=lambda idx, objs: copied.append(idx),
        )
        self.assertTrue(ret)
        self.assertEqual(c.write_queue.qsize(), 1)
        self.assertEqual(len(copied), 1)  # copy_func invoked once

    def _make_evict_throttle_cache(self, backlog):
        # Minimal cache for exercising evict()'s write-through throttle loop only.
        # The loop runs before any tree work; evictable_leaves is the first thing
        # the body touches, so a sentinel there proves the throttle was escaped.
        cache = object.__new__(EICPagedHiRadixCache)
        cache.ongoing_write_through = {i: object() for i in range(backlog)}
        cache.writing_check = mock.Mock()
        return cache

    def test_evict_throttle_is_bounded_when_write_thread_stalls(self):
        # The watchdog SIGQUIT crash: a hung EIC mset stops ongoing_write_through
        # from draining, and evict()'s throttle spun forever on all PP0 ranks. PP1
        # then blocked in broadcast, forward_ct froze, and the 300s scheduler
        # watchdog killed all 8 ranks. The throttle must break out on a deadline.
        cache = self._make_evict_throttle_cache(backlog=51)  # > 50, never drains
        params = SimpleNamespace(num_tokens=1, swa_num_tokens=0)
        escaped = RuntimeError("throttle escaped")

        class Sentinel:
            def __iter__(self):
                raise escaped

        cache.evictable_leaves = Sentinel()
        with mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache.time.perf_counter",
            # start, two spins under the deadline, then past 30s -> break
            side_effect=[0.0, 1.0, 2.0, 40.0],
        ), mock.patch("sglang.srt.mem_cache.eic_hiradix_cache.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                EICPagedHiRadixCache.evict(cache, params)
        self.assertIs(ctx.exception, escaped)  # reached the body, i.e. broke out
        self.assertEqual(cache.writing_check.call_count, 2)  # bounded spins
        self.assertEqual(len(cache.ongoing_write_through), 51)  # never drained

    def _remote_cache(self, page_size=256, budget=2048, memoized=()):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = dict.fromkeys(memoized)
        cache.tp_size = 1
        cache.pp_size = 1
        cache.pp_group = None
        cache.page_size = page_size
        cache.eic_check_max_num = budget
        cache.root_node = object()
        cache._insert_remote_node = mock.Mock()
        cache.cache_controller = mock.Mock()
        return cache

    def test_match_from_remote_probes_only_a_full_page_of_remainder(self):
        cache = self._remote_cache()
        node = SimpleNamespace(id=1, content_hash=None)
        cache._match_for_remote_fetch = lambda root, key: (0, node)
        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [
            256,
            256,
        ]
        reqs = [
            mock.Mock(
                rid=r, origin_input_ids=list(range(n)), output_ids=[9], extra_key=None
            )
            for r, n in (("under", 255), ("exact", 256), ("over", 257))
        ]
        with mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
            return_value=False,
        ):
            EICPagedHiRadixCache.match_from_remote(cache, reqs)

        probed = cache.cache_controller.batch_find_longest_prefix_in_eic.call_args[0][0]
        self.assertEqual([len(k) for k in probed], [256, 257])

    def test_match_from_remote_retries_a_miss_then_memoizes_the_hit(self):
        cache = self._remote_cache()
        node = SimpleNamespace(id=1, content_hash=None)
        cache._match_for_remote_fetch = lambda root, key: (512, node)
        req = mock.Mock(
            rid="R", origin_input_ids=list(range(4096)), output_ids=[9], extra_key=None
        )
        patch_hash = mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
            return_value=False,
        )

        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [0]
        with patch_hash:
            EICPagedHiRadixCache.match_from_remote(cache, [req])
        cache._insert_remote_node.assert_not_called()
        self.assertEqual(cache.match_req_set, {})

        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [768]
        with patch_hash:
            EICPagedHiRadixCache.match_from_remote(cache, [req])
        key = cache._insert_remote_node.call_args[0][1]
        self.assertEqual(len(key), 768)
        self.assertEqual(key.token_ids[0], 512)
        self.assertIn("R", cache.match_req_set)

        cache._match_for_remote_fetch = mock.Mock(
            side_effect=AssertionError("memoized rid must be skipped")
        )
        with patch_hash:
            EICPagedHiRadixCache.match_from_remote(cache, [req])
        self.assertEqual(
            cache.cache_controller.batch_find_longest_prefix_in_eic.call_count, 2
        )

    def test_match_from_remote_reduces_even_when_fetches_is_empty(self):
        # A lagging PP stage has an empty fetches; skipping its reduce would
        # orphan PP0's isend onto the next round's num_ready recv.
        def reduce_shapes(memoized):
            cache = self._remote_cache(memoized=memoized)
            cache.pp_size = 2
            cache.pp_group = object()
            node = SimpleNamespace(id=1, content_hash=None)
            cache._match_for_remote_fetch = lambda root, key: (0, node)
            cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [
                256,
                256,
            ]
            shapes = []

            def fake_reduce(t):
                shapes.append(tuple(t.shape))
                if t.dim() == 0:
                    t.fill_(2)

            cache._reduce_min = fake_reduce
            reqs = [
                mock.Mock(
                    rid=r,
                    origin_input_ids=list(range(4096)),
                    output_ids=[9],
                    extra_key=None,
                )
                for r in ("x", "y")
            ]
            with mock.patch(
                "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
                return_value=False,
            ):
                EICPagedHiRadixCache.match_from_remote(cache, reqs)
            return shapes

        self.assertEqual(reduce_shapes(()), reduce_shapes(("x", "y")))

    def test_eic_check_max_num_is_bounded_by_default(self):
        def probe_budget(cfg):
            cache = object.__new__(EICPagedHiRadixCache)
            cache.page_size = 256
            with mock.patch.object(
                EICPagedHiRadixCache.__bases__[0], "init_hyper_params", lambda *a: None
            ):
                EICPagedHiRadixCache.init_hyper_params(cache, cfg)
            return cache.eic_check_max_num

        self.assertEqual(probe_budget({}), 2048)
        self.assertEqual(probe_budget({"eic_check_max_num": 512}), 512)
        self.assertEqual(probe_budget({"eic_check_max_num": -1}), -1)

    def test_real_tree_node_carries_content_hash(self):
        # Every EIC hash path (_need_calculate_hash, _split_node, write_backup)
        # reads node.content_hash off nodes built by RadixCache._insert_helper,
        # not just EIC-created ones. The suite's SimpleNamespace fakes invent the
        # field, so only a real TreeNode catches it going missing -- without it
        # the first non-EIC-inserted node raises AttributeError.
        from sglang.srt.mem_cache.eic_hiradix_cache import _need_calculate_hash
        from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

        node = TreeNode()
        self.assertIsNone(node.content_hash)
        node.key = RadixKey(list(range(512)), None)
        self.assertTrue(_need_calculate_hash(node, 256))

    def test_insert_helper_matches_radix_cache_insert_contract(self):
        # EICHiRadixCache inherits RadixCache.insert, which unpacks
        # (prefix_len, last_node) from _insert_helper. Returning a bare int
        # crashed the first cache_unfinished_req of a real server.
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

        cache = object.__new__(EICPagedHiRadixCache)
        cache.disable = False
        cache.is_eagle = False
        cache.page_size = 1
        cache.root_node = TreeNode()
        cache.evictable_size_ = 0
        cache._update_leaf_status = lambda node: None
        cache.inc_hit_count = mock.Mock()
        cache.cache_controller = SimpleNamespace(write_policy="write_through")
        # One hash per page; page_size is 1.
        cache.calculate_hash_fn = lambda key, page_size, prev: list(key.token_ids)

        def insert(ids):
            return cache.insert(
                InsertParams(key=RadixKey(ids, None), value=torch.tensor(ids))
            )

        first = insert([1, 2, 3, 4])
        self.assertEqual(first.prefix_len, 0)
        self.assertEqual(first.last_device_node.key.token_ids, [1, 2, 3, 4])
        again = insert([1, 2, 3, 4])
        self.assertEqual(again.prefix_len, 4)
        self.assertIs(again.last_device_node, first.last_device_node)
        split = insert([1, 2, 9])
        self.assertEqual(split.prefix_len, 2)
        self.assertEqual(split.last_device_node.key.token_ids, [9])

    def test_match_from_remote_probe_budget_caps_a_long_queue(self):
        cache = object.__new__(EICPagedHiRadixCache)
        cache.match_req_set = {}
        cache.tp_size = 1
        cache.pp_size = 1
        cache.pp_group = None
        cache.page_size = 256
        cache.eic_check_max_num = 5  # each req below is 4 pages
        cache.root_node = object()
        cache._insert_remote_node = mock.Mock()
        node = SimpleNamespace(id=1, content_hash=None)
        cache._match_for_remote_fetch = lambda root, key: (0, node)
        cache.cache_controller = mock.Mock()
        cache.cache_controller.batch_find_longest_prefix_in_eic.return_value = [0, 0]

        reqs = [
            mock.Mock(
                rid=r,
                origin_input_ids=list(range(1024)),
                output_ids=[9],
                extra_key=None,
            )
            for r in ("a", "b", "c")
        ]
        with mock.patch(
            "sglang.srt.mem_cache.eic_hiradix_cache._need_calculate_hash",
            return_value=False,
        ):
            EICPagedHiRadixCache.match_from_remote(cache, reqs)

        probed = cache.cache_controller.batch_find_longest_prefix_in_eic.call_args[0][0]
        self.assertEqual(len(probed), 2)
        self.assertEqual(len(cache.match_req_set), 0)

    def test_hicache_host_stats_read_a_real_eic_host_pool(self):
        # _log_hicache_stats reads logical_size off tree_cache.token_to_kv_pool_host,
        # which for EIC is a flat EIC*TokenToKVPoolHost, not a HostPoolGroup with an
        # anchor to proxy from. Build the pool through its real __init__: a
        # hand-rolled SimpleNamespace would invent the attribute and hide that EIC
        # forces enable_hierarchical_cache while the reporter has no try/except, so
        # a missing logical_size is an AttributeError on every prefill report.
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )
        from sglang.srt.mem_cache.eic_memory_pool import EICBaseTokenToKVPoolHost

        device_pool = SimpleNamespace(store_dtype=torch.bfloat16, size=8192)
        host = EICBaseTokenToKVPoolHost.__new__(EICBaseTokenToKVPoolHost)
        host.get_size_per_token = lambda: 1024
        with mock.patch(
            "sglang.srt.mem_cache.eic_memory_pool.get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
        ), mock.patch.dict(os.environ, {"MY_HOST_IP": "10.0.0.1"}):
            EICBaseTokenToKVPoolHost.__init__(
                host,
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=256,
                extra_info={},
            )
        host.free_slots = host.free_slots[:-2960]

        rep = object.__new__(SchedulerMetricsReporter)
        rep.stats = SimpleNamespace()
        rep.scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            tree_cache=SimpleNamespace(token_to_kv_pool_host=host),
        )
        SchedulerMetricsReporter._log_hicache_stats(rep)
        self.assertEqual(rep.stats.hicache_host_total_tokens, 16384)
        self.assertEqual(rep.stats.hicache_host_used_tokens, 2960)

    def test_write_thread_frees_slots_when_mset_raises(self):
        from sglang.srt.mem_cache.eic_memory_pool import EICKVClient

        freed = []
        c = object.__new__(EICKVClient)
        c.write_queue = Queue()
        c.kv_cache_write_mem_pool = SimpleNamespace(
            check_data_ptr_allocated=lambda p: True,
            free_to_mempool=freed.append,
        )
        c._async_set_impl = mock.Mock(side_effect=RuntimeError("backend down"))
        values = [torch.zeros(2), torch.zeros(2)]
        c.write_queue.put((["a", "b"], values, None))
        c.write_queue.put(None)

        with self.assertRaises(TypeError):
            EICKVClient._write_thread(c)

        self.assertEqual(freed, [v.data_ptr() for v in values])

    def test_memory_pool_refuses_a_region_the_client_will_not_register(self):
        # register_memory's bool return was dropped. An unregistered region still
        # "works": the client falls back to a CPU memcpy of whatever pointer it was
        # handed, which for a device pointer is a segfault on its IO thread in the
        # middle of a write. Fail at construction instead.
        from sglang.srt.mem_cache import eic_memory_pool

        fake_eic = SimpleNamespace(
            MemoryInfo=lambda: SimpleNamespace(type=None, cuda_id=None),
            MemoryType=SimpleNamespace(MEMORY_CUDA=1),
            IOBuffers=lambda: SimpleNamespace(append=lambda *args: None),
        )
        conn = mock.Mock()
        real_zeros = torch.zeros

        def zeros_without_pinning(*args, **kwargs):
            # pin_memory needs CUDA; this test only exercises the registration check.
            kwargs.pop("pin_memory", None)
            return real_zeros(*args, **kwargs)

        with mock.patch.object(eic_memory_pool, "eic", fake_eic), mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ), mock.patch.object(torch, "zeros", zeros_without_pinning):
            conn.register_memory.return_value = False
            with self.assertRaises(RuntimeError):
                eic_memory_pool.FlexibleKVCacheMemoryPool(
                    conn, (2, 2), torch.float16, "cpu"
                )

            conn.register_memory.return_value = True
            pool = eic_memory_pool.FlexibleKVCacheMemoryPool(
                conn, (2, 2), torch.float16, "cpu"
            )
        self.assertEqual(pool.left_count(), len(pool.free_data_addr))

    def test_kvset_gpu_direct_stays_off_until_the_client_can_register_cuda(self):
        # eic 1.5.2 registers every region as byterpc HolderType::UserNormal, so a
        # device write pool reaches AssembleWriteSglist with no MR and the client
        # memcpy's device memory on its IO thread: SIGSEGV in be::Engine::Transmit,
        # server down. Deriving this flag from config again re-arms that crash.
        import ast
        import inspect

        from sglang.srt.mem_cache import eic_memory_pool

        assigns = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(eic_memory_pool)))
            if isinstance(node, ast.Assign)
            and any(
                getattr(target, "id", None) == "G_EnableKVSetGPUDirect"
                for target in node.targets
            )
        ]
        self.assertTrue(assigns)
        for assign in assigns:
            self.assertIsInstance(assign.value, ast.Constant)
            self.assertFalse(assign.value.value)

    def test_scheduler_eic_gate_admits_only_caches_with_admission_hooks(self):
        # The scheduler calls these hooks on tree_cache whenever enable_eic_cache is
        # set. The gate also admitted EICChunkCache (the PD decode-save path), which
        # has none of them, so aborting a waiting req there raised AttributeError on
        # release_load_admit.
        import ast
        import inspect

        from sglang.srt.managers import scheduler
        from sglang.srt.mem_cache import eic_hiradix_cache

        hooks = ("release_load_admit", "check_load_back_progress", "prefix_loading")
        gate = next(
            node.value
            for node in ast.walk(ast.parse(inspect.getsource(scheduler)))
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "self.enable_eic_cache"
            and "isinstance" in ast.unparse(node.value)
        )
        call = next(
            n
            for n in ast.walk(gate)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "isinstance"
        )
        admitted = call.args[1]
        names = (
            [ast.unparse(e) for e in admitted.elts]
            if isinstance(admitted, ast.Tuple)
            else [ast.unparse(admitted)]
        )
        self.assertTrue(names)
        for name in names:
            cls = getattr(scheduler, name)
            for hook in hooks:
                self.assertTrue(callable(getattr(cls, hook, None)), f"{name}.{hook}")
        self.assertIn(eic_hiradix_cache.EICHiRadixCache.__name__, names)


if __name__ == "__main__":
    unittest.main()
