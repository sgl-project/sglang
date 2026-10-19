import shutil
import tempfile
import time
from array import array
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    drop_hicache_atexit_pin,
    match_prefix,
    node_device_value,
    node_host_value,
    node_parent,
    node_token_ids,
    write_backup,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures


def aux_storage_key_transfers(cache, node_id):
    transfers = []
    if ComponentType.SWA in cache.tree_components:
        value = node_device_value(cache, node_id, ComponentType.SWA)
        if value is not None:
            num_pages = len(value) // cache.page_size
            if num_pages > 0:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.SWA,
                        keys=cache.tree_core.get_hash_values(node_id)[-num_pages:],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                )
    return transfers or None


class ExtraPoolFullComponent(FullComponent):
    """FULL component that also stores every page under a pool of its own,
    staged through the anchor's host pool."""

    EXTRA_POOL = PoolName.DRAFT

    def buffer_mode_host_pool_entries(self, host_pool_group):
        anchor = host_pool_group.anchor_entry
        return [
            PoolEntry(
                name=self.EXTRA_POOL,
                host_pool=anchor.host_pool,
                device_pool=anchor.device_pool,
                layer_mapper=anchor.layer_mapper,
            )
        ]

    def buffer_backup_keys(self, node, hash_values):
        return {self.EXTRA_POOL: list(hash_values)}

    def build_hicache_transfers(self, node, phase, **kwargs):
        if phase == CacheTransferPhase.BACKUP_HOST:
            # Two transfers of the pool, each keyed by its own pages: the
            # storage write must keep the keys per transfer.
            value = node.component_data[BASE_COMPONENT_TYPE].value
            keys = list(node.hash_value)
            split = len(keys) // 2
            page_size = self.tree_core.page_size
            halves = [
                (value[: split * page_size], keys[:split]),
                (value[split * page_size :], keys[split:]),
            ]
            return [
                PoolTransfer(name=self.EXTRA_POOL, device_indices=rows, keys=part)
                for rows, part in halves
                if part
            ]
        return super().build_hicache_transfers(node, phase, **kwargs)


class UnifiedRadixCacheStorageFixtures(UnifiedRadixCacheFixtures):
    def _make_storage_dir(self):
        storage_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        return storage_dir

    def _write_path_to_l3(self, cache, node):
        """Offload every node on root->node path from host to L3 storage."""
        for n in self._path_chain(cache, node):
            cache.write_backup_storage(n)

    def _ongoing_l3_backups(self, cache):
        """Storage writes in flight (buffer mode tracks them on the pipeline)."""
        if cache.buffer_pipeline is not None:
            return cache.buffer_pipeline.ongoing_backup
        return cache.ongoing_backup

    def _flush_l3_backups(self, cache, timeout: float = 10.0):
        """Wait for backup threads to finish, then drain acks (release locks)."""
        deadline = time.time() + timeout
        while self._ongoing_l3_backups(cache) and time.time() < deadline:
            cache.drain_storage_control_queues()
            if self._ongoing_l3_backups(cache):
                time.sleep(0.01)
        cache.drain_storage_control_queues()
        self.assertFalse(
            self._ongoing_l3_backups(cache), "L3 backups did not complete in time"
        )

    def _run_prefetch_to_completion(self, cache, req_id, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Host memory is reserved (and IO started) by the scheduler-thread
            # drain once the L3 hit count is known, and the buffer-mode fill
            # commits at its H2D ack in loading_check — pump the full event
            # round like the real scheduler loop does.
            cache.check_hicache_events()
            if cache.check_prefetch_progress(req_id):
                # Buffer mode parks a completed fetch as a staged prefetch;
                # consume it like the PrefillAdder would so callers see the
                # span tree-resident.
                if (
                    cache.buffer_pipeline is not None
                    and cache.buffer_pipeline.has_staged(req_id)
                ):
                    self._consume_staged_prefetch(cache, req_id, timeout=timeout)
                return
            time.sleep(0.01)
        self.fail(f"prefetch {req_id} did not complete in time")

    def _consume_staged_prefetch(
        self,
        cache,
        req_id,
        prefix_len=None,
        timeout: float = 10.0,
        *,
        extra_key=None,
        cache_salt=None,
        last_node=None,
        on_dispatched=None,
        return_req: bool = False,
    ):
        """Simulate the PrefillAdder consuming a staged prefetch at admission:
        init_load_back (buffer dispatch: device alloc + queued H2D), the batch
        start_loading flush, then pump until the ack commit lands. Returns
        the spliced device indices (empty on degrade). ``on_dispatched`` runs
        between the two, where the H2D is still in flight."""
        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        f = cache.buffer_pipeline.staged_prefetches[req_id]
        if prefix_len is None:
            prefix_len = f.matched_len
        req = mock.Mock()
        req.rid = req_id.rid
        req.cache_request_handle = req_id
        req.extra_key = extra_key
        req.cache_salt = cache_salt
        req.prefix_len = prefix_len
        req.last_node = cache.root_node_handle() if last_node is None else last_node
        joint = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    array("q", f.key_tokens), extra_key=extra_key, cache_salt=cache_salt
                )
            )
        )
        req.prefix_len = joint.device_prefix_len
        joint_covers_span = joint.device_prefix_len >= len(f.key_tokens)
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.tree_cache = cache
        with (
            mock.patch.object(
                cache.tree_core,
                "match_full_device_prefix",
                wraps=cache.tree_core.match_full_device_prefix,
            ) as full_match,
            mock.patch.object(
                cache.tree_core,
                "collect_full_device_indices",
                wraps=cache.tree_core.collect_full_device_indices,
            ) as collect_indices,
        ):
            self.assertTrue(
                cache.buffer_pipeline.prepare_staged_prefetch(req),
                "production admission FULL rematch unexpectedly deferred",
            )
            with adder._lock_node(req.last_node):
                loaded = cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=None,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
            if joint_covers_span:
                # A joint match past the staged span is kept without a rematch.
                full_match.assert_not_called()
            else:
                full_match.assert_called_once()
                # A load-back committing here reads the path again for its
                # slot-ownership check.
                collect_indices.assert_called()
        loaded_len, last_node = (0, req.last_node) if loaded is None else loaded
        path = cache.path_device_indices(last_node)
        new_indices = path[len(path) - loaded_len :]
        if on_dispatched is not None:
            on_dispatched()
        # Batch formation flushes the queued load into the batch's producer.
        cache.ready_to_load_host_cache()
        self._pump_hicache_until(
            cache,
            lambda: not cache.buffer_pipeline.ongoing_buffer_load_back,
            "staged-prefetch consumption did not commit",
            timeout=timeout,
        )
        return (new_indices, req) if return_req else new_indices

    def _all_page_hashes(self, cache, node):
        hashes = []
        for n in self._path_chain(cache, node):
            hashes.extend(cache.tree_core.get_hash_values(n))
        return hashes

    def _init_buffer_hicache(
        self,
        cache,
        storage_dir,
        prefetch_policy: str = "wait_complete",
        storage_extra: Optional[dict] = None,
        context_length: Optional[int] = None,
    ):
        if self.cfg.has_mamba:
            self.skipTest(
                "load-back requests here are Mocks; buffer_only Mamba is "
                "covered by TestUnifiedHybridBufferOnlyBitExact"
            )
        self._init_hicache(
            cache,
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
            host_memory_mode="buffer_only",
            prefetch_policy=prefetch_policy,
            storage_extra=storage_extra,
            context_length=context_length,
        )

    def _build_buffer_hicache_fixture(self, storage_dir, **init_kwargs):
        fixture = build_fixture(self.cfg)
        self._init_buffer_hicache(fixture[0], storage_dir, **init_kwargs)
        return fixture

    def _pump_hicache_until(self, cache, cond, msg, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            cache.check_hicache_events()
            if cond():
                return
            time.sleep(0.01)
        self.fail(msg)

    def _host_avail_sizes(self, cache):
        group = cache.cache_controller.mem_pool_host
        return {entry.name: entry.host_pool.available_size() for entry in group.entries}

    def _storage_exists_count(self, cache, page_hashes, pool_transfers=None):
        """Ground-truth longest-prefix existence count from the backend,
        folded across pools like the prefetch hit query."""
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo

        backend = cache.cache_controller.storage_backend
        extra_info = HiCacheStorageExtraInfo(prefix_keys=None)
        if pool_transfers:
            result = backend.batch_exists_v2(page_hashes, pool_transfers, extra_info)
            return min(result.kv_hit_pages, len(page_hashes))
        return backend.batch_exists(page_hashes, extra_info)

    def _buffer_backup_and_wait(self, cache, node):
        # Parent-first over the whole path, mirroring the production trigger
        # (_inc_hit_count fires per matched node on the insert walk): the SWA
        # component may have split the leaf at the window boundary.
        pipeline = cache.buffer_pipeline
        chain = self._path_chain(cache, node)
        write_backup(cache, node)
        for n in chain:
            self.assertIn(n, pipeline.inflight_backup_node_ids)
        self._pump_hicache_until(
            cache,
            lambda: (
                not pipeline.inflight_backup_node_ids and not pipeline.ongoing_backup
            ),
            "buffer backup pipeline did not drain",
        )

    def _produce_buffer_l3(
        self, storage_dir, seq, marker=None, *, extra_key=None, cache_salt=None
    ):
        """Producer tree in buffer mode: insert seq and push it to L3."""
        prod, prod_alloc, prod_rtp = self._build_buffer_hicache_fixture(storage_dir)
        self._insert(
            prod,
            prod_alloc,
            prod_rtp,
            seq,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        key = RadixKey(array("q", seq), extra_key=extra_key, cache_salt=cache_salt)
        leaf = prod.match_prefix(MatchPrefixParams(key=key)).last_device_node
        expected = None
        if marker is not None:
            m = prod.match_prefix(MatchPrefixParams(key=key))
            self._fill_full_kv(
                prod_alloc, prod.path_device_indices(m.last_device_node), marker=marker
            )
            expected = self._snapshot_full_kv(
                prod_alloc, prod.path_device_indices(m.last_device_node)
            )
        self._buffer_backup_and_wait(prod, leaf)
        return leaf, expected

    def _buffer_swa_seq(self, min_pages=4):
        """Sequence long enough for SWA prefetch (one full window + 1)."""
        num_pages = min_pages
        if self.cfg.has_swa:
            sw_pages = (
                self.cfg.sliding_window_size + self.cfg.page_size - 1
            ) // self.cfg.page_size
            num_pages = max(num_pages, sw_pages + 1)
        return self._make_seq(1, num_pages)

    def _drain_buffer_writes(self, cache):
        pipeline = cache.buffer_pipeline
        self._pump_hicache_until(
            cache,
            lambda: (
                not pipeline.inflight_backup_node_ids and not pipeline.ongoing_backup
            ),
            "buffer backup pipeline did not drain",
        )
        self.assertFalse(pipeline.pending_write_queue)
        self.assertFalse(pipeline._queued_span_refs)
        self.assertEqual(pipeline.write_backlog_tokens_, 0)
        self.assertEqual(pipeline.write_staged_tokens_, 0)

    @staticmethod
    def _writes_by_pool(write_storage, extra):
        """Key tuples of the KV writes and of the ``extra`` pool's writes
        among the storage writes recorded by ``write_storage``."""
        kv_writes, extra_writes = set(), set()
        for call in write_storage.call_args_list:
            host_indices, _, hash_value = call.args[:3]
            transfers = call.kwargs.get("extra_pools") or []
            names = [transfer.name for transfer in transfers]
            if hash_value:
                assert extra not in names
                kv_writes.add(tuple(hash_value))
            elif extra in names:
                # The pool's transfers reach storage as one contiguous span.
                assert host_indices.numel() == 0 and names == [extra]
                extra_writes.add(tuple(transfers[0].keys))
        return kv_writes, extra_writes

    def _assert_masked_full_is_reused(self, storage_dir, seq, live_tail: int):
        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        cons.enable_storage_metrics = True
        cons.storage_metrics_collector = mock.Mock()

        # Masked state: nodes born with live FULL under SWA tombstones (a
        # sibling insert whose SWA ring had slid past the span). live_tail is
        # how far short of the span end the ring stopped.
        live_from = len(seq) - live_tail
        value = self._alloc(cons_alloc, len(seq))
        cons.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=value[: len(seq)],
                component_evicted_seqlens={ComponentType.SWA: live_from},
            )
        )
        masked = match_prefix(cons, seq)
        self.assertEqual(masked.device_prefix_len, 0, "unified match not masked")
        self.assertEqual(masked.full_kv_hit_length, len(seq), "live FULL not resident")
        masked_full = value[: len(seq)].clone()
        # Model the real SWA-eviction path: tombstoned positions have their
        # allocator mapping cleared, while the live trailing portion remains.
        cons_alloc.free_swa(masked_full[:live_from])
        self._fill_full_kv(cons_alloc, masked_full, marker=7)
        masked_kv = self._snapshot_full_kv(cons_alloc, masked_full)
        tail_full = masked_full[live_from:]
        tail_swa = cons_alloc.full_to_swa_index_mapping[tail_full].clone()
        self.assertTrue(bool((tail_swa > 0).all()), "tail SWA not left resident")

        avail0 = self._host_avail_sizes(cons)
        req_id = CacheRequestHandle(f"masked-overlap-{live_tail}", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        # A parked SWA-only retry has no FULL suffix, but must re-pin its prefix.
        with mock.patch.object(cons.swa_kv_pool_host, "alloc", return_value=None):
            self._pump_hicache_until(
                cons,
                lambda: bool(cons.buffer_pipeline.pending_hit_allocs),
                "SWA-only hit was not parked on the staging shortfall",
            )
            self.assertEqual(
                cons.ongoing_prefetch[req_id].operation.storage_hit_count, 0
            )
            self.assertEqual(
                cons.buffer_pipeline._prefetch_prefix_ctx[req_id][0], list(seq)
            )
            self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
            self.assertEqual(cons.buffer_pipeline.anchor_locked_tokens_, 0)
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        self.assertIn(req_id, cons.buffer_pipeline.anchor_locks)
        cons.pop_prefetch_loaded_tokens(req_id)
        staged = cons.buffer_pipeline.staged_prefetches[req_id]
        self.assertEqual(staged.matched_len, len(seq))
        self.assertEqual(staged.num_tokens, 0, "resident FULL must not be re-fetched")
        window_tokens = sum(
            len(t.host_indices)
            for t in staged.aux_xfers
            if t.name == PoolName.SWA and t.host_indices is not None
        )
        self.assertGreater(window_tokens, live_tail, "window not partly tombstoned")
        full_avail0 = cons_alloc.full_available_size()
        mapping_before = cons_alloc.full_to_swa_index_mapping[masked_full].clone()
        dispatched = {}

        def capture_swa_mapping():
            load = cons.cache_controller.load_queue[-1]
            swa_xfer = next(t for t in load.pool_transfers if t.name == PoolName.SWA)
            dispatched["device"] = swa_xfer.device_indices.clone()
            full_window = masked_full[-window_tokens:]
            old_window = mapping_before[-window_tokens:]
            expected = old_window.clone()
            expected[old_window <= 0] = swa_xfer.device_indices
            self.assertTrue(
                torch.equal(
                    cons_alloc.full_to_swa_index_mapping[full_window],
                    expected,
                ),
                "admitted request did not fill the missing SWA mappings",
            )

        spliced, req = self._consume_staged_prefetch(
            cons,
            req_id,
            prefix_len=0,
            on_dispatched=capture_swa_mapping,
            return_req=True,
        )
        self.assertEqual(int(spliced.numel()), 0)
        self.assertEqual(req.prefix_len, len(seq), "FULL rematch was lost")
        self.assertEqual(len(dispatched["device"]), window_tokens - live_tail)
        self.assertEqual(cons_alloc.full_available_size(), full_avail0)
        k, v = self._snapshot_full_kv(cons_alloc, masked_full)
        self.assertTrue(torch.equal(k, masked_kv[0]))
        self.assertTrue(torch.equal(v, masked_kv[1]))
        self.assertTrue(
            torch.equal(cons_alloc.full_to_swa_index_mapping[tail_full], tail_swa),
            "live tail changed while dropping the staged overlap",
        )
        self.assertEqual(cons.buffer_pipeline.ongoing_buffer_load_back, {})
        self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        after = match_prefix(cons, seq)
        self.assertEqual(after.full_kv_hit_length, len(seq))
        self.assertEqual(after.device_prefix_len, len(seq))
        # Only the fetched SWA tail made the resident FULL reusable.
        self.assertEqual((req.storage_hit_start, req.storage_hit_length), (0, len(seq)))
        cons.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_not_called()
        cons.sanity_check()

    def _assert_sibling_head_is_trimmed(self, storage_dir, seq, window_sized: bool):
        _, (expected_k, expected_v) = self._produce_buffer_l3(
            storage_dir, seq, marker=9
        )

        cons, cons_alloc, cons_rtp = self._build_buffer_hicache_fixture(storage_dir)
        avail0 = self._host_avail_sizes(cons)

        req_id = CacheRequestHandle(f"growth-trim-{len(seq)}", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)

        # Sibling publishes the first page of the span while the hold parks.
        head = seq[: self.cfg.page_size]
        self._insert(cons, cons_alloc, cons_rtp, head)
        sib = match_prefix(cons, head)
        self.assertEqual(sib.device_prefix_len, len(head))
        self._fill_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node), marker=3
        )
        head_k, head_v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node)
        )

        if window_sized:
            # FULL shrank below a window; the complete SWA staging must survive.
            staged = cons.buffer_pipeline.staged_prefetches[req_id]
            aux_max = max(
                (
                    t.host_indices.numel()
                    for t in staged.aux_xfers
                    if t.host_indices is not None
                ),
                default=0,
            )
            self.assertGreater(aux_max, len(seq) - len(head))

        # The surfaced host hit is the splice-able tail, not the full span.
        req = SimpleNamespace(
            rid=req_id.rid,
            cache_request_handle=req_id,
            prefix_len=sib.device_prefix_len,
            kv=SimpleNamespace(cache_protected_len=sib.device_prefix_len),
        )
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual(req.host_hit_length, len(seq) - len(head))
        self.assertTrue(cons.buffer_pipeline.has_staged(req_id))

        spliced = self._consume_staged_prefetch(cons, req_id, prefix_len=len(head))
        self.assertEqual(int(spliced.numel()), len(seq) - len(head))

        m = match_prefix(cons, seq)
        self.assertEqual(m.device_prefix_len, len(seq))
        self.assertTrue(
            torch.equal(
                cons.path_device_indices(m.last_device_node)[: len(head)],
                cons.path_device_indices(sib.last_device_node),
            )
        )
        k, v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(m.last_device_node)[len(head) :]
        )
        self.assertTrue(torch.equal(k, expected_k[len(head) :]))
        self.assertTrue(torch.equal(v, expected_v[len(head) :]))
        hk, hv = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(m.last_device_node)[: len(head)]
        )
        self.assertTrue(torch.equal(hk, head_k))
        self.assertTrue(torch.equal(hv, head_v))

        # The whole bounce (trimmed head included) frees at the ack.
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        cons.sanity_check()

    def _patch_tp_prefetch_sync(self, cache, drop_swa: bool):
        """Fake all_reduce so _reduce_prefetch_ack runs the tp>1 path."""
        import torch.distributed as dist

        cc = cache.cache_controller

        # Fake _reduce_prefetch_ack to drop SWA pool_hits when drop_swa is True.
        def fake_reduce(ack):
            if drop_swa and ack.pool_hits is not None:
                if PoolName.SWA.value in ack.pool_hits:
                    ack.pool_hits[PoolName.SWA.value] = 0

        p_reduce = mock.patch.object(
            cc, "_reduce_prefetch_ack", side_effect=fake_reduce
        )
        p_reduce.start()
        self.addCleanup(p_reduce.stop)

        # Make all_reduce no-op.  The real all_reduce fails in unit tests with tp_world_size=2.
        p_dist = mock.patch.object(dist, "all_reduce", return_value=None)
        p_dist.start()
        self.addCleanup(p_dist.stop)

    def _swa_host_on_path(self, cache, seq):
        m = match_prefix(cache, seq)
        node = m.last_host_node
        while node != cache.root_node_handle():
            if node_host_value(cache, node, ComponentType.SWA) is not None:
                return True
            node = node_parent(cache, node)
        return False

    def _l3_produce(self, storage_dir, seq, marker=None):
        """Producer tree: insert seq and push it to L3. With ``marker``, fill its
        KV first and return the snapshot a consumer should read back."""
        prod, prod_alloc, prod_rtp = build_fixture(self.cfg)
        self._init_hicache(
            prod, storage_backend="file", storage_dir=storage_dir, prefetch_threshold=1
        )
        self._insert(prod, prod_alloc, prod_rtp, seq)
        leaf = match_prefix(prod, seq).last_device_node
        expected = None
        if marker is not None:
            indices = prod.path_device_indices(leaf)
            self._fill_full_kv(prod_alloc, indices, marker=marker)
            expected = self._snapshot_full_kv(prod_alloc, indices)
        self._backup_node(prod, leaf)
        self._write_path_to_l3(prod, leaf)
        self._flush_l3_backups(prod)
        return expected

    def _l3_consumer(self, storage_dir):
        fixture = build_fixture(self.cfg)
        self._init_hicache(
            fixture[0],
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
        )
        return fixture

    def _consume_prefetch(self, cons, seq, req_id):
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._run_prefetch_to_completion(cons, req_id)

    def _setup_swa_tp_prefetch(self):
        """Skip non-SWA fixtures; produce one full SWA window+1 page to L3.

        Returns (storage_dir, seq) or None when this fixture cannot exercise
        SWA L3 prefetch (then the caller skips).
        """
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("SWA-only fixture required")
        if self._skip_unsupported_hicache_test():
            return None
        sw_pages = (
            self.cfg.sliding_window_size + self.cfg.page_size - 1
        ) // self.cfg.page_size
        seq = self._make_seq(1, sw_pages + 1)

        storage_dir = self._make_storage_dir()
        self._l3_produce(storage_dir, seq)

        # Baseline (single rank) must actually adopt SWA, else the TP assertions
        # below would be vacuous -> skip.
        base, _, _ = self._l3_consumer(storage_dir)
        self._consume_prefetch(base, seq, CacheRequestHandle("base", 0))
        if not self._swa_host_on_path(base, seq):
            self.skipTest("fixture does not exercise SWA L3 prefetch")
        return storage_dir, seq


class UnifiedRadixPrefetchCorruptionFixtures:
    def _init_hicache(self, cache, *, write_policy="write_through"):
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

        orig_get_mha_host_pool_cls = assembler.get_mha_host_pool_cls

        def get_mha_host_pool_cls_wrapper(device_pool):
            host_pool_cls = orig_get_mha_host_pool_cls(device_pool)

            def kv_host_pool_wrapper(*args, **kwargs):
                kwargs["pin_memory"] = False
                return host_pool_cls(*args, **kwargs)

            return kv_host_pool_wrapper

        patcher = mock.patch.object(
            assembler,
            "get_mha_host_pool_cls",
            side_effect=get_mha_host_pool_cls_wrapper,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_write_policy=write_policy,
        )
        server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, self.cfg.page_size)
        set_global_server_args_for_scheduler(server_args)
        cache.init_hicache(server_args, cache.cache_init_params)
        self.addCleanup(drop_hicache_atexit_pin, cache)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0

    def _insert_device(self, cache, allocator, ids):
        """Insert a device-only chain (no auto-backup) and return its leaf id."""
        key = RadixKey(array("q", ids)).page_aligned(self.ps)
        val = allocator.alloc(len(key))
        self.assertIsNotNone(val)
        val = val.to(dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=val, prev_prefix_len=0))
        return cache.match_prefix(MatchPrefixParams(key=key)).last_device_node

    def _attach_host_child(self, cache, parent_id, start_token):
        """Mimic a prefetch commit: hang a backed-up host chain under `parent_id`."""
        ps = self.ps
        child_key = RadixKey(
            array("q", list(range(start_token, start_token + 2 * ps)))
        ).page_aligned(ps)
        host_idx = cache.cache_controller.mem_pool_host.alloc(len(child_key))
        self.assertIsNotNone(host_idx, "host pool alloc failed")
        host_idx = host_idx.to(dtype=torch.int64)
        hashes = [f"h{i}" for i in range(len(child_key) // ps)]
        res = cache.tree_core.insert_host(parent_id, child_key, host_idx, hashes)
        if res.host_insert_dropped:
            cache.cache_controller.mem_pool_host.free(host_idx)
        return res.inserted_host_node


class UnifiedRadixCacheStorageAttachBackfillFixtures:
    def _build_two_level_tree(self, *, storage_on_from_the_start: bool):
        """A parent node plus a child extending it, built with storage on or off."""
        cache, allocator, _ = build_fixture(self.cfg)
        cache.enable_storage = storage_on_from_the_start
        for tokens in (self.prefix_tokens, self.prefix_tokens + self.suffix_tokens):
            value = allocator.alloc(len(tokens))
            self.assertIsNotNone(value)
            cache.insert(InsertParams(key=RadixKey(tokens), value=value))
        return cache

    @staticmethod
    def _hashes_by_token_ids(cache):
        """Every non-root node's token ids mapped to its hash chain."""
        hashes = {}
        root = cache.root_node_handle()
        for node in cache.tree_core.get_all_node_ids():
            if node == root:
                continue
            hash_values = cache.tree_core.get_hash_values(node)
            hashes[tuple(node_token_ids(cache, node))] = (
                hash_values
                if cache.tree_core.get_last_hash_value(node) is not None
                else None
            )
        return hashes


class AnchorLockOutcomePolicyFixtures:
    def _make_pipeline(self, cache, cap_tokens=10_000):
        from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline.anchor_locks = {}
        pipeline.anchor_locked_tokens_ = 0
        pipeline.anchor_lock_cap_tokens = cap_tokens
        pipeline._anchor_lock_cap_skips = 0
        pipeline._prefetch_prefix_ctx = {self._REQ: (list(self._PREFIX), None, None)}
        pipeline._cache = cache
        return pipeline

    def _make_cache(self, live_match_len, pinned_len=None):
        cache = mock.MagicMock()
        cache.tree_core.is_eagle = False
        cache.tree_core.match_full_device_prefix.return_value = (
            live_match_len,
            99,
            live_match_len if pinned_len is None else pinned_len,
        )
        return cache
