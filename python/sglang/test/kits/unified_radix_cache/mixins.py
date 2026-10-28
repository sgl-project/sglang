"""Shared test steps for the UnifiedRadixCache unit tests: building requests,
backing up, evicting, and loading back nodes."""

import json
from array import array
from typing import Optional

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
    supports_hicache,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    drop_hicache_atexit_pin,
    match_prefix,
    node_children,
    node_device_value,
    node_host_value,
    node_parent,
    node_token_ids,
    write_backup,
)


class UnifiedRadixCacheFixtures:
    cfg: CacheConfig
    _rid: int = 0

    def _make_req(self, req_to_token_pool):
        sp = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=sp,
        )
        self._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _apply_match_to_req(self, req, match):
        req.prefix_len = match.device_prefix_len
        req.last_node = match.last_device_node
        req.last_host_node = match.last_host_node
        req.best_match_node = match.best_match_node
        req.host_hit_length = match.host_hit_length
        req.swa_host_hit_length = match.swa_host_hit_length
        req.mamba_host_hit_length = match.mamba_host_hit_length

    def _make_seq(self, start: int, num_pages: int) -> list[int]:
        """Page-aligned token sequence of num_pages pages."""
        page_size = self.cfg.page_size
        return list(range(start, start + num_pages * page_size))

    def _alloc(self, allocator, need_size):
        if not (self.cfg.has_swa and self.cfg.page_size > 1):
            return allocator.alloc(need_size)

        # SWATokenToKVPoolAllocator.alloc() asserts page_size == 1, and
        # alloc_extend() requires batch tensors unsuitable for unit tests.
        # Replicate alloc_extend's core logic here.
        ps = self.cfg.page_size
        aligned = ((need_size + ps - 1) // ps) * ps
        if aligned > allocator.full_attn_allocator.available_size():
            return None
        if aligned > allocator.swa_attn_allocator.available_size():
            return None
        full_indices = allocator.full_attn_allocator.alloc(aligned)
        swa_indices = allocator.swa_attn_allocator.alloc(aligned)
        assert full_indices is not None and swa_indices is not None
        allocator.full_to_swa_index_mapping[full_indices] = swa_indices
        return full_indices[:need_size]

    def _make_inflight_req(
        self,
        cache,
        allocator,
        req_to_token_pool,
        input_ids,
        output_ids=(),
        *,
        allocated=False,
    ):
        """A request whose input and output KV sits in its own slots, committed
        but not yet in the tree. Returns the request and those slots."""
        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = array("q", input_ids)
        req.output_ids = array("q", output_ids)
        req.full_untruncated_fill_ids = array("q", [*input_ids, *output_ids])
        req.extend_end = len(req.full_untruncated_fill_ids)
        kv_indices = self._alloc(allocator, req.extend_end)
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, req.extend_end)), kv_indices
        )
        req.kv.kv_committed_len = req.extend_end
        if allocated:
            req.kv.kv_allocated_len = req.extend_end
        req.last_node = cache.root_node_handle()
        req.kv.cache_protected_len = 0
        req.lock = None
        req.extra_key = None
        return req, kv_indices

    def _insert(
        self,
        cache,
        allocator,
        req_to_token_pool,
        tokens,
        priority=0,
        *,
        extra_key=None,
        cache_salt=None,
    ):
        """Insert tokens, attaching mamba data when the config has mamba."""
        key = RadixKey(array("q", tokens), extra_key=extra_key, cache_salt=cache_salt)
        value = self._alloc(allocator, len(tokens))
        params = InsertParams(key=key, value=value[: len(key)], priority=priority)
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        return cache.insert(params)

    def _path_chain(self, cache, node):
        """Nodes from root to node, excluding root."""
        chain = []
        cur = node
        while cur != cache.root_node_handle():
            chain.append(cur)
            cur = node_parent(cache, cur)
        chain.reverse()
        return chain

    def _skip_unsupported_hicache_test(self):
        """Runtime skip for tests that call hicache helpers without ``requires_hicache``."""
        if not supports_hicache(self.cfg):
            self.skipTest("HiCache unit fixture does not support SWA + Mamba stacks")
        return False

    def _simulate_backup(self, cache, node):
        """Simulate D->H backup over the whole root->node path (parent-first)."""
        for ancestor in self._path_chain(cache, node):
            for ct in (ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA):
                if ct not in self.cfg.components:
                    continue
                value = node_device_value(cache, ancestor, ct)
                if value is not None and node_host_value(cache, ancestor, ct) is None:
                    cache.tree_core.set_component_host_value_raw(
                        ancestor, ct, value.clone()
                    )
            # A real backup registers duplicate tracking at its ack.
            cache.tree_core.update_duplicate_tracking(ancestor)

    def _simulate_backup_tree(self, cache):
        """Backup all non-root nodes (simulates write-through)."""
        root = cache.root_node_handle()
        stack = [root]
        while stack:
            node = stack.pop()
            if node != root:
                self._simulate_backup(cache, node)
            stack.extend(node_children(cache, node))

    def _init_hicache(
        self,
        cache,
        *,
        write_policy: str = "write_through",
        storage_backend: Optional[str] = None,
        storage_dir: Optional[str] = None,
        prefetch_threshold: Optional[int] = None,
        prefetch_policy: str = "wait_complete",
        host_memory_mode: str = "cache",
        storage_extra: Optional[dict] = None,
        context_length: Optional[int] = None,
    ):
        storage_extra_config = None
        if storage_backend == "file":
            from sglang.srt.runtime_context import get_parallel

            # The file-backend storage config records TP/PP rank/size. These unit
            # fixtures run without initializing distributed parallel state, so
            # force the local single-rank topology the fixture represents.
            parallel_override = get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1
            )
            parallel_override.__enter__()
            self.addCleanup(parallel_override.__exit__, None, None, None)

            assert storage_dir is not None, "file backend needs a storage_dir"
            # HiCacheFile reads the directory from this env var.
            cm = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(storage_dir)
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
            extra = {}
            if prefetch_threshold is not None:
                extra["prefetch_threshold"] = prefetch_threshold
            if storage_extra:
                extra.update(storage_extra)
            storage_extra_config = json.dumps(extra) if extra else None

        # See _init_hicache: production kernel IO backend, pinned pools.
        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="kernel",
            hicache_write_policy=write_policy,
            hicache_storage_backend=storage_backend,
            hicache_storage_backend_extra_config=storage_extra_config,
            hicache_storage_prefetch_policy=prefetch_policy,
            hicache_host_memory_mode=host_memory_mode,
            context_length=context_length,
        )
        # See build_fixture for why _mamba_cache_chunk_size is preset.
        server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, self.cfg.page_size)
        set_global_server_args_for_scheduler(server_args)
        cache.init_hicache(server_args, cache.cache_init_params)
        self.addCleanup(drop_hicache_atexit_pin, cache)
        self.addCleanup(cache.release_host_resources)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0
        if storage_backend is not None:
            # Unit fixtures size host/device pools equally, which makes the
            # production prefetch capacity limit (host - device) zero.  Keep the
            # L3 tests focused on storage round trips by allowing one fixture
            # worth of prefetch tokens.
            cache.cache_controller.prefetch_capacity_limit = max(
                cache.cache_controller.prefetch_capacity_limit,
                cache.cache_controller.mem_pool_host.size,
            )
            # Background prefetch/backup threads are daemon; stop them per-test.
            self.addCleanup(cache.cache_controller._stop_storage_threads)

    def _build_hicache_fixture(self, tree_core_backend: Optional[str] = None):
        fixture = build_fixture(self.cfg, tree_core_backend=tree_core_backend)
        cache, _, _ = fixture
        self._init_hicache(cache)
        return fixture

    def _backup_node(self, cache, node):
        # Parent-first backup over the whole path: one insert can span several
        # nodes, so a single-node backup would leave an unbacked ancestor.
        backed_up = 0
        for ancestor in self._path_chain(cache, node):
            if cache.tree_core.is_backuped(ancestor):
                continue
            backed_up = write_backup(cache, ancestor, write_back=True)
            self.assertGreater(backed_up, 0)
        cache.writing_check(write_back=True)
        self.assertTrue(cache.tree_core.is_backuped(node))
        return backed_up

    def _backup_tree(self, cache):
        root = cache.root_node_handle()
        stack = [root]
        while stack:
            node = stack.pop()
            children = node_children(cache, node)
            stack.extend(reversed(children))
            if node != root:
                self._backup_node(cache, node)

    def _load_back_node(self, cache, node, req=None):
        loaded = cache.load_back(node, req=req)
        self.assertTrue(loaded)
        producer_id = cache.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for ack in list(cache.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cache.loading_check()
        return node_device_value(cache, node, ComponentType.FULL)

    def _get_full_kv_pool(self, allocator):
        kv_pool = allocator.get_kvcache()
        return getattr(kv_pool, "full_kv_pool", kv_pool)

    def _fill_full_kv(self, allocator, indices, marker):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        # `buf[indices]` is an advanced-index copy — assign, never `fill_()`.
        kv_pool.get_key_buffer(layer_id)[indices] = marker
        kv_pool.get_value_buffer(layer_id)[indices] = marker + 1

    def _snapshot_full_kv(self, allocator, indices):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        return (
            kv_pool.get_key_buffer(layer_id)[indices].float().cpu().clone(),
            kv_pool.get_value_buffer(layer_id)[indices].float().cpu().clone(),
        )

    def _build_chain_pages(self, cache, allocator, req_to_token_pool, num_pages):
        """Insert an incremental chain of single-page extensions.

        Returns the chain root-to-leaf. Length may differ from num_pages
        when the radix tree merges or splits nodes.
        """
        seq: list[int] = []
        for i in range(num_pages):
            seq = seq + self._make_seq(1000 * (i + 1), 1)
            self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        chain: list = []
        cur = m.last_device_node
        while cur != cache.root_node_handle():
            chain.append(cur)
            cur = node_parent(cache, cur)
        chain.reverse()
        return chain

    def _release_ongoing_load_back_locks(self, cache):
        for node, lock_params, host_lock_params in list(
            cache.ongoing_load_back.values()
        ):
            cache.dec_lock_ref(node, lock_params)
            cache.dec_host_lock_ref(node, host_lock_params)
        cache.ongoing_load_back.clear()

    def _finish_pending_loads(self, cache):
        producer_id = cache.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for ack in list(cache.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cache.loading_check()

    def _match_tokens_for_chain(self, cache, chain):
        tokens: list[int] = []
        for node in chain:
            tokens.extend(node_token_ids(cache, node))
        return tokens

    def _set_aux_host_tombstone(self, cache, node, component_type):
        value = node_device_value(cache, node, component_type)
        self.assertIsNotNone(value)
        if node_host_value(cache, node, component_type) is None:
            cache.tree_core.set_component_host_value_raw(
                node, component_type, value.clone()
            )
        cache.tree_core.set_component_device_value_raw(node, component_type, None)
        if cache.tree_core.is_node_in_device_lru(node, component_type):
            cache.tree_core.remove_node_from_device_lru(node, component_type)
        cache.tree_core.insert_node_into_host_lru(node, component_type)
        cache.tree_core.set_component_evictable_size(
            component_type,
            cache.tree_core.component_evictable_size(component_type) - len(value),
        )

    def _swa_finalize_setup(self):
        """Build a SWA chain long enough to fill at least the window
        plus one extra page, and host-back every node so we can flip
        SWA tombstones at will."""
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        window_pages = (sw + ps - 1) // ps
        chain_pages = window_pages + 2
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        chain = self._build_chain_pages(
            cache, allocator, req_to_token_pool, chain_pages
        )
        if len(chain) <= window_pages:
            self.skipTest("chain collapsed below the window length")
        self._simulate_backup_tree(cache)
        return cache, allocator, req_to_token_pool, chain, window_pages

    def _swa_anchor_chain_tokens(self, num_pages: int) -> list[int]:
        """Reproduce the token sequence used by _build_chain_pages."""
        tokens: list[int] = []
        for i in range(num_pages):
            tokens.extend(self._make_seq(1000 * (i + 1), 1))
        return tokens

    def _swa_anchor_setup(self):
        """Chain layout (root-to-leaf):

            ... top padding ...
            chain[-(window_pages + 1)] = N : SWA tombstone
            chain[-(window_pages)]     = Y : SWA host-only, FULL host-backed
            chain[-(window_pages - 1)..-2] : SWA device, FULL.host=None
            chain[-1]                  = X : SWA device, FULL.host=None

        Anchor=X has window_pages of device pages below N, so the SWA
        load_back walker accumulates n_swa >= sliding_window_size and
        exits before reaching N. Anchor=Y has only 1 page and walks into
        N's tombstone.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        if ps >= sw:
            self.skipTest("test scenario requires ps < sw")
        window_pages = (sw + ps - 1) // ps
        chain_pages = window_pages + 3
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(
            cache, allocator, req_to_token_pool, chain_pages
        )
        if len(chain) < chain_pages:
            self.skipTest("chain too short")
        self._backup_tree(cache)

        x = chain[-1]
        y = chain[-window_pages]
        n = chain[-(window_pages + 1)]

        cache.tree_core.set_component_device_value_raw(n, ComponentType.SWA, None)
        cache.tree_core.set_component_host_value_raw(n, ComponentType.SWA, None)
        cache.tree_core.set_component_device_value_raw(y, ComponentType.SWA, None)
        # Strip FULL.host on X + intermediates so last_host_node walks past
        # them to Y. Y.FULL untouched preserves the leaf-up evict invariant.
        for node in chain[-(window_pages - 1) :]:
            cache.tree_core.set_component_host_value_raw(node, ComponentType.FULL, None)

        tokens = self._swa_anchor_chain_tokens(len(chain))
        return cache, chain, n, y, x, tokens
