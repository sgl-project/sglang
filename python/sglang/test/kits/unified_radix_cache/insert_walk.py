from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    TreeLock,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_children,
    node_device_lock_ref,
    node_host_lock_ref,
    node_parent,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures


class ResumableInsertWalkSWAFixtures(UnifiedRadixCacheFixtures):
    def _recover_swa_tombstone_with_tracking(self, *, lock_full: bool):
        cache, allocator, _ = build_fixture(self.cfg)
        seq = list(range(1, self.cfg.sliding_window_size + 1))
        key = RadixKey(array("q", seq))
        evicted = self._alloc(allocator, len(seq))
        # Window eviction already released the peers below the floor.
        allocator.free_swa(evicted)
        cache.insert(
            InsertParams(
                key=key,
                value=evicted,
                component_evicted_seqlens={ComponentType.SWA: len(seq)},
            )
        )
        (leaf,) = node_children(cache, cache.root_node_handle())
        lock_result = cache.inc_lock_ref(leaf) if lock_full else None
        try:
            result = cache.insert(
                InsertParams(
                    key=key,
                    value=self._alloc(allocator, len(seq)),
                    track_adopted_ranges=True,
                )
            )
        finally:
            if lock_result is not None:
                cache.dec_lock_ref(leaf, lock_result.to_dec_params())
        return result, len(seq)


class SegmentLockProtocolFixtures(UnifiedRadixCacheFixtures):
    @staticmethod
    def _swa_ref(cache, node_id):
        return node_device_lock_ref(cache, node_id, ComponentType.SWA)

    @staticmethod
    def _segment(cache, leaf_id, window):
        """Node ids from leaf up to the position-based window boundary."""
        nodes, covered, cur = [], 0, leaf_id
        while not cache.tree_core.is_root(cur) and covered < window:
            nodes.append(cur)
            covered += cache.tree_core.get_node_key_length(cur)
            cur = node_parent(cache, cur)
        return nodes

    @staticmethod
    def _full_path(cache, leaf_id):
        """Non-root node ids protected by a Full device lock."""
        nodes, cur = [], leaf_id
        while not cache.tree_core.is_root(cur):
            nodes.append(cur)
            cur = node_parent(cache, cur)
        return nodes

    def _match_leaf(self, cache, seq):
        m = match_prefix(cache, seq)
        return m.last_device_node

    @staticmethod
    def _deepest(cache):
        """Structurally deepest node — bypasses SWA match validation, which
        never adopts a holed window (simulates the stale-relock drift case)."""
        node_id = cache.root_node_handle()
        while True:
            children = node_children(cache, node_id)
            if not children:
                return node_id
            node_id = children[0]

    def _assert_protocol_violation(self, fn, fragment):
        """Both cores report ownership violations as ordinary Python exceptions."""
        try:
            fn()
        except (AssertionError, RuntimeError) as exc:
            self.assertIn(fragment, str(exc))
        else:
            self.fail(f"protocol violation went unreported: {fragment}")

    def _locked_path_state(self, cache, allocator, leaf):
        """Every lock count on the leaf's path plus every free-slot count."""
        refs = [
            (node_device_lock_ref(cache, n, ct), node_host_lock_ref(cache, n, ct))
            for n in self._full_path(cache, leaf)
            for ct in self.cfg.components
        ]
        free_slots = [
            allocator.full_attn_allocator.available_size(),
            allocator.swa_attn_allocator.available_size(),
        ]
        if cache.cache_controller is not None:
            free_slots.append(cache.cache_controller.mem_pool_host.available_size())
        return refs, free_slots

    def _leaf_with_lock_path(self, *, hicache=False):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        if hicache:
            self._init_hicache(cache, write_policy="write_back")
        seq = self._make_seq(1, 2 * self.cfg.sliding_window_size)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        self.assertFalse(cache.tree_core.is_root(leaf))
        return cache, allocator, leaf

    def _assert_unanchored_release_rejected(
        self, cache, allocator, leaf, release, unlock
    ):
        """The anchor check runs before any lock count or free list changes."""
        before = self._locked_path_state(cache, allocator, leaf)
        self._assert_protocol_violation(release, "lock receipt anchored on node")
        if cache._tree_core_backend == "rust":
            # A Rust panic poisons the core; the crate's own test checks state.
            return
        self.assertEqual(self._locked_path_state(cache, allocator, leaf), before)
        unlock()
        cache.sanity_check()


class SegmentLockFuzzFixtures(UnifiedRadixCacheFixtures):
    def _lock_skips(self, rng):
        """The decode hold opts the Mamba lock out; exercise both receipts."""
        if self.cfg.has_mamba and rng.random() < 0.5:
            return (ComponentType.MAMBA,)
        return ()

    def _run_seed(self, seed: int, steps: int = 120):
        import random as _random

        rng = _random.Random(seed)
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        chains: list[list[int]] = []
        held: list[list] = []  # [node_id, receipt, released] entries

        for step in range(steps):
            op = rng.random()
            try:
                if op < 0.35 or not chains:
                    if chains and rng.random() < 0.6:
                        base = rng.choice(chains)
                        cut = rng.randrange(1, len(base) + 1)
                        seq = base[:cut] + self._make_seq(
                            1000 * (step + 1), rng.randrange(2, 12)
                        )
                    else:
                        seq = self._make_seq(1000 * (step + 1), rng.randrange(4, 20))
                    if allocator.available_size() < len(seq):
                        cache.evict(EvictParams(num_tokens=len(seq) * 2))
                    if allocator.available_size() < len(seq):
                        continue
                    swa_evict = rng.randrange(0, len(seq)) if rng.random() < 0.3 else 0
                    value = self._alloc(allocator, len(seq))
                    # Release the evicted prefix's SWA peers first, as the
                    # scheduler does before inserting a window-trimmed request.
                    allocator.free_swa(value[:swa_evict])
                    params = InsertParams(
                        key=RadixKey(array("q", seq)),
                        value=value,
                        component_evicted_seqlens={ComponentType.SWA: swa_evict},
                    )
                    if self.cfg.has_mamba:
                        req = self._make_req(req_to_token_pool)
                        params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
                    cache.insert(params)
                    chains.append(seq)
                elif op < 0.6:
                    seq = rng.choice(chains)
                    m = match_prefix(cache, seq)
                    node_id = m.last_device_node
                    if cache.tree_core.is_root(node_id):
                        continue
                    receipt = cache.inc_lock_ref(
                        node_id, skip_lock_components=self._lock_skips(rng)
                    )
                    held.append([node_id, receipt, False])
                elif op < 0.8 and held:
                    idx = rng.randrange(len(held))
                    node_id, receipt, released = held.pop(idx)
                    cache.dec_lock_ref(
                        node_id, receipt.to_dec_params(), skip_swa=released
                    )
                elif op < 0.9 and held:
                    idx = rng.randrange(len(held))
                    node_id, receipt, released = held[idx]
                    if (
                        released
                        or receipt.component_lock_uuids[ComponentType.SWA] is None
                    ):
                        continue
                    cache.release_swa(TreeLock(node_id, receipt.to_dec_params()))
                    held[idx][2] = True
                else:
                    cache.evict(
                        EvictParams(
                            num_tokens=rng.randrange(0, 32),
                            swa_num_tokens=rng.randrange(0, 32),
                            mamba_num=rng.randrange(0, 4) if self.cfg.has_mamba else 0,
                        )
                    )
            except AssertionError:
                raise
            cache.sanity_check()

        # Drain remaining locks; the tree must come back exactly balanced.
        for node_id, receipt, released in held:
            cache.dec_lock_ref(node_id, receipt.to_dec_params(), skip_swa=released)
        cache.sanity_check()


class StreamingSessionLockLifecycleFixtures:
    def _session(self, session_id):
        return SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
        )

    def _turn(self, tokens, session):
        req = Req(
            rid="turn",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.session = session
        req.extra_key = None
        req.output_ids = array("q")
        req.refresh_fill_ids()
        return req

    def _admitted_turn(self, cache, allocator, pool, tokens, session):
        """A first turn with its own row and the root lock admission took."""
        req = self._turn(tokens, session)
        pool.alloc([req])
        pool.write(
            (req.kv.req_pool_idx, slice(0, len(tokens))),
            allocator.alloc(len(tokens)),
        )
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.last_node = cache.root_node_handle()
        req.lock = cache.lock(req.last_node)
        return req

    def _later_turn(self, cache, allocator, pool):
        """A finished first turn, then a second turn on the slot's record."""
        session = self._session("s")
        first = self._admitted_turn(cache, allocator, pool, list(range(1, 9)), session)
        cache.maybe_hand_to_session(first)
        cache.checkpoint(first, up_to=8)
        first.finished_reason = FINISH_LENGTH(length=0)
        self.assertTrue(cache.session.try_cache_finished_req(first))

        tokens = list(range(1, 13))
        req = self._turn(tokens, session)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        req.last_node = match.last_device_node
        pool.write((req.kv.req_pool_idx, slice(8, 12)), allocator.alloc(4))
        req.kv.kv_committed_len = 12
        req.kv.kv_allocated_len = 12
        cache.maybe_hand_to_session(req)
        return cache.session.slots["s"], req
