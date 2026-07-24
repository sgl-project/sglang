"""Component-level tests for the window-granular strict SWA HiCache capture.

Exercises the real methods with a minimal fake host pool and small CPU bf16
tensors — no GPU / full unified pool needed.

Covers:
  * capture geometry + byte identity: stride == page, one window [B-win, B) per
    page boundary keyed (rid, B); the page's first half is never captured;
    per-request offset + per-layer indexing correct.
  * binding: a node's SWA host_value is the single window at its end boundary
    (len == win), not the node's full value; falls back when the window is
    missing.
  * restore: LOAD_BACK maps only the window's (last n_tokens) full indices to
    the restored SWA slots.
"""

import types
import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.swa_component import (
    SWAComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)

SWA = ComponentType.SWA


FULL = BASE_COMPONENT_TYPE


class _FakeHostPool:
    def __init__(
        self, *, win, head_dim, num_pages, layers, elem=2, device_buffers=None
    ):
        # `win` here is the host-page row count == device ring size (slot_page_size).
        self.slot_page_size = win
        self.item_bytes = win * head_dim * elem
        self.layer_num = layers
        self.data_refs = [
            torch.zeros(num_pages, self.item_bytes, dtype=torch.uint8)
            for _ in range(layers)
        ]
        # Only decode-source capture consults device_buffers; prefill leaves None.
        self.device_buffers = device_buffers
        self._capture_staging = {}
        self._num_pages = num_pages
        self._next_page = 0
        self.freed = []
        # H2 capture-done handshake probe: decode-source capture must record the
        # completion event AFTER its D2H copies so cross-stream consumers can
        # gate on it. Count calls; gpu_device None keeps the real event helper a
        # no-op if ever routed through the real pool.
        self.gpu_device = None
        self._capture_done_calls = 0

    def record_capture_done(self, stream=None):
        self._capture_done_calls += 1

    def alloc(self, n):
        assert n == self.slot_page_size
        if self._next_page >= self._num_pages:
            return None
        p = self._next_page
        self._next_page += 1
        start = p * self.slot_page_size
        return torch.arange(start, start + n, dtype=torch.int64)

    def free(self, idx):
        self.freed.append(idx)


class _FakePool:
    def __init__(self, host_pool, win, start_layer=0, ring=None):
        # `win` is the sliding window; `ring` (>= win) is the device ring size,
        # which carries speculative slack (ring = window + spec_extra). Default
        # ring == win reproduces the no-speculation geometry.
        self._swa_host_pool = host_pool
        self.unified_swa_ring_size = ring if ring is not None else win
        self.unified_swa_window = win
        self.start_layer = start_layer


def _fake_fb(ext, seqs, rids):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        extend_seq_lens_cpu=ext,
        seq_lens_cpu=torch.tensor(seqs),
        req_pool_indices=torch.tensor(rids),
    )


def _cd(value=None, host_value=None):
    return types.SimpleNamespace(value=value, host_value=host_value)


def _fake_fb_decode(seqs, rids):
    return types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(
            is_decode_or_idle=lambda: True, is_extend=lambda: False
        ),
        seq_lens_cpu=torch.tensor(seqs),
        req_pool_indices=torch.tensor(rids),
        batch_size=len(seqs),
    )


class TestSwaBindWindow(unittest.TestCase):
    """Bind the single window at node_end; fall back when it is absent."""

    def _fake_self(self, host):
        calls = []
        return (
            types.SimpleNamespace(
                _swa_kv_pool_host=host,
                _capture_rid=5,
                component_type=SWA,
                _attach_swa_host_value=lambda node, hv: calls.append(hv),
            ),
            calls,
        )

    def test_binds_single_window_not_node_length(self):
        win, head_dim = 2, 4
        host = _FakeHostPool(win=win, head_dim=head_dim, num_pages=4, layers=1)
        # stage a window tile at boundary B=4 keyed (rid=5, 4)
        tile = torch.arange(10, 10 + win, dtype=torch.int64)
        host._capture_staging[(5, 4)] = tile
        me, calls = self._fake_self(host)
        # node covers one page [0,4): SWA value length == page == 4 (256-analogue)
        node = types.SimpleNamespace(
            component_data={SWA: _cd(value=torch.arange(4), host_value=None)}
        )
        SWAComponent._bind_captured_swa_host(me, node, swa_start=0)
        # Co-lifetime: bind stashes a PENDING page (attached later, together with
        # Full host_value, via the coordinated BACKUP_HOST), not host_value now.
        pending = getattr(node, "_swa_pending_host", None)
        self.assertIsNotNone(pending)
        self.assertEqual(len(pending), win)  # window (2), not node length (4)
        self.assertTrue(torch.equal(pending, tile.to(torch.int64)))
        self.assertEqual(len(calls), 0)  # _attach deferred, not called at bind
        # tile consumed from staging
        self.assertNotIn((5, 4), host._capture_staging)

    def test_missing_window_is_i6_noop(self):
        win, head_dim = 2, 4
        host = _FakeHostPool(win=win, head_dim=head_dim, num_pages=4, layers=1)
        me, calls = self._fake_self(host)
        node = types.SimpleNamespace(
            component_data={SWA: _cd(value=torch.arange(4), host_value=None)}
        )
        SWAComponent._bind_captured_swa_host(me, node, swa_start=0)
        self.assertEqual(calls, [])  # nothing bound
        self.assertIsNone(getattr(node, "_swa_pending_host", None))


class TestSwaRestoreWindowMapping(unittest.TestCase):
    """infra: LOAD_BACK maps only the window's (last n_tokens) full indices."""

    def test_maps_only_window_full_indices(self):
        win = 2
        mapping_calls = []
        restore_calls = []
        allocator = types.SimpleNamespace(
            set_full_to_swa_mapping=lambda full, swa: mapping_calls.append(
                (full.clone(), swa.clone())
            )
        )
        me = types.SimpleNamespace(
            component_type=SWA,
            cache=types.SimpleNamespace(token_to_kv_pool_allocator=allocator),
            _restore_device_value=lambda n, v: restore_calls.append(v.clone()),
        )
        me._gather_window_full_indices = (
            lambda n, nt: SWAComponent._gather_window_full_indices(me, n, nt)
        )
        # node: full value length 4 (page node), SWA host_value == window (2)
        full_val = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node = types.SimpleNamespace(
            component_data={
                SWA: _cd(value=None, host_value=torch.tensor([0, 0])),
                FULL: _cd(value=full_val),
            }
        )
        device_indices = torch.tensor([700, 701], dtype=torch.int64)
        xfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([0, 0]),
            device_indices=device_indices,
            nodes_to_load=[node],
        )
        SWAComponent.commit_hicache_transfer(
            me, node, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )
        self.assertEqual(len(mapping_calls), 1)
        mapped_full, mapped_swa = mapping_calls[0]
        # only the LAST win (=2) full indices [102,103] are mapped
        self.assertTrue(torch.equal(mapped_full, full_val[-win:]))
        self.assertTrue(torch.equal(mapped_swa, device_indices))
        self.assertEqual(len(restore_calls), 1)
        self.assertTrue(torch.equal(restore_calls[0], device_indices))


class TestSwaRestoreSplitWindow(unittest.TestCase):
    """R.1 (Phase 4-prime.R): after a node split, a child shorter than the
    sliding window still owns the whole window host_value [B-win, B). Restore
    must gather the window full indices across the child AND its ancestors (in
    token order), not just the child own (shorter) full value. Regression for
    the set_full_to_swa_mapping length-mismatch assert."""

    def test_window_spans_parent_and_child(self):
        mapping_calls = []
        restore_calls = []
        allocator = types.SimpleNamespace(
            set_full_to_swa_mapping=lambda full, swa: mapping_calls.append(
                (full.clone(), swa.clone())
            )
        )
        root = types.SimpleNamespace(component_data={}, parent=None)
        me = types.SimpleNamespace(
            component_type=SWA,
            cache=types.SimpleNamespace(
                token_to_kv_pool_allocator=allocator, root_node=root
            ),
            _restore_device_value=lambda n, v: restore_calls.append(v.clone()),
        )
        me._gather_window_full_indices = (
            lambda n, nt: SWAComponent._gather_window_full_indices(me, n, nt)
        )
        # parent holds full tokens [B-4, B-2); child holds [B-2, B). The child
        # keeps the whole win=4 window host_value (parent.host_value is None
        # after redistribute_on_node_split).
        parent = types.SimpleNamespace(
            parent=root,
            component_data={
                SWA: _cd(value=None, host_value=None),
                FULL: _cd(value=torch.tensor([100, 101], dtype=torch.int64)),
            },
        )
        child = types.SimpleNamespace(
            parent=parent,
            component_data={
                SWA: _cd(value=None, host_value=torch.tensor([0, 0, 0, 0])),
                FULL: _cd(value=torch.tensor([102, 103], dtype=torch.int64)),
            },
        )
        device_indices = torch.tensor([700, 701, 702, 703], dtype=torch.int64)
        xfer = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.tensor([0, 0, 0, 0]),
            device_indices=device_indices,
            nodes_to_load=[child],
        )
        SWAComponent.commit_hicache_transfer(
            me, child, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )
        self.assertEqual(len(mapping_calls), 1)
        mapped_full, mapped_swa = mapping_calls[0]
        # window full indices, token order: parent tail [100,101] ++ child [102,103]
        self.assertTrue(
            torch.equal(
                mapped_full, torch.tensor([100, 101, 102, 103], dtype=torch.int64)
            )
        )
        self.assertTrue(torch.equal(mapped_swa, device_indices))
        self.assertEqual(len(restore_calls), 1)
        self.assertTrue(torch.equal(restore_calls[0], device_indices))


class TestStrictMatchValidatorI2Prime(unittest.TestCase):
    """R.6 / I2-prime: in strict mode a node whose SWA truth lives only in the
    per-request device ring (cd.value present, host_value None) must NOT extend
    a reuse match. The device ring is recycled across requests and is not a
    durable cross-request truth; the match must truncate so reuse restores from
    host or recomputes (I6). Best-effort (non-strict) keeps trusting device."""

    def _validator(self, strict):
        me = types.SimpleNamespace(
            sliding_window_size=4,
            component_type=SWA,
            _swa_kv_pool_host=object(),  # host pool wired => feature on, not device-only
            _strict_bit_exact=strict,
            cache=types.SimpleNamespace(cache_controller=object()),
        )
        return SWAComponent.create_match_validator(me)

    def _node(self, key_len, value, host_value):
        return types.SimpleNamespace(
            key=list(range(key_len)),
            backuped=True,
            evicted=False,
            component_data={SWA: _cd(value=value, host_value=host_value)},
        )

    def test_strict_device_only_node_truncates_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=[1, 2, 3, 4], host_value=None)
        self.assertFalse(v(node))

    def test_strict_host_node_extends_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=None, host_value=[0, 0, 0, 0])
        self.assertTrue(v(node))

    def test_strict_device_and_host_extends_match(self):
        v = self._validator(strict=True)
        node = self._node(4, value=[1, 2, 3, 4], host_value=[0, 0, 0, 0])
        self.assertTrue(v(node))

    def test_non_strict_trusts_device_value(self):
        v = self._validator(strict=False)
        node = self._node(4, value=[1, 2, 3, 4], host_value=None)
        self.assertTrue(v(node))


class TestReuseAnchorHostClamp(unittest.TestCase):
    """Mine 1: on cross-request reuse, the FULL device anchor
    (`best_match_device_value_len`) must not extend past the SWA host-gated
    `best_match_node` boundary, because the device-only validator trusts the
    per-request (recycled-across-requests) SWA device ring `cd.value` even
    without a durable host copy. `for_reuse=True` (scheduler reuse-match)
    clamps; `for_reuse=False` (self-match, e.g. `cache_unfinished_req`) must
    NOT clamp, or the I2' self-match invariant it depends on breaks and
    `cache_unfinished_req` trips its `new_prefix_len <= len(new_indices)`
    assertion.

    Builds a minimal 2-node chain root -> node_a -> node_b directly on real
    `UnifiedTreeNode`/`RadixKey` objects (matching the plain SimpleNamespace
    fake-component style used elsewhere in this file), with fake FULL/SWA
    components whose validators reproduce the real host-gated vs
    device-only-trusts-`cd.value` semantics from `TestStrictMatchValidatorI2Prime`
    above:
      * node_a: SWA has BOTH device value and durable host_value -> valid
        under every validator (host-gated boundary).
      * node_b: SWA has ONLY a device value (host_value=None) -> the
        per-request device ring; device-only validator still accepts it
        (I2'-required for self-match), but the host-gated validator rejects
        it (durable-copy requirement), so best_match_node stops at node_a.
    Both nodes are FULL-device-resident, so their FULL chunks are appended
    into `value` and `best_match_value_len` is well-defined at node_a.
    """

    PAGE_SIZE = 2

    def _make_full_component(self):
        def create_match_validator(match_device_only=False):
            return (
                lambda node: node.component_data[ComponentType.FULL].value is not None
            )

        return types.SimpleNamespace(
            component_type=ComponentType.FULL,
            create_match_validator=create_match_validator,
        )

    def _make_swa_component(self):
        def create_match_validator(match_device_only=False):
            if match_device_only:
                # I2'-required for self-match: trusts the per-request device
                # ring slot even without a durable host copy.
                return (
                    lambda node: node.component_data[ComponentType.SWA].value
                    is not None
                )
            # Host-gated (strict): only a durable host copy extends the match.
            return (
                lambda node: node.component_data[ComponentType.SWA].host_value
                is not None
            )

        return types.SimpleNamespace(
            component_type=ComponentType.SWA,
            create_match_validator=create_match_validator,
        )

    def _build_chain(self):
        """root -> node_a (device+host SWA) -> node_b (device-only SWA)."""
        tree_components = (ComponentType.FULL, ComponentType.SWA)
        root = UnifiedTreeNode(tree_components)

        node_a = UnifiedTreeNode(tree_components)
        node_a.parent = root
        node_a.key = RadixKey(array("q", [1, 2]))
        node_a.component_data[ComponentType.FULL].value = torch.tensor(
            [100, 101], dtype=torch.int64
        )
        node_a.component_data[ComponentType.SWA].value = torch.tensor(
            [900, 901], dtype=torch.int64
        )
        node_a.component_data[ComponentType.SWA].host_value = torch.tensor(
            [1, 1], dtype=torch.int64
        )
        root.children[(1, 2)] = node_a

        node_b = UnifiedTreeNode(tree_components)
        node_b.parent = node_a
        node_b.key = RadixKey(array("q", [3, 4]))
        node_b.component_data[ComponentType.FULL].value = torch.tensor(
            [102, 103], dtype=torch.int64
        )
        # Stale per-request SWA device ring slot: device value present, but
        # NOT durably backed on host (recycled across requests).
        node_b.component_data[ComponentType.SWA].value = torch.tensor(
            [902, 903], dtype=torch.int64
        )
        node_b.component_data[ComponentType.SWA].host_value = None
        node_a.children[(3, 4)] = node_b

        return root, node_a, node_b

    def _run_match_prefix_helper(self, for_reuse: bool):
        root, node_a, node_b = self._build_chain()
        fake_self = types.SimpleNamespace(
            root_node=root,
            page_size=self.PAGE_SIZE,
            # Non-None -> triggers the separate device-match validator path
            # (device_validators distinct from host-gated validators).
            cache_controller=object(),
            _components_tuple=(
                self._make_full_component(),
                self._make_swa_component(),
            ),
        )
        key = RadixKey(array("q", [1, 2, 3, 4]))
        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        ) = UnifiedRadixCache._match_prefix_helper(fake_self, key, for_reuse=for_reuse)
        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = torch.tensor([], dtype=torch.int64)
        return node_a, node_b, best_match_node, best_match_device_node, device_indices

    def test_reuse_clamps_device_anchor_to_host_gated_node(self):
        """for_reuse=True: device residency (node_b) extends past the
        host-gated best_match_node (node_a) -> the FULL device anchor must be
        clamped to node_a's boundary, excluding the stale node_b region."""
        (
            node_a,
            node_b,
            best_match_node,
            best_match_device_node,
            device_indices,
        ) = self._run_match_prefix_helper(for_reuse=True)

        # Host-gated boundary is unaffected by the flag: still node_a.
        self.assertIs(best_match_node, node_a)
        # Uncached the device-only validators trust node_b's stale ring slot.
        self.assertIs(best_match_device_node, node_b)
        # But the clamp must cap the returned device anchor at node_a's chunk
        # only (page_size=2 tokens), never reaching into node_b's stale region.
        self.assertEqual(len(device_indices), self.PAGE_SIZE)
        self.assertTrue(torch.equal(device_indices, torch.tensor([100, 101])))

    def test_self_match_does_not_clamp_device_anchor(self):
        """for_reuse=False (self-match, e.g. cache_unfinished_req): the same
        tree shape must NOT be clamped, so device_indices still covers both
        node_a and node_b (the request's own, not-yet-host-backed nodes),
        matching the I2' self-match invariant `cache_unfinished_req` relies on."""
        (
            node_a,
            node_b,
            best_match_node,
            best_match_device_node,
            device_indices,
        ) = self._run_match_prefix_helper(for_reuse=False)

        self.assertIs(best_match_node, node_a)
        self.assertIs(best_match_device_node, node_b)
        # NOT clamped: covers both node_a and node_b's chunks.
        self.assertEqual(len(device_indices), 2 * self.PAGE_SIZE)
        self.assertTrue(torch.equal(device_indices, torch.tensor([100, 101, 102, 103])))


class TestLoadBackCollectsHostBackedNodes(unittest.TestCase):
    """Mine 2 (warm reuse): a window node that is BOTH device-resident
    (`cd.value` present, a per-request device-ring slot recycled across
    requests) AND host-backed (`cd.host_value` present, the durable
    cross-request truth) must still be restored from host on reuse.

    Before this fix, two places disagreed and silently dropped it:
      * `build_hicache_transfers(LOAD_BACK)` treated `cd.value is not None`
        as "device exists, skip it", so the node was never collected into
        `nodes_to_load` / `host_indices` -- no SWA transfer was built for it.
      * `finalize_match_result` counted it as `n_swa += len(cd.value)` (not
        `swa_host_hit`), so `swa_host_hit_length` stayed 0 and the
        `load_back` gate never opened in the first place.

    Both must use the SAME host-backed predicate (`cd.host_value is not
    None`, strict mode) so that whenever the gate opens, the transfer that
    later runs actually contains this node. `finalize_match_result`
    additionally gates on `for_reuse=True`: self-match (`for_reuse=False`,
    e.g. `cache_unfinished_req`) must keep the OLD behavior, since the
    request's own freshly-computed nodes aren't host-backed yet and
    falsely opening the gate there would be wrong.
    """

    WIN = 4

    def _node_and_root(self, *, value, host_value):
        root = types.SimpleNamespace(component_data={}, parent=None)
        node = types.SimpleNamespace(
            parent=root,
            component_data={SWA: _cd(value=value, host_value=host_value)},
        )
        return root, node

    def _comp(self, *, strict):
        return types.SimpleNamespace(
            sliding_window_size=self.WIN,
            component_type=SWA,
            _swa_kv_pool_host=object(),  # host pool wired -> feature on
            _strict_bit_exact=strict,
            _unified_positional_swa=False,
            cache=types.SimpleNamespace(cache_controller=object()),
        )

    def _device_and_host_backed_node(self):
        return self._node_and_root(
            value=torch.tensor([900, 901, 902, 903], dtype=torch.int64),
            host_value=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        )

    def test_device_and_host_backed_node_is_collected_by_build(self):
        """The build-side predicate change: previously skipped, now
        collected into nodes_to_load whenever cd.host_value is not None,
        regardless of cd.value."""
        root, node = self._device_and_host_backed_node()
        comp = self._comp(strict=True)
        comp.cache.root_node = root

        transfers = SWAComponent.build_hicache_transfers(
            comp, node, CacheTransferPhase.LOAD_BACK
        )

        self.assertIsNotNone(transfers)
        xfer = transfers[0]
        self.assertIn(node, xfer.nodes_to_load)
        self.assertTrue(
            torch.equal(xfer.host_indices, node.component_data[SWA].host_value)
        )

    def test_finalize_counts_host_backed_node_on_reuse(self):
        """The finalize-side predicate change, gated on for_reuse=True: a
        device+host-resident node must count into swa_host_hit_length so the
        load_back gate opens, matching the build-side predicate above."""
        root, node = self._device_and_host_backed_node()
        comp = self._comp(strict=True)
        comp.cache.root_node = root
        result = MatchResult(
            device_indices=torch.tensor([], dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=0,
        )

        out = SWAComponent.finalize_match_result(
            comp,
            result=result,
            params=MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4])), for_reuse=True
            ),
            value_chunks=[],
            best_value_len=0,
        )

        self.assertGreater(out.swa_host_hit_length, 0)

    def test_finalize_self_match_keeps_old_behavior(self):
        """Guard: for_reuse=False (self-match, e.g. cache_unfinished_req)
        must NOT count the same device+host-resident node into
        swa_host_hit_length -- cd.value stays trusted first, so warm
        self-match does not falsely open the load_back gate."""
        root, node = self._device_and_host_backed_node()
        comp = self._comp(strict=True)
        comp.cache.root_node = root
        result = MatchResult(
            device_indices=torch.tensor([], dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            host_hit_length=0,
        )

        out = SWAComponent.finalize_match_result(
            comp,
            result=result,
            params=MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4])), for_reuse=False
            ),
            value_chunks=[],
            best_value_len=0,
        )

        self.assertEqual(out.swa_host_hit_length, 0)

    def test_non_strict_build_keeps_skipping_device_resident_node(self):
        """Guard: best-effort (non-strict) mode is unaffected by the
        strict-mode-scoped build change -- a device-resident node stays
        skipped even when also host-backed."""
        root, node = self._device_and_host_backed_node()
        comp = self._comp(strict=False)
        comp.cache.root_node = root

        transfers = SWAComponent.build_hicache_transfers(
            comp, node, CacheTransferPhase.LOAD_BACK
        )

        # Skipped (device exists), and nothing else host-only above it in
        # this single-node chain -> no transfer at all.
        self.assertIsNone(transfers)


class TestLoadBackMappingLengths(unittest.TestCase):
    """S3 contract: commit_hicache_transfer(LOAD_BACK) must feed equal-length
    index tensors to set_full_to_swa_mapping for every node. Correct device
    allocation (device_indices == sum host_value) maps cleanly; a device
    under-allocation raises the S3 diagnostic AssertionError with the exact
    sizes (not the opaque allocator full==swa assert)."""

    WIN = 128

    def _me(self, mapping_calls):
        allocator = types.SimpleNamespace(
            set_full_to_swa_mapping=lambda full, swa: mapping_calls.append(
                (int(full.numel()), int(swa.numel()))
            )
        )
        root = types.SimpleNamespace(component_data={}, parent=None)
        me = types.SimpleNamespace(
            component_type=SWA,
            _capture_rid=5,
            _swa_kv_pool_host=None,
            cache=types.SimpleNamespace(
                token_to_kv_pool_allocator=allocator, root_node=root
            ),
            _restore_device_value=lambda n, v: None,
        )
        me._gather_window_full_indices = (
            lambda n, nt: SWAComponent._gather_window_full_indices(me, n, nt)
        )
        return me, root

    def _window_node(self, root, base):
        # a page-boundary node: FULL value length == WIN, SWA host_value == WIN.
        full_val = torch.arange(base, base + self.WIN, dtype=torch.int64)
        return types.SimpleNamespace(
            parent=root,
            component_data={
                SWA: _cd(
                    value=None, host_value=torch.zeros(self.WIN, dtype=torch.int64)
                ),
                FULL: _cd(value=full_val),
            },
        )

    def _xfer(self, nodes, device_len):
        total = self.WIN * len(nodes)
        return PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.zeros(total, dtype=torch.int64),
            device_indices=torch.arange(device_len, dtype=torch.int64) + 900,
            nodes_to_load=nodes,
        )

    def test_correct_alloc_multi_node_maps_equal_lengths(self):
        mapping_calls = []
        me, root = self._me(mapping_calls)
        nodes = [self._window_node(root, 100), self._window_node(root, 300)]
        xfer = self._xfer(nodes, device_len=self.WIN * len(nodes))
        SWAComponent.commit_hicache_transfer(
            me, nodes[-1], CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )
        self.assertEqual(len(mapping_calls), 2)
        for full_n, swa_n in mapping_calls:
            self.assertEqual(full_n, swa_n)
            self.assertEqual(full_n, self.WIN)

    def test_device_under_allocation_raises_diagnostic(self):
        mapping_calls = []
        me, root = self._me(mapping_calls)
        nodes = [self._window_node(root, 100), self._window_node(root, 300)]
        # device_indices short by 64 for the 2nd node -> swa_chunk < window_full
        xfer = self._xfer(nodes, device_len=self.WIN * len(nodes) - 64)
        with self.assertRaises(AssertionError) as ctx:
            SWAComponent.commit_hicache_transfer(
                me, nodes[-1], CacheTransferPhase.LOAD_BACK, transfers=[xfer]
            )
        msg = str(ctx.exception)
        self.assertIn("index-length mismatch", msg)
        self.assertIn("swa_chunk=64", msg)
        self.assertIn("window_full=128", msg)


class TestSparseSwaReuseClamp(unittest.TestCase):
    """Task A4 (I2-inv): under stride>1, non-stride pages have a Full-host copy
    (tree backbone) but NO SWA window. On cross-request reuse the SWA validator
    must reject such pages so the reuse boundary clamps to the nearest page that
    has a SWA window (a stride boundary or the tail). This runtime clamp is the
    load-bearing path that REPLACES the retired insert-time SWA>=Full guard
    (former S5): reuse correctness relies on the clamp, not on an insert-time
    superset invariant."""

    def _reuse_validator(self):
        me = types.SimpleNamespace(
            sliding_window_size=4,
            component_type=SWA,
            _swa_kv_pool_host=object(),  # feature on (not device-only hicache)
            _strict_bit_exact=True,
            cache=types.SimpleNamespace(cache_controller=object()),
        )
        # reuse match == match_device_only=False (scheduler cross-request reuse).
        return SWAComponent.create_match_validator(me, match_device_only=False)

    def _node(self, key_len, *, value, host_value):
        return types.SimpleNamespace(
            key=list(range(key_len)),
            backuped=True,
            evicted=False,
            component_data={SWA: _cd(value=value, host_value=host_value)},
        )

    def test_non_stride_page_without_swa_window_clamps(self):
        # sparse intermediate page: Full-host exists but no SWA window at all.
        v = self._reuse_validator()
        self.assertFalse(v(self._node(4, value=None, host_value=None)))

    def test_non_stride_page_with_only_device_ring_clamps(self):
        # even if the per-request device ring still holds this page, strict
        # reuse never trusts it as cross-request truth -> still clamp.
        v = self._reuse_validator()
        self.assertFalse(v(self._node(4, value=[1, 2, 3, 4], host_value=None)))

    def test_stride_page_with_swa_host_extends(self):
        # a stride boundary / tail page has a durable SWA window -> reuse OK.
        v = self._reuse_validator()
        self.assertTrue(v(self._node(4, value=None, host_value=[0, 0, 0, 0])))


class TestSwaStagingCleanupRobustToRetract(unittest.TestCase):
    """B0-3: decode capture stages (req_pool_idx, B) across many steps; the
    retract/abort path caches with is_insert=False (prepare_for_caching_req and
    thus _capture_rid never run). cleanup_after_caching_req must still free this
    request's staging keyed by req.req_pool_idx, else it leaks and can mis-bind a
    stale window once req_pool_idx is recycled."""

    def _hp(self):
        hp = _FakeHostPool(win=2, head_dim=4, num_pages=64, layers=2)
        hp._capture_staging = {
            (7, 4): torch.arange(2, dtype=torch.int64),
            (7, 8): torch.arange(2, dtype=torch.int64),
            (9, 4): torch.arange(2, dtype=torch.int64),  # another live request
        }
        return hp

    def _call(self, hp, capture_rid, req_pool_idx):
        fake_self = types.SimpleNamespace(
            _swa_kv_pool_host=hp,
            _capture_rid=capture_rid,
        )
        req = types.SimpleNamespace(req_pool_idx=req_pool_idx)
        SWAComponent.cleanup_after_caching_req(fake_self, req, is_finished=True)

    def test_frees_by_req_pool_idx_when_capture_rid_stale(self):
        hp = self._hp()
        # is_insert=False: _capture_rid is None (stale), but req_pool_idx=7 owns
        # the staging and must be freed; req 9 untouched.
        self._call(hp, capture_rid=None, req_pool_idx=7)
        self.assertNotIn((7, 4), hp._capture_staging)
        self.assertNotIn((7, 8), hp._capture_staging)
        self.assertIn((9, 4), hp._capture_staging)

    def test_insert_path_behavior_unchanged(self):
        hp = self._hp()
        # is_insert=True: _capture_rid == req_pool_idx == 7.
        self._call(hp, capture_rid=7, req_pool_idx=7)
        self.assertNotIn((7, 4), hp._capture_staging)
        self.assertIn((9, 4), hp._capture_staging)


class _FakeDecodePool:
    def __init__(self, host_pool, win):
        self._swa_host_pool = host_pool
        self.unified_swa_ring_size = win
        self.unified_swa_window = win
        self.start_layer = 0


class _IKey:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _INode:
    def __init__(self, keylen):
        self.key = _IKey(keylen)
        self.component_data = {SWA: _cd(value=None, host_value=None)}


class _ICache:
    def __init__(self, page_size):
        self.page_size = page_size
        self.splits = []

    def _split_node(self, key, child, split_len):
        # Mirror UnifiedRadixCache._split_node's shape: return a fresh parent
        # covering [start, start+split_len); the original object becomes the
        # tail child (its key shrinks) and keeps any plain attributes it held.
        parent = _INode(split_len)
        child.key = _IKey(len(child.key) - split_len)
        self.splits.append(split_len)
        return parent


class TestSwaBindInteriorWindows(unittest.TestCase):
    """Stride model: interior stride windows become host-only carrier nodes so
    cross-request reuse clamps to the nearest stride page, not the chunk end."""

    def _fake_self(self, host):
        return types.SimpleNamespace(
            _swa_kv_pool_host=host,
            _capture_rid=5,
            component_type=SWA,
            cache=_ICache(page_size=2),
        )

    def _stage(self, host, rid, B, win):
        tile = torch.arange(100 + B, 100 + B + win, dtype=torch.int64)
        host._capture_staging[(rid, B)] = tile
        return tile

    def test_binds_interior_stride_windows_as_pending(self):
        win = 2
        host = _FakeHostPool(win=win, head_dim=4, num_pages=8, layers=1)
        self._stage(host, 5, 4, win)
        self._stage(host, 5, 8, win)
        # Boundaries that must be ignored: region_end (12), region_start (0),
        # and another request's window.
        self._stage(host, 5, 12, win)
        self._stage(host, 5, 0, win)
        self._stage(host, 7, 6, win)

        me = self._fake_self(host)
        region = _INode(12)  # tombstone span [0, 12)
        SWAComponent._bind_interior_captured_swa_hosts(me, region, 0, 12)

        # Two carrier nodes created (at B=8 then B=4), split offsets from anchor.
        self.assertEqual(me.cache.splits, [8, 4])
        # Interior windows consumed; excluded boundaries left intact.
        self.assertNotIn((5, 4), host._capture_staging)
        self.assertNotIn((5, 8), host._capture_staging)
        self.assertIn((5, 12), host._capture_staging)
        self.assertIn((5, 0), host._capture_staging)
        self.assertIn((7, 6), host._capture_staging)

    def test_pending_page_is_the_window_bytes(self):
        win = 2
        host = _FakeHostPool(win=win, head_dim=4, num_pages=8, layers=1)
        t4 = self._stage(host, 5, 4, win)
        me = self._fake_self(host)

        created = []
        orig_split = me.cache._split_node

        def _tracking_split(key, child, split_len):
            parent = orig_split(key, child, split_len)
            created.append((split_len, parent))
            return parent

        me.cache._split_node = _tracking_split
        region = _INode(8)
        SWAComponent._bind_interior_captured_swa_hosts(me, region, 0, 8)

        self.assertEqual(len(created), 1)
        split_len, parent = created[0]
        self.assertEqual(split_len, 4)  # node ends at boundary B=4
        pending = getattr(parent, "_swa_pending_host", None)
        self.assertIsNotNone(pending)
        self.assertEqual(len(pending), win)  # one window, not node length
        self.assertTrue(torch.equal(pending, t4.to(torch.int64)))

    def test_no_staged_interior_is_noop(self):
        win = 2
        host = _FakeHostPool(win=win, head_dim=4, num_pages=8, layers=1)
        me = self._fake_self(host)
        region = _INode(8)
        SWAComponent._bind_interior_captured_swa_hosts(me, region, 0, 8)
        self.assertEqual(me.cache.splits, [])


class _FakeHostLRU:
    def __init__(self):
        self._nodes = set()

    def in_list(self, node):
        return id(node) in self._nodes

    def insert_mru(self, node):
        self._nodes.add(id(node))

    def remove_node(self, node):
        self._nodes.discard(id(node))


class TestSwaR1InteriorCarrierPendingLifetime(unittest.TestCase):
    """R1: an interior stride carrier's captured page (``_swa_pending_host``)
    has a Full(base)-tracked lifetime, decoupled from the SWA device ring. The
    device SWA tombstone (SWA pool pressure) ALWAYS fires before the finish-time
    coordinated BACKUP_HOST, so the pending page must:

      (1) SURVIVE the device tombstone for an interior carrier, while a plain
          (non-carrier) pending page is still freed there (co-lifetime I3);
      (2) still be promotable to a durable ``host_value`` by that later
          coordinated BACKUP_HOST *after* the tombstone (the path that actually
          makes finer-than-chunk stride reuse durable);
      (3) be freed exactly once, at true node removal
          (``free_pending_host_on_remove``), so the SWA host pool never leaks
          and never double-frees.

    Deterministic: exercises the real SWAComponent methods against fake pools
    (no model, no device), building a genuinely-flagged carrier via the real
    ``_bind_interior_captured_swa_hosts`` bind path.
    """

    WIN = 2

    def _host(self):
        return _FakeHostPool(win=self.WIN, head_dim=4, num_pages=8, layers=1)

    def _make_carrier(self, host):
        """Build a genuine interior carrier through the real bind path, so it is
        actually flagged ``_swa_interior_carrier`` and owns a real pending page.
        Returns (carrier_node, window_tile)."""
        me = types.SimpleNamespace(
            _swa_kv_pool_host=host,
            _capture_rid=5,
            component_type=SWA,
            cache=_ICache(page_size=2),
        )
        tile = torch.arange(104, 104 + self.WIN, dtype=torch.int64)
        host._capture_staging[(5, 4)] = tile
        created = {}
        orig_split = me.cache._split_node

        def _tracking_split(key, child, split_len):
            parent = orig_split(key, child, split_len)
            created["node"] = parent
            return parent

        me.cache._split_node = _tracking_split
        region = _INode(8)
        SWAComponent._bind_interior_captured_swa_hosts(me, region, 0, 8)
        carrier = created["node"]
        carrier.id = 1  # real UnifiedTreeNode has .id (used by stride dbg logs)
        # Precondition: the bind path really produced a flagged carrier.
        self.assertTrue(getattr(carrier, "_swa_interior_carrier", False))
        pending = getattr(carrier, "_swa_pending_host", None)
        self.assertIsNotNone(pending)
        self.assertTrue(torch.equal(pending, tile.to(torch.int64)))
        return carrier, tile

    def _evict_self(self, host):
        return types.SimpleNamespace(
            component_type=SWA,
            _swa_kv_pool_host=host,
            cache=types.SimpleNamespace(
                token_to_kv_pool_allocator=types.SimpleNamespace(
                    free_swa=lambda v: None, free=lambda v: None
                ),
                component_evictable_size_={SWA: 1000},
                host_lru_lists={SWA: _FakeHostLRU()},
            ),
        )

    # (1a) leak-guard branch (real interior shape: no device SWA, no host_value)
    def test_leak_guard_branch_preserves_carrier_pending(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        cd = carrier.component_data[SWA]
        self.assertIsNone(cd.value)  # interior carrier: never a device SWA slot
        self.assertIsNone(cd.host_value)
        me = self._evict_self(host)

        SWAComponent.evict_component(me, carrier, target=EvictLayer.DEVICE)

        # R1: pending survives the device tombstone; host pool untouched.
        surviving = getattr(carrier, "_swa_pending_host", None)
        self.assertIsNotNone(surviving)
        self.assertTrue(torch.equal(surviving, tile.to(torch.int64)))
        self.assertEqual(host.freed, [])

    # (1b) contrast: a plain (non-carrier) pending page IS freed at the tombstone
    def test_leak_guard_branch_frees_plain_pending(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        carrier._swa_interior_carrier = False  # strip flag -> plain pending page
        me = self._evict_self(host)

        SWAComponent.evict_component(me, carrier, target=EvictLayer.DEVICE)

        # Co-lifetime (I3): a not-yet-promoted plain page must not outlive here.
        self.assertIsNone(getattr(carrier, "_swa_pending_host", None))
        self.assertEqual(len(host.freed), 1)
        self.assertTrue(torch.equal(host.freed[0], tile.to(torch.int64)))

    # (1c) defensive: even if a carrier also holds a device SWA value, the
    # device-branch free must not drag its pending page down with it.
    def test_device_branch_preserves_carrier_pending(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        cd = carrier.component_data[SWA]
        cd.value = torch.tensor([900, 901], dtype=torch.int64)
        carrier.component_data[FULL] = _cd(
            value=torch.tensor([100, 101], dtype=torch.int64)
        )
        me = self._evict_self(host)

        freed, _ = SWAComponent.evict_component(me, carrier, target=EvictLayer.DEVICE)

        self.assertEqual(freed, 2)  # device ring tombstoned
        self.assertIsNone(cd.value)
        surviving = getattr(carrier, "_swa_pending_host", None)
        self.assertIsNotNone(surviving)  # R1: pending kept
        self.assertTrue(torch.equal(surviving, tile.to(torch.int64)))
        self.assertEqual(host.freed, [])  # pending NOT host-freed

    # (2) after the tombstone, the coordinated BACKUP_HOST still promotes the
    # survived pending page to a durable host_value (the reuse-enabling path).
    def test_coordinated_backup_promotes_survived_pending(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        cd = carrier.component_data[SWA]
        # State right after a device tombstone: no device SWA, no host copy yet.
        self.assertIsNone(cd.value)
        self.assertIsNone(cd.host_value)
        carrier.component_data[FULL] = _cd(
            value=torch.tensor([100, 101], dtype=torch.int64)
        )
        attached = []
        me = types.SimpleNamespace(
            component_type=SWA,
            _swa_kv_pool_host=host,
            _strict_bit_exact=True,
            cache=types.SimpleNamespace(cache_controller=object()),
            _attach_swa_host_value=lambda node, hv: attached.append(hv),
        )

        transfers = SWAComponent.build_hicache_transfers(
            me, carrier, CacheTransferPhase.BACKUP_HOST
        )
        self.assertIsNotNone(transfers)
        self.assertEqual(len(transfers), 1)
        # host-only adopt of the pre-staged page: no redundant dev->host copy.
        self.assertIsNone(transfers[0].device_indices)
        self.assertTrue(torch.equal(transfers[0].host_indices, tile.to(torch.int64)))

        SWAComponent.commit_hicache_transfer(
            me, carrier, CacheTransferPhase.BACKUP_HOST, transfers=transfers
        )
        # Promoted: host_value adopted (via _attach), pending ownership dropped.
        self.assertEqual(len(attached), 1)
        self.assertTrue(torch.equal(attached[0], tile.to(torch.int64)))
        self.assertIsNone(getattr(carrier, "_swa_pending_host", None))

    # (3) removal is the single true free chokepoint (no leak).
    def test_removal_frees_survived_carrier_pending(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        me = types.SimpleNamespace(_swa_kv_pool_host=host)

        SWAComponent.free_pending_host_on_remove(me, carrier)

        self.assertIsNone(getattr(carrier, "_swa_pending_host", None))
        self.assertEqual(len(host.freed), 1)
        self.assertTrue(torch.equal(host.freed[0], tile.to(torch.int64)))

    # (3b) once promoted, removal is a no-op -> no double free of the host page.
    def test_removal_after_promotion_is_noop(self):
        host = self._host()
        carrier, _ = self._make_carrier(host)
        carrier._swa_pending_host = None  # already adopted by host_value
        me = types.SimpleNamespace(_swa_kv_pool_host=host)

        SWAComponent.free_pending_host_on_remove(me, carrier)

        self.assertEqual(host.freed, [])

    # (1)+(3) end-to-end lifetime: tombstone survives, then removal frees once.
    def test_tombstone_then_removal_frees_exactly_once(self):
        host = self._host()
        carrier, tile = self._make_carrier(host)
        me_evict = self._evict_self(host)

        SWAComponent.evict_component(me_evict, carrier, target=EvictLayer.DEVICE)
        self.assertIsNotNone(getattr(carrier, "_swa_pending_host", None))
        self.assertEqual(host.freed, [])  # survived tombstone

        me_rm = types.SimpleNamespace(_swa_kv_pool_host=host)
        SWAComponent.free_pending_host_on_remove(me_rm, carrier)
        self.assertIsNone(getattr(carrier, "_swa_pending_host", None))
        self.assertEqual(len(host.freed), 1)  # freed exactly once at removal
        self.assertTrue(torch.equal(host.freed[0], tile.to(torch.int64)))


if __name__ == "__main__":
    unittest.main()
