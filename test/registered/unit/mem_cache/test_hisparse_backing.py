"""Unit tests for the HiSparse backing seam.

HiSparse has one user-facing switch (`--enable-hisparse`) and picks its logical
KV pool from the cache configuration. Two things must hold for that to be safe:

- the resolution matrix -- every cache configuration either resolves to exactly
  one backing or is rejected at startup with both legal recipes named, so a
  half-supported combination can never boot;
- one indexer expansion ratio per backing -- the capacity model and the pool
  construction read the same helper, or the pool overruns the memory that was
  reserved for it.

Plus a drift guard: whatever the protocol declares, every backing implements.
"""

import inspect
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.hisparse_coordinator import (  # noqa: E402
    PrivateHostHiSparseCoordinator,
)
from sglang.srt.managers.hisparse_hicache_admission import AdmissionLedger  # noqa: E402
from sglang.srt.managers.hisparse_hicache_coordinator import (  # noqa: E402
    HiCacheHiSparseCoordinator,
    _AdmittedNodes,
    _ExpandedIndexerPages,
    _PendingAdmission,
)
from sglang.srt.managers.hisparse_protocol import HiSparseCoordinator  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.sparsity import (  # noqa: E402
    HiSparseBacking,
    create_hisparse_coordinator,
    hisparse_backing,
    hisparse_indexer_expansion_ratio,
    hisparse_indexer_regions,
    hisparse_indexer_top_k,
    resolve_hisparse_backing,
)
from sglang.srt.mem_cache.unified_cache.component_type import (  # noqa: E402
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.full import (  # noqa: E402
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: E402
    UnifiedTreeCore,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _resolve(**overrides):
    """Resolve a HiCache-shaped configuration with the given fields replaced."""
    kwargs = dict(
        enable_hisparse=True,
        disable_radix_cache=False,
        enable_hierarchical_cache=True,
        hicache_write_policy="write_back",
    )
    kwargs.update(overrides)
    return resolve_hisparse_backing(**kwargs)


def _flags_for(backing: HiSparseBacking) -> dict:
    """The minimal server-args flags that resolve to `backing`."""
    if backing is HiSparseBacking.PRIVATE_HOST:
        return dict(enable_hisparse=True, disable_radix_cache=True)
    return dict(
        enable_hisparse=True,
        enable_hierarchical_cache=True,
        hicache_write_policy="write_back",
    )


def _callable_parameters(signature: inspect.Signature) -> list[tuple[str, str]]:
    """(name, kind) per parameter, ignoring annotations and defaults.

    Callers bind by name, so a rename or a positional-only parameter breaks them;
    a differing annotation or default does not.
    """
    return [
        (name, param.kind.name)
        for name, param in signature.parameters.items()
        if name != "self"
    ]


def _server_args(test_case, **overrides) -> ServerArgs:
    override = get_context().override_server_args(**overrides)
    server_args = override.install()
    test_case.addCleanup(override.restore)
    return server_args


class TestHiSparseBackingResolution(CustomTestCase):
    def test_off_resolves_to_no_backing(self):
        self.assertIsNone(_resolve(enable_hisparse=False))
        # ... whatever the cache configuration says.
        self.assertIsNone(_resolve(enable_hisparse=False, disable_radix_cache=True))

    def test_radix_disabled_is_private_host(self):
        self.assertIs(_resolve(disable_radix_cache=True), HiSparseBacking.PRIVATE_HOST)
        # The radix-off branch wins before any hicache field is consulted, so a
        # stale --hicache-write-policy cannot flip it.
        self.assertIs(
            _resolve(
                disable_radix_cache=True,
                enable_hierarchical_cache=False,
                hicache_write_policy="write_through",
            ),
            HiSparseBacking.PRIVATE_HOST,
        )

    def test_radix_plus_write_back_hicache_is_hicache(self):
        self.assertIs(_resolve(), HiSparseBacking.HICACHE)

    def test_radix_without_host_tier_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            _resolve(enable_hierarchical_cache=False)
        message = str(caught.exception)
        # Both legal recipes must be in the message: the user has to be able to
        # pick one without reading the source.
        self.assertIn("--enable-hierarchical-cache", message)
        self.assertIn("--disable-radix-cache", message)

    def test_write_through_is_rejected(self):
        for policy in ("write_through", "write_through_selective"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError) as caught:
                    _resolve(hicache_write_policy=policy)
                self.assertIn("write_back", str(caught.exception))


class TestIndexerExpansionRatio(CustomTestCase):
    def test_off_needs_no_expansion(self):
        self.assertEqual(hisparse_indexer_expansion_ratio(_server_args(self)), 1.0)

    def test_private_host_uses_host_to_device_ratio(self):
        server_args = _server_args(
            self,
            enable_hisparse=True,
            disable_radix_cache=True,
            hisparse_config='{"top_k": 2048, "host_to_device_ratio": 3}',
        )
        self.assertIs(hisparse_backing(server_args), HiSparseBacking.PRIVATE_HOST)
        self.assertEqual(hisparse_indexer_expansion_ratio(server_args), 3.0)

    def test_private_host_rejects_a_fractional_ratio(self):
        # The private-host pool multiplies a token count by this; a fraction
        # would surface as a non-integral buffer size inside pool allocation.
        server_args = _server_args(
            self,
            enable_hisparse=True,
            disable_radix_cache=True,
            hisparse_config='{"host_to_device_ratio": 2.5}',
        )
        with self.assertRaises(ValueError):
            hisparse_indexer_expansion_ratio(server_args)

    def test_hicache_covers_the_base_region_plus_both_tiers(self):
        """`2 + hicache_ratio`, and the leading 2 is the part that gets lost.

        One pool for the base region (a KV page id IS its indexer page id) plus
        `1 + hicache_ratio` for the expanded region, which holds a copy of every
        admitted prefix and is therefore bounded by the two tiers that can hold
        those prefixes. Sizing the total at `1 + hicache_ratio` -- the natural
        misreading, since that IS the two tiers -- leaves the expanded region one
        pool short and silently halves admission depth.
        """
        server_args = _server_args(
            self,
            enable_hisparse=True,
            enable_hierarchical_cache=True,
            hicache_write_policy="write_back",
            hicache_ratio=4.0,
        )
        self.assertIs(hisparse_backing(server_args), HiSparseBacking.HICACHE)
        self.assertEqual(hisparse_indexer_expansion_ratio(server_args), 6.0)

    def test_hicache_expansion_ratio_override_wins(self):
        server_args = _server_args(
            self,
            enable_hisparse=True,
            enable_hierarchical_cache=True,
            hicache_write_policy="write_back",
            hicache_ratio=4.0,
            hisparse_config='{"expansion_ratio": 2.5}',
        )
        self.assertEqual(hisparse_indexer_expansion_ratio(server_args), 2.5)

    def test_hicache_rejects_a_non_positive_override(self):
        server_args = _server_args(
            self,
            enable_hisparse=True,
            enable_hierarchical_cache=True,
            hicache_write_policy="write_back",
            hisparse_config='{"expansion_ratio": 0}',
        )
        with self.assertRaises(ValueError):
            hisparse_indexer_expansion_ratio(server_args)


# Every implemented backing, keyed by the enum member it answers for. A further
# backing lands by adding one row; the conformance tests then cover it.
BACKING_IMPLEMENTATIONS = {
    HiSparseBacking.PRIVATE_HOST: PrivateHostHiSparseCoordinator,
    HiSparseBacking.HICACHE: HiCacheHiSparseCoordinator,
}

# Backings the configuration layer already accepts -- they resolve, size the pool
# and pass startup validation -- but that no coordinator implements yet. Listing
# one here is what keeps the completeness check meaningful while a port is in
# progress, and `test_pending_backings_fail_loudly` makes the list
# self-tightening: implementing a backing without moving it out turns that test
# red.
PENDING_BACKINGS = set()


class TestIndexerTopK(CustomTestCase):
    """One top-k rule for both backings: the model's `index_topk` wins.

    Both backings size their swap-in geometry from this number, so resolving it in
    two places is how they would end up with two different geometries for the same
    checkpoint.
    """

    @staticmethod
    def _model_config(index_topk=None):
        text_config = SimpleNamespace()
        if index_topk is not None:
            text_config.index_topk = index_topk
        return SimpleNamespace(hf_text_config=text_config)

    def test_config_top_k_used_when_the_model_declares_none(self):
        top_k = hisparse_indexer_top_k(
            server_args=_server_args(self, hisparse_config='{"top_k": 1024}'),
            model_config=self._model_config(),
        )
        self.assertEqual(top_k, 1024)

    def test_the_model_value_wins_over_the_config(self):
        # Not a mirror of the one-liner: this is the direction of precedence both
        # backings' geometry depends on, and flipping it would size every buffer
        # from a number the indexer does not use.
        top_k = hisparse_indexer_top_k(
            server_args=_server_args(self, hisparse_config='{"top_k": 512}'),
            model_config=self._model_config(index_topk=2048),
        )
        self.assertEqual(top_k, 2048)


class TestIndexerRegions(CustomTestCase):
    """The base/expanded split of the indexer buffer.

    Every consumer -- the expanded-page allocator and the hybrid page table --
    derives page ids from this one place, so an off-by-one here does not raise:
    the indexer silently scores whatever keys live at the wrong page.
    """

    def test_base_region_covers_the_whole_attention_pool(self):
        # A KV page id doubles as its base indexer page id, so the base region has
        # to span every page the allocator can hand out. Paged allocators start at
        # 1, so a token loc reaches device_pool_size + page_size - 1. The two
        # counts must also partition the buffer: a gap wastes pages, an overlap
        # aliases two requests onto one page.
        base_pages, expanded_pages = hisparse_indexer_regions(
            page_size=64, num_indexer_pages=400, device_pool_size=64 * 100
        )
        self.assertEqual(base_pages, 101)
        self.assertEqual(expanded_pages, 400 - 101)
        self.assertEqual(base_pages + expanded_pages, 400)

    def test_buffer_without_room_for_an_expanded_region_is_rejected(self):
        # Sized for the pool alone: admission would have no private pages to copy
        # an evicted prefix into, which must fail at startup rather than at the
        # first eviction.
        with self.assertRaises(ValueError) as caught:
            hisparse_indexer_regions(
                page_size=64, num_indexer_pages=101, device_pool_size=64 * 100
            )
        self.assertIn("expansion_ratio", str(caught.exception))


class TestHiCacheDeferredAdmission(CustomTestCase):
    def test_pending_admission_requires_flushed_frees(self):
        """Deferred admissions must resume after the allocator closes its free group,
        while an open group must reject admission without consuming a retry."""
        allocator = PagedTokenToKVPoolAllocator(
            size=256,
            page_size=64,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        pending = _PendingAdmission(
            req=SimpleNamespace(
                rid="pending",
                kv=SimpleNamespace(req_pool_idx=1, cache_protected_len=64),
            )
        )
        coordinator = object.__new__(HiCacheHiSparseCoordinator)
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.page_size = 64
        coordinator._pending_admission = {1: pending}
        coordinator._eviction_queue = []
        coordinator._indexer_pages = _ExpandedIndexerPages(num_pages=0, device="cpu")

        allocator.free_group_begin()
        try:
            with self.assertRaisesRegex(AssertionError, "outside the allocator"):
                coordinator.admit_pending()
            self.assertEqual(pending.attempts, 0)
        finally:
            allocator.free_group_end()

        coordinator.admit_pending()
        self.assertEqual(pending.attempts, 1)
        self.assertIn(1, coordinator._pending_admission)


class TestHiCacheSplitHostLocks(CustomTestCase):
    PAGE_SIZE = 64
    PREFIX_LEN = 192
    NUM_INDEXER_PAGES = 16

    def setUp(self):
        self.allocator = PagedTokenToKVPoolAllocator(
            size=1024,
            page_size=self.PAGE_SIZE,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.PAGE_SIZE,
        )
        self.cache = object.__new__(UnifiedRadixCache)
        self.cache.disable = False
        self.cache.enable_session_radix_cache = False
        self.core = UnifiedTreeCore(
            params=params,
            components={ComponentType.FULL: FullComponent(self.cache, params)},
        )
        self.core.enable_hicache = True
        self.core.is_write_back = True
        self.cache.tree_core = self.core
        self.node = self._add_node(key_start=0)
        self.host_rows = torch.arange(self.PREFIX_LEN, dtype=torch.int64)

    def _add_node(self, *, key_start):
        return self.core._add_new_node(
            parent=self.core.root_node,
            key=RadixKey(list(range(key_start, key_start + self.PREFIX_LEN))),
            value=self.allocator.alloc(self.PREFIX_LEN),
        )

    def _coordinator(self, *, request_count=2, matched_node=None):
        matched_node = self.node if matched_node is None else matched_node
        coord = object.__new__(HiCacheHiSparseCoordinator)
        coord.device = "cpu"
        coord.page_size = self.PAGE_SIZE
        coord.layer_num = 1
        coord.device_buffer_size = self.PAGE_SIZE
        coord.token_to_kv_pool_allocator = self.allocator
        device_rows = matched_node.component_data[ComponentType.FULL].value
        coord.req_to_token_pool = SimpleNamespace(
            req_to_token=device_rows.int().repeat(request_count, 1)
        )
        coord.req_to_host_pool = torch.full_like(
            coord.req_to_token_pool.req_to_token, -1
        )
        coord.req_device_buffer_locs = torch.stack(
            [self.allocator.alloc(self.PAGE_SIZE).int() for _ in range(request_count)]
        )
        coord.req_device_buffer_tokens = torch.full(
            (1, request_count, self.PAGE_SIZE), -1, dtype=torch.int32
        )
        coord._lru_init = torch.arange(self.PAGE_SIZE, dtype=torch.int16)
        coord.lru_slots = coord._lru_init.repeat(1, request_count, 1)
        coord._indexer_page_offset = 17
        coord._indexer_pages = _ExpandedIndexerPages(
            num_pages=self.NUM_INDEXER_PAGES, device="cpu"
        )
        coord.req_to_indexer_page = torch.stack(
            [
                coord._indexer_pages.alloc(self.PREFIX_LEN // self.PAGE_SIZE)
                + coord._indexer_page_offset
                for _ in range(request_count)
            ]
        )
        coord.decode_producer_stream = None
        coord._eviction_queue = []
        coord._pending_admission = {}
        coord._announced_first_eviction = True
        coord.ledger = AdmissionLedger(
            device_pool_tokens=1024,
            temp_slot_tokens=self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            chunk_tokens=self.PAGE_SIZE,
        )
        coord._req_nodes = {}
        for slot in range(request_count):
            coord.ledger.claim_node(matched_node)
            coord.ledger.activate(
                slot, self.PREFIX_LEN, rid=str(slot), decode_reserve=0
            )
            coord._req_nodes[slot] = _AdmittedNodes(matched=(matched_node,))
        coord.item_size_bytes = 16
        coord._host_binding = torch.zeros((1, 2), dtype=torch.int64)
        self.cache.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                size=1024,
                data_refs=[torch.zeros((1024, 16), dtype=torch.uint8)],
                token_stride_size=16,
            )
        )
        coord.set_tree_cache(self.cache)
        return coord

    def _demote(self):
        self.core.commit_backup(
            node_id=self.node.id, host_indices=self.host_rows, comp_xfers={}
        )
        result = self.core.demote(self.node.id)
        for indices in result.device_frees[ComponentType.FULL]:
            self.allocator.free(indices)
        result.device_frees.clear()
        result.host_frees.clear()

    def _split_and_restore_prefix(self):
        parent, _ = self.core._split_node(
            key=self.node.key, child=self.node, split_len=2 * self.PAGE_SIZE
        )
        ancestor, _ = self.core._split_node(
            key=parent.key, child=parent, split_len=self.PAGE_SIZE
        )
        for node in (ancestor, parent):
            if node.component_data[ComponentType.FULL].value is None:
                self.core._unevict_node_on_insert(
                    node, self.allocator.alloc(self.PAGE_SIZE)
                )
        self.core.inc_lock_ref(parent.id)
        return ancestor, parent, self.node

    def _reclaim_host(self):
        with envs.SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM.override(False):
            result = self.core.drive_host_eviction(ComponentType.FULL, self.PREFIX_LEN)
        chunks = result.host_frees[ComponentType.FULL]
        freed = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
        result.device_frees.clear()
        result.host_frees.clear()
        return freed

    def _finish(self, coord, slot):
        coord.request_finished(
            SimpleNamespace(rid=str(slot), kv=SimpleNamespace(req_pool_idx=slot))
        )

    def _assert_shared_ownership_and_release(self, coord, fragments):
        self.assertEqual(self._reclaim_host().numel(), 0)
        self.assertEqual(coord.ledger._claimed_tokens, self.PREFIX_LEN)
        self.assertEqual(coord.ledger._host_locked_tokens, self.PREFIX_LEN)
        self.assertTrue(
            all(
                n.component_data[ComponentType.FULL].host_lock_ref == 2
                for n in fragments
            )
        )
        free_before = self.allocator.available_size()
        self._finish(coord, 0)
        self.assertEqual(self._reclaim_host().numel(), 0)
        self.assertTrue(
            all(
                n.component_data[ComponentType.FULL].host_lock_ref == 1
                for n in fragments
            )
        )
        self._finish(coord, 1)
        self.assertEqual(coord.ledger._claimed_tokens, 0)
        self.assertEqual(coord.ledger._host_locked_tokens, 0)
        self.assertFalse(coord.ledger._node_claims)
        self.assertEqual(coord._indexer_pages.available(), self.NUM_INDEXER_PAGES)
        self.assertEqual(
            self.allocator.available_size(), free_before + 2 * self.PAGE_SIZE
        )
        self.assertTrue(torch.equal(self._reclaim_host().sort().values, self.host_rows))

    def test_shared_host_rows_survive_repeated_split_until_last_request_finishes(self):
        """Reloading split prefix fragments must not reclaim host rows still read
        by either active request; finishing both requests must release every lock."""
        coord = self._coordinator()
        self._demote()
        coord._sync_evictions()
        fragments = self._split_and_restore_prefix()
        self._assert_shared_ownership_and_release(coord, fragments)

    def test_pending_eviction_pins_split_fragments_until_request_locks_take_over(self):
        """A split between eviction callback and application must retain its
        temporary host locks, then hand all fragments to the affected requests."""
        coord = self._coordinator()
        self._demote()
        fragments = self._split_and_restore_prefix()
        self.assertEqual(self._reclaim_host().numel(), 0)
        coord._sync_evictions()
        self.assertTrue(torch.equal(coord.req_to_host_pool[0], self.host_rows.int()))
        self._assert_shared_ownership_and_release(coord, fragments)

    def test_unmatched_eviction_releases_all_temporary_split_locks(self):
        """An unrelated eviction must leave no fragment pinned after attribution
        finds no active request referencing its rows."""
        coord = self._coordinator(
            request_count=1, matched_node=self._add_node(key_start=1000)
        )
        self._demote()
        fragments = self._split_and_restore_prefix()
        coord._sync_evictions()
        self.assertTrue(
            all(
                n.component_data[ComponentType.FULL].host_lock_ref == 0
                for n in fragments
            )
        )
        self.assertTrue(torch.equal(self._reclaim_host().sort().values, self.host_rows))
        self.assertFalse(coord._req_nodes[0].host_locks)
        self.assertEqual(coord.ledger._host_locked_tokens, 0)
        self._finish(coord, 0)
        self.assertEqual(coord.ledger._claimed_tokens, 0)

    def test_device_only_split_preserves_shared_prefix_accounting(self):
        """Splitting a claimed device prefix must not bill it a second time when
        another request claims the new parent before any host backup exists."""
        coord = self._coordinator()
        ancestor, _, _ = self._split_and_restore_prefix()
        coord.ledger.claim_node(ancestor)
        self.assertEqual(coord.ledger._claimed_tokens, self.PREFIX_LEN)
        coord.ledger.release_node(ancestor)
        for slot in range(2):
            self._finish(coord, slot)
        self.assertFalse(coord.ledger._node_claims)
        self.assertEqual(coord.ledger._claimed_tokens, 0)


class TestBackingsImplementTheProtocol(CustomTestCase):
    """The compatibility guard between the backings.

    Two implementations behind one protocol only stay interchangeable if nothing
    drifts, and structural typing is not checked at runtime: a backing can lag a
    method, or keep a stale positional signature, and every call site still
    imports fine -- the failure shows up as a TypeError mid-forward on whichever
    configuration exercises that path.
    """

    @staticmethod
    def _protocol_methods():
        return sorted(
            name
            for name, value in vars(HiSparseCoordinator).items()
            if callable(value) and not name.startswith("_")
        )

    def test_the_protocol_is_not_vacuous(self):
        # A renamed/emptied protocol would make every check below trivially pass.
        self.assertGreater(len(self._protocol_methods()), 10)

    def test_every_backing_is_accounted_for(self):
        # The factory dispatches on this enum, so a member that is neither
        # implemented nor listed as pending is a backing whose conformance
        # nothing checks.
        self.assertEqual(
            sorted(b.value for b in HiSparseBacking),
            sorted(b.value for b in (set(BACKING_IMPLEMENTATIONS) | PENDING_BACKINGS)),
        )
        self.assertEqual(
            set(BACKING_IMPLEMENTATIONS) & PENDING_BACKINGS,
            set(),
            "a backing cannot be both implemented and pending",
        )

    def test_pending_backings_fail_loudly(self):
        """A pending backing must refuse construction, not half-build something.

        Its configuration already resolves and sizes the pool, so without this the
        failure would surface as an AttributeError deep in a forward. This also
        makes PENDING_BACKINGS self-tightening: once a backing is implemented the
        factory stops raising, and this test fails until the row moves.
        """
        for backing in PENDING_BACKINGS:
            with self.subTest(backing=backing):
                with self.assertRaises(NotImplementedError):
                    create_hisparse_coordinator(
                        server_args=_server_args(self, **_flags_for(backing)),
                        # Not a config stand-in: the factory only reads the model's
                        # own indexer top-k off this, and no bag carries it.
                        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
                        req_to_token_pool=None,
                        token_to_kv_pool_allocator=None,
                        device="cuda",
                        tp_group=None,
                        pp_size=1,
                        is_speculative=False,
                    )

    def test_backings_implement_every_protocol_method(self):
        for backing, impl in BACKING_IMPLEMENTATIONS.items():
            missing = [
                name
                for name in self._protocol_methods()
                if not callable(getattr(impl, name, None))
            ]
            with self.subTest(backing=backing):
                self.assertEqual(missing, [], f"{impl.__name__} is missing {missing}")

    def test_backings_match_the_protocol_signatures(self):
        """Presence is not enough: a caller passes keywords, so a backing that
        kept positional-only parameters -- or renamed one -- breaks only when that
        configuration actually runs."""
        for backing, impl in BACKING_IMPLEMENTATIONS.items():
            for name in self._protocol_methods():
                expected = inspect.signature(getattr(HiSparseCoordinator, name))
                actual = inspect.signature(getattr(impl, name))
                with self.subTest(backing=backing, method=name):
                    self.assertEqual(
                        _callable_parameters(expected),
                        _callable_parameters(actual),
                        f"{impl.__name__}.{name} does not match the protocol",
                    )

    def test_backings_declare_their_identity(self):
        # Runtime consumers read coordinator.backing instead of re-resolving the
        # configuration, so a backing that forgot to declare it -- or copied a
        # sibling's value -- would silently send every caller down the wrong branch.
        for backing, impl in BACKING_IMPLEMENTATIONS.items():
            with self.subTest(backing=backing):
                self.assertIs(impl.backing, backing)


if __name__ == "__main__":
    unittest.main()
