"""CPU unit tests for the HiSparse x UnifiedRadixCache (Scheme 1) host-tier logic.

Pure-Python coverage that needs no GPU: exercises the methods unbound against
lightweight stubs (or real UnifiedTreeNode where cheap), so the token<->host-slot
arithmetic, the owned/discard accounting, the host-prefix gather, the sliced host
free, and the allocator deferred-release quarantine are all pinned independent of
a running model.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.mem_cache.sparsity.host_ledger import HiSparseHostLedger
from sglang.srt.mem_cache.unified_cache.component_type import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGatherHostPrefixIndices(CustomTestCase):
    """UnifiedTreeCore.gather_host_prefix_indices: root-first concat of the
    backed-up chain ending at a node."""

    @staticmethod
    def _node(host_value):
        n = UnifiedTreeNode(tree_components=(ComponentType.FULL, ComponentType.MAMBA))
        if host_value is not None:
            n.component_data[BASE_COMPONENT_TYPE].host_value = torch.tensor(host_value)
        return n

    def _core(self, root):
        # node_by_id is identity (we pass node objects as ids); only the three
        # attrs gather reads are needed.
        return SimpleNamespace(
            node_by_id=lambda nid: nid,
            root_node=root,
            device=torch.device("cpu"),
        )

    def test_concatenates_chain_root_first(self):
        root = self._node(None)
        a = self._node([0, 1, 2, 3])
        a.parent = root
        b = self._node([4, 5, 6, 7])
        b.parent = a
        core = self._core(root)
        out = UnifiedTreeCore.gather_host_prefix_indices(core, b, BASE_COMPONENT_TYPE)
        self.assertEqual(out.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        # shallower node yields only its prefix
        out_a = UnifiedTreeCore.gather_host_prefix_indices(core, a, BASE_COMPONENT_TYPE)
        self.assertEqual(out_a.tolist(), [0, 1, 2, 3])

    def test_root_gives_empty(self):
        root = self._node(None)
        core = self._core(root)
        out = UnifiedTreeCore.gather_host_prefix_indices(
            core, root, BASE_COMPONENT_TYPE
        )
        self.assertEqual(out.numel(), 0)


class TestRecordHostInsert(CustomTestCase):
    """HiSparseCoordinator.record_host_insert -> host_ledger.hand_to_tree."""

    def _ledger(self, allocated_len=256):
        pool = _FakeHostPool(0, page_size=4)
        table = torch.zeros((1, allocated_len), dtype=torch.int64)
        allocated = torch.tensor([allocated_len])
        for i in range(allocated_len):
            table[0, i] = i
        return HiSparseHostLedger(pool, table, allocated)

    def _coord(self, ledger=None):
        calls = []
        ledger = ledger or self._ledger()
        coord = SimpleNamespace(
            host_ledger=ledger,
            backup_index_k_range=lambda req, s, e, *, written_by_forward: calls.append(
                (s, e, written_by_forward)
            ),
        )
        return coord, ledger, calls

    def _rec(self, coord, req, matched, key_len, wbf=True):
        HiSparseCoordinator.record_host_insert(
            coord, req, matched_len=matched, key_len=key_len, written_by_forward=wbf
        )

    @staticmethod
    def _req(slot=0):
        return SimpleNamespace(kv=SimpleNamespace(req_pool_idx=slot))

    def test_first_insert_sets_owned_no_discard(self):
        coord, ledger, calls = self._coord()
        self._rec(coord, self._req(), matched=0, key_len=10, wbf=False)
        rec = ledger.snapshot()[0]
        self.assertEqual(rec.owned_len, 10)
        self.assertEqual(rec.discarded, [])
        self.assertEqual(calls, [(0, 10, False)])  # backup_index_k_range forwarded

    def test_rematch_of_own_node_is_not_a_discard(self):
        coord, ledger, _ = self._coord()
        self._rec(coord, self._req(), matched=0, key_len=10, wbf=False)
        self._rec(coord, self._req(), matched=10, key_len=20, wbf=True)
        rec = ledger.snapshot()[0]
        self.assertEqual(rec.owned_len, 20)
        self.assertEqual(rec.discarded, [])

    def test_discard_when_foreign_node_covers_range(self):
        coord, ledger, _ = self._coord()
        self._rec(coord, self._req(), matched=0, key_len=10, wbf=False)
        # a node this req did NOT hand over covers [10,16): discard it
        self._rec(coord, self._req(), matched=16, key_len=24, wbf=True)
        rec = ledger.snapshot()[0]
        self.assertEqual(rec.owned_len, 24)
        self.assertEqual(rec.discarded, [(10, 16)])

    def test_borrowed_prefix_rematch_not_discarded(self):
        ledger = self._ledger()
        ledger.adopt(0, 8, torch.arange(8))
        coord, _, _ = self._coord(ledger)
        # matched==borrowed is a re-match of the borrowed prefix, not a discard
        self._rec(coord, self._req(), matched=8, key_len=20, wbf=True)
        rec = ledger.snapshot()[0]
        self.assertEqual(rec.owned_len, 20)
        self.assertEqual(rec.discarded, [])

    def test_ledger_verify_is_clean(self):
        coord, ledger, _ = self._coord()
        self._rec(coord, self._req(), matched=0, key_len=10, wbf=False)
        self._rec(coord, self._req(), matched=16, key_len=24, wbf=True)
        self.assertEqual(ledger.verify(), [])


class TestLoadBackPrefixIndexK(CustomTestCase):
    """HiSparseCoordinator.load_back_prefix_index_k restores the adopted prefix."""

    def _coord(self, slot, borrowed):
        table = torch.full((4, 32), -1, dtype=torch.int64)
        ledger = HiSparseHostLedger(
            _FakeHostPool(0, page_size=4), table, torch.zeros(4, dtype=torch.int64)
        )
        if borrowed:
            ledger.adopt(slot, borrowed, torch.arange(100, 100 + borrowed))
        calls = []
        coord = object.__new__(HiSparseCoordinator)
        coord.host_ledger = ledger
        coord.req_to_host_pool = table
        coord.is_dsv4_hisparse = False
        coord._has_pending_index_k_backup = False
        coord.indexer_host_pool = SimpleNamespace(
            load_to_device_per_layer=lambda dev, host, device, layer_id, io_backend: (
                calls.append((host.tolist(), device.tolist(), layer_id))
            )
        )
        coord.mem_pool_device = SimpleNamespace(layer_num=2)
        coord.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(4 * 32, dtype=torch.int32).reshape(4, 32)
        )
        return coord, calls

    def test_restores_borrowed_rows_for_every_layer(self):
        coord, calls = self._coord(slot=2, borrowed=8)
        coord.load_back_prefix_index_k(
            SimpleNamespace(kv=SimpleNamespace(req_pool_idx=2))
        )
        host = list(range(100, 108))
        device = list(range(64, 72))
        self.assertEqual(calls, [(host, device, 0), (host, device, 1)])

    def test_no_borrow_loads_nothing(self):
        coord, calls = self._coord(slot=2, borrowed=0)
        coord.load_back_prefix_index_k(
            SimpleNamespace(kv=SimpleNamespace(req_pool_idx=2))
        )
        self.assertEqual(calls, [])


class TestAllocatedHostIndicesStart(CustomTestCase):
    """HiSparseHostPoolMixin.allocated_host_indices honors start= and drops -1."""

    def _call(self, row, allocated_len, start):
        pool = SimpleNamespace(_round_up_to_page_size=lambda n: n)  # page_size=1
        return HiSparseHostPoolMixin.allocated_host_indices(
            pool, row, 0, allocated_len, start=start
        )

    def test_start_zero_filters_negatives(self):
        row = torch.tensor([[10, 11, -1, 13, 14, -1]])
        self.assertEqual(self._call(row, 6, 0).tolist(), [10, 11, 13, 14])

    def test_start_offset_only_tail(self):
        row = torch.tensor([[10, 11, -1, 13, 14, -1]])
        self.assertEqual(self._call(row, 6, 3).tolist(), [13, 14])

    def test_start_beyond_len_is_empty(self):
        row = torch.tensor([[10, 11, -1, 13, 14, -1]])
        self.assertEqual(self._call(row, 6, 6).numel(), 0)


class _FakeEvent:
    def __init__(self, done):
        self._done = done

    def query(self):
        return self._done

    def synchronize(self):
        self._done = True


class TestPagedQuarantine(CustomTestCase):
    """PagedTokenToKVPoolAllocator deferred-release: freed pages wait in
    quarantine until the barrier event's query() passes."""

    def _alloc_stub(self):
        stub = SimpleNamespace(
            need_sort=False,
            free_pages=torch.tensor([5, 6, 7]),
            staged_pages=[],
            num_staged_pages=0,
            _reuse_guard_event=None,
            _staged_guard_event=None,
            _release_barrier=None,
            _quarantine=[],
            _quarantined_pages=0,
            page_size=1,
        )
        # _release_page_ids / _drain_quarantine call self._release_page_ids_now;
        # bind the real one so the unbound-on-stub pattern reaches it.
        stub._release_page_ids_now = lambda *p, _s=stub: (
            PagedTokenToKVPoolAllocator._release_page_ids_now(_s, *p)
        )
        return stub

    def test_quarantine_then_release_on_query(self):
        stub = self._alloc_stub()
        ev = _FakeEvent(False)
        PagedTokenToKVPoolAllocator.set_release_barrier(stub, ev)
        self.assertIs(stub._release_barrier, ev)

        # barrier not done -> freed pages are held, not returned
        PagedTokenToKVPoolAllocator._release_page_ids(stub, torch.tensor([1, 2]))
        self.assertEqual(stub._quarantined_pages, 2)
        self.assertEqual(stub.free_pages.tolist(), [5, 6, 7])

        # non-blocking drain while still pending keeps them
        PagedTokenToKVPoolAllocator._drain_quarantine(stub)
        self.assertEqual(stub._quarantined_pages, 2)

        # event completes -> drain returns them to the free list
        ev._done = True
        PagedTokenToKVPoolAllocator._drain_quarantine(stub)
        self.assertEqual(stub._quarantined_pages, 0)
        self.assertEqual(sorted(stub.free_pages.tolist()), [1, 2, 5, 6, 7])

    def test_immediate_release_when_barrier_done(self):
        stub = self._alloc_stub()
        PagedTokenToKVPoolAllocator.set_release_barrier(stub, _FakeEvent(True))
        PagedTokenToKVPoolAllocator._release_page_ids(stub, torch.tensor([1, 2]))
        self.assertEqual(stub._quarantined_pages, 0)
        self.assertEqual(sorted(stub.free_pages.tolist()), [1, 2, 5, 6, 7])

    def test_inert_when_no_barrier(self):
        stub = self._alloc_stub()
        PagedTokenToKVPoolAllocator._release_page_ids(stub, torch.tensor([1, 2]))
        self.assertEqual(stub._quarantined_pages, 0)
        self.assertEqual(sorted(stub.free_pages.tolist()), [1, 2, 5, 6, 7])

    def test_available_size_counts_quarantine(self):
        # Rank-deterministic: must not depend on when each rank's event completes.
        stub = self._alloc_stub()
        PagedTokenToKVPoolAllocator.set_release_barrier(stub, _FakeEvent(False))
        PagedTokenToKVPoolAllocator._release_page_ids(stub, torch.tensor([1, 2]))
        self.assertEqual(PagedTokenToKVPoolAllocator.available_size(stub), 5)

    def test_alloc_shortfall_uses_quarantine_under_need_sort(self):
        # need_sort (PD) stages drained pages; alloc must merge them after the
        # block-drain or it fails while available_size() reports enough.
        alloc = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
        alloc.page_size = 1
        alloc.need_sort = True
        alloc.debug_mode = False
        alloc.device = "cpu"
        alloc.free_pages = torch.tensor([5, 6, 7])
        alloc.staged_pages = []
        alloc.num_staged_pages = 0
        alloc._reuse_guard_event = None
        alloc._staged_guard_event = None
        alloc._release_barrier = None
        alloc._quarantine = []
        alloc._quarantined_pages = 0
        alloc.set_release_barrier(_FakeEvent(False))
        alloc._release_page_ids(torch.tensor([1, 2]))
        self.assertEqual(alloc.available_size(), 5)
        out = alloc.alloc(5)
        self.assertIsNotNone(out)
        self.assertEqual(sorted(out.tolist()), [1, 2, 5, 6, 7])
        self.assertEqual(alloc._quarantined_pages, 0)
        self.assertEqual(alloc.available_size(), 0)


class _FakeHostPool:
    def __init__(self, available, page_size=4):
        self.available = available
        self.page_size = page_size
        self._next = 1000

    def available_size(self):
        return self.available

    def alloc_paged_token_slots(self, table, allocated, idx, start, num_tokens):
        """Mirror HiSparseHostPoolMixin: page-granular slots, token-granular view."""
        allocated_len = int(allocated[idx])
        page_end = -(-(start + num_tokens) // self.page_size) * self.page_size
        if page_end > allocated_len:
            n = page_end - allocated_len
            locs = torch.arange(self._next, self._next + n, dtype=torch.int64)
            self._next += n
            table[idx, allocated_len:page_end] = locs
            allocated[idx] = page_end
        return table[idx, start : start + num_tokens]


class TestEnsureHostFree(CustomTestCase):
    """HiSparseCoordinator._ensure_host_free: decode-time host allocs reclaim
    tree-held slots under host-radix, and are inert without an evict hook."""

    @staticmethod
    def _coord(available, evict_fn=None):
        return SimpleNamespace(
            mem_pool_host=_FakeHostPool(available), _host_evict_fn=evict_fn
        )

    def test_evicts_only_the_shortfall(self):
        calls = []
        coord = self._coord(available=1)
        coord._host_evict_fn = calls.append
        HiSparseCoordinator._ensure_host_free(coord, 4)
        self.assertEqual(calls, [3])

    def test_no_evict_when_enough(self):
        calls = []
        coord = self._coord(available=4, evict_fn=calls.append)
        HiSparseCoordinator._ensure_host_free(coord, 4)
        self.assertEqual(calls, [])

    def test_inert_without_hook(self):
        HiSparseCoordinator._ensure_host_free(self._coord(available=0), 4)


class TestNextStepHostDemand(CustomTestCase):
    """HiSparseCoordinator.next_step_host_demand: the per-step host check, as a
    KVDemand, that lets decode past the clip retract instead of failing alloc."""

    PAGE = 64

    def _coord(self, allocated_lens, available, evictable=0):
        pool = _FakeHostPool(available, page_size=self.PAGE)
        evicted = []

        def evict(n):
            freed = min(n, evictable - sum(evicted))
            evicted.append(freed)
            pool.available += freed

        coord = SimpleNamespace(
            req_to_host_pool_allocated_len=torch.tensor(allocated_lens),
            page_size=self.PAGE,
            mem_pool_host=pool,
            _host_evict_fn=evict,
            host_token_len=lambda n: n,
        )
        coord._ensure_host_free = lambda n: HiSparseCoordinator._ensure_host_free(
            coord, n
        )
        coord.host_available = lambda: HiSparseCoordinator.host_available(coord)
        coord.host_reclaim = lambda n: HiSparseCoordinator.host_reclaim(coord, n)
        return coord, pool, evicted

    @staticmethod
    def _req(slot, seqlen):
        return SimpleNamespace(kv=SimpleNamespace(req_pool_idx=slot), seqlen=seqlen)

    def _demand(self, coord, reqs):
        return HiSparseCoordinator.next_step_host_demand(coord, reqs)

    def _fits(self, coord, reqs):
        demand = self._demand(coord, reqs)
        return coord.host_reclaim(demand.host_tokens) >= demand.host_tokens

    def test_mid_page_needs_nothing(self):
        # Next backup lands inside an allocated page: no host demand at all.
        coord, _, evicted = self._coord([128], available=0)
        self.assertEqual(self._demand(coord, [self._req(0, 100)]).host_tokens, 0)
        self.assertTrue(self._fits(coord, [self._req(0, 100)]))
        self.assertEqual(evicted, [])

    def test_page_boundary_asks_one_page(self):
        coord, _, evicted = self._coord([128], available=self.PAGE)
        self.assertEqual(
            self._demand(coord, [self._req(0, 129)]).host_tokens, self.PAGE
        )
        self.assertTrue(self._fits(coord, [self._req(0, 129)]))
        self.assertEqual(evicted, [])

    def test_page_boundary_reclaims_tree_cache(self):
        coord, _, evicted = self._coord([128], available=0, evictable=self.PAGE)
        self.assertTrue(self._fits(coord, [self._req(0, 129)]))
        self.assertEqual(evicted, [self.PAGE])

    def test_past_clip_exhausted_host_retracts_until_it_fits(self):
        # Two requests past the 4096-token reservation both cross a page boundary
        # with host and tree empty: the check fails (scheduler retracts), and once
        # one victim's pages return the survivor fits -- retract_decode's
        # convergence.
        allocated = 1000 + 4200 - 8  # page-aligned in the real pool; only the
        allocated -= allocated % self.PAGE  # boundary position matters here
        coord, pool, _ = self._coord([allocated, allocated], available=0)
        reqs = [self._req(0, allocated + 1), self._req(1, allocated + 1)]
        self.assertEqual(self._demand(coord, reqs).host_tokens, 2 * self.PAGE)
        self.assertFalse(self._fits(coord, reqs))
        pool.available += allocated  # retract request 1: its host slots return
        self.assertTrue(self._fits(coord, reqs[:1]))

    def test_unslotted_request_is_skipped(self):
        coord, _, _ = self._coord([0], available=0)
        self.assertEqual(self._demand(coord, [self._req(None, 10)]).host_tokens, 0)


class TestHiSparseCheckDecodeCapacity(CustomTestCase):
    """HiSparseTokenToKVPoolAllocator.check_decode_capacity folds the host demand
    into the per-step decode capacity gate the retract loop converges on."""

    def _alloc(
        self, logical, hot, host_demand=None, binding=True, available=0, reclaim_to=None
    ):
        alloc = object.__new__(HiSparseTokenToKVPoolAllocator)
        alloc.logical_attn_allocator = SimpleNamespace(available_size=lambda: logical)
        alloc.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: hot)
        if host_demand is not None or binding:
            reclaims = []

            def host_reclaim(n):
                reclaims.append(n)
                return available if reclaim_to is None else reclaim_to

            coord = SimpleNamespace(
                next_step_host_demand=lambda reqs: SimpleNamespace(
                    host_tokens=0 if host_demand is None else host_demand
                ),
                host_reclaim=host_reclaim,
                host_available=lambda: available if reclaim_to is None else reclaim_to,
                host_can_bind=binding,
                set_host_radix=lambda enabled: None,
            )
            alloc.set_host_pool(coord)
            alloc._test_reclaims = reclaims
        return alloc

    def _check(self, alloc, num_tokens=8, requests=("r",)):
        return alloc.check_decode_capacity(
            num_tokens=num_tokens, tree_cache=None, requests=list(requests)
        )

    def test_host_shortfall_fails_the_check(self):
        alloc = self._alloc(64, 64, host_demand=64)
        self.assertFalse(self._check(alloc))
        # The step check may reclaim: one attempt, for the whole demand.
        self.assertEqual(alloc._test_reclaims, [64])

    def test_reclaim_lets_the_step_through(self):
        # The step check is allowed to evict (a shortfall retracts, so evicting
        # is the cheaper answer); free-after-reclaim is what it prices.
        alloc = self._alloc(64, 64, host_demand=64, reclaim_to=64)
        self.assertTrue(self._check(alloc))
        self.assertEqual(alloc._test_reclaims, [64])

    def test_device_shortfall_short_circuits_host_check(self):
        alloc = self._alloc(4, 64, host_demand=64)
        self.assertFalse(self._check(alloc))
        self.assertEqual(alloc._test_reclaims, [])

    def test_passes_when_both_fit(self):
        self.assertTrue(self._check(self._alloc(64, 64, host_demand=64, available=64)))

    def test_without_host_pool_only_device_counts(self):
        self.assertTrue(self._check(self._alloc(64, 64, binding=False)))


class TestHostFitsIsPure(CustomTestCase):
    """allocator.host_fits is a predicate: it prices the host pool and never
    evicts. Taking the slots belongs to the caller that has decided to admit."""

    def _alloc(self, available, binding=True):
        alloc = object.__new__(HiSparseTokenToKVPoolAllocator)
        alloc.logical_attn_allocator = SimpleNamespace(available_size=lambda: 64)
        alloc.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: 64)
        reclaims = []
        coord = SimpleNamespace(
            host_can_bind=binding,
            host_available=lambda: available,
            host_reclaim=lambda n: reclaims.append(n) or available,
            set_host_radix=lambda enabled: None,
        )
        alloc.set_host_pool(coord)
        return alloc, reclaims

    def test_prices_available_without_reclaiming(self):
        alloc, reclaims = self._alloc(available=32)
        self.assertFalse(alloc.host_fits(KVDemand(host_tokens=33)))
        self.assertTrue(alloc.host_fits(KVDemand(host_tokens=32)))
        self.assertEqual(reclaims, [])

    def test_demand_without_host_component_always_fits(self):
        alloc, reclaims = self._alloc(available=0)
        self.assertTrue(alloc.host_fits(KVDemand(full_tokens=10)))
        self.assertEqual(reclaims, [])

    def test_unbound_pool_always_fits(self):
        alloc, reclaims = self._alloc(available=0, binding=False)
        self.assertTrue(alloc.host_fits(KVDemand(host_tokens=64)))
        self.assertEqual(reclaims, [])


class TestHostCanBind(CustomTestCase):
    """HiSparseCoordinator.host_can_bind: the one place that decides whether host
    capacity must be priced. A tied pool still binds under host-radix, because
    the slots the L2 tree retains belong to no live request."""

    @staticmethod
    def _coord(decoupled=False, radix=False):
        coord = SimpleNamespace(host_pool_decoupled=decoupled)
        coord.set_host_radix = lambda enabled: HiSparseCoordinator.set_host_radix(
            coord, enabled
        )
        coord.set_host_radix(radix)
        return coord

    @staticmethod
    def _binds(coord):
        return HiSparseCoordinator.host_can_bind.fget(coord)

    def test_tied_without_radix_does_not_bind(self):
        self.assertFalse(self._binds(self._coord()))

    def test_decoupled_pool_binds(self):
        self.assertTrue(self._binds(self._coord(decoupled=True)))

    def test_tied_pool_binds_under_host_radix(self):
        # The regression this guards: host size == logical size is not enough to
        # rule out a host shortfall once the tree retains slots.
        self.assertTrue(self._binds(self._coord(radix=True)))

    def test_decoupled_with_radix_binds(self):
        self.assertTrue(self._binds(self._coord(decoupled=True, radix=True)))

    def test_allocator_prices_host_only_when_the_coordinator_says_so(self):
        alloc = object.__new__(HiSparseTokenToKVPoolAllocator)
        alloc.logical_attn_allocator = SimpleNamespace(available_size=lambda: 64)
        alloc.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: 64)
        coord = SimpleNamespace(
            next_step_host_demand=lambda reqs: SimpleNamespace(host_tokens=64),
            host_reclaim=lambda n: 0,
            host_available=lambda: 0,
            set_host_radix=lambda enabled: None,
            host_can_bind=True,
        )
        alloc.set_host_pool(coord)
        self.assertFalse(
            alloc.check_decode_capacity(num_tokens=8, tree_cache=None, requests=["r"])
        )


class TestHostGrowthReserved(CustomTestCase):
    """Coordinator host accounting: full-output host target and the decode
    growth still owed by admitted requests."""

    @staticmethod
    def _req(idx, prompt, max_new, finished=False):
        return SimpleNamespace(
            origin_input_ids=list(range(prompt)),
            sampling_params=SimpleNamespace(max_new_tokens=max_new),
            kv=SimpleNamespace(req_pool_idx=idx),
            finished=lambda: finished,
        )

    def _coord(self, allocated):
        coord = SimpleNamespace(
            mem_pool_host=_FakeHostPool(0, page_size=4),
            req_to_host_pool_allocated_len=torch.tensor(allocated),
        )
        coord.host_target = lambda req, fill_len: HiSparseCoordinator.host_target(
            coord, req, fill_len
        )
        coord.host_growth_reserved = lambda reqs: (
            HiSparseCoordinator.host_growth_reserved(coord, reqs)
        )
        return coord

    def test_target_is_page_rounded_prompt_plus_output(self):
        coord = self._coord([0])
        self.assertEqual(coord.host_target(self._req(0, 10, 5), 10), 16)

    def test_rebootstrap_target_counts_emitted_tokens(self):
        # A retracted request refills prompt + emitted tokens, so a prompt-only
        # target under-counts and the gate admits a fill that then fails to alloc.
        from sglang.srt.disaggregation.utils import pre_alloc_fill_len

        req = self._req(0, 10, 5)
        req.output_ids = list(range(7))
        coord = self._coord([0])
        fill_len = pre_alloc_fill_len(req)  # 10 + 7 - 1 pending input
        self.assertEqual(coord.host_target(req, fill_len), 24)  # 21 -> 24

    def test_growth_sums_active_requests(self):
        running = self._req(0, 10, 5)  # target 16, allocated 12 -> owes 4
        waiting = self._req(1, 8, 8)  # target 16, allocated 8 -> owes 8
        moving = self._req(2, 4, 0)  # target 4, allocated 4 -> owes 0
        coord = self._coord([12, 8, 4])
        self.assertEqual(coord.host_growth_reserved([running, waiting, moving]), 12)

    def test_growth_skips_finished_and_unslotted(self):
        done = self._req(0, 10, 5, finished=True)
        no_slot = self._req(None, 10, 5)
        coord = self._coord([0])
        self.assertEqual(coord.host_growth_reserved([done, no_slot]), 0)


class TestHostDemandSeesSamePassGrowth(CustomTestCase):
    """DecodePreallocQueue._hisparse_host_demand: the prealloc queue is part of
    the reservation.

    A request admitted earlier in the same pass keeps its req_pool_idx and stays
    queued until the pass ends, so the next request must charge its growth --
    otherwise both pass the gate and the second fails in alloc_paged_token_slots.
    A queued request without a slot (this one, or the not-yet-admitted) must not
    self-count.
    """

    FILL = 64
    PAGE = 8

    @staticmethod
    def _req(idx, prompt, max_new):
        return SimpleNamespace(
            origin_input_ids=list(range(prompt)),
            sampling_params=SimpleNamespace(max_new_tokens=max_new),
            kv=SimpleNamespace(req_pool_idx=idx),
            finished=lambda: False,
        )

    def _queue(self, queued):
        coord = SimpleNamespace(
            mem_pool_host=_FakeHostPool(0, page_size=self.PAGE),
            req_to_host_pool_allocated_len=torch.zeros(4, dtype=torch.int64),
        )
        coord.host_target = lambda req, fill_len: HiSparseCoordinator.host_target(
            coord, req, fill_len
        )
        coord.host_growth_reserved = lambda reqs: (
            HiSparseCoordinator.host_growth_reserved(coord, reqs)
        )
        return SimpleNamespace(
            scheduler=SimpleNamespace(
                hisparse_coordinator=coord,
                running_batch=SimpleNamespace(reqs=[]),
                waiting_queue=[],
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            queue=SimpleNamespace(items=lambda: list(queued)),
        )

    def _demand(self, queue, req, extra=()):
        return DecodePreallocQueue._hisparse_host_demand(
            queue, req, self.FILL, 0, extra_reserved=extra
        )

    def test_second_request_charges_the_first_requests_growth(self):
        first = self._req(0, self.FILL, 8)  # slotted: admitted this pass
        second = self._req(None, self.FILL, 8)  # itself, not yet admitted
        # Page-aligned target of one request: 64 prompt + 8 output.
        target = 72
        alone = self._demand(self._queue([]), second)
        together = self._demand(
            self._queue([SimpleNamespace(req=first), SimpleNamespace(req=second)]),
            second,
        )
        self.assertEqual(alone.host_tokens, target)
        self.assertEqual(together.host_tokens, target + target)

    def test_an_unslotted_queued_request_adds_nothing(self):
        second = self._req(None, self.FILL, 8)
        alone = self._demand(self._queue([]), second)
        self.assertEqual(
            self._demand(self._queue([SimpleNamespace(req=second)]), second),
            alone,
        )

    def test_resume_path_charges_requests_already_resumed(self):
        # Resumed requests leave the retracted queue only after the loop, so the
        # resume path charges them through extra_reserved instead.
        first = self._req(0, self.FILL, 8)  # resumed and re-slotted this pass
        second = self._req(None, self.FILL, 8)
        queue = self._queue([])
        alone = self._demand(queue, second)
        together = self._demand(queue, second, extra=[first])
        self.assertEqual(together.host_tokens, alone.host_tokens + 72)


class TestHiSparseHostAdmit(CustomTestCase):
    """DecodePreallocQueue._hisparse_host_admit takes the slots: it reclaims
    once, prices what is free afterwards, and defers the request otherwise."""

    FILL = 64

    @classmethod
    def _req(cls):
        # pre_alloc_fill_len reads the fill off these: FILL prompt, no output.
        return SimpleNamespace(
            rid="r",
            kv=SimpleNamespace(req_pool_idx=None),
            finished=lambda: False,
            origin_input_ids=[0] * cls.FILL,
            output_ids=[],
        )

    def _queue(self, available, binding=True):
        reclaims = []

        def host_reclaim(n):
            reclaims.append(n)
            return available

        coord = SimpleNamespace(
            host_can_bind=binding,
            host_reclaim=host_reclaim,
            host_available=lambda: available,
            # Demand is the whole fill plus 100 output tokens.
            host_target=lambda req, fill_len: fill_len + 100,
            host_growth_reserved=lambda reqs: 0,
        )
        queue = SimpleNamespace(
            scheduler=SimpleNamespace(
                hisparse_coordinator=coord,
                running_batch=SimpleNamespace(reqs=[]),
                waiting_queue=[],
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            queue=SimpleNamespace(items=lambda: []),
        )
        # Unbound calls still go through `self._hisparse_host_demand`, so the
        # stub needs the real method bound onto it.
        queue._hisparse_host_demand = (
            lambda req, fill_len, total_prefix_len, *, extra_reserved=(): (
                DecodePreallocQueue._hisparse_host_demand(
                    queue,
                    req,
                    fill_len,
                    total_prefix_len,
                    extra_reserved=extra_reserved,
                )
            )
        )
        return queue, reclaims

    def test_shortfall_defers_after_exactly_one_reclaim(self):
        queue, reclaims = self._queue(available=0)
        self.assertFalse(DecodePreallocQueue._hisparse_host_admit(queue, self._req()))
        self.assertEqual(reclaims, [self.FILL + 100])

    def test_enough_free_admits(self):
        queue, reclaims = self._queue(available=self.FILL + 100)
        self.assertTrue(DecodePreallocQueue._hisparse_host_admit(queue, self._req()))
        self.assertEqual(reclaims, [self.FILL + 100])

    def test_no_host_component_needs_no_reclaim(self):
        queue, reclaims = self._queue(available=0)
        # The borrowed prefix covers the whole target: nothing new to take.
        self.assertTrue(
            DecodePreallocQueue._hisparse_host_admit(
                queue, self._req(), self.FILL + 100
            )
        )
        self.assertEqual(reclaims, [])

    def test_unbound_pool_admits_without_pricing(self):
        queue, reclaims = self._queue(available=0, binding=False)
        self.assertTrue(DecodePreallocQueue._hisparse_host_admit(queue, self._req()))
        self.assertEqual(reclaims, [])


class TestHostBorrowBeforeSlotIndex(CustomTestCase):
    """The match-time host borrow must survive until pre-alloc assigns a slot.

    A fresh decode request has no req_pool_idx when it matches the host radix, so
    the ledger cannot key the receipt yet; it rides on the match instead. Keying
    it under the (None) slot let the receipt vanish and the adopt assert.
    """

    def _queue(self):
        pool = _FakeHostPool(0, page_size=4)
        table = torch.zeros((1, 8), dtype=torch.int64)
        ledger = HiSparseHostLedger(pool, table, torch.zeros(1, dtype=torch.int64))
        gathered, dropped = [], []
        tree = SimpleNamespace(
            get_host_prefix_indices=lambda node: (
                gathered.append(node),
                torch.arange(16, dtype=torch.int64),
            )[1],
            dec_host_lock_ref=lambda node, params: dropped.append((node, params)),
        )
        q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        q.scheduler = SimpleNamespace(
            enable_hisparse=True,
            enable_hisparse_host_radix=True,
            hisparse_coordinator=SimpleNamespace(host_ledger=ledger),
        )
        q.tree_cache = tree
        return q, ledger, gathered, dropped

    @staticmethod
    def _req(slot=None, node=7, params="receipt"):
        req = SimpleNamespace(rid="r0", kv=SimpleNamespace(req_pool_idx=slot))
        if node is not None:
            req._hisparse_pending_host_lock = (node, params)
        return req

    def test_gather_uses_the_pending_receipt_not_the_slot(self):
        q, _, gathered, _ = self._queue()
        out = q._gather_hisparse_host_prefix(self._req(node=11))
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 16)
        self.assertEqual(gathered, [11])

    def test_no_receipt_gathers_nothing(self):
        q, _, gathered, _ = self._queue()
        self.assertIsNone(q._gather_hisparse_host_prefix(self._req(node=None)))
        self.assertEqual(gathered, [])

    def test_receipt_is_filed_only_once_a_slot_exists(self):
        q, ledger, _, _ = self._queue()
        req = self._req(slot=None)
        q._file_host_lock(req)
        self.assertFalse(ledger.tracks(0))
        self.assertEqual(req._hisparse_pending_host_lock, (7, "receipt"))
        req.kv.req_pool_idx = 0
        q._file_host_lock(req)
        self.assertIsNone(req._hisparse_pending_host_lock)
        self.assertEqual(ledger.borrowed_node(0), 7)

    def test_release_before_adoption_unlocks_from_the_req(self):
        q, ledger, _, dropped = self._queue()
        req = self._req(slot=None)
        q._release_host_prefix_ref(req)
        self.assertEqual(dropped, [(7, "receipt")])
        self.assertIsNone(req._hisparse_pending_host_lock)
        self.assertFalse(ledger.tracks(0))

    def test_release_after_adoption_unlocks_from_the_ledger(self):
        q, ledger, _, dropped = self._queue()
        req = self._req(slot=0)
        q._file_host_lock(req)
        q._release_host_prefix_ref(req)
        self.assertEqual(dropped, [(7, "receipt")])
        self.assertEqual(ledger.pop_nodes(0), [])


class TestHostTokenCapacityConfig(CustomTestCase):
    """hisparse-config host_token_capacity parsing."""

    def _parse(self, cfg):
        from unittest import mock

        from sglang.srt.mem_cache.sparsity import factory

        with mock.patch.object(
            factory, "get_memory", return_value=SimpleNamespace(hisparse_config=cfg)
        ):
            return factory._parse_sparse_config()

    def test_default_is_none(self):
        self.assertIsNone(self._parse('{"top_k": 2048}').host_token_capacity)

    def test_parsed_and_not_left_in_extra(self):
        cfg = self._parse('{"top_k": 2048, "host_token_capacity": 4600000}')
        self.assertEqual(cfg.host_token_capacity, 4600000)
        self.assertNotIn("host_token_capacity", cfg.sparse_extra_config)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            self._parse('{"top_k": 2048, "host_token_capacity": 0}')


class TestUsesHisparseHostRadix(CustomTestCase):
    """The single predicate every host-radix branch gates on. Getting it wrong
    silently routes HiSparse to the wrong tree, so pin each disqualifier."""

    @staticmethod
    def _args(**over):
        base = dict(
            enable_hisparse=True,
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            enable_hierarchical_cache=True,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def _uses(self, arch="DeepseekV32ForCausalLM", **over):
        from sglang.srt.arg_groups.hisparse_hook import uses_hisparse_host_radix

        return uses_hisparse_host_radix(self._args(**over), {"architectures": [arch]})

    def test_all_three_flags_on_a_dsa_decode_engine(self):
        self.assertTrue(self._uses())

    def test_hisparse_off(self):
        self.assertFalse(self._uses(enable_hisparse=False))

    def test_prefill_engine(self):
        self.assertFalse(self._uses(disaggregation_mode="prefill"))

    def test_single_engine(self):
        self.assertFalse(self._uses(disaggregation_mode="null"))

    def test_decode_radix_cache_off(self):
        self.assertFalse(self._uses(disaggregation_decode_enable_radix_cache=False))

    def test_hierarchical_cache_off(self):
        self.assertFalse(self._uses(enable_hierarchical_cache=False))

    def test_dsv4_is_excluded(self):
        # C4 host layout plus SWA/C128 layers need prefix recovery the host tree
        # cannot provide, so dsv4 must fall back even with every flag on.
        self.assertFalse(self._uses(arch="DeepseekV4ForCausalLM"))


class TestReleaseHostAfterInsert(CustomTestCase):
    """HiSparseCoordinator.release_host_after_insert: the ledger names the ranges
    and the host locks; the coordinator frees and unlocks."""

    def _coord(self, allocated_len=256):
        pool = _FakeHostPool(0, page_size=4)
        table = torch.zeros((1, allocated_len), dtype=torch.int64)
        allocated = torch.tensor([allocated_len])
        for i in range(allocated_len):
            table[0, i] = i
        ledger = HiSparseHostLedger(pool, table, allocated)
        allocated[:] = 0
        table[:] = -1
        ledger.alloc(0, 0, allocated_len)
        ledger._records.clear()
        freed, dropped = [], []
        coord = SimpleNamespace(
            host_ledger=ledger,
            req_to_host_pool=table,
            _host_borrow_release_fn=lambda node, params: dropped.append((node, params)),
            _free_host_range=lambda idx, start, end: (
                freed.append((start, end)) if end > start else None
            ),
        )
        return coord, ledger, freed, dropped

    @staticmethod
    def _req(slot=0):
        return SimpleNamespace(rid="r0", kv=SimpleNamespace(req_pool_idx=slot))

    def _release(self, coord, req=None, idx=0):
        HiSparseCoordinator.release_host_after_insert(coord, req or self._req(), idx)

    def test_no_insert_no_borrow_frees_everything(self):
        coord, _, freed, _ = self._coord()
        self._release(coord)
        self.assertEqual(freed, [(0, 256)])

    def test_borrowed_prefix_is_never_freed(self):
        coord, ledger, freed, _ = self._coord()
        ledger.adopt(0, 64, torch.arange(64))
        self._release(coord)
        self.assertEqual(freed, [(64, 256)])

    def test_owned_range_is_left_to_evict_host(self):
        coord, ledger, freed, _ = self._coord()
        ledger.hand_to_tree(0, 0, 192)
        self._release(coord)
        self.assertEqual(freed, [(192, 256)])

    def test_discarded_holes_are_freed_separately(self):
        coord, ledger, freed, _ = self._coord()
        ledger.adopt(0, 64, torch.arange(64))
        ledger.hand_to_tree(0, 128, 192)  # [64,128) skipped by the insert
        self._release(coord)
        self.assertEqual(freed, [(64, 128), (192, 256)])

    def test_discard_past_allocated_is_clamped(self):
        # Defensive: a discard range is always inside the allocated span, but
        # clamping keeps a stale one from freeing slots this request never had.
        coord, ledger, freed, _ = self._coord()
        ledger.adopt(0, 128, torch.arange(128))
        ledger.hand_to_tree(0, 4096, 4096)
        self._release(coord)
        self.assertEqual(freed, [(128, 256)])

    def test_fully_owned_frees_nothing(self):
        coord, ledger, freed, _ = self._coord()
        ledger.hand_to_tree(0, 0, 256)
        self._release(coord)
        self.assertEqual(freed, [])

    def test_state_is_reset_for_a_resumed_request(self):
        # A retract-resumed request never re-matches, so a stale borrow length
        # would make the next teardown skip that many live slots.
        coord, ledger, _, _ = self._coord()
        ledger.adopt(0, 64, torch.arange(64))
        ledger.hand_to_tree(0, 128, 128)
        self._release(coord)
        self.assertFalse(ledger.tracks(0))
        self.assertEqual(ledger.allocated(0), 0)

    def test_borrow_refs_are_dropped(self):
        coord, ledger, _, dropped = self._coord()
        ledger.borrow_node(0, 7, "receipt")
        self._release(coord)
        self.assertEqual(dropped, [(7, "receipt")])
        # The borrow is gone, so the node id must not survive to be gathered.
        self.assertIsNone(ledger.borrowed_node(0))

    def test_caller_passes_the_slot_index(self):
        # release_kv_cache nulls req.kv.req_pool_idx before this runs, so the
        # index comes from the caller, not from the request.
        coord, _, freed, _ = self._coord()
        req = self._req()
        req.kv.req_pool_idx = None
        self._release(coord, req, idx=0)
        self.assertEqual(freed, [(0, 256)])


class TestRustTreeCoreFallback(CustomTestCase):
    """HiSparse host-radix must land on the Python TreeCore: the Rust one has no
    gather_host_prefix_indices, so a host-prefix match would raise at runtime."""

    def _reason(self, hisparse_host_kv_pool):
        from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
            _rust_fallback_reason,
        )

        params = SimpleNamespace(
            enable_session_radix_cache=False,
            tree_slots=None,
            component_registry_override=None,
            hisparse_host_kv_pool=hisparse_host_kv_pool,
            # Only read by the checks after the HiSparse clause.
            token_to_kv_pool_allocator=None,
        )
        return _rust_fallback_reason(params)

    def test_hisparse_host_pool_forces_python(self):
        reason = self._reason(object())
        self.assertIsNotNone(reason)
        self.assertIn("HiSparse", reason)

    def test_without_the_host_pool_the_check_does_not_fire(self):
        # Later checks (platform, torch version) may still fall back; this only
        # pins that the HiSparse clause is not what rejected it.
        reason = self._reason(None)
        if reason is not None:
            self.assertNotIn("HiSparse", reason)

    def test_the_gather_is_not_on_the_interface(self):
        # Deliberately not abstract: the Rust core cannot implement it, and
        # making it abstract would leave RustUnifiedTreeCore un-instantiable.
        from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
            UnifiedTreeCoreInterface,
        )

        self.assertNotIn(
            "gather_host_prefix_indices",
            UnifiedTreeCoreInterface.__abstractmethods__,
        )


class TestRequestRetirerHostRadixSplit(CustomTestCase):
    """RequestRetirer.release: under host-radix the host half of the HiSparse
    teardown runs after release_kv_cache, so the checkpoint insert still sees
    the request's host table."""

    def _release(self, host_radix: bool, checkpoint: bool, staging: bool = False):
        from unittest import mock

        from sglang.srt.managers.scheduler_components import request_retirer

        calls = []

        def release_request_device(req):
            # A staging request is freed whole, so no host half is owed.
            calls.append("staging" if staging else "device")
            return None if staging else 0

        coord = SimpleNamespace(
            release_request=lambda req: calls.append("full"),
            release_request_device=release_request_device,
            release_host_after_insert=lambda req, idx: calls.append("host"),
        )
        retirer = request_retirer.RequestRetirer(
            tree_cache=SimpleNamespace(),
            beam_coordinator=SimpleNamespace(),
            hisparse_coordinator=coord,
            uocr_runtime=None,
            send_abort_output=lambda abort_req, req: None,
            stream_output=lambda *a, **k: None,
            enable_hisparse_host_radix=host_radix,
        )
        req = SimpleNamespace(
            finished_reason=None, kv=SimpleNamespace(holds_kv=True, holds_mamba=False)
        )

        def fake_release(req, tree_cache, checkpoint):
            calls.append(f"release(checkpoint={checkpoint})")
            req.kv.holds_kv = False

        with (
            mock.patch.object(
                request_retirer, "release_kv_cache", side_effect=fake_release
            ),
            mock.patch.object(request_retirer, "discard_kv_cache_backup"),
            mock.patch.object(request_retirer, "record_exit"),
        ):
            retirer.release(req, checkpoint=checkpoint)
        return calls

    def test_host_radix_splits_teardown_around_the_insert(self):
        self.assertEqual(
            self._release(host_radix=True, checkpoint=True),
            ["device", "release(checkpoint=True)", "host"],
        )

    def test_without_host_radix_tears_down_before_release(self):
        self.assertEqual(
            self._release(host_radix=False, checkpoint=True),
            ["full", "release(checkpoint=True)"],
        )

    def test_host_radix_staging_request_skips_the_host_half(self):
        # _release_staging already freed the host slots; a second host release
        # would find neither req_pool_idx nor the stash and assert.
        self.assertEqual(
            self._release(host_radix=True, checkpoint=True, staging=True),
            ["staging", "release(checkpoint=True)"],
        )


class TestRebootstrapPrefixReuse(CustomTestCase):
    """A retracted request may reuse only what the host tree still holds.

    Retraction releases the request's locks, so the tree may evict the prefix
    before re-admission: the match is best-effort and a miss falls back to a
    full recompute. Native HiSparse keeps no host-tier tree, so it never reuses.
    """

    @staticmethod
    def _mixin(host_radix: bool) -> DecodePreallocQueue:
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(enable_hisparse_host_radix=host_radix)
        return queue

    def test_host_radix_allows_best_effort_reuse(self):
        self.assertTrue(self._mixin(True)._rebootstrap_may_reuse_prefix())

    def test_native_hisparse_recomputes_from_token_zero(self):
        self.assertFalse(self._mixin(False)._rebootstrap_may_reuse_prefix())


if __name__ == "__main__":
    unittest.main()
