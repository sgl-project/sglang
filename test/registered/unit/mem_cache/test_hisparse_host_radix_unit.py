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

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
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
    """HiSparseCoordinator.record_host_insert owned/discard bookkeeping."""

    def _coord(self):
        calls = []
        stub = SimpleNamespace(
            backup_index_k_range=lambda req, s, e, *, written_by_forward: calls.append(
                (s, e, written_by_forward)
            )
        )
        return stub, calls

    def _rec(self, coord, req, matched, key_len, wbf=True):
        HiSparseCoordinator.record_host_insert(
            coord, req, matched_len=matched, key_len=key_len, written_by_forward=wbf
        )

    def test_first_insert_sets_owned_no_discard(self):
        coord, calls = self._coord()
        req = SimpleNamespace(hisparse_borrowed_prefix_len=0)
        self._rec(coord, req, matched=0, key_len=10, wbf=False)
        self.assertEqual(req._hisparse_host_owned_len, 10)
        self.assertIsNone(getattr(req, "_hisparse_host_discarded", None))
        self.assertEqual(calls, [(0, 10, False)])  # backup_index_k_range forwarded

    def test_rematch_of_own_node_is_not_a_discard(self):
        coord, _ = self._coord()
        req = SimpleNamespace(hisparse_borrowed_prefix_len=0)
        self._rec(coord, req, matched=0, key_len=10, wbf=False)  # own [0,10)
        self._rec(coord, req, matched=10, key_len=20, wbf=True)  # finish re-matches it
        self.assertEqual(req._hisparse_host_owned_len, 20)
        self.assertIsNone(getattr(req, "_hisparse_host_discarded", None))

    def test_discard_when_foreign_node_covers_range(self):
        coord, _ = self._coord()
        req = SimpleNamespace(hisparse_borrowed_prefix_len=0)
        self._rec(coord, req, matched=0, key_len=10, wbf=False)  # owned -> 10
        # a node this req did NOT hand over covers [10,16): discard it
        self._rec(coord, req, matched=16, key_len=24, wbf=True)
        self.assertEqual(req._hisparse_host_owned_len, 24)
        self.assertEqual(req._hisparse_host_discarded, [(10, 16)])

    def test_borrowed_prefix_rematch_not_discarded(self):
        coord, _ = self._coord()
        req = SimpleNamespace(hisparse_borrowed_prefix_len=8)
        # matched==borrowed is a re-match of the borrowed prefix, not a discard
        self._rec(coord, req, matched=8, key_len=20, wbf=True)
        self.assertEqual(req._hisparse_host_owned_len, 20)
        self.assertIsNone(getattr(req, "_hisparse_host_discarded", None))


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

    def available_size(self):
        return self.available


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


class TestHostDecodeCapacity(CustomTestCase):
    """HiSparseCoordinator.host_decode_capacity_ok: the per-step host check that
    lets decode past CLIP_MAX_NEW_TOKEN retract instead of failing the alloc."""

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
        return coord, pool, evicted

    @staticmethod
    def _req(slot, seqlen):
        return SimpleNamespace(kv=SimpleNamespace(req_pool_idx=slot), seqlen=seqlen)

    def _ok(self, coord, reqs):
        return HiSparseCoordinator.host_decode_capacity_ok(coord, reqs)

    def test_mid_page_needs_nothing(self):
        # Next backup lands inside an allocated page: no host demand at all.
        coord, _, evicted = self._coord([128], available=0)
        self.assertTrue(self._ok(coord, [self._req(0, 100)]))
        self.assertEqual(evicted, [])

    def test_page_boundary_with_free_page(self):
        coord, _, evicted = self._coord([128], available=self.PAGE)
        self.assertTrue(self._ok(coord, [self._req(0, 129)]))
        self.assertEqual(evicted, [])

    def test_page_boundary_reclaims_tree_cache(self):
        coord, _, evicted = self._coord([128], available=0, evictable=self.PAGE)
        self.assertTrue(self._ok(coord, [self._req(0, 129)]))
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
        self.assertFalse(self._ok(coord, reqs))
        pool.available += allocated  # retract request 1: its host slots return
        self.assertTrue(self._ok(coord, reqs[:1]))

    def test_unslotted_request_is_skipped(self):
        coord, _, _ = self._coord([0], available=0)
        self.assertTrue(self._ok(coord, [self._req(None, 10)]))


class TestHiSparseCheckDecodeCapacity(CustomTestCase):
    """HiSparseTokenToKVPoolAllocator.check_decode_capacity folds the host check
    into the per-step decode capacity gate the retract loop converges on."""

    def _alloc(self, logical, hot, host_fn=None):
        alloc = object.__new__(HiSparseTokenToKVPoolAllocator)
        alloc.logical_attn_allocator = SimpleNamespace(available_size=lambda: logical)
        alloc.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: hot)
        if host_fn is not None:
            alloc.set_host_decode_capacity_fn(host_fn)
        return alloc

    def _check(self, alloc, num_tokens=8, requests=("r",)):
        return alloc.check_decode_capacity(
            num_tokens=num_tokens, tree_cache=None, requests=list(requests)
        )

    def test_host_shortfall_fails_the_check(self):
        self.assertFalse(self._check(self._alloc(64, 64, host_fn=lambda r: False)))

    def test_device_shortfall_short_circuits_host_check(self):
        calls = []
        alloc = self._alloc(4, 64, host_fn=lambda r: calls.append(r) or True)
        self.assertFalse(self._check(alloc))
        self.assertEqual(calls, [])

    def test_passes_when_both_fit(self):
        self.assertTrue(self._check(self._alloc(64, 64, host_fn=lambda r: True)))

    def test_without_host_fn_only_device_counts(self):
        self.assertTrue(self._check(self._alloc(64, 64)))


class TestHostGrowthReserved(CustomTestCase):
    """DecodePreallocQueue host-gate helpers: full-output host target and the
    decode growth still owed by admitted requests."""

    @staticmethod
    def _req(idx, prompt, max_new, finished=False):
        return SimpleNamespace(
            origin_input_ids=list(range(prompt)),
            sampling_params=SimpleNamespace(max_new_tokens=max_new),
            kv=SimpleNamespace(req_pool_idx=idx),
            finished=lambda: finished,
        )

    def _queue(self, allocated, running=(), waiting=(), transfer=()):
        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        q.scheduler = SimpleNamespace(
            hisparse_coordinator=SimpleNamespace(
                mem_pool_host=_FakeHostPool(0, page_size=4),
                req_to_host_pool_allocated_len=torch.tensor(allocated),
            ),
            running_batch=SimpleNamespace(reqs=list(running)),
            waiting_queue=list(waiting),
        )
        q.transfer_queue = SimpleNamespace(queue=list(transfer))
        return q

    def test_target_is_page_rounded_prompt_plus_output(self):
        q = self._queue([0])
        self.assertEqual(q._hisparse_host_target(self._req(0, 10, 5), 10), 16)

    def test_rebootstrap_target_counts_emitted_tokens(self):
        # A retracted request refills prompt + emitted tokens, so a prompt-only
        # target under-counts and the gate admits a fill that then fails to alloc.
        from sglang.srt.disaggregation.utils import pre_alloc_fill_len

        req = self._req(0, 10, 5)
        req.output_ids = list(range(7))
        q = self._queue([0])
        fill_len = pre_alloc_fill_len(req)  # 10 + 7 - 1 pending input
        self.assertEqual(q._hisparse_host_target(req, fill_len), 24)  # 21 -> 24

    def test_growth_sums_active_requests(self):
        running = self._req(0, 10, 5)  # target 16, allocated 12 -> owes 4
        waiting = self._req(1, 8, 8)  # target 16, allocated 8 -> owes 8
        moving = self._req(2, 4, 0)  # target 4, allocated 4 -> owes 0
        q = self._queue([12, 8, 4], [running], [waiting], [moving])
        self.assertEqual(q._hisparse_host_growth_reserved(), 12)

    def test_growth_skips_finished_and_unslotted(self):
        done = self._req(0, 10, 5, finished=True)
        no_slot = self._req(None, 10, 5)
        q = self._queue([0], [done, no_slot])
        self.assertEqual(q._hisparse_host_growth_reserved(), 0)


class TestHiSparseMaxNewTokens(CustomTestCase):
    """The generation budget follows the logical pool, not the device rows."""

    def _alloc(self, size_full):
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )

        alloc = HiSparseTokenToKVPoolAllocator.__new__(HiSparseTokenToKVPoolAllocator)
        alloc._size_full = size_full
        alloc.page_size = 64
        return alloc

    def _budget(self, alloc, input_tokens, max_new, device_rows):
        return alloc.max_new_tokens_for_memory(
            input_tokens,
            max_new,
            token_capacity=device_rows,
            sliding_window_size=None,
            chunk_size=None,
        )

    def test_prompt_longer_than_device_rows_keeps_output(self):
        # 131K prompt, 78K device rows, 2.2M logical rows.
        alloc = self._alloc(2_204_160)
        self.assertEqual(self._budget(alloc, 131_100, 2048, 78_720), 2048)

    def test_still_clipped_by_logical_pool(self):
        alloc = self._alloc(1024)
        self.assertEqual(self._budget(alloc, 896, 2048, 512), 1024 - 896 - 64 - 1)


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
    """The four-range host teardown. Freeing a borrowed or tree-owned range
    corrupts another request's prefix; skipping the tail leaks host slots."""

    def _coord(self, allocated_len):
        freed = []
        coord = SimpleNamespace(
            req_to_host_pool_allocated_len=torch.tensor([allocated_len]),
            req_to_host_pool=torch.zeros((1, allocated_len), dtype=torch.int64),
            _host_borrow_release_fn=None,
            borrowed_prefix_len=HiSparseCoordinator.borrowed_prefix_len,
            _free_host_range=lambda idx, start, end: (
                freed.append((start, end)) if end > start else None
            ),
        )
        return coord, freed

    @staticmethod
    def _req(**over):
        req = SimpleNamespace(
            rid="r0",
            kv=SimpleNamespace(req_pool_idx=0),
            hisparse_borrowed_prefix_len=0,
            _hisparse_host_owned_len=0,
            _hisparse_host_discarded=None,
        )
        for k, v in over.items():
            setattr(req, k, v)
        return req

    def _release(self, coord, req):
        HiSparseCoordinator.release_host_after_insert(coord, req)

    def test_no_insert_no_borrow_frees_everything(self):
        coord, freed = self._coord(256)
        self._release(coord, self._req())
        self.assertEqual(freed, [(0, 256)])

    def test_borrowed_prefix_is_never_freed(self):
        coord, freed = self._coord(256)
        self._release(coord, self._req(hisparse_borrowed_prefix_len=64))
        self.assertEqual(freed, [(64, 256)])

    def test_owned_range_is_left_to_evict_host(self):
        coord, freed = self._coord(256)
        self._release(coord, self._req(_hisparse_host_owned_len=192))
        self.assertEqual(freed, [(192, 256)])

    def test_discarded_holes_are_freed_separately(self):
        coord, freed = self._coord(256)
        self._release(
            coord,
            self._req(
                _hisparse_host_owned_len=192,
                _hisparse_host_discarded=[(64, 128)],
            ),
        )
        self.assertEqual(freed, [(64, 128), (192, 256)])

    def test_discard_past_allocated_is_clamped(self):
        # Defensive: a discard range is always inside the allocated span, but
        # clamping keeps a stale one from freeing slots this request never had.
        coord, freed = self._coord(256)
        self._release(
            coord,
            self._req(
                _hisparse_host_owned_len=256,
                _hisparse_host_discarded=[(128, 4096)],
            ),
        )
        self.assertEqual(freed, [(128, 256)])

    def test_fully_owned_frees_nothing(self):
        coord, freed = self._coord(256)
        self._release(coord, self._req(_hisparse_host_owned_len=256))
        self.assertEqual(freed, [])

    def test_state_is_reset_for_a_resumed_request(self):
        # A retract-resumed request never re-matches, so a stale borrow length
        # would make the next teardown skip that many live slots.
        coord, _ = self._coord(256)
        req = self._req(hisparse_borrowed_prefix_len=64, _hisparse_host_owned_len=128)
        self._release(coord, req)
        self.assertEqual(req.hisparse_borrowed_prefix_len, 0)
        self.assertEqual(req._hisparse_host_owned_len, 0)
        self.assertIsNone(req._hisparse_host_discarded)
        self.assertEqual(int(coord.req_to_host_pool_allocated_len[0]), 0)
        self.assertTrue(bool((coord.req_to_host_pool == -1).all()))

    def test_borrow_refs_are_dropped_by_rid(self):
        coord, _ = self._coord(256)
        dropped = []
        coord._host_borrow_release_fn = dropped.append
        self._release(coord, self._req())
        self.assertEqual(dropped, ["r0"])

    def test_falls_back_to_the_stashed_pool_index(self):
        # release_kv_cache nulls req.kv.req_pool_idx before this runs.
        coord, freed = self._coord(256)
        req = self._req(_hisparse_host_pool_idx=0)
        req.kv.req_pool_idx = None
        self._release(coord, req)
        self.assertEqual(freed, [(0, 256)])

    def test_missing_pool_index_asserts(self):
        coord, _ = self._coord(256)
        req = self._req()
        req.kv.req_pool_idx = None
        with self.assertRaises(AssertionError):
            self._release(coord, req)


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
            return not staging

        coord = SimpleNamespace(
            release_request=lambda req: calls.append("full"),
            release_request_device=release_request_device,
            release_host_after_insert=lambda req: calls.append("host"),
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


if __name__ == "__main__":
    unittest.main()
