"""Allocator agreement for unified memory sharded by logical page.

The tests give each rank its own arena and replay identical logical allocations
and frees. They check owner-local binding, admission under owner pressure, and
consistency of the mirrored class lists.
"""

import random
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.alloc.kv_shard import ShardedOwner
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.utils.invariants import resolve_level
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_test_unified_pools,
    setup_allocator_context,
)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PS = 4
# One rank's token pages by sizing; the arena holds a few more.
FULL_PAGES = 24
SLOTS = 8
# A cache without nodes: a continuation reads the base its request recorded.
_NO_TREE = SimpleNamespace(rotation_base_of=lambda node: None)


def _signal_invariants(case, level: InvariantCheckLevel = InvariantCheckLevel.WARN):
    """Run ``case`` with the allocator's signal-only checks on, at least at
    ``level``: some checks of the sharded shortcuts run only from WARN up, and
    the CPU stage this file runs in sets no level."""
    level = max(level, resolve_level())
    override = envs.SGLANG_INVARIANT_CHECK.override(int(level))
    override.__enter__()
    case.addCleanup(override.__exit__, None, None, None)


class _Chain:
    """One request's logical tokens: the same on every rank."""

    def __init__(self, ids: List[int], base: int, slot: Optional[int]):
        self.ids = ids
        self.base = base
        self.slot = slot
        # Evictable once cached; a chain cut mid-page is finished.
        self.cached = False

    @property
    def extendable(self) -> bool:
        return len(self.ids) % PS == 0


class _Rank:
    """One rank of the shard group: its own storage, built like the runtime's."""

    def __init__(self, rank: int, shard_size: int, device: str = "cpu"):
        state = SimpleNamespace(
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        )
        self.rank = rank
        self.shard_size = shard_size
        self.device = device
        self.spec = PageShardSpec(
            shard_rank=rank,
            shard_size=shard_size,
            page_size=PS,
            max_prefix_tokens=8 * FULL_PAGES * PS * shard_size,
            chunk_tokens=16 * PS,
        )
        bundle = build_test_unified_pools(
            device=device,
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            use_mla_backend=True,
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            page_size=PS,
            start_layer=0,
            end_layer=2,
            full_attention_layer_ids=[1],
            full_max_total_num_tokens=FULL_PAGES * PS,
            mamba_layer_ids=[0],
            mamba2_cache_params=state,
            max_mamba_cache_size=SLOTS,
            model_context_len=FULL_PAGES * PS * shard_size,
            extra_max_context_len=1,
            max_num_reqs=SLOTS,
            lazy_compaction=True,
            disable_overlap_schedule=True,
            kv_shard_spec=self.spec,
        )
        self.bundle = bundle
        self.storage = bundle.storage
        self.allocator = bundle.token_to_kv_pool_allocator
        self.owner = self.allocator.full_attn_allocator
        self.state_name = self.allocator.mamba_allocator.sub_pool_name
        self.rows = bundle.arena.entries(self.owner.sub_pool_name)

    def take(self, runs, lru):
        """One extend of ``runs`` (prefix_len, seq_len, base, last_loc), with
        the eviction the prefix cache's reclaim runs over ``lru``: nothing
        when the demand fits now, nothing when the allocator says evicting
        everything would not help, else a walk it stops. Returns the logical
        ids (None when refused), the bases, the evicted chains, whether the
        demand fits after the walk, and whether the gate promised it would."""
        allocator, storage = self.allocator, self.storage
        victims, fits, promised = [], [], []

        def make_room(num_tokens):
            asked = KVDemand.same_tokens(allocator, num_tokens)
            demand = {
                name: pages
                for name, pages in allocator.page_demands(asked).items()
                if pages > 0
            }
            if allocator.fits_after_frees(demand):
                fits.append(True)
                promised.append(True)
                return
            chains = [chain for _, chain in lru]
            promised.append(
                allocator.admits(
                    asked,
                    evictable=self.credit(chains),
                    owner_class_evictable=lambda shard_size: self.class_credit(chains),
                )
            )
            if promised[0]:
                allocator.free_group_begin()
                for cid, chain in lru:
                    if allocator.fits_after_frees(demand):
                        break
                    self.free(chain, how="tree")
                    victims.append(cid)
                allocator.free_group_end()
            fits.append(allocator.units_fit(demand))

        prefix = torch.tensor([run[0] for run in runs], dtype=torch.int64)
        seq = torch.tensor([run[1] for run in runs], dtype=torch.int64)
        reqs = [
            SimpleNamespace(last_node=None, kv_rotation_base=run[2]) for run in runs
        ]
        last_loc = torch.tensor([run[3] for run in runs], dtype=torch.int64)
        out = storage.take_extend(
            rows=None,
            prefix_lens=prefix.to(self.device),
            prefix_lens_cpu=prefix,
            seq_lens=seq.to(self.device),
            seq_lens_cpu=seq,
            last_loc=last_loc.to(self.device),
            num_tokens=int((seq - prefix).sum()),
            make_room=make_room,
            tree_cache=_NO_TREE,
            batch=SimpleNamespace(reqs=reqs),
        )
        return (
            None if out is None else out.tolist(),
            [req.kv_rotation_base for req in reqs] if out is not None else None,
            victims,
            fits[0],
            promised[0],
        )

    def credit(self, cached) -> Dict[str, int]:
        """What a cache holding the ``cached`` chains could evict: their
        pages, as tokens (a cache holds whole pages), and their state slots."""
        return {
            self.owner.sub_pool_name: sum(self.class_credit(cached)) * PS,
            self.state_name: sum(c.slot is not None for c in cached),
        }

    def class_credit(self, cached) -> List[int]:
        """The ``cached`` chains' pages per owner class, read off their ids."""
        pages = [0] * self.shard_size
        for chain in cached:
            for page in {t // PS for t in chain.ids}:
                pages[page % self.shard_size] += 1
        return pages

    def free(self, chain: _Chain, *, how: str, start: int = 0) -> None:
        """Give back ``chain.ids[start:]`` by one of the storage's entries, and
        with a whole chain its state slot."""
        storage, allocator = self.storage, self.allocator
        ids = torch.tensor(chain.ids[start:], dtype=torch.int64)
        if how == "tree":
            storage.free_token_rows(ids)
        elif how == "units":
            storage.free_units(self.owner.sub_pool_name, ids)
        elif how == "owner":
            storage.free_owner_rows(ids)
        elif how == "tokens":
            storage.free_tokens(ids)
        elif how == "segment":
            allocator.free_segment(ids, start_pos=start)
        else:
            (row,) = storage.take_rows(1)
            table = storage.requests.req_to_token
            table[row, : len(chain.ids)] = torch.tensor(chain.ids, dtype=table.dtype)
            storage.free_row(ReqKvInfo(req_pool_idx=row), [(start, len(chain.ids))])
            storage.release_rows([row])
        if start == 0 and chain.slot is not None:
            storage.free_state_slots(self.state_name, torch.tensor([chain.slot]))

    def class_lists(self):
        return tuple(
            tuple(pages.tolist())
            for pages in self.allocator.kv_shard.ids.class_free_pages
        )


class _ShardGroup:
    """N ranks replaying one logical workload; what they report is compared
    after every operation."""

    def __init__(self, case, shard_size: int, seed: int):
        setup_allocator_context()
        self.case = case
        self.shard_size = shard_size
        self.rng = random.Random(seed)
        self.ranks = [_Rank(rank, shard_size) for rank in range(shard_size)]
        self.chains: Dict[int, _Chain] = {}
        self.next_id = 0
        self.loose_slots: List[int] = []
        self.launches: List[list] = []
        self.unwritten: List[int] = []
        self.taken = self.refused = self.evicted = 0
        # Admissions only the per-class eviction credit grants.
        self.class_credit_helped = 0

    def _same(self, results, what):
        self.case.assertEqual(
            len({repr(result) for result in results}), 1, f"ranks differ on {what}"
        )
        return results[0]


class TestShardGroupAgrees(CustomTestCase):
    def setUp(self):
        _signal_invariants(self)

    def test_a_request_larger_than_one_rank_is_served(self):
        """The shard group holds N times one rank's pages: a prompt that no
        single rank's arena holds is admitted and taken, a stripe per rank."""
        for shard_size in (2, 4):
            with self.subTest(shard_size=shard_size):
                group = _ShardGroup(self, shard_size, 0)
                per_rank = group.ranks[0].allocator.kv_shard.per_rank_pages
                pages = per_rank + shard_size
                for rank in group.ranks:
                    self.assertTrue(
                        rank.allocator.admits(KVDemand(pages * PS), empty_pool=True)
                    )
                ids, *_ = group._same(
                    [rank.take([(0, pages * PS, None, 0)], []) for rank in group.ranks],
                    "the take",
                )
                self.assertEqual(len(ids), pages * PS)
                held = [rank.owner.live_page_count for rank in group.ranks]
                self.assertEqual(sum(held), pages)
                self.assertLessEqual(max(held) - min(held), 1)
                self.assertLess(max(held), pages)

    def test_admission_credits_no_eviction_that_misses_the_fullest_rank(self):
        """The cache reports one evictable count for all owner classes. When
        every evictable page belongs to one rank, evicting them frees nothing
        on the other: admission must not split the credit between the ranks."""
        group = _ShardGroup(self, 2, 0)
        ranks = group.ranks
        owner = ranks[0].owner.sub_pool_name
        per_rank = ranks[0].allocator.kv_shard.per_rank_pages
        # One-page chains alternate between the classes (least-full seeding).
        by_class = {0: [], 1: []}
        for _ in range(2 * (per_rank - 4)):
            ids, bases, *_ = group._same(
                [rank.take([(0, PS, None, 0)], []) for rank in ranks], "a take"
            )
            by_class[bases[0]].append(ids)
        self.assertEqual(len(by_class[0]), len(by_class[1]))
        # Class 0's pages are cached; class 1's are pinned by running requests.
        credit = {owner: len(by_class[0]) * PS}
        fits_now = KVDemand(8 * PS)
        needs_room_on_both = KVDemand(12 * PS)
        for rank in ranks:
            allocator = rank.allocator
            self.assertTrue(allocator.admits(fits_now, evictable=credit))
            self.assertFalse(allocator.admits(needs_room_on_both, evictable=credit))
        for ids in by_class[0]:
            for rank in ranks:
                rank.storage.free_tokens(torch.tensor(ids, dtype=torch.int64))
        for rank in ranks:
            self.assertFalse(rank.allocator.ensure_capacity(needs_room_on_both))
        # Once nothing is pinned the whole credit counts.
        credit = {owner: len(by_class[1]) * PS}
        for rank in ranks:
            self.assertTrue(
                rank.allocator.admits(KVDemand(2 * per_rank * PS), evictable=credit)
            )


class TestShardedChainSurface(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.rank = _Rank(1, 2)
        self.allocator = self.rank.allocator

    def _take(self, pages: int) -> torch.Tensor:
        ids, *_ = self.rank.take([(0, pages * PS, None, 0)], [])
        return torch.tensor(ids, dtype=torch.int64)

    def test_scheduler_seams_read_the_shard(self):
        allocator = self.allocator
        self.assertEqual(page_interleave_shard_size(allocator), 2)
        self.assertIs(allocator.shard_spec, self.rank.spec)
        self.assertEqual(allocator.page_size, PS)
        ids = self._take(3)
        free = allocator.get_all_free_pages()
        self.assertEqual(int(free.numel()), sum(allocator.class_free_page_counts()))
        self.assertFalse(bool(torch.isin(ids // PS, free).any()))

    def test_the_memory_clip_prices_admissions_class_rounding(self):
        """Admission charges a request one rounding page per owner class. The
        arrival clip prices the same, so it lets in no prompt that admission
        could then never fit whole, even in an empty pool."""
        allocator = self.allocator
        capacity_pages = 2 * allocator.kv_shard.per_rank_pages
        slots = allocator.fresh_request_state_slots
        admitted = []
        for pages in range(capacity_pages - 8, capacity_pages + 2):
            clipped = allocator.max_new_tokens_for_memory(
                pages * PS,
                0,
                token_capacity=10**9,
                sliding_window_size=None,
                chunk_size=16 * PS,
                prefill_retire_lag=0,
                eviction_interval=1,
            )
            # The prompt's pages and one rounding page per class.
            whole = KVDemand((pages + 2) * PS, {}, slots)
            self.assertEqual(
                clipped is not None, allocator.admits(whole, empty_pool=True)
            )
            admitted.append(clipped is not None)
        self.assertIn(True, admitted)
        self.assertIn(False, admitted)

    def test_a_take_without_a_rotation_base_is_refused(self):
        with self.assertRaises(AssertionError):
            self.allocator.alloc(PS)
        self.assertEqual(self.rank.owner.live_page_count, 0)
        self.assertEqual(self.allocator.kv_shard.live_pages(), [0, 0])


class TestShardedControlPathShortcuts(CustomTestCase):
    """The sharded chain's host shortcuts: a kept answer never outlives the
    state it was computed on, and a free splits its pages by owner class as
    the sharding allocator's own free does."""

    def setUp(self):
        _signal_invariants(self)

    def _two_ranks(self, pages: int = 6):
        """A 2-rank group holding one chain of ``pages`` pages: the ranks, the
        chain's tokens and its rotation base."""
        setup_allocator_context()
        ranks = [_Rank(rank, 2) for rank in range(2)]
        for rank in ranks:
            ids, bases, *_ = rank.take([(0, pages * PS, None, 0)], [])
            # The scheduler's capacity read: the counts are kept from here on.
            rank.allocator.available_size()
        return ranks, torch.tensor(ids, dtype=torch.int64), bases[0]

    def _shard_state(self, rank: _Rank):
        """What a free changes on one rank: the class lists, the band's live
        pages and its id table."""
        return (
            rank.class_lists(),
            rank.owner.live_page_count,
            rank.owner.ids.virtual_to_physical.tolist(),
        )

    def test_the_idle_audit_compares_the_band_with_its_class(self):
        """A class list written behind a band's back is reported on the rank
        that owns the class."""
        ranks, tokens, _ = self._two_ranks()
        page_class = int(tokens[0]) // PS % 2
        for rank in ranks:
            allocator = rank.allocator
            self.assertEqual(allocator.verify_byte_accounting(), [])
            allocator.kv_shard.ids.free(tokens[:1])
            report = allocator.verify_byte_accounting()
            self.assertEqual(
                any("its class holds" in line for line in report),
                rank.spec.shard_rank == page_class,
                report,
            )

    def test_clear_moves_the_class_lists_epoch(self):
        """`clear` refills the class lists; the snapshot every capacity
        question is asked of, kept under their epoch, must follow."""
        ranks, _, _ = self._two_ranks()
        for rank in ranks:
            allocator, shard = rank.allocator, rank.allocator.kv_shard
            owner = rank.owner.sub_pool_name
            whole = 2 * shard.per_rank_pages
            self.assertEqual(shard.live_pages(), [3, 3])
            self.assertEqual(allocator._shard_snapshot().band(owner).live, 3)
            self.assertFalse(allocator.units_fit({owner: whole}))
            key = allocator.pricing_key()
            allocator.clear()
            self.assertNotEqual(allocator.pricing_key(), key)
            self.assertEqual(shard.live_pages(), [0, 0])
            self.assertEqual(allocator._shard_snapshot().band(owner).live, 0)
            self.assertTrue(allocator.units_fit({owner: whole}))
            self.assertFalse(allocator.units_fit({owner: whole + 1}))

    def test_a_free_outside_the_id_space_is_refused(self):
        """A free that names a page outside the logical id space (a reserved
        page, a negative id, one past the space) raises on every rank before
        a class list or a band changes."""
        probe, _, _ = self._two_ranks()
        span = probe[0].allocator.kv_shard._page_span
        frees = {
            "reserved page 0": lambda tokens: torch.tensor([0]),
            "reserved page 1": lambda tokens: torch.tensor([PS]),
            "page -1": lambda tokens: torch.tensor([-PS]),
            "past the id space": lambda tokens: torch.tensor([span * PS]),
            "a reserved page among live ones": lambda tokens: torch.cat(
                (tokens[: 2 * PS], torch.arange(PS))
            ),
        }
        for label, bad in frees.items():
            ranks, tokens, _ = self._two_ranks()
            for rank in ranks:
                with self.subTest(free=label, rank=rank.rank):
                    before = self._shard_state(rank)
                    with self.assertRaisesRegex(RuntimeError, "logical id space"):
                        rank.allocator.free_segment(bad(tokens), start_pos=0)
                    with self.assertRaisesRegex(RuntimeError, "logical id space"):
                        rank.allocator.free(bad(tokens))
                    self.assertEqual(self._shard_state(rank), before)
                    self.assertEqual(rank.allocator.verify_byte_accounting(), [])

    def test_the_classed_free_leaves_the_stock_frees_lists(self):
        """`ShardedOwner.give_back` splits freed pages by owner class itself
        and hands them over pre-split (`free_classed`): the class lists are
        the ones the sharding allocator's own `free` of those pages leaves."""
        rng = random.Random(11)
        for shard_size in (2, 3, 8):
            for need_sort in (False, True):
                with self.subTest(shard_size=shard_size, need_sort=need_sort):
                    self._replay_frees_both_ways(rng, shard_size, need_sort)

    def _replay_frees_both_ways(self, rng, shard_size: int, need_sort: bool):
        stock, classed = (
            PageInterleavePoolAllocator(
                size=FULL_PAGES * PS,
                physical_page_size=PS,
                shard_size=shard_size,
                dtype=torch.int64,
                device="cpu",
                kvcache=None,
                need_sort=need_sort,
                shard_spec=PageShardSpec(
                    shard_rank=0,
                    shard_size=shard_size,
                    page_size=PS,
                    max_prefix_tokens=8 * FULL_PAGES * PS * shard_size,
                    chunk_tokens=16 * PS,
                ),
            )
            for _ in range(2)
        )
        # What `_split_by_class` reads of a sharded owner.
        span = shard_size * (FULL_PAGES + 1)
        owner = SimpleNamespace(
            shard_size=shard_size,
            _page_span=span,
            _class_key_ends=torch.arange(1, shard_size + 1) * span,
        )
        zero = torch.zeros(1, dtype=torch.int64)

        def class_lists(ids):
            return [
                pages.tolist()
                for pages in (*ids.class_free_pages, *ids.class_release_pages)
            ]

        live, freed = [], 0
        for _ in range(150):
            if live and (rng.random() < 0.5 or len(live) > 10):
                # One token of each page, as `Band._page_reps` names them.
                reps = live.pop(rng.randrange(len(live)))[::PS]
                if rng.random() < 0.3:
                    reps = reps.flip(0)
                stock.free(reps)
                split = ShardedOwner._split_by_class(owner, reps // PS)
                self.assertEqual(sum(part.numel() for part in split), reps.numel())
                classed.free_classed(split)
                freed += reps.numel()
            else:
                seq = torch.tensor([rng.choice((1, 2, 3, 5)) * PS])
                taken = [
                    ids.alloc_extend(
                        zero, zero, seq, seq, zero, int(seq), rotation_bases=[None]
                    )
                    for ids in (stock, classed)
                ]
                if taken[0] is None:
                    self.assertIsNone(taken[1])
                    continue
                self.assertEqual(taken[0].tolist(), taken[1].tolist())
                live.append(taken[0])
            self.assertEqual(class_lists(stock), class_lists(classed))
        self.assertGreater(freed, 50)


if __name__ == "__main__":
    unittest.main()
