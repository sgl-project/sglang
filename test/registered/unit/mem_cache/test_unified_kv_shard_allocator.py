"""The ranks of a KV shard group agree without asking each other.

Under logical-page KV sharding of unified memory each rank of the shard group
runs its own storage: logical page ids from the sharding allocator, the pages
it owns bound in its own full-attention band, and compaction that nobody else
sees. Nothing is exchanged between ranks, so every scheduler-visible answer
must follow from the logical alloc/free stream alone.

Here N ranks are N storages built as the runtime builds them (an MLA full
band and a state band over one arena each, nothing shared) and one random
logical workload is replayed on all of them through the storage and allocator
API: takes with rotation bases, continuations, whole and tail frees through
every free entry, free groups, state slots, eviction walks gated and stopped
by the allocator as the prefix cache's reclaim does, forwards in flight and
opportunistic flushes. After every operation:

- every capacity answer and every logical id is the same on all ranks;
- the sharding allocators' class lists are the same;
- rank r's band holds exactly the live pages of class r, and each live token
  is stored on exactly one rank;
- bytes written through the write translation are read back through it,
  wherever rank-local compaction moved the page;
- a take the mirrored capacity search said yes to succeeds on every rank.
"""

import random
import unittest
from array import array
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.managers.scheduler_components.state_check import (
    FailureReporter,
    StateCheckError,
)
from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    evictable_by_sub_pool,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.alloc.kv_shard import ShardedOwner
from sglang.srt.mem_cache.page_interleave import PageShardSpec
from sglang.srt.mem_cache.prefill_budget import ChainPrefillBudget
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.invariants import resolve_level
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
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


class _Event:
    """A forward's done event that already fired: nothing here waits a stream."""

    def query(self):
        return True


def _marker(tokens: torch.Tensor) -> torch.Tensor:
    """Four bytes naming each logical token, as stored in its first bytes."""
    return torch.stack(
        (tokens & 255, (tokens >> 8) & 255, (tokens >> 16) & 255, tokens * 0 + 0xA5),
        dim=1,
    ).to(torch.uint8)


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

    # -- the workload's operations --

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

    def write(self, ids: List[int]) -> None:
        tokens = torch.tensor(ids, dtype=torch.int64)
        physical = self.allocator.translate_write_loc(tokens)
        owned = physical > 0
        self.rows[physical[owned], :4] = _marker(tokens[owned])

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

    def take_slot(self) -> Optional[int]:
        slot = self.storage.take_state_slots(self.state_name, 1)
        return None if slot is None else int(slot[0])

    def launch(self, written: List[int], slots: List[int]):
        steps = self.allocator.forward_steps
        launch = steps.forward_launched()
        event = _Event()
        steps.forward_recorded(event)
        self.allocator.set_latest_forward_done_event(event)
        self.allocator.set_inflight_forward(
            event,
            torch.tensor(written, dtype=torch.int64),
            torch.tensor(slots, dtype=torch.int64),
        )
        return launch

    # -- what every rank must agree on --

    def answers(self, demands, credit):
        allocator = self.allocator
        return (
            allocator.available_size(),
            allocator.full_available_size(),
            allocator.estimated_state_slots_available(),
            allocator.size,
            allocator.conserve_full_available_size(),
            tuple(sorted(allocator.usage().items())),
            tuple(allocator.ensure_capacity(demand) for demand in demands),
            tuple(allocator.admits(demand, evictable=credit) for demand in demands),
            tuple(allocator.admits(demand, empty_pool=True) for demand in demands),
        )

    def class_lists(self):
        return tuple(
            tuple(pages.tolist())
            for pages in self.allocator.kv_shard.ids.class_free_pages
        )

    def stored(self, ids: List[int]) -> torch.Tensor:
        """Per token: whether this rank stores it, its marker checked."""
        tokens = torch.tensor(ids, dtype=torch.int64)
        physical = self.allocator.translate_write_loc(tokens)
        owned = physical > 0
        assert torch.equal(owned, (tokens // PS) % self.shard_size == self.rank), (
            f"rank {self.rank} stores pages of another class"
        )
        assert torch.equal(self.rows[physical[owned], :4], _marker(tokens[owned])), (
            f"rank {self.rank} lost bytes of a page it owns"
        )
        return owned


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

    def _lru(self):
        return [(cid, c) for cid, c in self.chains.items() if c.cached]

    # -- operations --

    def take(self):
        rng = self.rng
        runs, continued = [], []
        for _ in range(rng.choice((1, 1, 1, 2))):
            open_chains = [
                cid
                for cid, c in self.chains.items()
                if c.extendable and not c.cached and cid not in continued
            ]
            if open_chains and rng.random() < 0.45:
                cid = rng.choice(open_chains)
                chain = self.chains[cid]
                continued.append(cid)
                prefix, base, last = len(chain.ids), chain.base, chain.ids[-1]
                pages = rng.choice((1, 1, 2, 3))
            else:
                continued.append(None)
                prefix, base, last = 0, None, 0
                pages = rng.choice((1, 2, 3, 5, 8, 13))
            # A chunk ends on a page; only a request's last one may not.
            cut = rng.choice((0, 0, 0, 1, 3))
            runs.append((prefix, prefix + pages * PS - cut, base, last))
        lru = self._lru()
        ids, bases, victims, fits, promised = self._same(
            [rank.take(runs, lru) for rank in self.ranks], "a take"
        )
        for cid in victims:
            del self.chains[cid]
        self.evicted += len(victims)
        if promised:
            self.case.assertTrue(fits, "an eviction the gate promised fell short")
        if fits:
            self.case.assertIsNotNone(ids, "a take refused after the search said yes")
        if ids is None:
            self.refused += 1
            return
        self.taken += 1
        for rank in self.ranks:
            rank.write(ids)
        at = 0
        for (prefix, seq, _, _), cid, base in zip(runs, continued, bases):
            new = ids[at : at + seq - prefix]
            at += seq - prefix
            if cid is None:
                slot = self._same([rank.take_slot() for rank in self.ranks], "a slot")
                self.chains[self.next_id] = _Chain(new, base, slot)
                self.next_id += 1
            else:
                self.case.assertEqual(base, self.chains[cid].base)
                self.chains[cid].ids += new
        self.unwritten += ids

    def free(self):
        rng = self.rng
        cid = rng.choice(list(self.chains))
        chain = self.chains[cid]
        how = rng.choice(("tree", "units", "owner", "tokens", "segment", "row"))
        pages = len(chain.ids) // PS
        tail = rng.random() < 0.35 and pages >= 2 and how in ("segment", "row")
        start = rng.randint(1, pages - 1) * PS if tail else 0
        grouped = rng.random() < 0.3
        for rank in self.ranks:
            if grouped:
                rank.allocator.free_group_begin()
            rank.free(chain, how=how, start=start)
            if grouped:
                if rng.random() < 0.5:
                    rank.allocator.flush_deferred_full_frees()
                rank.allocator.free_group_end()
        if tail:
            del chain.ids[start:]
        else:
            del self.chains[cid]

    def slot(self):
        if self.loose_slots and self.rng.random() < 0.5:
            slot = self.loose_slots.pop(self.rng.randrange(len(self.loose_slots)))
            for rank in self.ranks:
                rank.storage.free_state_slots(rank.state_name, torch.tensor([slot]))
            return
        slot = self._same([rank.take_slot() for rank in self.ranks], "a slot")
        if slot is not None:
            self.loose_slots.append(slot)

    def forward(self):
        if self.launches and self.rng.random() < 0.6:
            launches = self.launches.pop(0)
            for rank, launch in zip(self.ranks, launches):
                rank.allocator.forward_steps.forward_processed(launch)
            return
        written = [
            t for t in self.unwritten if any(t in c.ids for c in self.chains.values())
        ]
        slots = [c.slot for c in self.chains.values() if c.slot is not None]
        self.unwritten = []
        self.launches.append([rank.launch(written, slots) for rank in self.ranks])
        if not written:
            return
        for rank in self.ranks:
            # The ledger names the pages the forward writes on this rank: no
            # logical id stands in for one of the band's own pages.
            ledger = rank.owner._inflight_write_sets()[-1][1]
            tokens = torch.tensor(written, dtype=torch.int64)
            physical = rank.allocator.translate_write_loc(tokens)
            self.case.assertEqual(ledger, set((physical // PS).tolist()))

    def evict_everything_admitted(self):
        """An admitted demand is one the take serves once the credit is
        evicted: admission never credits an eviction that does not help."""
        demand = KVDemand(self.rng.randint(1, 40) * PS, {}, self.rng.randint(0, 2))
        credit = self._credit()
        cached = self._cached()
        by_count, by_class = self._same(
            [
                (
                    rank.allocator.admits(demand, evictable=credit),
                    rank.allocator.admits(
                        demand,
                        evictable=credit,
                        owner_class_evictable=lambda shard_size: rank.class_credit(
                            cached
                        ),
                    ),
                )
                for rank in self.ranks
            ],
            "an admission",
        )
        # The count alone prices pinned pages as one rank's: never more
        # generous than the per-class credit.
        self.case.assertTrue(by_class or not by_count)
        self.class_credit_helped += by_class and not by_count
        if not by_class:
            # Exact: what the per-class credit refuses does not fit even
            # with every cached chain gone.
            kept = [
                have - gone
                for have, gone in zip(
                    self.ranks[0].allocator.kv_shard.live_pages(),
                    self.ranks[0].class_credit(cached),
                )
            ]
            rank = self.ranks[0]
            allocator, owner = rank.allocator, rank.owner.sub_pool_name
            freed = max(allocator.kv_shard.live_pages()) - max(kept)
            self.case.assertFalse(
                allocator._admits_on(
                    allocator._shard_snapshot(),
                    demand.state_slots,
                    allocator._per_rank(allocator.page_demands(demand)),
                    {owner: freed, rank.state_name: credit[rank.state_name]},
                    False,
                )
            )
            return
        for cid, chain in self._lru():
            for rank in self.ranks:
                rank.free(chain, how="tree")
            del self.chains[cid]
        for rank in self.ranks:
            self.case.assertTrue(rank.allocator.ensure_capacity(demand))

    def _cached(self):
        return [c for c in self.chains.values() if c.cached]

    def _credit(self):
        return self.ranks[0].credit(self._cached())

    # -- the comparison --

    def check(self):
        case, rng = self.case, self.rng
        demands = [
            KVDemand(rng.randint(0, 60) * PS, {}, rng.randint(0, 3)) for _ in range(3)
        ]
        self._same(
            [rank.answers(demands, self._credit()) for rank in self.ranks], "capacity"
        )
        self._same([rank.class_lists() for rank in self.ranks], "the class lists")
        live_tokens = sum(len(c.ids) for c in self.chains.values())
        live_pages = [0] * self.shard_size
        for chain in self.chains.values():
            for page in {t // PS for t in chain.ids}:
                live_pages[page % self.shard_size] += 1
        for rank in self.ranks:
            allocator = rank.allocator
            case.assertEqual(allocator.kv_shard.live_pages(), live_pages)
            case.assertEqual(rank.owner.live_page_count, live_pages[rank.rank])
            case.assertEqual(allocator.verify_byte_accounting(), [])
            capacity, available = allocator.usage()[rank.owner.sub_pool_name]
            case.assertEqual(capacity - available, sum(live_pages) * PS)
        if live_tokens:
            ids = [t for c in self.chains.values() for t in c.ids]
            holders = sum(rank.stored(ids).to(torch.int64) for rank in self.ranks)
            case.assertTrue(
                bool((holders == 1).all()), "a token on no or several ranks"
            )

    def run(self, steps: int):
        rng = self.rng
        self.check()
        for _ in range(steps):
            roll = rng.random()
            if roll < 0.45 or not self.chains:
                self.take()
            elif roll < 0.62:
                self.free()
            elif roll < 0.72:
                for chain in rng.sample(list(self.chains.values()), 1):
                    chain.cached = True
            elif roll < 0.8:
                self.slot()
            elif roll < 0.9:
                self.forward()
            elif roll < 0.95:
                self.evict_everything_admitted()
            else:
                for rank in self.ranks:
                    rank.allocator.flush_opportunistic()
            self.check()

    def drain(self):
        """Back to idle: every forward processed, every id returned."""
        for launches in self.launches:
            for rank, launch in zip(self.ranks, launches):
                rank.allocator.forward_steps.forward_processed(launch)
        for chain in self.chains.values():
            for rank in self.ranks:
                rank.free(chain, how="tree")
        self.chains.clear()
        for slot in self.loose_slots:
            for rank in self.ranks:
                rank.storage.free_state_slots(rank.state_name, torch.tensor([slot]))
        self.check()


class TestShardGroupAgrees(CustomTestCase):
    def setUp(self):
        _signal_invariants(self)

    def test_ranks_agree_on_a_random_workload(self):
        for shard_size in (2, 4):
            taken = refused = evicted = moves = 0
            compacted_apart = False
            for seed in range(6):
                with self.subTest(shard_size=shard_size, seed=seed):
                    group = _ShardGroup(self, shard_size, 20261109 + seed)
                    group.run(160)
                    group.drain()
                    shard = group.ranks[0].allocator.kv_shard
                    full = shard_size * shard.per_rank_pages * PS
                    for rank in group.ranks:
                        allocator = rank.allocator
                        self.assertEqual(rank.owner.live_page_count, 0)
                        self.assertEqual(allocator.full_available_size(), full)
                        self.assertEqual(
                            allocator.usage()[rank.owner.sub_pool_name], (full, full)
                        )
                    taken += group.taken
                    refused += group.refused
                    evicted += group.evicted
                    per_rank = [rank.owner._stats_n_flush_moves for rank in group.ranks]
                    moves += sum(per_rank)
                    compacted_apart |= len(set(per_rank)) > 1
            # The workload reached what it is here for: refusals, eviction
            # walks, and compaction that differed between ranks.
            self.assertGreater(taken, 100)
            self.assertGreater(refused, 0)
            self.assertGreater(evicted, 0)
            self.assertGreater(moves, 0)
            self.assertTrue(compacted_apart)

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


class TestEvictionCreditPerClass(CustomTestCase):
    """Admission credits the cache's evictable pages per owner class, so the
    pages a running request pins count on the ranks that hold them."""

    def setUp(self):
        _signal_invariants(self)

    def test_a_request_past_one_rank_finishes_under_eviction_pressure(self):
        """A warm cache and one chunked request larger than one rank's pages:
        every chunk is admitted by the gate the prefix cache's reclaim asks,
        which then evicts what the chunk needs. Priced by the evictable count
        alone, the request's own pinned pages would cap it at one rank's
        pages and it could never finish."""
        for shard_size in (2, 4):
            with self.subTest(shard_size=shard_size):
                group = _ShardGroup(self, shard_size, 0)
                ranks = group.ranks
                per_rank = ranks[0].allocator.kv_shard.per_rank_pages
                # One-page chains fill all but a few pages of every rank.
                lru = []
                for cid in range(shard_size * (per_rank - 3)):
                    ids, bases, *_ = group._same(
                        [rank.take([(0, PS, None, 0)], []) for rank in ranks], "warm"
                    )
                    chain = _Chain(ids, bases[0], None)
                    chain.cached = True
                    lru.append((cid, chain))
                pages = per_rank + 2 * shard_size
                held, base, count_alone_refused = [], None, False
                while len(held) < pages * PS:
                    chunk = min(2, pages - len(held) // PS) * PS
                    asked = KVDemand(chunk + shard_size * PS)
                    cached = [chain for _, chain in lru]
                    count_alone_refused |= not all(
                        rank.allocator.ensure_capacity(asked)
                        or rank.allocator.admits(asked, evictable=rank.credit(cached))
                        for rank in ranks
                    )
                    run = (len(held), len(held) + chunk, base, held[-1] if held else 0)
                    ids, bases, victims, fits, promised = group._same(
                        [rank.take([run], lru) for rank in ranks], "a chunk"
                    )
                    self.assertTrue(promised and fits, f"stalled at {len(held) // PS}")
                    self.assertIsNotNone(ids)
                    lru = [(cid, chain) for cid, chain in lru if cid not in victims]
                    held += ids
                    base = bases[0]
                self.assertTrue(count_alone_refused)
                self.assertEqual(
                    sum(rank.owner.live_page_count for rank in ranks),
                    pages + len(lru),
                )
                self.assertGreater(pages, per_rank)

    def _tree(self, rank: "_Rank") -> UnifiedRadixCache:
        return UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=rank.storage.requests,
                token_to_kv_pool_allocator=rank.allocator,
                page_size=PS,
                eviction_policy="lru",
                tree_slots=tree_slots_of((ComponentType.FULL,)),
                tree_core_backend="python",
            )
        )

    def test_the_tree_counts_evictable_pages_per_owner_class(self):
        """The prefix tree names its evictable pages per owner class from each
        node's rotation base and position alone; the count is the ids' own
        classes, through inserts, a split and locks."""
        setup_allocator_context()
        shard_size = 4
        rank = _Rank(1, shard_size)
        tree = self._tree(rank)
        allocator = rank.allocator
        nodes = {}

        def insert(tokens, prefix_len=0, base=None, last=0):
            req = SimpleNamespace(last_node=None, kv_rotation_base=base)
            prefix = torch.tensor([prefix_len], dtype=torch.int64)
            seq = torch.tensor([len(tokens)], dtype=torch.int64)
            ids = alloc_paged_token_slots_extend(
                tree,
                prefix,
                prefix,
                seq,
                seq,
                torch.tensor([last], dtype=torch.int64),
                len(tokens) - prefix_len,
                batch=SimpleNamespace(reqs=[req]),
            )
            return ids, req.kv_rotation_base

        def cache(tokens, ids, base):
            result = tree.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=ids.to(torch.int64),
                    rotation_base=base,
                )
            )
            self.assertFalse(result.rotation_tail_declined)
            nodes[result.last_device_node] = None
            return result.last_device_node

        def expect(evictable_ids):
            pages = [0] * shard_size
            for page in {int(t) // PS for t in evictable_ids}:
                pages[page % shard_size] += 1
            self.assertEqual(tree.full_evictable_pages_by_class(shard_size), pages)
            self.assertEqual(sum(pages) * PS, tree.full_evictable_size())
            self.assertEqual(
                evictable_by_sub_pool(tree)[rank.owner.sub_pool_name], sum(pages) * PS
            )

        expect([])
        a = list(range(100, 100 + 5 * PS))
        a_ids, a_base = insert(a)
        a_node = cache(a, a_ids, a_base)
        expect(a_ids.tolist())
        b = list(range(300, 300 + 3 * PS))
        b_ids, b_base = insert(b)
        b_node = cache(b, b_ids, b_base)
        expect(a_ids.tolist() + b_ids.tolist())
        # A continuation of the first chain past its second page: the split
        # keeps both halves on the chain's base.
        c_tail = list(range(500, 500 + 2 * PS))
        c_ids, c_base = insert(
            a[: 2 * PS] + c_tail,
            prefix_len=2 * PS,
            base=a_base,
            last=int(a_ids[2 * PS - 1]),
        )
        self.assertEqual(c_base, a_base)
        c_node = cache(
            a[: 2 * PS] + c_tail, torch.cat((a_ids[: 2 * PS], c_ids)), c_base
        )
        everything = a_ids.tolist() + b_ids.tolist() + c_ids.tolist()
        expect(everything)
        # A lock pins the path to the root: the shared two pages and the tail.
        receipt = tree.inc_lock_ref(c_node)
        expect(a_ids[2 * PS :].tolist() + b_ids.tolist())
        other = tree.inc_lock_ref(b_node)
        expect(a_ids[2 * PS :].tolist())
        tree.dec_lock_ref(c_node, receipt.to_dec_params())
        expect(a_ids.tolist() + c_ids.tolist())
        tree.dec_lock_ref(b_node, other.to_dec_params())
        expect(everything)
        self.assertIsNotNone(a_node)

    def test_a_continuation_chunk_is_priced_as_its_take_asks(self):
        """A chunk's take asks the cache to make room for one rounding page
        per owner class; the chunk is admitted at that price, so the eviction
        an admitted chunk relies on is one the cache's gate lets through."""
        setup_allocator_context()
        ranks = {shard_size: _Rank(0, shard_size) for shard_size in (2, 4)}
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        for shard_size, rank in ranks.items():
            with self.subTest(shard_size=shard_size):
                allocator = rank.allocator
                tree = MagicMock()
                tree.full_evictable_size.return_value = 0
                tree.supports_mamba.return_value = False
                tree.supports_prefix_sharing.return_value = False
                tree.window_checkpoint_interval.return_value = 0
                tree.full_evictable_pages_by_class.return_value = [0] * shard_size
                budget = ChainPrefillBudget(allocator, tree)
                priced = []
                chunk_fits = budget._chunk_fits

                def record(tokens, headroom, **kwargs):
                    priced.append(kwargs["extra_tokens"])
                    return chunk_fits(tokens, headroom, **kwargs)

                budget._chunk_fits = record
                budget._others_hold_memory = lambda req: False
                chunk = budget.fit_chunk(
                    SimpleNamespace(),
                    extend_input_len=8 * PS,
                    max_new_tokens=0,
                    chunk_limit=4 * PS,
                )
                self.assertEqual(chunk, 4 * PS)
                self.assertEqual(set(priced), {(shard_size - 1) * PS})
                # The take's own ask: the chunk and a page per owner class.
                seen = []
                rank.storage.take_extend(
                    rows=None,
                    prefix_lens=torch.tensor([0]),
                    prefix_lens_cpu=torch.tensor([0]),
                    seq_lens=torch.tensor([chunk]),
                    seq_lens_cpu=torch.tensor([chunk]),
                    last_loc=torch.tensor([0]),
                    num_tokens=chunk,
                    make_room=seen.append,
                    tree_cache=_NO_TREE,
                    batch=SimpleNamespace(
                        reqs=[SimpleNamespace(last_node=None, kv_rotation_base=None)]
                    ),
                )
                self.assertEqual(seen, [chunk + shard_size * PS])
                self.assertEqual(
                    budget.request_demand(
                        chunk, 0, chunk_limit=4 * PS, extra_tokens=priced[0]
                    ).full_tokens,
                    chunk + shard_size * PS,
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

    def test_the_idle_page_sweep_reads_the_class_lists(self):
        """The scheduler's page sweep compares the rows requests hold with the
        free logical pages of every owner class: quiet while a request's pages
        are held, loud once a row names a freed one."""
        ids = self._take(3)
        table = self.rank.storage.requests
        table.req_to_token[0, : ids.numel()] = ids.to(table.req_to_token.dtype)
        holder = SimpleNamespace(
            rid="a",
            kv=ReqKvInfo(
                req_pool_idx=0,
                kv_committed_len=ids.numel(),
                kv_allocated_len=ids.numel(),
            ),
        )
        checker = SimpleNamespace(
            req_to_token_pool=table,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=SimpleNamespace(session_records=lambda: {}),
            get_request_lists=lambda: {ListKind.DECODING: [holder]},
            reporter=FailureReporter(),
        )
        checker._requests_owning_rows = lambda: (
            SchedulerInvariantChecker._requests_owning_rows(checker)
        )
        SchedulerInvariantChecker._check_kv_page_invariants(checker)
        self.allocator.free(ids)
        with self.assertRaisesRegex(StateCheckError, "use-after-free"):
            SchedulerInvariantChecker._check_kv_page_invariants(checker)

    def test_prefill_admission_reads_the_shard(self):
        """The admission pass of a sharded chain: page-aligned chunk ends, one
        page per owner class reserved per request, the gather scratch's
        budget, and a budget in logical tokens."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        allocator, spec = self.allocator, self.rank.spec
        tree = MagicMock()
        tree.full_evictable_size.return_value = 0
        tree.supports_mamba.return_value = False
        tree.supports_prefix_sharing.return_value = False
        tree.window_checkpoint_interval.return_value = 0
        running = MagicMock()
        running.reqs = []
        adder = PrefillAdder(
            page_size=PS,
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=spec.chunk_tokens,
        )
        self.assertEqual(adder.kv_shard_size, 2)
        self.assertEqual(adder.kv_shard_granule, PS)
        self.assertEqual(adder.per_req_token_overhead, 2 * PS)
        self.assertIs(adder.kv_shard_scratch_spec, spec)
        self.assertIsInstance(adder.memory_budget, ChainPrefillBudget)
        self.assertEqual(adder.memory_budget.kv_shard_size, 2)
        self.assertTrue(adder._kv_shard_reserve_scratch(prefix_len=0, extend_len=PS))
        self.assertFalse(
            adder._kv_shard_reserve_scratch(prefix_len=0, extend_len=spec.chunk_tokens)
        )

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
