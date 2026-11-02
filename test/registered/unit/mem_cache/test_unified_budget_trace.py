"""Admission and budget answers of unified pools, recorded and compared.

For the same pools, histories and demands, the answers every admission path
gives (the allocator's admission rule, decode preallocation with and without
the scheduler's budgets, the ceiling, the take check, the empty-pool clip, and
the prefill budget's checks, chunks and reservations) and the window facts
they read (retained lengths, the prefix tree's window slots) must stay what
the stored record says. Set ``SGLANG_RECORD_BUDGET_TRACE=1`` to rewrite it.
"""

import itertools
import json
import os
import random
import unittest
from pathlib import Path

from test_unified_chain_realize import _allocator
from test_unified_window_sub_pools import _storage

from sglang.srt.mem_cache.base_prefix_cache import evictable_by_sub_pool
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_RECORD = Path(__file__).with_name("unified_budget_trace.json")


def _cache(storage, sliding_window_size):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=storage.allocator,
            page_size=storage.allocator.page_size,
            sliding_window_size=sliding_window_size,
            tree_slots=storage.tree_slots(),
            tree_core_backend="python",
        )
    )


def _history(allocator, seed):
    """Take and free some tokens so admission sees holes and live pages."""
    rng = random.Random(seed)
    held = []
    for _ in range(rng.randint(2, 6)):
        got = allocator.alloc(allocator.page_size * rng.randint(1, 4))
        if got is not None:
            held.append(got)
    for ids in held:
        if rng.random() < 0.4:
            allocator.free(ids)


def _windows(names, values):
    return {name: value for name, value in zip(names, values)}


def _allocator_answers(allocator, names, state):
    page = allocator.page_size
    out = {}
    evictables = {
        "none": {},
        "some": {
            allocator.full_attn_allocator.sub_pool_name: 4 * page,
            **{name: 2 * page for name in names},
            **({state: 1} if state is not None else {}),
        },
    }
    window_choices = [0, 8, 32]
    for full, slots, ev_name in itertools.product(
        [0, 8, 24, 64], [0, 1] if state is not None else [0], evictables
    ):
        for values in itertools.product(window_choices, repeat=len(names)):
            if len(names) > 1 and len(set(values)) > 2:
                continue
            windows = _windows(names, values)
            key = f"admits/{full}/{list(values)}/{slots}/{ev_name}"
            out[key] = allocator.admits(
                full, windows, slots, evictable=dict(evictables[ev_name])
            )
            demand = KVDemand(full, windows, slots)
            out[f"ceiling/{full}/{list(values)}/{slots}"] = (
                allocator.prealloc_ceiling_fits(demand)
            )
            out[f"take/{full}/{list(values)}/{slots}"] = allocator.ensure_capacity(
                demand
            )
            for full_budget, swa_budget in itertools.product(
                [16, 64, 200], [None, 8, 32]
            ):
                out[
                    f"prealloc/{full}/{list(values)}/{slots}/{ev_name}/"
                    f"{full_budget}/{swa_budget}"
                ] = allocator.prealloc_fits(
                    demand,
                    evictable=dict(evictables[ev_name]),
                    full_budget_tokens=full_budget,
                    swa_budget_tokens=swa_budget,
                )
    for input_tokens, max_new, chunk in itertools.product(
        [4, 40, 120], [8, 64], [None, 16]
    ):
        out[f"clip/{input_tokens}/{max_new}/{chunk}"] = (
            allocator.max_new_tokens_for_memory(
                input_tokens,
                max_new,
                token_capacity=allocator.size_full,
                sliding_window_size=16,
                chunk_size=chunk,
            )
        )
    return out


def _budget_answers(allocator, cache):
    budget = allocator.create_prefill_budget(cache)
    out = {
        "remaining_swa": int(budget.remaining_swa),
        "has_capacity": budget.has_capacity(),
    }
    for extend, max_new, chunk in itertools.product([4, 24, 80], [4, 32], [None, 16]):
        key = f"{extend}/{max_new}/{chunk}"
        out[f"check/{key}"] = list(
            budget.check_prefill(
                extend_input_len=extend,
                total_tokens=extend + max_new,
                max_new_tokens=max_new,
                input_tokens=extend,
                chunk_limit=chunk,
            )
        )
        out[f"alloc/{key}"] = budget.can_allocate_prefill(
            paged_input=-(-extend // allocator.page_size) * allocator.page_size,
            extend_input_len=extend,
            max_new_tokens=max_new,
            chunk_limit=chunk,
        )
        if chunk is not None:
            out[f"fit/{key}"] = budget.fit_chunk(
                extend_input_len=extend, max_new_tokens=max_new, chunk_limit=chunk
            )
    budget.reserve(16, 8)
    out["after_reserve/remaining_swa"] = int(budget.remaining_swa)
    out["after_reserve/check"] = list(
        budget.check_prefill(
            extend_input_len=24,
            total_tokens=32,
            max_new_tokens=8,
            input_tokens=24,
            chunk_limit=16,
        )
    )
    out["evictable"] = evictable_by_sub_pool(cache, allocator)
    return out


def trace():
    record = {}
    for windows in [(16,), (16, 8), (16, 8, 4)]:
        for seed in range(3):
            storage = _storage(windows, lazy=False)
            allocator = storage.allocator
            cache = _cache(storage, max(windows))
            _history(allocator, seed)
            names = list(allocator.window_pool_names())
            entry = {
                "window_pools": names,
                "retained": dict(storage.retained_window_lengths()),
                "tree_windows": [
                    [slot.component, slot.window]
                    for slot in storage.tree_slots()
                    if slot.window is not None
                ],
            }
            entry.update(_allocator_answers(allocator, names, None))
            entry.update(
                {f"budget/{k}": v for k, v in _budget_answers(allocator, cache).items()}
            )
            record[f"storage{list(windows)}/{seed}"] = entry
    for chain in ("pair", "tri"):
        for seed in range(3):
            allocator = _allocator("cpu", chain, lazy=True)
            # As the builder binds it: the model's window is 32 tokens.
            allocator.storage.bind_window_retention(32, None)
            _history(allocator, seed)
            names = list(allocator.window_pool_names())
            state = (
                None
                if allocator.mamba_allocator is None
                else allocator.mamba_allocator.sub_pool_name
            )
            entry = {"window_pools": names}
            entry.update(_allocator_answers(allocator, names, state))
            record[f"{chain}/{seed}"] = entry
    return record


def _values(entry):
    """An entry's answers in key order: the record keeps values only, its
    keys are rebuilt from the same grid."""
    return [entry[key] for key in sorted(entry)]


class TestBudgetTrace(CustomTestCase):
    def test_admission_and_budgets_match_the_record(self):
        got = json.loads(json.dumps(trace(), sort_keys=True))
        if os.environ.get("SGLANG_RECORD_BUDGET_TRACE") == "1":
            compact = {config: _values(entry) for config, entry in got.items()}
            _RECORD.write_text(
                json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n"
            )
        want = json.loads(_RECORD.read_text())
        self.assertEqual(sorted(got), sorted(want))
        for config, values in want.items():
            with self.subTest(config=config):
                keys = sorted(got[config])
                self.assertEqual(len(keys), len(values))
                differ = {
                    key: (got[config][key], value)
                    for key, value in zip(keys, values)
                    if got[config][key] != value
                }
                self.assertEqual(differ, {})


if __name__ == "__main__":
    unittest.main()
