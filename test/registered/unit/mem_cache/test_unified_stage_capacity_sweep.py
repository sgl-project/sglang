"""Pipeline stages under unified memory, swept over random take and free
sequences: every answer is the conjunction of the stages' own answers."""

import random
import unittest
from typing import List

from test_unified_stage_capacity import (
    _STATE_STAGES,
    _admits_alone,
    _fits_alone,
    _on_every_stage,
    _stages,
)

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, stage="weekly", runner_config="cpu")

# With a sliding window (a float between the ends): the first stage mostly
# full attention, the second mostly window layers.
_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=tuple(range(4)),
        swa_layer_ids=(4,),
        state_layer_ids=(5,),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
    ),
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=(6,),
        swa_layer_ids=(7, 8, 9, 10),
        state_layer_ids=(11, 12),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
        start_layer=6,
    ),
)


def _probe_demands(allocator, rng, count=12) -> List[dict]:
    owner, windows, state = allocator.sub_pool_roles()
    probes = []
    for _ in range(count):
        pages = rng.randrange(0, 80)
        demands = {owner: pages}
        for window in windows:
            demands[window] = min(pages, rng.randrange(0, 8))
        if state is not None:
            demands[state] = rng.randrange(0, 24)
        probes.append(demands)
    return probes


class TestStageCapacityLockstep(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def _check_lockstep_simulation(self, geometries, seed):
        """As the stages take and free the same units, every take is decided alike and
        every answer is the conjunction of the stages' answers on their real layouts."""
        stages = _stages(geometries)
        a0, a1 = stages
        rng = random.Random(seed)
        owner, windows, state = a0.sub_pool_roles()
        tokens, slots = [], []
        disagreements = refusals = 0
        for _ in range(160):
            roll = rng.random()
            if roll < 0.4:
                n = rng.randrange(1, 24) * a0.page_size
                taken = _on_every_stage(stages, lambda a: a.alloc(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is None:
                    refusals += 1
                else:
                    tokens.append(taken)
            elif roll < 0.6 and state is not None:
                n = rng.randrange(1, 6)
                taken = _on_every_stage(stages, lambda a: a.alloc_state_slots(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is not None:
                    slots.append(taken)
            elif roll < 0.85 and tokens:
                pair = tokens.pop(rng.randrange(len(tokens)))
                for allocator, ids in zip(stages, pair):
                    allocator.free(ids)
            elif slots:
                pair = slots.pop(rng.randrange(len(slots)))
                for allocator, ids in zip(stages, pair):
                    allocator.mamba_allocator.free(ids)
            self.assertEqual(a0.stage_counts(), a1.stage_counts())
            for demands in _probe_demands(a0, rng, count=4):
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                disagreements += fit0 != fit1
                self.assertEqual(a0.units_fit(demands), fit0 and fit1, demands)
                self.assertEqual(a1.units_fit(demands), fit0 and fit1, demands)
                evictable = {
                    name: rng.randrange(0, 16) * (1 if name == state else 4)
                    for name in (owner, *windows, state)
                    if name is not None
                }
                demand = KVDemand(
                    demands[owner] * a0.page_size,
                    {w: demands[w] * a0.page_size for w in windows},
                    demands.get(state, 0),
                )
                alone = _admits_alone(a0, demand, evictable) and _admits_alone(
                    a1, demand, evictable
                )
                for allocator in stages:
                    self.assertEqual(
                        allocator.admits(demand, evictable=evictable),
                        alone,
                        (demands, evictable),
                    )
        self.assertGreater(disagreements, 0)
        self.assertGreater(refusals, 0)
        for allocator in stages:
            self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_lockstep_simulation_state_chain(self):
        self._check_lockstep_simulation(_STATE_STAGES, seed=0)

    def test_lockstep_simulation_window_chain(self):
        self._check_lockstep_simulation(_WINDOW_STAGES, seed=1)


if __name__ == "__main__":
    unittest.main()
