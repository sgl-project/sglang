"""Capacity pricing over a band snapshot agrees with what the allocator then does.

The unified chain allocator decides capacity on a host-only snapshot of its
bands, then realizes what the price needs (end flushes, a float relocation) and
takes. For every state reached by a fixed list of seeded operation streams and
by the corner states (holes, a closed move gate, pending reuse behind an unfired
forward, a pending host-cache reservation, a float with interior holes, a full
allocator):

- on a chain of two token ends and on one whose window floats,
  `ensure_capacity(n, n)` holds exactly when an allocation of n tokens then
  succeeds, and asking it moves no page;
- a refused allocation leaves every band's live pages as they were;
- the snapshot the chain keeps between decisions equals a fresh one after
  writes and after move-gate flips.

A price that its realization does not deliver, or a realization the price does
not credit, turns this red.
"""

import random
import unittest
from typing import Dict, List, Tuple

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.token_demand import token_demand
from sglang.test.unified_allocator_fixtures import chain_snapshot
from sglang.test.unified_allocator_trace import (
    BuilderAdapter,
    HostModel,
    TraceConfig,
    apply_op,
    corner_states,
    draw_op,
)

register_cuda_ci(est_time=40, stage="extra-a", runner_config="1-gpu-small")

_SEEDS = (0, 1, 2, 3)
_STEPS = 60
_CHECKPOINTS = (15, 30, 45, 60)
# Demands around what fits as the chain stands, where realizations matter.
_PAGES_PAST_AVAILABLE = (0, 1, 2, 4, 8)


def _pressure_states(config: TraceConfig) -> List[Tuple[str, List[dict]]]:
    """Fill the chain until a request is refused, then punch holes and (on
    window chains) retire windows, so pricing must credit realizations."""
    ps = config.page_size
    probe, model, ops = BuilderAdapter(config), HostModel(), []
    for req in range(256):
        op = {"op": "alloc", "req": req, "n": 4 * ps}
        if apply_op(probe, model, op) is None:
            break
        ops.append(op)
    out = [("filled", list(ops))]
    reqs = sorted(model.requests)
    for req in reqs[::3]:
        op = {"op": "free", "req": req}
        apply_op(probe, model, op)
        ops.append(op)
    out.append(("filled_then_holes", list(ops)))
    if config.chain in ("swa", "tri"):
        for req in sorted(model.requests)[::2]:
            op = {"op": "retire", "req": req, "upto": len(model.requests[req].tokens)}
            apply_op(probe, model, op)
            ops.append(op)
        out.append(("filled_holes_retired", list(ops)))
    return out


def _states(config: TraceConfig) -> List[Tuple[str, List[dict]]]:
    out = list(corner_states(config).items())
    out += _pressure_states(config)
    for seed in _SEEDS:
        model, rng, ops = HostModel(), random.Random(seed), []
        probe = BuilderAdapter(config)
        for step in range(1, _STEPS + 1):
            op = draw_op(rng, model, config)
            apply_op(probe, model, op)
            ops.append(op)
            if step in _CHECKPOINTS:
                out.append((f"seed{seed}@{step}", list(ops)))
    return out


def _rebuild(config: TraceConfig, ops: List[dict]) -> BuilderAdapter:
    adapter, model = BuilderAdapter(config), HostModel()
    for op in ops:
        apply_op(adapter, model, op)
    return adapter


def _live(adapter: BuilderAdapter) -> Dict[str, int]:
    return {b.sub_pool_name: b.allocated_count() for b in adapter.allocator.bands}


class TestUnifiedPricingProperty(CustomTestCase):
    def _each_state(self, chain: str):
        for page_size in (1, 4):
            for lazy in (False, True):
                config = TraceConfig(chain=chain, page_size=page_size, lazy=lazy)
                for name, ops in _states(config):
                    available = _rebuild(config, ops).allocator.available_size()
                    for extra in _PAGES_PAST_AVAILABLE:
                        tokens = max(page_size, available + extra * page_size)
                        yield config, name, ops, tokens

    def _allocate(self, adapter: BuilderAdapter, tokens: int) -> bool:
        before = _live(adapter)
        out = adapter.allocator.alloc(tokens)
        if out is None:
            self.assertEqual(_live(adapter), before, "a refused allocation took pages")
        return out is not None

    def _prices_exactly(self, chain: str):
        for config, name, ops, tokens in self._each_state(chain):
            with self.subTest(config=config.key(), state=name, tokens=tokens):
                adapter = _rebuild(config, ops)
                before = chain_snapshot(adapter.allocator)
                priced = adapter.allocator.ensure_capacity(
                    token_demand(adapter.allocator, tokens, tokens)
                )
                self.assertEqual(chain_snapshot(adapter.allocator), before)
                self.assertEqual(self._allocate(adapter, tokens), priced)

    def test_two_token_ends_price_exactly(self):
        self._prices_exactly("swa")

    def test_float_chain_prices_exactly(self):
        self._prices_exactly("tri")

    def test_kept_snapshot_follows_writes_and_gates(self):
        """The chain keeps one pricing snapshot until a band is written or a
        move gate's answer changes, which no capacity field records."""
        for chain in ("swa", "tri"):
            with self.subTest(chain=chain):
                adapter = BuilderAdapter(
                    TraceConfig(chain=chain, page_size=1, lazy=True)
                )
                c = adapter.allocator
                kept = c.pricing_snapshot()
                self.assertIs(c.pricing_snapshot(), kept)
                ids = adapter.alloc(8)
                written = c.pricing_snapshot()
                self.assertIsNot(written, kept)
                self.assertEqual(written, chain_snapshot(c))
                adapter.free(ids[:4])
                self.assertEqual(c.pricing_snapshot(), chain_snapshot(c))
                adapter.gate(False)
                closed = c.pricing_snapshot()
                self.assertTrue(any(band.blocked for band in closed.bands))
                self.assertEqual(closed, chain_snapshot(c))
                adapter.gate(True)
                opened = c.pricing_snapshot()
                self.assertFalse(any(band.blocked for band in opened.bands))
                self.assertEqual(opened, chain_snapshot(c))
                self.assertEqual(c.verify_byte_accounting(), [])


if __name__ == "__main__":
    unittest.main()
