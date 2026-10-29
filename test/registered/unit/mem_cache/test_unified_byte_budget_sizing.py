# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A total device budget includes the chain, its sink and declared fixed storage.

The bs=1 floor rejects budgets that cannot hold the state slots and window of
one request, before constructing any arena.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.decl import FullLifetime, StateLifetime
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.memory.arena import KVArena, plan_bands
from sglang.srt.mem_cache.kv.plan import KVStoragePlan, PlannedSubPool, _unit
from sglang.srt.mem_cache.kv.sizing import check_bs1_floor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_DEV = "cpu"


def _swa_factory(**over):
    kw = dict(
        device=_DEV,
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=1,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[1, 3],
        full_attention_layer_ids=[0, 2],
        full_max_total_num_tokens=64,
        swa_max_total_num_tokens=32,
        enable_memory_saver=False,
        need_sort=False,
    )
    kw.update(over)
    return build_test_unified_pools(**kw)


def _entry_bytes():
    full = MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=2,
        head_dim=8,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    return full.entry_bytes()


class TestBudgetSizing(unittest.TestCase):
    def test_swa_factory_honors_the_budget_exactly(self):
        """Arena and fixed storage partition the complete budget, including
        bytes that a whole-token capacity cannot represent."""
        e = _entry_bytes()
        for budget in (
            96 * e + 512,  # deliberately NOT a token-count multiple
            (64 + 32) * e + (e - 2),  # almost one more entry
        ):
            with self.subTest(budget=budget):
                bundle = _swa_factory(total_bytes=budget)
                self.assertEqual(
                    bundle.arena.total_bytes + bundle.storage.sized.fixed_bytes, budget
                )

    def test_fallback_is_the_token_count_resum(self):
        e = _entry_bytes()
        bundle = _swa_factory()
        self.assertEqual(
            bundle.arena.total_bytes, (64 + 32) * e + bundle.storage.plan.sink_bytes()
        )


class TestReservedFloorIsOneSourceOfTruth(unittest.TestCase):
    """The bs=1 floor charges the slot-0 sink, and MUST charge exactly what
    the arena actually reserves.

    Regression (Falcon-H1 boot): the floor hand-copied the formula as
    `page_size * max(entry_bytes)`, applying the page multiplier to the MAMBA
    spec, which the pool deliberately excludes because it is page_size=1. At
    page_size=256 that over-charged the sink 256x and a healthy config failed
    to boot with real headroom to spare.
    """

    def _specs(self, page_size):
        full = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=8,
            store_dtype=torch.float16,
            grow_direction="down",
        )
        # A state entry vastly larger than a KV token entry: the real ratio
        # (~139 MB vs ~45 KB) is what made the over-charge fatal.
        mamba = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            conv_state_shapes=((4, 256),),
            conv_dtype=torch.float16,
            temporal_state_shape=(4, 256, 64),
            temporal_dtype=torch.float16,
            grow_direction="up",
        )
        return full, mamba

    @staticmethod
    def _plan(full, mamba, page_size) -> KVStoragePlan:
        """The two specs as a plan lays them out; only units and specs matter."""
        return KVStoragePlan(
            page_size=page_size,
            sub_pools=tuple(
                PlannedSubPool(
                    name=spec.name,
                    lifetime=lifetime,
                    unit=_unit(lifetime, page_size),
                    layer_ids=(0, 1),
                    geometry=None,
                    spec=spec,
                    external=None,
                )
                for spec, lifetime in ((mamba, StateLifetime()), (full, FullLifetime()))
            ),
        )

    def test_mamba_entry_is_not_multiplied_by_page_size(self):
        full, mamba = self._specs(page_size=256)
        got = self._plan(full, mamba, 256).sink_bytes()
        self.assertEqual(got, max(mamba.entry_bytes(), 256 * full.entry_bytes()))
        self.assertLess(got, 256 * mamba.entry_bytes())  # the bug's value

    def test_floor_sink_equals_what_the_pool_reserves(self):
        """Pin the two against each other so the formula cannot drift again."""
        for page_size in (1, 4, 256):
            with self.subTest(page_size=page_size):
                full, mamba = self._specs(page_size)
                plan = self._plan(full, mamba, page_size)
                floor = plan.sink_bytes()
                pool = KVArena(
                    bands=plan_bands(plan),
                    total_bytes=floor + 64 * mamba.entry_bytes(),
                    sink_bytes=floor,
                    device=_DEV,
                )
                # min_slot_index is ceil(reserved_floor / entry_bytes) per side.
                for spec in (full, mamba):
                    self.assertEqual(
                        pool.first_entry(spec.name),
                        -(-floor // spec.entry_bytes()),
                    )


class TestBs1FeasibilityFloor(unittest.TestCase):
    def test_infeasible_budget_raises_before_construction(self):
        """The buffer cannot hold one sliding window plus the sink, so boot
        must fail loud instead of livelocking later."""
        with self.assertRaises(RuntimeError) as ctx:
            _swa_factory(
                total_bytes=8 * _entry_bytes(),
                model_context_len=4096,
                sliding_window_size=4096,
            )
        self.assertIn("bs=1 floor", str(ctx.exception))
        self.assertIn("swa_window_kv", str(ctx.exception))

    def test_feasible_floor_inputs_are_not_rejected(self):
        """Floor inputs that must NOT raise: charging the full-attention token
        side, or a window the context never clamps, failed these at boot."""
        e = _entry_bytes()
        for case, model_context_len, sliding_window_size in (
            ("context_longer_than_the_pool", 1_000_000, 16),
            ("feasible_config", 64, 16),
            ("window_larger_than_context", 64, 10_000),
        ):
            with self.subTest(case=case):
                # The request table is fixed storage even when context is much
                # longer than the KV capacity; reserve its two rows explicitly.
                table_bytes = 2 * 4 * model_context_len
                bundle = _swa_factory(
                    total_bytes=200 * e + table_bytes,
                    model_context_len=model_context_len,
                    sliding_window_size=sliding_window_size,
                )
                self.assertEqual(
                    bundle.arena.total_bytes + bundle.storage.sized.fixed_bytes,
                    200 * e + table_bytes,
                )

    def test_exact_floor_passes(self):
        """Boundary: total == floor must NOT raise (>= is the contract)."""
        check_bs1_floor(total_bytes=16, floor_terms=[("a", 8), ("b", 8)], where="test")


if __name__ == "__main__":
    unittest.main()
