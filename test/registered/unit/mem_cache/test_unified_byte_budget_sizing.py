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
"""Byte-budget buffer sizing for the unified 2-pool factories.

Derived properties pinned here:

  * Budget honored EXACTLY: with ``unified_total_bytes`` set, the swa pair's
    buffer is that many bytes (the mamba pair adds the state pool's bytes on
    top — the budget is captured AFTER the state carve-out). Sizing from the
    ratio-derived token counts instead re-introduces the configurator's
    rounding: the swa split floors the budget by the cell size and then
    page-aligns EACH side's token count, so the re-sum reconstructs less
    than the profiled budget by up to about one page of tokens per side.
  * Fallback: without the budget, sizing is the historical token-count re-sum,
    bit-for-bit.
  * bs=1 feasibility floor: a budget that cannot fit ONE worst-case request
    (full KV at max context, plus one SWA window / the state slots a single
    running request locks) raises at BOOT, before any pool construction —
    under-sizing is a retract LIVELOCK at runtime, not a perf bug.
  * The 4096-byte alignment exists because the factories ``.view()`` the whole
    uint8 buffer as the KV dtype; an unaligned budget must be floored, never
    rounded up (rounding up overcommits profiled memory).

    python -m pytest test/registered/unit/mem_cache/test_unified_byte_budget_sizing.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
    _check_bs1_feasibility_floor,
    _reserved_floor_bytes,
    init_unified_swa_pools,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

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
    return init_unified_swa_pools(**kw)


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
        e = _entry_bytes()
        budget = 96 * e + 512  # deliberately NOT a token-count multiple
        bundle = _swa_factory(unified_total_bytes=budget)
        self.assertEqual(bundle.unified_memory_pool.total_bytes, budget)

    def test_fallback_is_the_token_count_resum(self):
        e = _entry_bytes()
        bundle = _swa_factory()
        self.assertEqual(bundle.unified_memory_pool.total_bytes, (64 + 32) * e)

    def test_budget_beats_resum_on_rounding(self):
        """The property that motivates the whole phase: the re-sum cannot
        represent a budget that is not a whole-token multiple per side, so it
        strands bytes the buffer could have held."""
        e = _entry_bytes()
        budget = (64 + 32) * e + (e - 2)  # almost one more entry
        bundle = _swa_factory(unified_total_bytes=budget)
        self.assertEqual(bundle.unified_memory_pool.total_bytes, budget)
        self.assertGreater(budget, (64 + 32) * e)


class TestReservedFloorIsOneSourceOfTruth(unittest.TestCase):
    """The bs=1 floor charges the slot-0 sink, and MUST charge exactly what
    `UnifiedKVPool` actually reserves.

    Regression (GPU eval_434/436, Falcon-H1 boot): the floor hand-copied the
    formula as `page_size * max(entry_bytes)`, applying the page multiplier to
    the MAMBA spec. The pool deliberately excludes mamba (it is page_size=1),
    so with page_size=256 and a ~139 MB state entry the floor over-charged the
    sink by 256x — ~33 GiB of phantom requirement — and a healthy config
    failed to boot with 25 GiB of real headroom.
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
        # A state entry vastly larger than a KV token entry — the real ratio
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

    def test_mamba_entry_is_not_multiplied_by_page_size(self):
        full, mamba = self._specs(page_size=256)
        got = _reserved_floor_bytes([full, mamba], 256)
        self.assertEqual(got, max(mamba.entry_bytes(), 256 * full.entry_bytes()))
        self.assertLess(got, 256 * mamba.entry_bytes())  # the bug's value

    def test_floor_sink_equals_what_the_pool_reserves(self):
        """Pin the two against each other so the formula cannot drift again."""
        for page_size in (1, 4, 256):
            with self.subTest(page_size=page_size):
                full, mamba = self._specs(page_size)
                floor = _reserved_floor_bytes([full, mamba], page_size)
                pool = UnifiedKVPool(
                    total_bytes=floor + 64 * mamba.entry_bytes(),
                    sub_pool_specs=[full, mamba],
                    device=_DEV,
                    enable_memory_saver=False,
                    page_size=page_size,
                )
                # min_slot_index is ceil(reserved_floor / entry_bytes) per side.
                for spec in (full, mamba):
                    self.assertEqual(
                        pool.min_slot_index(spec.name),
                        -(-floor // spec.entry_bytes()),
                    )


class TestBs1FeasibilityFloor(unittest.TestCase):
    def test_infeasible_budget_raises_before_construction(self):
        """The buffer cannot hold one sliding window plus the sink, so boot
        must fail loud instead of livelocking later."""
        with self.assertRaises(RuntimeError) as ctx:
            _swa_factory(
                unified_total_bytes=8 * _entry_bytes(),
                model_context_len=4096,
                sliding_window_size=4096,
            )
        self.assertIn("bs=1 floor", str(ctx.exception))
        self.assertIn("swa_window_kv", str(ctx.exception))

    def test_context_longer_than_the_pool_is_not_rejected(self):
        """REGRESSION: the floor must NOT charge the full-attention token side.
        `TpModelWorker.get_worker_info` clamps max_req_len to the pool, so a
        context far larger than the buffer is refused at admission, not a
        livelock -- and it is an ordinary way to serve a long-context model on
        one GPU. Charging it here made such configs fail at boot."""
        e = _entry_bytes()
        bundle = _swa_factory(
            unified_total_bytes=200 * e,
            model_context_len=1_000_000,  # far beyond what the buffer holds
            sliding_window_size=16,
        )
        self.assertEqual(bundle.unified_memory_pool.total_bytes, 200 * e)

    def test_feasible_config_boots_with_floor_inputs_present(self):
        e = _entry_bytes()
        bundle = _swa_factory(
            unified_total_bytes=200 * e,
            model_context_len=64,
            sliding_window_size=16,
        )
        self.assertEqual(bundle.unified_memory_pool.total_bytes, 200 * e)

    def test_window_term_is_clamped_to_context(self):
        """A window larger than the context must charge at most the context —
        otherwise short-context models over-raise."""
        e = _entry_bytes()
        bundle = _swa_factory(
            unified_total_bytes=200 * e,
            model_context_len=64,
            sliding_window_size=10_000,  # window >> context
        )
        self.assertIsNotNone(bundle)

    def test_floor_message_itemizes_terms(self):
        with self.assertRaises(RuntimeError) as ctx:
            _check_bs1_feasibility_floor(
                total_bytes=10,
                floor_terms=[("a", 8), ("b", 8)],
                factory="test",
            )
        msg = str(ctx.exception)
        self.assertIn("a=8", msg)
        self.assertIn("b=8", msg)
        self.assertIn("16", msg)

    def test_exact_floor_passes(self):
        """Boundary: total == floor must NOT raise (>= is the contract)."""
        _check_bs1_feasibility_floor(
            total_bytes=16, floor_terms=[("a", 8), ("b", 8)], factory="test"
        )


if __name__ == "__main__":
    unittest.main()
