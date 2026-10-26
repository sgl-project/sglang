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

The SWA factory's ``total_bytes`` sets the exact buffer size. The Mamba pair's
``unified_total_bytes`` adds state bytes because its budget excludes that carve-out.
Without an explicit budget, sizing falls back to the token-count re-sum.
Sizing from the ratio-derived token counts instead would re-introduce the
configurator's rounding, which floors by the cell size and then page-aligns
EACH side, losing up to about a page of tokens per side.

The bs=1 feasibility floor raises at BOOT, before any pool construction: a
budget that cannot fit ONE worst-case request (full KV at max context, plus
one SWA window / the state slots a running request locks) is a retract
LIVELOCK at runtime, not a perf bug.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPoolMXFP8,
    _check_bs1_feasibility_floor,
    _reserved_floor_bytes,
    init_unified_mamba_pools,
    init_unified_swa_pools,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
        """The buffer IS the budget, including values the token-count re-sum
        cannot represent (not a whole-token multiple per side)."""
        e = _entry_bytes()
        for budget in (
            96 * e + 512,  # deliberately NOT a token-count multiple
            (64 + 32) * e + (e - 2),  # almost one more entry
        ):
            with self.subTest(budget=budget):
                bundle = _swa_factory(total_bytes=budget)
                self.assertEqual(bundle.unified_memory_pool.total_bytes, budget)

    def test_fallback_is_the_token_count_resum(self):
        e = _entry_bytes()
        bundle = _swa_factory()
        self.assertEqual(bundle.unified_memory_pool.total_bytes, (64 + 32) * e)


class TestReservedFloorIsOneSourceOfTruth(unittest.TestCase):
    """The bs=1 floor charges the slot-0 sink, and MUST charge exactly what
    `UnifiedKVPool` actually reserves.

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
                bundle = _swa_factory(
                    total_bytes=200 * e,
                    model_context_len=model_context_len,
                    sliding_window_size=sliding_window_size,
                )
                self.assertEqual(bundle.unified_memory_pool.total_bytes, 200 * e)

    def test_exact_floor_passes(self):
        """Boundary: total == floor must NOT raise (>= is the contract)."""
        _check_bs1_feasibility_floor(
            total_bytes=16, floor_terms=[("a", 8), ("b", 8)], factory="test"
        )


class TestBlockScaledPool(unittest.TestCase):
    def _pool(self, page_size, **over):
        full = MHASubPoolSpec(
            name="full",
            layer_num=1,
            head_num=1,
            head_dim=32,
            store_dtype=torch.uint8,
            kv_cache_dtype=torch.float8_e4m3fn,
            scale_block_size=32,
            grow_direction="down",
        )
        return UnifiedKVPool(
            total_bytes=8 * 128 * full.entry_bytes(),
            sub_pool_specs=[full, replace(full, name="swa", grow_direction="up")],
            device=_DEV,
            enable_memory_saver=False,
            page_size=page_size,
            **over,
        )

    def test_post_capture_backing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "block-scaled"):
            self._pool(128, post_capture_active=True)

    def test_page_size_must_match_the_interleaved_layout(self):
        self._pool(128)
        with self.assertRaisesRegex(ValueError, "page_size 128"):
            self._pool(64)

    def test_mamba_factory_builds_a_scaled_full_pool(self):
        params = SimpleNamespace(
            shape=SimpleNamespace(conv=[(3, 8)], temporal=(0, 0, 0)),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
            layers=[1],
        )
        with get_parallel().override(attn_dcp_size=1, attn_tp_size=1):
            bundle = init_unified_mamba_pools(
                device=_DEV,
                kv_cache_dtype=torch.float8_e4m3fn,
                scale_block_size=32,
                head_num=1,
                head_dim=32,
                v_head_dim=32,
                page_size=128,
                start_layer=0,
                end_layer=2,
                is_draft_worker=False,
                use_mla_backend=False,
                mamba_layer_ids=[1],
                full_attention_layer_ids=[0],
                mamba2_cache_params=params,
                model_context_len=256,
                extra_max_context_len=4,
                max_total_num_tokens=1024,
                max_mamba_cache_size=4,
                max_num_reqs=4,
                enable_memory_saver=False,
                enable_mamba_extra_buffer=False,
                enable_mamba_extra_buffer_lazy=False,
                speculative_num_draft_tokens=None,
                disable_overlap_schedule=True,
                need_sort=False,
            )
        self.assertIsInstance(
            bundle.token_to_kv_pool.full_kv_pool, UnifiedMHATokenToKVPoolMXFP8
        )


if __name__ == "__main__":
    unittest.main()
