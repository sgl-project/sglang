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
    MHASubPoolSpec,
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


if __name__ == "__main__":
    unittest.main()
