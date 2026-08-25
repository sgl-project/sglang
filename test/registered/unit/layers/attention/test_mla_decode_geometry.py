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
"""Stage-1 split budget for the gfx950 MLA decode geometry.

Stage-1 launches `batch * head_tiles * kv_splits` workgroups, and crossing the
bucket's budget costs a step rather than a proportional slice: at batch 24 on a 68k
context, 21 splits measured 358 us against 528 us for 22. Which is also why the
budget is divided down with floor and not round -- `round(512/12) = 43` would put
batch 12 at 516 blocks, just over.

The budget lives with the geometry rather than as a global constant, since halving
`num_warps` moved the cliff from 512 blocks to 1024. These tests pin the two
together, so retuning one without the other, or rounding the division up, fails here
instead of costing 50% at one batch size.

    python -m pytest test/registered/unit/layers/attention/test_mla_decode_geometry.py -v
"""

import unittest

from sglang.kernels.ops.attention import decode_attention as da
from sglang.kernels.ops.attention.decode_attention import (
    _MLA_BLOCK_N,
    _MLA_BUCKET_BATCH_FREE,
    _MLA_BUCKETS,
    _fwd_grouped_kernel_stage1,
    _grouped_head_tiles,
    _keep_scheduler_splits,
    _mla_bucket,
    _mla_kv_splits,
    _mla_split_budget,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# gfx950 full-GPU. Passed in rather than read from the device so this stays a CPU test.
CORE_COUNT = 256

# The batch sizes cuda-graph capture walks, and the split count separately
# measured as optimal at each on a 68k context (head_tiles == 1, i.e. K3 at tp 8).
MEASURED_OPTIMUM = {
    1: 112,
    2: 112,
    3: 85,
    4: 64,
    5: 51,
    6: 85,
    7: 73,
    8: 64,
    10: 51,
    12: 42,
    14: 36,
    16: 32,
    20: 25,
    24: 21,
    28: 36,
    32: 32,
}

MAX_KV_SPLITS = 256


class TestMlaDecodeGeometry(unittest.TestCase):
    def test_rule_reproduces_measured_optimum(self):
        # a budget plus two constants, not a fit, so it hits the measured optimum
        for batch, want in MEASURED_OPTIMUM.items():
            with self.subTest(batch=batch):
                self.assertEqual(
                    _mla_kv_splits(batch, 1, MAX_KV_SPLITS, CORE_COUNT), want
                )

    def test_stage1_split_count_stays_runtime(self):
        # A constexpr here would compile one stage-1 variant per rung of the capture
        # ladder (21 for the default one), and the count only feeds kv_len_per_split.
        params = {p.name: p for p in _fwd_grouped_kernel_stage1.params}
        self.assertFalse(params["forced_kv_splits"].is_constexpr)
        self.assertTrue(params["USE_FORCED"].is_constexpr)

    def test_batch_free_geometry_is_pinned(self):
        # Deterministic inference runs on this geometry and BLOCK_N/num_warps reorder
        # the fp32 accumulation, so retuning either moves those numbers.
        self.assertEqual(
            (
                _MLA_BLOCK_N,
                _MLA_BUCKET_BATCH_FREE.num_warps,
                _MLA_BUCKET_BATCH_FREE.num_stages,
            ),
            (32, 2, 2),
        )

    def test_head_tiles_matches_the_grid(self):
        # The budget is divided by the same head_tiles the grid is launched with; a
        # BLOCK_H that drifts between the two silently mis-sizes the budget.
        for head_num, kv_group_num, want in ((16, 16, 1), (128, 128, 8), (8, 8, 1)):
            with self.subTest(head_num=head_num):
                self.assertEqual(_grouped_head_tiles(head_num, kv_group_num), want)

    def test_never_crosses_the_budget(self):
        for head_tiles in (1, 2):
            for batch in range(1, 1025):
                splits = _mla_kv_splits(batch, head_tiles, MAX_KV_SPLITS, CORE_COUNT)
                budget = _mla_split_budget(_mla_bucket(batch).num_warps, CORE_COUNT)
                with self.subTest(batch=batch, head_tiles=head_tiles):
                    if batch * head_tiles <= budget:
                        self.assertLessEqual(batch * head_tiles * splits, budget)
                    else:
                        # already past the budget, so 1 is the floor
                        self.assertEqual(splits, 1)

    def test_low_batch_is_capped_not_scaled(self):
        # below 6 the budget stops binding, and more splits stopped paying at 112
        # whatever the batch, so the cap sits on top of the budget instead of scaling
        self.assertEqual(_mla_kv_splits(1, 1, MAX_KV_SPLITS, CORE_COUNT), 112)
        self.assertEqual(_mla_kv_splits(2, 1, MAX_KV_SPLITS, CORE_COUNT), 112)
        self.assertLess(
            1 * 112, _mla_split_budget(_mla_bucket(1).num_warps, CORE_COUNT)
        )

    def test_caller_max_kv_splits_wins(self):
        for cap in (1, 4, 16):
            with self.subTest(max_kv_splits=cap):
                self.assertLessEqual(_mla_kv_splits(1, 1, cap, CORE_COUNT), cap)

    def test_at_least_one_split(self):
        for batch in (1, 4096):
            with self.subTest(batch=batch):
                self.assertGreaterEqual(
                    _mla_kv_splits(batch, 1, MAX_KV_SPLITS, CORE_COUNT), 1
                )

    def test_buckets_are_ordered_and_total(self):
        bounds = [b.batch_max for b in _MLA_BUCKETS]
        self.assertIsNone(bounds[-1], "the last bucket has to be the catch-all")
        finite = bounds[:-1]
        self.assertEqual(finite, sorted(finite))
        self.assertTrue(all(b is not None for b in finite))

    def test_wider_workgroup_gets_a_tighter_budget(self):
        # the budget divides by num_warps because that is where the cliff moved:
        # halving the warps took it from 512 blocks to 1024
        budgets = [
            _mla_split_budget(w, CORE_COUNT)
            for w in sorted({b.num_warps for b in _MLA_BUCKETS}, reverse=True)
        ]
        self.assertEqual(budgets, sorted(budgets))

    def test_budget_follows_the_partition_size(self):
        # A CPX partition exposes 32 of the 256 CUs while is_gfx95_supported() still
        # says yes, so a budget pinned to the whole GPU would oversubscribe it 8x.
        for warps in (1, 2, 4):
            with self.subTest(num_warps=warps):
                self.assertEqual(
                    _mla_split_budget(warps, 32) * 8,
                    _mla_split_budget(warps, 256),
                )
        self.assertEqual(
            _mla_kv_splits(8, 1, MAX_KV_SPLITS, 0), 0, "no core count, no budget"
        )

    def test_splits_shrink_on_a_partition(self):
        # Same batch, smaller partition -> fewer splits, never more.
        for batch in (8, 16, 32, 64):
            with self.subTest(batch=batch):
                self.assertLessEqual(
                    _mla_kv_splits(batch, 1, MAX_KV_SPLITS, 32),
                    _mla_kv_splits(batch, 1, MAX_KV_SPLITS, 256),
                )


class TestKeepSchedulerSplits(unittest.TestCase):
    """Which configs decline the batch-wide count.

    Through override_server_args, so the flags resolve the way a launched server
    resolves them; poking the cached decision keeps passing after they move namespace.
    """

    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)
        # resolved once per process, so clear it at both ends
        da._KEEP_SCHEDULER_SPLITS = None
        self.addCleanup(setattr, da, "_KEEP_SCHEDULER_SPLITS", None)

    def test_a_plain_config_takes_the_batch_wide_count(self):
        self._publish()
        self.assertFalse(_keep_scheduler_splits())

    def test_deterministic_inference_keeps_the_scheduler_splits(self):
        self._publish(enable_deterministic_inference=True)
        self.assertTrue(_keep_scheduler_splits())

    def test_an_explicit_split_tile_size_keeps_them(self):
        self._publish(triton_attention_split_tile_size=256)
        self.assertTrue(_keep_scheduler_splits())

    def test_the_static_kv_splits_env_keeps_them(self):
        self._publish()
        with envs.SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS.override(True):
            self.assertTrue(_keep_scheduler_splits())

    def test_the_geometry_rule_does_not_read_the_config(self):
        # _mla_kv_splits answers for a device, not for a config; the decline lives one
        # level up, so a deterministic config elsewhere cannot rewrite the pins above
        self._publish(enable_deterministic_inference=True)
        self.assertEqual(_mla_kv_splits(24, 1, MAX_KV_SPLITS, CORE_COUNT), 21)


if __name__ == "__main__":
    unittest.main()
