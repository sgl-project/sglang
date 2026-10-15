"""CPU tests: the MiniMax-M3 mono kernel constants at TP 2, 4 and 8.

TP is read once at import (``config.TP``) because the constants derived from it
are bound by value across about nine modules, so there is no later point at
which changing it would propagate. Varying it therefore means a fresh
interpreter per TP rather than a reload, which is what these tests do.

The TP4 column is a regression gate: those are the literal values PR #44319 was
written with, before TP became a parameter.
"""

import json
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Dumped in a child interpreter: importing config binds TP by value.
_PROBE = """
import json
from sglang.kernels.ops.mono.gfx950.minimax_m3 import config as c
from sglang.kernels.ops.mono.gfx950.minimax_m3 import layout as l
own = c.IndexHeads()
print(json.dumps({
    "TP": c.TP,
    "LOCAL_Q_HEADS": c.LOCAL_Q_HEADS,
    "LOCAL_KV_HEADS": c.LOCAL_KV_HEADS,
    "LOCAL_INDEX_HEADS": c.LOCAL_INDEX_HEADS,
    "INTER": c.INTER,
    "O_K": c.O_K,
    "MAX_WIDTH": c.MAX_WIDTH,
    "SCALE_COLS": c.SCALE_COLS,
    "PADDED_SCALE_COLS": c.PADDED_SCALE_COLS,
    "own_rows": own.rows,
    "own_count": own.count,
    "own_cp": own.cp,
    "own_score_blocks": own.score_blocks,
    "own_score_words": own.score_words,
    "own_iq_off": own.iq_off,
    "own_ik_off": own.ik_off,
    "H": l.H,
    "H_SUB": l.H_SUB,
    "SPLIT_PASSES": l.SPLIT_PASSES,
    "O_K_SUB": l.O_K_SUB,
    "KV_PAGE_BYTES": l.KV_PAGE_BYTES,
    "ISCORE_BLOCKS": l.ISCORE_BLOCKS,
    "N_SPLIT": l.N_SPLIT,
    "THREADS": c.THREADS,
    "MOE_SLOTS": c.MOE_SLOTS,
    "MERGE_SLOTS": l.MERGE_SLOTS,
    "MERGE_PASSES": l.MERGE_PASSES,
    "MERGE_ROUNDS_WHOLE": l.MERGE_ROUNDS_WHOLE,
    "MID_COPY_ROUNDS": l.MID_COPY_ROUNDS,
    "MID8_SLOTS": l.MID8_SLOTS,
    "MID8_ROUNDS": l.MID8_ROUNDS,
    "WIDE_FROM": l.WIDE_FROM,
    "HIDDEN": c.HIDDEN,
    "BLOCKS": c.BLOCKS,
    "SH_TASKS": l.SH_TASKS,
    "MAX_QKV_ROWS": c.MAX_QKV_ROWS,
    "cp_rows": c.qkv_rows(c.TOTAL_INDEX_HEADS),
    # lists, not dicts: JSON would turn the int widths into string keys
    "pool": [l.pool_words(s) for s in range(1, c.MAX_TOKENS + 1)],
    "wide": [l.wide_moe(s) for s in range(1, c.MAX_TOKENS + 1)],
}))
"""


def probe(tp: int) -> dict:
    """``config`` / ``layout`` constants as a build for ``tp`` sees them."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={**__import__("os").environ, "SGLANG_TEST_MONO_TP": str(tp)},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestMonoM3PerTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = {tp: probe(tp) for tp in (2, 4, 8)}

    def test_tp4_is_the_original_build(self):
        """PR #44319's literal TP4-only values. These must not drift."""
        v = self.v[4]
        for k, want in {
            "TP": 4, "LOCAL_Q_HEADS": 16, "LOCAL_KV_HEADS": 1,
            "LOCAL_INDEX_HEADS": 1, "INTER": 768, "O_K": 2048,
            "MAX_WIDTH": 16, "SCALE_COLS": 24, "PADDED_SCALE_COLS": 24,
            "own_rows": 2560, "own_count": 1, "own_iq_off": 2304,
            "own_ik_off": 2432, "H": 16, "H_SUB": 16, "SPLIT_PASSES": 1,
            "O_K_SUB": 2048,
        }.items():  # fmt: skip
            self.assertEqual(v[k], want, f"TP4 {k}")

    def test_heads_partition_the_model(self):
        """Every TP's shares multiply back up to the model's totals."""
        for tp, v in self.v.items():
            self.assertEqual(v["LOCAL_Q_HEADS"] * tp, 64, f"TP{tp} q heads")
            self.assertEqual(v["INTER"] * tp, 3072, f"TP{tp} INTER")
            # 4 KV and 4 index heads: a rank past 4 ranks still gets one
            self.assertEqual(v["LOCAL_KV_HEADS"], max(1, 4 // tp), f"TP{tp} kv")
            self.assertEqual(v["LOCAL_INDEX_HEADS"], max(1, 4 // tp), f"TP{tp} idx")

    def test_a_split_pass_is_one_kv_head_and_its_gqa_group(self):
        """The split stage's thread map holds 16 rows, so a pass -- one KV head
        and the q heads sharing it -- must be at most that wide."""
        for tp, v in self.v.items():
            self.assertEqual(v["SPLIT_PASSES"], v["LOCAL_KV_HEADS"], f"TP{tp}")
            self.assertEqual(v["H_SUB"] * v["LOCAL_KV_HEADS"], v["H"], f"TP{tp}")
            self.assertLessEqual(v["H_SUB"], 16, f"TP{tp} GQA group over the map")
            self.assertEqual(v["O_K_SUB"] * v["SPLIT_PASSES"], v["O_K"], f"TP{tp}")
            # the cache is [page16][kv head][...], 2048 B a head
            self.assertEqual(
                v["KV_PAGE_BYTES"], v["LOCAL_KV_HEADS"] * 16 * 128, f"TP{tp}"
            )

    def test_default_index_heads_are_the_ranks_own(self):
        """``IndexHeads()`` is own mode at every TP -- not context parallelism,
        whose count coincides with own mode's at TP2 while the meaning differs."""
        for tp, v in self.v.items():
            self.assertFalse(v["own_cp"], f"TP{tp} default must not be CP")
            self.assertEqual(v["own_count"], v["LOCAL_INDEX_HEADS"], f"TP{tp}")
            # own mode gives each head every block
            self.assertEqual(v["own_score_blocks"], 1 << 13, f"TP{tp}")
            self.assertEqual(
                v["own_score_words"],
                v["own_count"] * v["own_score_blocks"],
                f"TP{tp}",
            )
            # the fused projection: q | k | v | index_q | index_k
            rows = (
                v["LOCAL_Q_HEADS"] + 2 * v["LOCAL_KV_HEADS"] + 1 + v["own_count"]
            ) * 128
            self.assertEqual(v["own_rows"], rows, f"TP{tp} fused projection rows")

    def test_score_region_holds_whichever_mode_needs_more(self):
        """``ISCORE_BLOCKS`` must cover own mode (every head, every block) and
        CP (every model head, a 1/TP stripe). Equal wherever a rank has one
        index head, so TP4 and TP8 keep the original size."""
        for tp, v in self.v.items():
            self.assertGreaterEqual(
                v["ISCORE_BLOCKS"], v["LOCAL_INDEX_HEADS"] * (1 << 13), f"TP{tp}"
            )
            self.assertGreaterEqual(
                v["ISCORE_BLOCKS"], 4 * ((1 << 13) // tp), f"TP{tp} cp"
            )
        self.assertEqual(self.v[4]["ISCORE_BLOCKS"], 1 << 13)
        self.assertEqual(self.v[8]["ISCORE_BLOCKS"], 1 << 13)

    def test_two_split_tasks_per_model_index_head(self):
        """N_SPLIT is 2 * TOTAL_INDEX_HEADS at every TP. The original `2 * TP`
        coincided only at TP4."""
        for tp, v in self.v.items():
            self.assertEqual(v["N_SPLIT"], 8, f"TP{tp}")

    def test_width_cap_follows_the_lds_budget(self):
        """LDS scales with H and INTER, so a smaller TP buys fewer tokens per
        step. TP2 is capped; TP4 and TP8 are not. Measured by COMPILE_ONLY
        builds -- see config.MAX_WIDTH."""
        self.assertEqual(self.v[2]["MAX_WIDTH"], 12)
        self.assertEqual(self.v[4]["MAX_WIDTH"], 16)
        self.assertEqual(self.v[8]["MAX_WIDTH"], 16)

    def test_tp8_moe_scale_columns_are_padded(self):
        """aiter's e8m0_shuffle rounds an MXFP4 scale's columns up to a multiple
        of 8, so TP8's 12 become 16 and anything indexing that tensor must use
        the padded stride."""
        self.assertEqual(self.v[8]["SCALE_COLS"], 12)
        self.assertEqual(self.v[8]["PADDED_SCALE_COLS"], 16)
        for tp in (2, 4):
            v = self.v[tp]
            self.assertEqual(v["PADDED_SCALE_COLS"], v["SCALE_COLS"], f"TP{tp}")

    def test_thread_maps_cover_the_work_they_walk(self):
        """Every TP-sized map is walked in whole THREADS-wide rounds, and the
        rounds must actually reach the end of it.

        This is the shape of three TP2 bugs at once: each map was written as a
        single round, which is correct only because at TP4 the work comes to
        exactly THREADS slots. At TP2 the dimension doubles, one round covers
        half, and the rest of the region is left for the next stage to read as
        NaN bits. Asserting coverage rather than the literal counts is what
        makes this a test and not a restatement of the arithmetic."""
        for tp, v in self.v.items():
            threads, msg = v["THREADS"], f"TP{tp}"
            # the merge: 4 dims of one head a thread
            self.assertEqual(v["MERGE_SLOTS"], v["H"] * 128 // 4, msg)
            self.assertGreaterEqual(
                v["MERGE_PASSES"] * threads, v["MERGE_SLOTS"], f"{msg} merge short"
            )
            # the narrow MoE mid passes: 2 elements a thread, then 8
            work = v["MOE_SLOTS"] * v["INTER"]
            self.assertGreaterEqual(
                v["MID_COPY_ROUNDS"] * threads * 2, work, f"{msg} mid copy short"
            )
            self.assertEqual(v["MID8_SLOTS"], work // 8, msg)
            self.assertGreaterEqual(
                v["MID8_ROUNDS"] * threads, v["MID8_SLOTS"], f"{msg} mxfp8 short"
            )
            # a round's tail must not split the 4-lane group the MXFP8 scale is
            # reduced over, and a round offset must not move a lane out of its
            # wave
            self.assertEqual(threads % 64, 0, msg)
            self.assertEqual(v["MID8_SLOTS"] % 4, 0, f"{msg} partial scale group")

        # and none of it may move at TP4, whose build is the regression gate
        v4 = self.v[4]
        self.assertEqual(v4["MERGE_PASSES"], 1)
        self.assertEqual(v4["MID_COPY_ROUNDS"], 4)
        self.assertEqual(v4["MID8_ROUNDS"], 1)
        self.assertEqual(v4["MID8_SLOTS"], 480)
        # TP2 is where every one of them needed a second round
        v2 = self.v[2]
        self.assertEqual(v2["MERGE_PASSES"], 2)
        self.assertEqual(v2["MID_COPY_ROUNDS"], 8)
        self.assertEqual(v2["MID8_ROUNDS"], 2)

    def test_the_narrow_pool_holds_what_the_down_stage_stages(self):
        """The narrow MoE keeps every token's mids three ways at once -- bf16 as
        they land, then MXFP8 and its E8M0 scales -- so the pool must cover all
        three, not just the bf16 xn.

        This is the bug class the thread-map test covers, in a sizing function
        rather than a thread map: the three mids come to MOE_SLOTS * INTER *
        25/32, which is 3000 at TP4 and slips under HIDDEN / 2 = 3072 by 2.4%.
        Sizing the pool from HIDDEN therefore looked right at TP4 and put 2928
        words per token out of bounds at TP2, where it is 6000. Assert the
        requirement, not the number."""
        for tp, v in self.v.items():
            inter, hidden = v["INTER"], v["HIDDEN"]
            mids = v["MOE_SLOTS"] * inter
            need = mids // 2 + mids // 4 + mids // 32
            for s, wide in enumerate(v["wide"], start=1):
                if wide:
                    continue  # the wide path stages by expert, sized separately
                pool = v["pool"][s - 1]
                self.assertGreaterEqual(
                    pool,
                    s * max(hidden // 2, need),
                    f"TP{tp} narrow width {s}: pool {pool} words does not "
                    f"cover {s} x {need} mid words",
                )

    def test_narrow_widths_are_only_those_that_fit_lds(self):
        """``WIDE_FROM`` is TP-dependent because the narrow pool is the budget's
        swing term. Measured by COMPILE_ONLY against the 160 KB gfx950 limit:
        TP2 narrow needs 168,416 B at width 3 and 192,416 B at width 4, so TP2
        goes wide from 3; TP4's mids are half the size and fit through 4."""
        self.assertEqual(self.v[2]["WIDE_FROM"], 3)
        self.assertEqual(self.v[4]["WIDE_FROM"], 5)
        self.assertEqual(self.v[8]["WIDE_FROM"], 5)
        for tp, v in self.v.items():
            narrow = [s for s, wide in enumerate(v["wide"], start=1) if not wide]
            self.assertEqual(
                narrow, list(range(1, v["WIDE_FROM"])), f"TP{tp} narrow widths"
            )

    def test_tp8_merge_still_needs_a_masked_tail(self):
        """A known gap, asserted so it cannot be forgotten: at H = 8 the merge
        fills half a CTA, so its rounds do not come out whole and post_attn
        refuses to build. TP2 and TP4 do come out whole."""
        self.assertFalse(self.v[8]["MERGE_ROUNDS_WHOLE"], "TP8 merge")
        for tp in (2, 4):
            self.assertTrue(self.v[tp]["MERGE_ROUNDS_WHOLE"], f"TP{tp} merge")

    def test_one_cta_per_shared_expert_task(self):
        """``has_sh = start("shared") < SH_TASKS`` with ``start`` taken mod
        BLOCKS: past BLOCKS tasks the guard admits every CTA and the tasks
        beyond it are silently never run. INTER scales as 1 / TP, so the
        requirement is loosest at the smallest TP."""
        for tp, v in self.v.items():
            self.assertLessEqual(v["SH_TASKS"], v["BLOCKS"], f"TP{tp} shared")

    def test_the_projection_bound_covers_both_index_head_modes(self):
        """``MAX_QKV_ROWS`` sizes SCRATCH_X8, so it must cover the widest
        projection a build can ask for -- CP's every-index-head rows, since
        own mode's are never more. It used to be ``qkv_rows(TP)``, which is
        that number only at TP4."""
        for tp, v in self.v.items():
            self.assertGreaterEqual(v["MAX_QKV_ROWS"], v["own_rows"], f"TP{tp} own")
            self.assertGreaterEqual(v["MAX_QKV_ROWS"], v["cp_rows"], f"TP{tp} cp")
        # and the TP4 build is unmoved by having said it that way
        self.assertEqual(self.v[4]["MAX_QKV_ROWS"], 2944)


if __name__ == "__main__":
    unittest.main()
