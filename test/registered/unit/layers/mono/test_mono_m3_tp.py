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


if __name__ == "__main__":
    unittest.main()
