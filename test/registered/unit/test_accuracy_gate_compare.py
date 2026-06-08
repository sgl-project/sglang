"""Unit tests for the offline DS-vs-DSA accuracy-gate compare (development/loop8/accuracy_gate.py).

`compare()` is the fail-closed gate that decides MMLU/NIAH DS-vs-DSA from two
per-side artifacts produced sequentially (one TP=8 server at a time, since GLM-5.1
can't run two TP=8 servers or TP=4). These tests pin: a clean pass; the MMLU
1.0 pp tolerance; the within-budget NIAH 5.0 pp tolerance; beyond-budget treated
as characterization-only; and fail-closed on incomparable artifacts (run_id /
prompt-set-hash / index_topk mismatch, missing/zero data).

    python -m pytest test/registered/unit/test_accuracy_gate_compare.py -v
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import unittest


def _load_gate():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "development", "loop8", "accuracy_gate.py")
    )
    spec = importlib.util.spec_from_file_location("_accuracy_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_accuracy_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


AG = _load_gate()


def _side(side, *, mmlu_hits, mmlu_total=200, niah_hits=(19, 18, 17),
          run_id="RID", index_topk=2048):
    lengths = (256, 512, 1024)
    return {
        "schema": AG.SCHEMA, "side": side, "run_id": run_id, "index_topk": index_topk,
        "mmlu": {"hits": mmlu_hits, "total": mmlu_total, "prompt_set_hash": "MMH"},
        "niah_within_budget": [
            {"length_words": L, "num_prompts": 20, "hits": h, "prompt_set_hash": f"NH{L}"}
            for L, h in zip(lengths, niah_hits)
        ],
        "niah_beyond_budget": [],
    }


class TestAccuracyGateCompare(unittest.TestCase):
    def test_clean_pass(self):
        dsa = _side("dsa", mmlu_hits=150, niah_hits=(19, 18, 17))
        ds = _side("ds", mmlu_hits=149, niah_hits=(19, 18, 16))  # within 1pp / 5pp
        v = AG.compare(dsa, ds)
        self.assertTrue(v["mandatory_pass"])
        self.assertTrue(v["mmlu"]["pass"])
        self.assertTrue(v["niah_within_budget"]["pass"])

    def test_mmlu_delta_fail(self):
        dsa = _side("dsa", mmlu_hits=150)          # 75.0%
        ds = _side("ds", mmlu_hits=144)            # 72.0% -> 3.0 pp > 1.0 pp
        v = AG.compare(dsa, ds)
        self.assertFalse(v["mmlu"]["pass"])
        self.assertFalse(v["mandatory_pass"])

    def test_niah_delta_fail(self):
        dsa = _side("dsa", mmlu_hits=150, niah_hits=(20, 20, 20))   # 100%
        ds = _side("ds", mmlu_hits=150, niah_hits=(20, 20, 17))     # 85% at L=1024 -> 15pp > 5pp
        v = AG.compare(dsa, ds)
        self.assertTrue(v["mmlu"]["pass"])
        self.assertFalse(v["niah_within_budget"]["pass"])
        self.assertFalse(v["mandatory_pass"])

    def test_run_id_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, run_id="RID-A")
        ds = _side("ds", mmlu_hits=150, run_id="RID-B")
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("run_id", str(cm.exception))

    def test_mmlu_prompt_hash_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        ds["mmlu"]["prompt_set_hash"] = "DIFFERENT"
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("MMLU prompt-set hash", str(cm.exception))

    def test_index_topk_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, index_topk=2048)
        ds = _side("ds", mmlu_hits=150, index_topk=4096)
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("index_topk", str(cm.exception))

    def test_niah_length_set_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        ds["niah_within_budget"].pop()  # drop a length
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("length set", str(cm.exception))

    def test_zero_mmlu_total_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=0, mmlu_total=0)
        ds = _side("ds", mmlu_hits=0, mmlu_total=0)
        with self.assertRaises(AG.GateError):
            AG.compare(dsa, ds)

    def test_beyond_budget_is_characterization_only(self):
        # A large beyond-budget recall gap must NOT fail the mandatory verdict.
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        dsa["niah_beyond_budget"] = [{"length_words": 16384, "num_prompts": 10, "hits": 9}]
        ds["niah_beyond_budget"] = [{"length_words": 16384, "num_prompts": 10, "hits": 1}]
        v = AG.compare(dsa, ds)
        self.assertTrue(v["mandatory_pass"])  # beyond-budget never gates
        self.assertEqual(len(v["niah_beyond_budget_characterization"]), 1)
        self.assertEqual(v["niah_beyond_budget_characterization"][0]["delta_pp"], 80.0)


if __name__ == "__main__":
    unittest.main()
