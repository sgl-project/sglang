"""Unit tests for the offline DS-vs-DSA accuracy-gate compare (development/loop8/accuracy_gate.py).

`compare()` is the fail-closed gate deciding MMLU/NIAH DS-vs-DSA from two per-side
artifacts produced sequentially (one TP=8 server at a time; GLM-5.1 can't run two
TP=8 or TP=4). These tests pin: a clean pass; the MMLU 1.0 pp / within-budget NIAH
5.0 pp tolerances; beyond-budget as characterization-only; and — critically — that
the gate FAILS CLOSED when requests didn't actually succeed (all-failed, partial-
served, usage missing, over-budget) or the server op-point / prompt set differs.

    python -m pytest test/registered/unit/test_accuracy_gate_compare.py -v
"""

from __future__ import annotations

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
WITHIN = (1024, 1536)
BEYOND = (4096, 16384, 65536)


def _server_info(side, *, tp_size=8, random_seed=20260607):
    return {
        "model_path": "/glm", "tp_size": tp_size, "page_size": 64,
        "kv_cache_dtype": "fp8_e4m3", "disable_radix_cache": True,
        "dsa_prefill_backend": "flashmla_kv", "dsa_decode_backend": "flashmla_kv",
        "attention_backend": "dsa", "random_seed": random_seed,
        "enable_double_sparsity": (side == "ds"),
        "mem_fraction_static": 0.7 if side == "ds" else 0.8,  # allowed diff
    }


def _side(side, *, mmlu_hits, mmlu_total=200, niah_hits=(19, 18), run_id="RID",
          index_topk=2048, mmlu_served=None, mmlu_err=None, within_served=None,
          within_err=None, usage_missing=False, max_prompt_tokens=1800,
          server_info=None):
    mmlu_served = mmlu_total if mmlu_served is None else mmlu_served
    within_served = 20 if within_served is None else within_served
    return {
        "schema": AG.SCHEMA, "side": side, "run_id": run_id, "index_topk": index_topk,
        "server_info": server_info or _server_info(side),
        "mmlu": {"hits": mmlu_hits, "total": mmlu_total, "served": mmlu_served,
                 "first_error": mmlu_err, "prompt_set_hash": "MMH"},
        "niah_within_budget": [
            {"length_words": L, "num_prompts": 20, "served": within_served, "hits": h,
             "max_prompt_tokens": max_prompt_tokens, "usage_missing": usage_missing,
             "first_error": within_err, "prompt_set_hash": f"NH{L}"}
            for L, h in zip(WITHIN, niah_hits)
        ],
        "niah_beyond_budget": [
            {"length_words": L, "num_prompts": 10, "served": 10, "hits": 5,
             "prompt_set_hash": f"BH{L}"} for L in BEYOND
        ],
    }


class TestAccuracyGateCompare(unittest.TestCase):
    def test_clean_pass(self):
        v = AG.compare(_side("dsa", mmlu_hits=150, niah_hits=(19, 18)),
                       _side("ds", mmlu_hits=149, niah_hits=(19, 17)))
        self.assertTrue(v["mandatory_pass"])

    def test_mmlu_delta_fail(self):
        v = AG.compare(_side("dsa", mmlu_hits=150), _side("ds", mmlu_hits=144))  # 3pp
        self.assertFalse(v["mmlu"]["pass"])
        self.assertFalse(v["mandatory_pass"])

    def test_niah_delta_fail(self):
        v = AG.compare(_side("dsa", mmlu_hits=150, niah_hits=(20, 20)),
                       _side("ds", mmlu_hits=150, niah_hits=(20, 17)))  # 15pp at L=1536
        self.assertFalse(v["niah_within_budget"]["pass"])
        self.assertFalse(v["mandatory_pass"])

    # ---- fail-closed: requests didn't actually succeed ----
    def test_all_failed_mmlu_fails_closed(self):
        # The R6 bug: 0/200 both sides -> delta 0 -> false pass. Must now raise.
        dsa = _side("dsa", mmlu_hits=0, mmlu_served=0, mmlu_err="HTTP 500")
        ds = _side("ds", mmlu_hits=0, mmlu_served=0, mmlu_err="HTTP 500")
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("MMLU", str(cm.exception))

    def test_all_failed_niah_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, within_served=0, within_err="HTTP 500", niah_hits=(0, 0))
        ds = _side("ds", mmlu_hits=150, within_served=0, within_err="HTTP 500", niah_hits=(0, 0))
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("NIAH", str(cm.exception))

    def test_partial_served_niah_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150, within_served=12)  # 12 != 20
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("served", str(cm.exception))

    def test_niah_usage_missing_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150, usage_missing=True)
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("usage", str(cm.exception))

    def test_niah_over_budget_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, max_prompt_tokens=4096)  # > index_topk 2048
        ds = _side("ds", mmlu_hits=150, max_prompt_tokens=4096)
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("within budget", str(cm.exception))

    # ---- fail-closed: incomparable artifacts ----
    def test_run_id_mismatch_fails_closed(self):
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(_side("dsa", mmlu_hits=150, run_id="A"),
                       _side("ds", mmlu_hits=150, run_id="B"))
        self.assertIn("run_id", str(cm.exception))

    def test_op_point_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, server_info=_server_info("dsa", tp_size=8))
        ds = _side("ds", mmlu_hits=150, server_info=_server_info("ds", tp_size=4))  # TP differs
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("op-point", str(cm.exception))

    def test_seed_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150, server_info=_server_info("dsa", random_seed=111))
        ds = _side("ds", mmlu_hits=150, server_info=_server_info("ds", random_seed=222))
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("op-point", str(cm.exception))

    def test_mmlu_prompt_hash_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        ds["mmlu"]["prompt_set_hash"] = "DIFF"
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("MMLU prompt-set hash", str(cm.exception))

    def test_index_topk_mismatch_fails_closed(self):
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(_side("dsa", mmlu_hits=150, index_topk=2048),
                       _side("ds", mmlu_hits=150, index_topk=4096))
        self.assertIn("index_topk", str(cm.exception))

    def test_niah_length_set_mismatch_fails_closed(self):
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        ds["niah_within_budget"].pop()
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(dsa, ds)
        self.assertIn("length set", str(cm.exception))

    def test_zero_mmlu_total_fails_closed(self):
        with self.assertRaises(AG.GateError):
            AG.compare(_side("dsa", mmlu_hits=0, mmlu_total=0, mmlu_served=0),
                       _side("ds", mmlu_hits=0, mmlu_total=0, mmlu_served=0))

    def test_beyond_budget_required_and_characterization_only(self):
        # A large beyond-budget gap must NOT fail the mandatory verdict...
        dsa = _side("dsa", mmlu_hits=150)
        ds = _side("ds", mmlu_hits=150)
        dsa["niah_beyond_budget"][0]["hits"] = 9
        ds["niah_beyond_budget"][0]["hits"] = 1
        v = AG.compare(dsa, ds)
        self.assertTrue(v["mandatory_pass"])
        self.assertEqual(len(v["niah_beyond_budget_characterization"]), len(BEYOND))
        # ...but a MISSING beyond-budget block fails closed (record must not omit it).
        ds2 = _side("ds", mmlu_hits=150)
        ds2["niah_beyond_budget"] = []
        with self.assertRaises(AG.GateError) as cm:
            AG.compare(_side("dsa", mmlu_hits=150), ds2)
        self.assertIn("beyond-budget", str(cm.exception))

    def test_default_mmlu_data_dir_resolves_repo_path(self):
        # Codex #4: empty AC12_MMLU_DATA_DIR must not reach os.makedirs("").
        prev = os.environ.pop("AC12_MMLU_DATA_DIR", None)
        try:
            d = AG._default_mmlu_data_dir()
            self.assertTrue(d.endswith(os.path.join("benchmark", "mmlu", "data")))
            self.assertTrue(os.path.isabs(d))
        finally:
            if prev is not None:
                os.environ["AC12_MMLU_DATA_DIR"] = prev


if __name__ == "__main__":
    unittest.main()
