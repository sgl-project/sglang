"""
Unit tests for python/sglang/srt/logit_slicing/.

All tests run on CPU with mocked tokenizers — no GPU or live server required.

Run:
    pytest test/srt/test_logit_slicing.py -v

These tests stub the sglang import chain so they work with any Python >= 3.9
that has torch installed, without needing a full sglang GPU installation.
"""

import importlib.util
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import torch

# ── Bootstrap: load our modules directly by file path ─────────────────────────
#
# sglang requires Python 3.10+ syntax but tests must run on Python 3.9 with
# only torch available.  We bypass the sglang package registry entirely by
# loading each of our three source files with importlib.util, registering them
# under their full dotted names in sys.modules.  processor.py has a try/except
# fallback for CustomLogitProcessor, so the heavy sglang chain is never hit.

_SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "python" / "sglang" / "srt" / "logit_slicing"


def _load(dotted_name: str, filename: str):
    """Load a source file and register it in sys.modules under dotted_name."""
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = dotted_name.rsplit(".", 1)[0]
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_schema_mod    = _load("sglang.srt.logit_slicing.schema",       "schema.py")
_anchor_mod    = _load("sglang.srt.logit_slicing.vocab_anchor", "vocab_anchor.py")
_proc_mod      = _load("sglang.srt.logit_slicing.processor",    "processor.py")

SimultaneousMultiIntentEntityLogitProcessor = _proc_mod.SimultaneousMultiIntentEntityLogitProcessor
IntentSchema = _schema_mod.IntentSchema
NERSchema    = _schema_mod.NERSchema
SlotSchema   = _schema_mod.SlotSchema

STRATEGY_FIRST_TOKEN  = _anchor_mod.STRATEGY_FIRST_TOKEN
STRATEGY_SINGLE_TOKEN = _anchor_mod.STRATEGY_SINGLE_TOKEN
STRATEGY_EXPLICIT     = _anchor_mod.STRATEGY_EXPLICIT
VocabAnchor           = _anchor_mod.VocabAnchor
build_anchor_config   = _anchor_mod.build_anchor_config

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 200


def _fake_tokenizer(label_map: Dict[str, List[int]], vocab_size: int = VOCAB_SIZE):
    """Return a mock tokenizer that encodes labels using a fixed map."""
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = 2

    def encode(text, add_special_tokens=False):
        return label_map.get(text, [0])

    tok.encode = encode
    return tok


def _make_schema() -> NERSchema:
    return NERSchema(
        intents=IntentSchema(labels=["book_flight", "cancel", "check_status"]),
        slots=[
            SlotSchema(name="departure_city", labels=["O", "B", "I"]),
            SlotSchema(name="date", labels=["O", "B"]),
        ],
    )


def _make_req():
    """Minimal stand-in for sglang Req — just needs customized_info attribute."""
    return SimpleNamespace(customized_info=None)


# ─────────────────────────────────────────────────────────────────────────────
# IntentSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentSchema(unittest.TestCase):
    def test_requires_at_least_two_labels(self):
        with self.assertRaises(ValueError):
            IntentSchema(labels=["only_one"])

    def test_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            IntentSchema(labels=["a", "a", "b"])

    def test_token_ids_before_anchoring_raises(self):
        schema = IntentSchema(labels=["a", "b"])
        with self.assertRaises(RuntimeError):
            schema.token_ids()

    def test_token_ids_after_anchoring(self):
        schema = IntentSchema(labels=["a", "b"], label_to_token_id={"a": 10, "b": 20})
        self.assertEqual(schema.token_ids(), [10, 20])

    def test_is_anchored(self):
        schema = IntentSchema(labels=["a", "b"])
        self.assertFalse(schema.is_anchored)
        schema.label_to_token_id = {"a": 10, "b": 20}
        self.assertTrue(schema.is_anchored)


# ─────────────────────────────────────────────────────────────────────────────
# SlotSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotSchema(unittest.TestCase):
    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            SlotSchema(name="", labels=["O", "B"])

    def test_requires_at_least_two_labels(self):
        with self.assertRaises(ValueError):
            SlotSchema(name="city", labels=["O"])

    def test_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            SlotSchema(name="city", labels=["O", "O"])


# ─────────────────────────────────────────────────────────────────────────────
# NERSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestNERSchema(unittest.TestCase):
    def test_rejects_duplicate_slot_names(self):
        with self.assertRaises(ValueError):
            NERSchema(
                intents=IntentSchema(labels=["a", "b"]),
                slots=[
                    SlotSchema(name="city", labels=["O", "B"]),
                    SlotSchema(name="city", labels=["O", "B"]),
                ],
            )

    def test_to_anchor_config_before_anchoring_raises(self):
        schema = _make_schema()
        with self.assertRaises(RuntimeError):
            schema.to_anchor_config()

    def test_to_anchor_config_round_trips_via_from_dict(self):
        schema = _make_schema()
        schema.intents.label_to_token_id = {"book_flight": 10, "cancel": 20, "check_status": 30}
        schema.slots[0].label_to_token_id = {"O": 40, "B": 50, "I": 60}
        schema.slots[1].label_to_token_id = {"O": 70, "B": 80}

        config = schema.to_anchor_config()
        restored = NERSchema.from_dict(config)

        self.assertEqual(restored.intents.labels, ["book_flight", "cancel", "check_status"])
        self.assertEqual(restored.intents.token_ids(), [10, 20, 30])
        self.assertEqual(restored.slots[0].name, "departure_city")
        self.assertEqual(restored.slots[0].token_ids(), [40, 50, 60])


# ─────────────────────────────────────────────────────────────────────────────
# VocabAnchor
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabAnchor(unittest.TestCase):
    def _make_tok_with_unique_ids(self):
        label_map = {
            "book_flight": [10],
            "cancel": [20],
            "check_status": [30],
            "O": [40],
            "B": [50],
            "I": [60],
        }
        return _fake_tokenizer(label_map)

    def test_first_token_strategy(self):
        schema = _make_schema()
        tok = self._make_tok_with_unique_ids()
        anchored = VocabAnchor().anchor(schema, tok, strategy=STRATEGY_FIRST_TOKEN)
        self.assertTrue(anchored.is_anchored)
        self.assertEqual(anchored.intents.label_to_token_id["book_flight"], 10)
        self.assertEqual(anchored.slots[0].label_to_token_id["B"], 50)

    def test_single_token_strategy_raises_on_multi_token(self):
        schema = _make_schema()
        # Make "book_flight" return two tokens
        label_map = {
            "book_flight": [10, 11],
            "cancel": [20],
            "check_status": [30],
            "O": [40],
            "B": [50],
            "I": [60],
        }
        tok = _fake_tokenizer(label_map)
        with self.assertRaises(ValueError, msg="single_token strategy should reject multi-token label"):
            VocabAnchor().anchor(schema, tok, strategy=STRATEGY_SINGLE_TOKEN)

    def test_explicit_strategy(self):
        schema = _make_schema()
        tok = _fake_tokenizer({})  # encode never called
        intent_override = {"book_flight": 10, "cancel": 20, "check_status": 30}
        slot_overrides = {
            "departure_city": {"O": 40, "B": 50, "I": 60},
            "date": {"O": 70, "B": 80},
        }
        anchored = VocabAnchor().anchor(
            schema, tok,
            strategy=STRATEGY_EXPLICIT,
            intent_override=intent_override,
            slot_overrides=slot_overrides,
        )
        self.assertEqual(anchored.intents.label_to_token_id["cancel"], 20)

    def test_explicit_strategy_missing_override_raises(self):
        schema = _make_schema()
        tok = _fake_tokenizer({})
        with self.assertRaises(ValueError):
            VocabAnchor().anchor(schema, tok, strategy=STRATEGY_EXPLICIT)

    def test_collision_detection_raises(self):
        schema = NERSchema(
            intents=IntentSchema(labels=["a", "b"]),
            slots=[],
        )
        # Both labels map to the same token
        label_map = {"a": [10], "b": [10]}
        tok = _fake_tokenizer(label_map)
        with self.assertRaises(ValueError, msg="Should raise on token ID collision"):
            VocabAnchor().anchor(schema, tok, strategy=STRATEGY_FIRST_TOKEN)

    def test_out_of_vocab_range_raises(self):
        schema = NERSchema(intents=IntentSchema(labels=["a", "b"]), slots=[])
        label_map = {"a": [10], "b": [9999]}  # 9999 > VOCAB_SIZE=200
        tok = _fake_tokenizer(label_map, vocab_size=100)
        with self.assertRaises(ValueError):
            VocabAnchor().anchor(schema, tok)

    def test_build_anchor_config_attaches_eos(self):
        schema = _make_schema()
        tok = self._make_tok_with_unique_ids()
        tok.eos_token_id = 2
        config = build_anchor_config(schema, tok)
        self.assertIn("schema", config)
        self.assertIn("eos_token_id", config)
        self.assertEqual(config["eos_token_id"], 2)

    def test_build_anchor_config_explicit_eos_overrides(self):
        schema = _make_schema()
        tok = self._make_tok_with_unique_ids()
        tok.eos_token_id = 2
        config = build_anchor_config(schema, tok, eos_token_id=99)
        self.assertEqual(config["eos_token_id"], 99)


# ─────────────────────────────────────────────────────────────────────────────
# SimultaneousMultiIntentEntityLogitProcessor
# ─────────────────────────────────────────────────────────────────────────────

class TestSMIELP(unittest.TestCase):
    """Tests for the logit processor on CPU with synthetic logit tensors."""

    def _make_schema_config(self):
        """Return a fully-anchored schema config dict with known token IDs."""
        return {
            "intent_labels": ["book_flight", "cancel", "check_status"],
            "intent_token_ids": [10, 20, 30],
            "entity_slots": [
                {
                    "name": "departure_city",
                    "bio_labels": ["O", "B", "I"],
                    "bio_token_ids": [40, 50, 60],
                },
                {
                    "name": "date",
                    "bio_labels": ["O", "B"],
                    "bio_token_ids": [70, 80],
                },
            ],
        }

    def _make_logits(self, batch_size: int, intent_winner: int, slot_winners: List[int]) -> torch.Tensor:
        """
        Build a [batch_size, VOCAB_SIZE] logit tensor where the 'winner' token IDs
        have logit=10.0 and all others have logit=0.0.
        """
        logits = torch.zeros(batch_size, VOCAB_SIZE)
        intent_token_ids = [10, 20, 30]
        slot_token_ids_list = [[40, 50, 60], [70, 80]]

        for b in range(batch_size):
            logits[b, intent_token_ids[intent_winner]] = 10.0
            for slot_idx, winner in enumerate(slot_winners):
                slot_tids = slot_token_ids_list[slot_idx]
                logits[b, slot_tids[winner]] = 10.0

        return logits

    def _run_processor(self, logits, schema_config, eos_id=2):
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        batch_size = logits.shape[0]
        reqs = [_make_req() for _ in range(batch_size)]
        custom_param_list = [
            {"__req__": reqs[i], "schema": schema_config, "eos_token_id": eos_id}
            for i in range(batch_size)
        ]
        out_logits = proc(logits.clone(), custom_param_list)
        return out_logits, reqs, custom_param_list

    # ── Intent classification ────────────────────────────────────────────────

    def test_intent_label_selected_correctly(self):
        # book_flight (index 0) should win
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        result = reqs[0].customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "book_flight")

    def test_cancel_intent_selected(self):
        logits = self._make_logits(1, intent_winner=1, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        result = reqs[0].customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "cancel")

    def test_intent_confidence_is_highest_for_winner(self):
        logits = self._make_logits(1, intent_winner=2, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        dist = reqs[0].customized_info["smielp"][0]["intent"]["distribution"]
        winner_prob = dist["check_status"]
        for label, prob in dist.items():
            if label != "check_status":
                self.assertGreater(winner_prob, prob)

    # ── Entity slot classification ───────────────────────────────────────────

    def test_slot_tag_selected_correctly_B(self):
        # departure_city: B wins (index 1 in [O, B, I] → token_id=50)
        logits = self._make_logits(1, intent_winner=0, slot_winners=[1, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        entities = reqs[0].customized_info["smielp"][0]["entities"]
        dep_city = next(e for e in entities if e["slot"] == "departure_city")
        self.assertEqual(dep_city["tag"], "B")

    def test_slot_tag_selected_correctly_O(self):
        # date: O wins (index 0 in [O, B] → token_id=70)
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        entities = reqs[0].customized_info["smielp"][0]["entities"]
        date_slot = next(e for e in entities if e["slot"] == "date")
        self.assertEqual(date_slot["tag"], "O")

    def test_all_slots_present_in_result(self):
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        entities = reqs[0].customized_info["smielp"][0]["entities"]
        slot_names = {e["slot"] for e in entities}
        self.assertEqual(slot_names, {"departure_city", "date"})

    # ── EOS forcing ──────────────────────────────────────────────────────────

    def test_eos_token_is_zero_after_processing(self):
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        out, _, _ = self._run_processor(logits, self._make_schema_config(), eos_id=2)
        self.assertAlmostEqual(float(out[0, 2]), 0.0)

    def test_all_non_eos_tokens_are_neg_inf(self):
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        out, _, _ = self._run_processor(logits, self._make_schema_config(), eos_id=2)
        for tok_id in range(VOCAB_SIZE):
            if tok_id == 2:
                continue
            self.assertTrue(
                math.isinf(float(out[0, tok_id])) and float(out[0, tok_id]) < 0,
                msg=f"Token {tok_id} expected -inf, got {float(out[0, tok_id])}",
            )

    # ── Batch processing ─────────────────────────────────────────────────────

    def test_batch_of_two_requests(self):
        # Request 0: intent=book_flight, Request 1: intent=cancel
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, 10] = 10.0   # book_flight
        logits[1, 20] = 10.0   # cancel

        proc = SimultaneousMultiIntentEntityLogitProcessor()
        schema = self._make_schema_config()
        reqs = [_make_req(), _make_req()]
        cpl = [
            {"__req__": reqs[0], "schema": schema, "eos_token_id": 2},
            {"__req__": reqs[1], "schema": schema, "eos_token_id": 2},
        ]
        proc(logits, cpl)

        self.assertEqual(reqs[0].customized_info["smielp"][0]["intent"]["label"], "book_flight")
        self.assertEqual(reqs[1].customized_info["smielp"][0]["intent"]["label"], "cancel")

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_empty_custom_param_list_returns_logits_unchanged(self):
        logits = torch.ones(1, VOCAB_SIZE)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        out = proc(logits.clone(), [])
        self.assertTrue(torch.equal(out, logits))

    def test_missing_schema_key_returns_empty_result(self):
        logits = torch.ones(1, VOCAB_SIZE)
        req = _make_req()
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        proc(logits, [{"__req__": req, "eos_token_id": 2}])  # no "schema" key
        result = req.customized_info["smielp"][0]
        self.assertIsNone(result["intent"])
        self.assertEqual(result["entities"], [])

    def test_missing_req_does_not_crash(self):
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        # No "__req__" key — processor should log warning and not raise
        proc(logits, [{"schema": self._make_schema_config(), "eos_token_id": 2}])

    def test_none_params_entry_does_not_crash(self):
        """None in custom_param_list must not crash; EOS is forced with default id=2."""
        logits = torch.zeros(1, VOCAB_SIZE)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        out = proc(logits, [None])
        self.assertEqual(float(out[0, 2]), 0.0)
        self.assertEqual(float(out[0, 0]), float("-inf"))

    def test_distribution_sums_to_one(self):
        logits = self._make_logits(1, intent_winner=0, slot_winners=[0, 0])
        _, reqs, _ = self._run_processor(logits, self._make_schema_config())
        result = reqs[0].customized_info["smielp"][0]
        intent_sum = sum(result["intent"]["distribution"].values())
        self.assertAlmostEqual(intent_sum, 1.0, places=5)
        for ent in result["entities"]:
            slot_sum = sum(ent["distribution"].values())
            self.assertAlmostEqual(slot_sum, 1.0, places=5)

    def test_output_key_constant(self):
        self.assertEqual(SimultaneousMultiIntentEntityLogitProcessor.OUTPUT_KEY, "smielp")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end schema → config → processor round-trip (CPU, no server)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndRoundTrip(unittest.TestCase):
    """Validates the full schema → anchor → config → processor pipeline."""

    def test_full_pipeline_no_server(self):
        label_map = {
            "book_flight": [10],
            "cancel":      [20],
            "O":           [40],
            "B":           [50],
            "I":           [60],
        }
        tok = _fake_tokenizer(label_map, vocab_size=200)
        tok.eos_token_id = 2

        schema = NERSchema(
            intents=IntentSchema(labels=["book_flight", "cancel"]),
            slots=[SlotSchema(name="city", labels=["O", "B", "I"])],
        )

        custom_params = build_anchor_config(schema, tok)
        self.assertIn("schema", custom_params)
        self.assertEqual(custom_params["eos_token_id"], 2)

        # Simulate logits where "book_flight" (token 10) and "B" (token 50) win
        logits = torch.zeros(1, 200)
        logits[0, 10] = 10.0  # book_flight intent
        logits[0, 50] = 10.0  # B tag for city slot

        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, **custom_params}])

        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "book_flight")
        city = next(e for e in result["entities"] if e["slot"] == "city")
        self.assertEqual(city["tag"], "B")


if __name__ == "__main__":
    unittest.main()
