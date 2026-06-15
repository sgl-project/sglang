"""
Phase 4 — Part A: Integration test for SMIELP with real Qwen2.5-0.5B-Instruct vocabulary.

Validates the full schema → anchor → processor pipeline using the actual Qwen2.5 token IDs
without requiring a live sglang server, GPU, or the transformers library.

Run:
    /Users/aliiii/.venv/bin/python -m unittest test.srt.test_logit_slicing_integration -v

Requirements: torch (no transformers, no sglang server)
"""

import importlib.util
import json
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "python" / "sglang" / "srt" / "logit_slicing"

def _load(dotted_name, filename):
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = dotted_name.rsplit(".", 1)[0]
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load in dependency order: schema first (no internal deps), then anchor, then processor.
_schema_mod = _load("sglang.srt.logit_slicing.schema",       "schema.py")
_anchor_mod = _load("sglang.srt.logit_slicing.vocab_anchor", "vocab_anchor.py")
_proc_mod   = _load("sglang.srt.logit_slicing.processor",    "processor.py")

SimultaneousMultiIntentEntityLogitProcessor = _proc_mod.SimultaneousMultiIntentEntityLogitProcessor
IntentSchema    = _schema_mod.IntentSchema
NERSchema       = _schema_mod.NERSchema
SlotSchema      = _schema_mod.SlotSchema
VocabAnchor     = _anchor_mod.VocabAnchor
build_anchor_config = _anchor_mod.build_anchor_config
STRATEGY_EXPLICIT   = _anchor_mod.STRATEGY_EXPLICIT
STRATEGY_FIRST_TOKEN = _anchor_mod.STRATEGY_FIRST_TOKEN
STRATEGY_SINGLE_TOKEN = _anchor_mod.STRATEGY_SINGLE_TOKEN
# ─────────────────────────────────────────────────────────────────────────────


# ── Real Qwen2.5-0.5B-Instruct vocabulary IDs (single-token labels) ───────────
#
# Derived by grepping the tokenizer.json vocab dict at:
#   ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/.../tokenizer.json
#
# All labels below tokenize to a SINGLE token in Qwen2.5.  All IDs within each
# classification head are unique (no collisions).
#
QWEN_BASE_VOCAB  = 151643   # base vocab (before added special tokens)
QWEN_VOCAB_SIZE  = 151936   # model config vocab_size — padded for GPU alignment; logit tensor dim
QWEN_EOS_ID      = 151643   # <|endoftext|>  (within the padded range)
QWEN_IM_END_ID   = 151645   # <|im_end|>  (typical Qwen generation terminator)

# Intent head token IDs
INTENT_LABELS   = ["book", "cancel", "status"]
INTENT_TOKEN_IDS = {
    "book":   2190,
    "cancel": 18515,
    "status": 2829,
}

# BIO slot token IDs (shared across slots — OK: each slot is an independent head)
BIO_LABELS   = ["O", "B", "I"]
BIO_TOKEN_IDS = {"O": 46, "B": 33, "I": 40}

# Presence slot (binary detection) — verifies multi-slot / multi-label support
PRESENCE_LABELS   = ["none", "present"]
PRESENCE_TOKEN_IDS = {"none": 6697, "present": 28744}


class MinimalQwenVocabTokenizer:
    """
    Minimal tokenizer wrapper backed by the Qwen2.5 tokenizer.json vocab dict.

    Only handles labels that exist as single tokens in the vocabulary.
    Sufficient for our integration tests (all chosen labels are single-token).
    Raises ValueError on any label that requires multi-token encoding.
    """

    _TOKENIZER_JSON = (
        pathlib.Path.home()
        / ".cache/huggingface/hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / "7ae557604adf67be50417f59c2c2f167def9a775"
        / "tokenizer.json"
    )

    def __init__(self):
        if not self._TOKENIZER_JSON.exists():
            raise FileNotFoundError(
                f"Qwen2.5 tokenizer.json not found at {self._TOKENIZER_JSON}.\n"
                "Run: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct"
            )
        with open(self._TOKENIZER_JSON) as f:
            data = json.load(f)
        self._vocab: dict = data["model"]["vocab"]
        # VocabAnchor uses vocab_size only for range-checking label token IDs.
        # All our labels are in the base vocab (< 151643), so base vocab is correct here.
        self.vocab_size = QWEN_BASE_VOCAB
        self.eos_token_id = QWEN_EOS_ID

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        # Multi-token labels are not supported in this minimal wrapper.
        # Use strategy='explicit' or rename the label to a single-token string.
        raise ValueError(
            f"MinimalQwenVocabTokenizer: '{text}' is not a single-token label "
            "in the Qwen2.5 vocab. Use strategy='explicit' to supply token IDs directly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_req():
    return SimpleNamespace(customized_info=None)


def _make_anchored_schema():
    """Return a NERSchema pre-anchored with real Qwen2.5 token IDs."""
    schema = NERSchema(
        intents=IntentSchema(
            labels=INTENT_LABELS,
            label_to_token_id=dict(INTENT_TOKEN_IDS),
        ),
        slots=[
            SlotSchema(
                name="city",
                labels=BIO_LABELS,
                label_to_token_id=dict(BIO_TOKEN_IDS),
            ),
            SlotSchema(
                name="date",
                labels=BIO_LABELS,
                label_to_token_id=dict(BIO_TOKEN_IDS),
            ),
            SlotSchema(
                name="hotel",
                labels=PRESENCE_LABELS,
                label_to_token_id=dict(PRESENCE_TOKEN_IDS),
            ),
        ],
    )
    assert schema.is_anchored
    return schema


def _logits_with_winner(vocab_size: int, batch_size: int, hot_ids: list[int]) -> torch.Tensor:
    """
    Build [batch_size, vocab_size] tensor where the specified token IDs have
    logit=20.0 (strong winner) and all others are 0.0.
    """
    logits = torch.zeros(batch_size, vocab_size)
    for tid in hot_ids:
        logits[:, tid] = 20.0
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRealVocabAnchoring(unittest.TestCase):
    """Validates VocabAnchor with the real Qwen2.5-0.5B tokenizer."""

    def _get_tokenizer(self):
        try:
            return MinimalQwenVocabTokenizer()
        except FileNotFoundError as e:
            self.skipTest(str(e))

    def test_first_token_strategy_single_token_labels(self):
        tok = self._get_tokenizer()
        schema = NERSchema(
            intents=IntentSchema(labels=INTENT_LABELS),
            slots=[SlotSchema(name="city", labels=BIO_LABELS)],
        )
        anchored = VocabAnchor().anchor(schema, tok, strategy=STRATEGY_FIRST_TOKEN)
        self.assertTrue(anchored.is_anchored)
        self.assertEqual(anchored.intents.label_to_token_id["book"],   2190)
        self.assertEqual(anchored.intents.label_to_token_id["cancel"], 18515)
        self.assertEqual(anchored.intents.label_to_token_id["status"], 2829)
        self.assertEqual(anchored.slots[0].label_to_token_id["O"], 46)
        self.assertEqual(anchored.slots[0].label_to_token_id["B"], 33)
        self.assertEqual(anchored.slots[0].label_to_token_id["I"], 40)

    def test_single_token_strategy_passes_for_single_token_labels(self):
        tok = self._get_tokenizer()
        schema = NERSchema(
            intents=IntentSchema(labels=INTENT_LABELS),
            slots=[SlotSchema(name="city", labels=BIO_LABELS)],
        )
        anchored = VocabAnchor().anchor(schema, tok, strategy=STRATEGY_SINGLE_TOKEN)
        self.assertTrue(anchored.is_anchored)

    def test_multi_token_label_raises_in_single_token_strategy(self):
        tok = self._get_tokenizer()
        # "book_flight" is not a single token — MinimalQwenVocabTokenizer raises ValueError
        schema = NERSchema(
            intents=IntentSchema(labels=["book_flight", "cancel"]),
            slots=[],
        )
        with self.assertRaises(ValueError):
            VocabAnchor().anchor(schema, tok, strategy=STRATEGY_SINGLE_TOKEN)

    def test_explicit_strategy_with_real_ids(self):
        tok = self._get_tokenizer()
        schema = NERSchema(
            intents=IntentSchema(labels=INTENT_LABELS),
            slots=[SlotSchema(name="city", labels=BIO_LABELS)],
        )
        anchored = VocabAnchor().anchor(
            schema, tok,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={"city": dict(BIO_TOKEN_IDS)},
        )
        self.assertEqual(anchored.intents.token_ids(), [2190, 18515, 2829])

    def test_vocab_range_is_real_qwen_size(self):
        tok = self._get_tokenizer()
        # tokenizer.vocab_size is the base vocab (151643); model config pads to 151936.
        # VocabAnchor range-checks use tokenizer.vocab_size, which is correct for label IDs.
        self.assertEqual(tok.vocab_size, QWEN_BASE_VOCAB)
        schema = NERSchema(
            intents=IntentSchema(labels=INTENT_LABELS),
            slots=[SlotSchema(name="city", labels=BIO_LABELS)],
        )
        # Anchoring with valid IDs must succeed against real vocab_size
        anchored = VocabAnchor().anchor(
            schema, tok,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={"city": dict(BIO_TOKEN_IDS)},
        )
        self.assertTrue(all(0 <= tid < QWEN_VOCAB_SIZE for tid in anchored.intents.token_ids()))

    def test_out_of_real_vocab_range_raises(self):
        tok = self._get_tokenizer()
        schema = NERSchema(
            intents=IntentSchema(labels=["a", "b"]),
            slots=[],
        )
        # tok.vocab_size = QWEN_BASE_VOCAB (151643); IDs at or above that are out of range.
        bad_ids = {"a": QWEN_BASE_VOCAB, "b": QWEN_BASE_VOCAB + 1}
        with self.assertRaises(ValueError, msg="IDs beyond vocab_size must be rejected"):
            VocabAnchor().anchor(schema, tok, strategy=STRATEGY_EXPLICIT, intent_override=bad_ids)

    def test_build_anchor_config_uses_real_eos(self):
        tok = self._get_tokenizer()
        schema = NERSchema(
            intents=IntentSchema(labels=INTENT_LABELS),
            slots=[SlotSchema(name="city", labels=BIO_LABELS)],
        )
        config = build_anchor_config(
            schema, tok,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={"city": dict(BIO_TOKEN_IDS)},
        )
        self.assertEqual(config["eos_token_id"], QWEN_EOS_ID)
        self.assertIn("schema", config)


class TestRealVocabClassification(unittest.TestCase):
    """Tests SMIELP with vocab_size=151643 (real Qwen2.5 scale)."""

    def _schema_config(self):
        return _make_anchored_schema().to_anchor_config()

    def test_logit_slicing_at_real_vocab_scale(self):
        # Allocate a [1, 151643] tensor — same shape as real Qwen2.5 decode output.
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, INTENT_TOKEN_IDS["cancel"]] = 20.0   # cancel wins
        logits[0, BIO_TOKEN_IDS["B"]] = 20.0           # B tag wins for city

        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_IM_END_ID}])

        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "cancel")
        city = next(e for e in result["entities"] if e["slot"] == "city")
        self.assertEqual(city["tag"], "B")

    def test_batch_of_three_distinct_intents(self):
        """Three requests in a batch, each with a different dominant intent."""
        logits = torch.zeros(3, QWEN_VOCAB_SIZE)
        for i, label in enumerate(["book", "cancel", "status"]):
            logits[i, INTENT_TOKEN_IDS[label]] = 20.0
            logits[i, BIO_TOKEN_IDS["O"]] = 20.0  # all slots = O

        proc = SimultaneousMultiIntentEntityLogitProcessor()
        reqs = [_make_req() for _ in range(3)]
        schema_cfg = self._schema_config()
        cpl = [
            {"__req__": reqs[i], "schema": schema_cfg, "eos_token_id": QWEN_IM_END_ID}
            for i in range(3)
        ]
        proc(logits, cpl)

        for i, expected_intent in enumerate(["book", "cancel", "status"]):
            result = reqs[i].customized_info["smielp"][0]
            self.assertEqual(
                result["intent"]["label"],
                expected_intent,
                msg=f"Request {i}: expected intent '{expected_intent}'",
            )

    def test_eos_forced_to_im_end(self):
        logits = _logits_with_winner(QWEN_VOCAB_SIZE, 1, [INTENT_TOKEN_IDS["book"]])
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        out = proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_IM_END_ID}])

        self.assertAlmostEqual(float(out[0, QWEN_IM_END_ID]), 0.0)
        # All non-EOS logits must be -inf
        non_eos_max = out[0, :QWEN_IM_END_ID].max().item()
        self.assertTrue(math.isinf(non_eos_max) and non_eos_max < 0)

    def test_presence_slot_binary_classification(self):
        """Validates a non-BIO binary slot (none vs present)."""
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, INTENT_TOKEN_IDS["book"]] = 20.0
        logits[0, PRESENCE_TOKEN_IDS["present"]] = 20.0  # hotel is present

        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_IM_END_ID}])

        result = req.customized_info["smielp"][0]
        hotel = next(e for e in result["entities"] if e["slot"] == "hotel")
        self.assertEqual(hotel["tag"], "present")
        self.assertGreater(hotel["confidence"], 0.99)

    def test_three_slots_all_in_result(self):
        logits = _logits_with_winner(QWEN_VOCAB_SIZE, 1, [INTENT_TOKEN_IDS["book"]])
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_EOS_ID}])
        entities = req.customized_info["smielp"][0]["entities"]
        self.assertEqual({e["slot"] for e in entities}, {"city", "date", "hotel"})

    def test_distribution_sums_to_one_at_real_scale(self):
        logits = _logits_with_winner(QWEN_VOCAB_SIZE, 1, [INTENT_TOKEN_IDS["cancel"]])
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_EOS_ID}])
        result = req.customized_info["smielp"][0]
        intent_sum = sum(result["intent"]["distribution"].values())
        self.assertAlmostEqual(intent_sum, 1.0, places=5)

    def test_customized_info_structure(self):
        """Validates the exact key/value shape that sglang's output streamer expects."""
        logits = _logits_with_winner(QWEN_VOCAB_SIZE, 1, [INTENT_TOKEN_IDS["book"]])
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"__req__": req, "schema": self._schema_config(), "eos_token_id": QWEN_EOS_ID}])

        # req.customized_info must be {str: [Any]} — one list element per output token.
        # For max_new_tokens=1, exactly one element.
        self.assertIsInstance(req.customized_info, dict)
        self.assertIn("smielp", req.customized_info)
        items = req.customized_info["smielp"]
        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 1)

        item = items[0]
        self.assertIn("intent", item)
        self.assertIn("entities", item)
        intent = item["intent"]
        self.assertIn("label", intent)
        self.assertIn("confidence", intent)
        self.assertIn("distribution", intent)

        for entity in item["entities"]:
            self.assertIn("slot", entity)
            self.assertIn("tag", entity)
            self.assertIn("confidence", entity)
            self.assertIn("distribution", entity)


class TestAnchorConfigSerialisation(unittest.TestCase):
    """Validates the NERSchema → to_anchor_config → from_dict round-trip with real IDs."""

    def test_round_trip_preserves_real_token_ids(self):
        schema = _make_anchored_schema()
        config = schema.to_anchor_config()

        # Verify wire format
        self.assertEqual(config["intent_labels"],    INTENT_LABELS)
        self.assertEqual(config["intent_token_ids"], [INTENT_TOKEN_IDS[l] for l in INTENT_LABELS])
        self.assertEqual(len(config["entity_slots"]), 3)

        # Round-trip via from_dict
        restored = NERSchema.from_dict(config)
        self.assertTrue(restored.is_anchored)
        self.assertEqual(restored.intents.token_ids(), [2190, 18515, 2829])
        city = next(s for s in restored.slots if s.name == "city")
        self.assertEqual(city.token_ids(), [46, 33, 40])

    def test_config_is_json_serialisable(self):
        """Anchor config must survive JSON round-trip (required for HTTP transport)."""
        schema = _make_anchored_schema()
        config = schema.to_anchor_config()
        dumped = json.dumps(config)
        restored_config = json.loads(dumped)
        # Should be able to reconstruct from JSON-roundtripped config
        schema2 = NERSchema.from_dict(restored_config)
        self.assertTrue(schema2.is_anchored)
        self.assertEqual(schema2.intents.token_ids(), [2190, 18515, 2829])


if __name__ == "__main__":
    unittest.main()
