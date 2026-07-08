"""Unit tests for srt/logit_slicing — no server, no model loading.

Covers Phase A (SimultaneousMultiIntentEntityLogitProcessor) and Phase B
(SMIELPWithHiddenStates) processors, schema validation, vocab anchoring, and
the backward-compatible sampler helper.  All tests run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_cpu_ci(est_time=30, suite="base-b-test-cpu")

import unittest
from typing import Dict, List
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from sglang.srt.logit_slicing import (
    IntentSchema,
    NERSchema,
    SimultaneousMultiIntentEntityLogitProcessor,
    SMIELPWithHiddenStates,
    SlotSchema,
    VocabAnchor,
    build_anchor_config,
    build_phase_b_config,
)
from sglang.srt.logit_slicing.vocab_anchor import (
    STRATEGY_EXPLICIT,
    STRATEGY_FIRST_TOKEN,
    STRATEGY_SINGLE_TOKEN,
)
from sglang.test.test_utils import CustomTestCase

# ── Shared constants ────────────────────────────────────────────────────────
VOCAB_SIZE = 200
HIDDEN_DIM = 64

# Token IDs used throughout tests
BOOK_ID = 10
CANCEL_ID = 20
STATUS_ID = 30
O_ID = 40
B_ID = 50
I_ID = 60
EOS_ID = 2


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_req():
    req = MagicMock()
    req.customized_info = {}
    return req


def _fake_tokenizer(label_map: Dict[str, List[int]], vocab_size: int = VOCAB_SIZE):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_token_id = EOS_ID
    tok.encode = lambda text, **kw: label_map.get(text, [0])
    return tok


def _explicit_schema():
    schema = NERSchema(
        intents=IntentSchema(labels=["book", "cancel", "status"]),
        slots=[SlotSchema(name="city", labels=["O", "B", "I"])],
    )
    VocabAnchor().anchor(
        schema,
        tokenizer=None,
        strategy=STRATEGY_EXPLICIT,
        intent_override={"book": BOOK_ID, "cancel": CANCEL_ID, "status": STATUS_ID},
        slot_overrides={"city": {"O": O_ID, "B": B_ID, "I": I_ID}},
    )
    return schema


def _make_config():
    return _explicit_schema().to_anchor_config()


def _make_logits(intent_winner_idx: int, slot_winner_idx: int) -> torch.Tensor:
    logits = torch.zeros(1, VOCAB_SIZE)
    intent_ids = [BOOK_ID, CANCEL_ID, STATUS_ID]
    slot_ids = [O_ID, B_ID, I_ID]
    logits[0, intent_ids[intent_winner_idx]] = 10.0
    logits[0, slot_ids[slot_winner_idx]] = 10.0
    return logits


# ── Schema validation ────────────────────────────────────────────────────────


class TestIntentSchema(CustomTestCase):
    def test_requires_at_least_two_labels(self):
        with self.assertRaises(ValueError):
            IntentSchema(labels=["only_one"])

    def test_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            IntentSchema(labels=["a", "a"])

    def test_valid_schema_created(self):
        s = IntentSchema(labels=["book", "cancel"])
        self.assertEqual(list(s.labels), ["book", "cancel"])


class TestSlotSchema(CustomTestCase):
    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            SlotSchema(name="", labels=["O", "B"])

    def test_requires_at_least_two_labels(self):
        with self.assertRaises(ValueError):
            SlotSchema(name="city", labels=["O"])


class TestNERSchema(CustomTestCase):
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
        schema = NERSchema(
            intents=IntentSchema(labels=["a", "b"]),
            slots=[],
        )
        with self.assertRaises(RuntimeError):
            schema.to_anchor_config()

    def test_to_anchor_config_round_trips_via_from_dict(self):
        schema = _explicit_schema()
        cfg = schema.to_anchor_config()
        restored = NERSchema.from_dict(cfg)
        self.assertEqual(cfg, restored.to_anchor_config())


# ── VocabAnchor ──────────────────────────────────────────────────────────────


class TestVocabAnchor(CustomTestCase):
    def test_explicit_strategy(self):
        schema = NERSchema(
            intents=IntentSchema(labels=["book", "cancel"]),
            slots=[],
        )
        VocabAnchor().anchor(
            schema,
            tokenizer=None,
            strategy=STRATEGY_EXPLICIT,
            intent_override={"book": 10, "cancel": 20},
        )
        self.assertEqual(schema.intents.label_to_token_id, {"book": 10, "cancel": 20})

    def test_first_token_strategy(self):
        label_map = {"book": [10], "cancel": [20]}
        tok = _fake_tokenizer(label_map)
        schema = NERSchema(intents=IntentSchema(labels=["book", "cancel"]), slots=[])
        VocabAnchor().anchor(schema, tok, strategy=STRATEGY_FIRST_TOKEN)
        self.assertEqual(schema.intents.label_to_token_id["book"], 10)

    def test_collision_detection_raises(self):
        schema = NERSchema(intents=IntentSchema(labels=["a", "b"]), slots=[])
        with self.assertRaises(ValueError):
            VocabAnchor().anchor(
                schema,
                tokenizer=None,
                strategy=STRATEGY_EXPLICIT,
                intent_override={"a": 5, "b": 5},
            )

    def test_explicit_requires_override(self):
        schema = NERSchema(intents=IntentSchema(labels=["a", "b"]), slots=[])
        with self.assertRaises(ValueError):
            VocabAnchor().anchor(schema, tokenizer=None, strategy=STRATEGY_EXPLICIT)

    def test_build_anchor_config_attaches_eos(self):
        schema = _explicit_schema()
        tok = MagicMock()
        tok.vocab_size = VOCAB_SIZE
        tok.eos_token_id = EOS_ID
        schema2 = NERSchema(
            intents=IntentSchema(labels=["book", "cancel", "status"]),
            slots=[SlotSchema(name="city", labels=["O", "B", "I"])],
        )
        cfg = build_anchor_config(
            schema2,
            tok,
            strategy=STRATEGY_EXPLICIT,
            intent_override={"book": BOOK_ID, "cancel": CANCEL_ID, "status": STATUS_ID},
            slot_overrides={"city": {"O": O_ID, "B": B_ID, "I": I_ID}},
        )
        self.assertEqual(cfg["eos_token_id"], EOS_ID)
        self.assertIn("schema", cfg)


# ── Phase A processor ────────────────────────────────────────────────────────


class TestPhaseAProcessor(CustomTestCase):
    def _run(self, logits, schema_config=None, eos_token_id=EOS_ID):
        if schema_config is None:
            schema_config = _make_config()
        req = _make_req()
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        proc(logits, [{"schema": schema_config, "eos_token_id": eos_token_id, "__req__": req}])
        return req.customized_info.get("smielp", [{}])[0]

    def test_intent_winner_book(self):
        result = self._run(_make_logits(intent_winner_idx=0, slot_winner_idx=0))
        self.assertEqual(result["intent"]["label"], "book")

    def test_intent_winner_cancel(self):
        result = self._run(_make_logits(intent_winner_idx=1, slot_winner_idx=0))
        self.assertEqual(result["intent"]["label"], "cancel")

    def test_slot_winner_B(self):
        result = self._run(_make_logits(intent_winner_idx=0, slot_winner_idx=1))
        city = next(e for e in result["entities"] if e["slot"] == "city")
        self.assertEqual(city["tag"], "B")

    def test_eos_forced_after_processing(self):
        logits = _make_logits(0, 0)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"schema": _make_config(), "eos_token_id": EOS_ID, "__req__": req}])
        self.assertEqual(float(logits[0, EOS_ID]), 0.0)
        self.assertTrue(all(v == float("-inf") for i, v in enumerate(logits[0].tolist()) if i != EOS_ID))

    def test_eos_clamped_when_out_of_range(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        req = _make_req()
        proc(logits, [{"schema": _make_config(), "eos_token_id": VOCAB_SIZE + 100, "__req__": req}])
        self.assertEqual(float(logits[0, VOCAB_SIZE - 1]), 0.0)

    def test_none_params_does_not_crash(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        out = proc(logits, [None])
        self.assertEqual(float(out[0, 2]), 0.0)

    def test_empty_param_list_returns_unchanged(self):
        logits = torch.ones(1, VOCAB_SIZE)
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        out = proc(logits.clone(), [])
        self.assertTrue(torch.equal(out, logits))

    def test_distribution_sums_to_one(self):
        result = self._run(_make_logits(0, 0))
        total = sum(result["intent"]["distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_batch_of_two(self):
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, BOOK_ID] = 10.0
        logits[1, CANCEL_ID] = 10.0
        cfg = _make_config()
        req0, req1 = _make_req(), _make_req()
        proc = SimultaneousMultiIntentEntityLogitProcessor()
        proc(
            logits,
            [
                {"schema": cfg, "eos_token_id": EOS_ID, "__req__": req0},
                {"schema": cfg, "eos_token_id": EOS_ID, "__req__": req1},
            ],
        )
        self.assertEqual(req0.customized_info["smielp"][0]["intent"]["label"], "book")
        self.assertEqual(req1.customized_info["smielp"][0]["intent"]["label"], "cancel")

    def test_output_key_constant(self):
        self.assertEqual(SimultaneousMultiIntentEntityLogitProcessor.OUTPUT_KEY, "smielp")


# ── Phase B processor ────────────────────────────────────────────────────────


def _orthogonal_embeddings(n: int, d: int) -> torch.Tensor:
    mat = torch.randn(n, d)
    q, _ = torch.linalg.qr(mat.T)
    return q.T[:n]


class TestPhaseBProcessor(CustomTestCase):
    def _make_label_embs(self, h: torch.Tensor, winner_idx: int, n: int):
        embs = _orthogonal_embeddings(n, HIDDEN_DIM)
        embs[winner_idx] = F.normalize(h.float(), dim=0)
        return embs

    def test_fallback_classifies_when_no_hidden_states(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, BOOK_ID] = 10.0
        req = _make_req()
        proc = SMIELPWithHiddenStates()
        proc(
            logits,
            [{"schema": _make_config(), "eos_token_id": EOS_ID, "__req__": req}],
            hidden_states=None,
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "book")
        self.assertEqual(result["mode"], "logit_slicing")
        self.assertIsNone(result["hidden_state_l2"])

    def test_embedding_mode_selects_winner(self):
        h = torch.randn(HIDDEN_DIM)
        schema_cfg = _make_config()
        label_embs = {
            "intent": self._make_label_embs(h, winner_idx=1, n=3),
            "city": self._make_label_embs(h, winner_idx=0, n=3),
        }
        logits = torch.zeros(1, VOCAB_SIZE)
        hs = h.unsqueeze(0)
        req = _make_req()
        proc = SMIELPWithHiddenStates()
        proc(
            logits,
            [{"schema": schema_cfg, "eos_token_id": EOS_ID, "__req__": req, "label_embeddings": label_embs}],
            hidden_states=hs,
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "cancel")
        self.assertEqual(result["mode"], "embedding")

    def test_hidden_state_l2_populated(self):
        h = torch.randn(HIDDEN_DIM)
        req = _make_req()
        proc = SMIELPWithHiddenStates()
        proc(
            torch.zeros(1, VOCAB_SIZE),
            [{"schema": _make_config(), "eos_token_id": EOS_ID, "__req__": req}],
            hidden_states=h.unsqueeze(0),
        )
        result = req.customized_info["smielp"][0]
        self.assertIsNotNone(result["hidden_state_l2"])
        self.assertAlmostEqual(result["hidden_state_l2"], float(h.norm().item()), places=4)

    def test_shape_mismatch_falls_back(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, BOOK_ID] = 10.0
        req = _make_req()
        proc = SMIELPWithHiddenStates()
        wrong_hs = torch.randn(2, HIDDEN_DIM)
        proc(
            logits,
            [{"schema": _make_config(), "eos_token_id": EOS_ID, "__req__": req}],
            hidden_states=wrong_hs,
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["mode"], "logit_slicing")

    def test_nan_hidden_state_falls_back(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, CANCEL_ID] = 10.0
        req = _make_req()
        proc = SMIELPWithHiddenStates()
        nan_hs = torch.full((1, HIDDEN_DIM), float("nan"))
        proc(
            logits,
            [{"schema": _make_config(), "eos_token_id": EOS_ID, "__req__": req}],
            hidden_states=nan_hs,
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["mode"], "logit_slicing")
        self.assertEqual(result["intent"]["label"], "cancel")

    def test_none_params_does_not_crash(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        proc = SMIELPWithHiddenStates()
        out = proc(logits, [None], hidden_states=None)
        self.assertEqual(float(out[0, 2]), 0.0)


# ── Sampler helper ───────────────────────────────────────────────────────────


class TestProcessorCapabilityDetection(CustomTestCase):
    def test_phase_b_detected_as_capable(self):
        from sglang.srt.layers.sampler import _processor_accepts_hidden_states

        self.assertTrue(_processor_accepts_hidden_states(SMIELPWithHiddenStates))

    def test_phase_a_detected_as_incapable(self):
        from sglang.srt.layers.sampler import _processor_accepts_hidden_states

        self.assertFalse(
            _processor_accepts_hidden_states(SimultaneousMultiIntentEntityLogitProcessor)
        )


if __name__ == "__main__":
    unittest.main()
