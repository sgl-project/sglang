"""
Phase B integration test — SMIELPWithHiddenStates at Qwen2.5-0.5B scale.

Uses:
  - Real Qwen2.5-0.5B vocab IDs (from tokenizer.json, no transformers needed)
  - Real model dimensions: vocab_size=151936, hidden_dim=896
  - Synthetic embedding matrix (random orthogonal, Qwen2.5 shape) via build_phase_b_config
  - No GPU, no server, no transformers library required

Run:
    /Users/aliiii/.venv/bin/python -m unittest test.srt.test_logit_slicing_phase_b_integration -v
"""

import importlib.util
import json
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "python" / "sglang" / "srt" / "logit_slicing"

def _load(dotted_name, filename):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / filename)
    mod  = importlib.util.module_from_spec(spec)
    mod.__package__ = dotted_name.rsplit(".", 1)[0]
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

_schema_mod   = _load("sglang.srt.logit_slicing.schema",           "schema.py")
_anchor_mod   = _load("sglang.srt.logit_slicing.vocab_anchor",     "vocab_anchor.py")
_proc_a_mod   = _load("sglang.srt.logit_slicing.processor",        "processor.py")
_proc_b_mod   = _load("sglang.srt.logit_slicing.processor_phase_b","processor_phase_b.py")

IntentSchema        = _schema_mod.IntentSchema
NERSchema           = _schema_mod.NERSchema
SlotSchema          = _schema_mod.SlotSchema
build_anchor_config = _anchor_mod.build_anchor_config
build_phase_b_config = _anchor_mod.build_phase_b_config
STRATEGY_EXPLICIT   = _anchor_mod.STRATEGY_EXPLICIT
SMIELPWithHiddenStates = _proc_b_mod.SMIELPWithHiddenStates
SimultaneousMultiIntentEntityLogitProcessor = _proc_a_mod.SimultaneousMultiIntentEntityLogitProcessor
# ─────────────────────────────────────────────────────────────────────────────

# ── Real Qwen2.5-0.5B-Instruct constants ──────────────────────────────────────
QWEN_BASE_VOCAB = 151643
QWEN_VOCAB_SIZE = 151936   # padded vocab (logit tensor width)
QWEN_HIDDEN_DIM = 896      # model hidden_size from config.json
QWEN_EOS_ID     = 151643
QWEN_IM_END_ID  = 151645

# Real single-token label IDs (from tokenizer.json vocab dict)
INTENT_LABELS    = ["book", "cancel", "status"]
INTENT_TOKEN_IDS = {"book": 2190, "cancel": 18515, "status": 2829}

BIO_LABELS    = ["O", "B", "I"]
BIO_TOKEN_IDS = {"O": 46, "B": 33, "I": 40}

PRESENCE_LABELS    = ["none", "present"]
PRESENCE_TOKEN_IDS = {"none": 6697, "present": 28744}


class MinimalQwenVocabTokenizer:
    """Reads Qwen2.5 tokenizer.json; encodes single-token labels only."""

    _JSON = (
        pathlib.Path.home()
        / ".cache/huggingface/hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / "7ae557604adf67be50417f59c2c2f167def9a775"
        / "tokenizer.json"
    )

    def __init__(self):
        if not self._JSON.exists():
            raise FileNotFoundError(str(self._JSON))
        with open(self._JSON) as f:
            data = json.load(f)
        self._vocab = data["model"]["vocab"]
        self.vocab_size    = QWEN_BASE_VOCAB
        self.eos_token_id  = QWEN_EOS_ID

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        raise ValueError(f"'{text}' is not a single-token label in Qwen2.5 vocab")


def _make_req():
    return SimpleNamespace(customized_info=None)


def _make_schema():
    return NERSchema(
        intents=IntentSchema(labels=INTENT_LABELS),
        slots=[
            SlotSchema(name="city",  labels=BIO_LABELS),
            SlotSchema(name="date",  labels=BIO_LABELS),
            SlotSchema(name="hotel", labels=PRESENCE_LABELS),
        ],
    )


def _make_embedding_matrix(vocab_size: int, hidden_dim: int, seed: int = 42) -> torch.Tensor:
    """
    Synthetic embedding matrix of shape [vocab_size, hidden_dim].
    Random normal — mimics model.embed_tokens.weight for test purposes.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(vocab_size, hidden_dim, generator=gen)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPhaseB(unittest.TestCase):
    """Validates build_phase_b_config at real Qwen2.5 scale."""

    def _get_tok(self):
        try:
            return MinimalQwenVocabTokenizer()
        except FileNotFoundError as e:
            self.skipTest(str(e))

    def test_returns_schema_eos_and_label_embeddings_keys(self):
        tok    = self._get_tok()
        schema = _make_schema()
        emb    = _make_embedding_matrix(QWEN_BASE_VOCAB, QWEN_HIDDEN_DIM)
        config = build_phase_b_config(
            schema, tok, emb,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={
                "city":  dict(BIO_TOKEN_IDS),
                "date":  dict(BIO_TOKEN_IDS),
                "hotel": dict(PRESENCE_TOKEN_IDS),
            },
            eos_token_id=QWEN_EOS_ID,
        )
        self.assertIn("schema",           config)
        self.assertIn("eos_token_id",     config)
        self.assertIn("label_embeddings", config)

    def test_label_embeddings_shapes(self):
        tok    = self._get_tok()
        schema = _make_schema()
        emb    = _make_embedding_matrix(QWEN_BASE_VOCAB, QWEN_HIDDEN_DIM)
        config = build_phase_b_config(
            schema, tok, emb,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={
                "city":  dict(BIO_TOKEN_IDS),
                "date":  dict(BIO_TOKEN_IDS),
                "hotel": dict(PRESENCE_TOKEN_IDS),
            },
        )
        le = config["label_embeddings"]
        self.assertEqual(le["intent"].shape, (3, QWEN_HIDDEN_DIM))   # 3 intents
        self.assertEqual(le["city"].shape,   (3, QWEN_HIDDEN_DIM))   # O, B, I
        self.assertEqual(le["date"].shape,   (3, QWEN_HIDDEN_DIM))
        self.assertEqual(le["hotel"].shape,  (2, QWEN_HIDDEN_DIM))   # none, present

    def test_label_embeddings_are_rows_from_embedding_matrix(self):
        """Each row in label_embeddings must equal the corresponding row in emb."""
        tok    = self._get_tok()
        schema = _make_schema()
        emb    = _make_embedding_matrix(QWEN_BASE_VOCAB, QWEN_HIDDEN_DIM)
        config = build_phase_b_config(
            schema, tok, emb,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={
                "city":  dict(BIO_TOKEN_IDS),
                "date":  dict(BIO_TOKEN_IDS),
                "hotel": dict(PRESENCE_TOKEN_IDS),
            },
        )
        le = config["label_embeddings"]
        # "cancel" token_id = 18515 → row 18515 of emb
        self.assertTrue(torch.allclose(le["intent"][1], emb[18515]))
        # "O" token_id = 46 → row 46 of emb
        self.assertTrue(torch.allclose(le["city"][0], emb[46]))


class TestPhaseBAtRealScale(unittest.TestCase):
    """SMIELPWithHiddenStates at Qwen2.5-0.5B dimensions (vocab=151936, hidden=896)."""

    def _make_config_explicit(self):
        emb    = _make_embedding_matrix(QWEN_BASE_VOCAB, QWEN_HIDDEN_DIM)
        schema = _make_schema()
        config = build_phase_b_config(
            schema,
            tokenizer=None,         # not needed for STRATEGY_EXPLICIT
            embedding_matrix=emb,
            strategy=STRATEGY_EXPLICIT,
            intent_override=dict(INTENT_TOKEN_IDS),
            slot_overrides={
                "city":  dict(BIO_TOKEN_IDS),
                "date":  dict(BIO_TOKEN_IDS),
                "hotel": dict(PRESENCE_TOKEN_IDS),
            },
            eos_token_id=QWEN_IM_END_ID,
        )
        return config, emb

    def test_embedding_mode_at_hidden_dim_896(self):
        config, emb = self._make_config_explicit()
        # Make hidden_state aligned with "cancel" embedding direction → cancel wins
        h_cancel = F.normalize(emb[INTENT_TOKEN_IDS["cancel"]], dim=0)
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        hidden = h_cancel.unsqueeze(0)

        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        proc(logits, [{"__req__": req, **config}], hidden_states=hidden)

        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "cancel")
        self.assertEqual(result["mode"], "embedding")

    def test_three_intents_correctly_separated(self):
        """Each intent embedding direction should uniquely select its label."""
        config, emb = self._make_config_explicit()
        for label, tid in INTENT_TOKEN_IDS.items():
            h    = F.normalize(emb[tid], dim=0).unsqueeze(0)
            logits = torch.zeros(1, QWEN_VOCAB_SIZE)
            req  = _make_req()
            SMIELPWithHiddenStates()(
                logits, [{"__req__": req, **config}], hidden_states=h
            )
            result = req.customized_info["smielp"][0]
            self.assertEqual(
                result["intent"]["label"], label,
                msg=f"Expected intent '{label}' when hidden state aligned with its embedding",
            )

    def test_bio_slot_classification_at_real_scale(self):
        config, emb = self._make_config_explicit()
        # Align hidden state with "B" token embedding → B wins for city and date
        h = F.normalize(emb[BIO_TOKEN_IDS["B"]], dim=0).unsqueeze(0)
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        req  = _make_req()
        SMIELPWithHiddenStates()(
            logits, [{"__req__": req, **config}], hidden_states=h
        )
        entities = req.customized_info["smielp"][0]["entities"]
        city = next(e for e in entities if e["slot"] == "city")
        date = next(e for e in entities if e["slot"] == "date")
        self.assertEqual(city["tag"], "B")
        self.assertEqual(date["tag"], "B")

    def test_eos_clamped_for_padded_vocab(self):
        """eos_token_id=151645 (> base vocab) must be within the 151936-wide logit tensor."""
        config, emb = self._make_config_explicit()
        h      = torch.randn(1, QWEN_HIDDEN_DIM)
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        req    = _make_req()
        out    = SMIELPWithHiddenStates()(logits, [{"__req__": req, **config}], hidden_states=h)
        # EOS (151645) within QWEN_VOCAB_SIZE (151936) — should not be clamped
        self.assertAlmostEqual(float(out[0, QWEN_IM_END_ID]), 0.0)
        self.assertTrue(all(math.isinf(float(out[0, i])) and float(out[0, i]) < 0
                            for i in range(QWEN_IM_END_ID - 3, QWEN_IM_END_ID)))

    def test_hidden_state_l2_norm_at_real_scale(self):
        config, _ = self._make_config_explicit()
        h     = torch.ones(1, QWEN_HIDDEN_DIM)   # L2 = sqrt(896) ≈ 29.93
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        req   = _make_req()
        SMIELPWithHiddenStates()(logits, [{"__req__": req, **config}], hidden_states=h)
        result = req.customized_info["smielp"][0]
        self.assertAlmostEqual(result["hidden_state_l2"], math.sqrt(QWEN_HIDDEN_DIM), places=2)

    def test_fallback_logit_slicing_with_real_vocab_ids(self):
        """When no label_embeddings provided, falls back to logit slicing at vocab=151936."""
        config, _ = self._make_config_explicit()
        # Remove label_embeddings to force fallback
        config_no_emb = {k: v for k, v in config.items() if k != "label_embeddings"}
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, INTENT_TOKEN_IDS["status"]] = 20.0
        req = _make_req()
        SMIELPWithHiddenStates()(
            logits,
            [{"__req__": req, **config_no_emb}],
            hidden_states=torch.randn(1, QWEN_HIDDEN_DIM),
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "status")
        self.assertEqual(result["mode"], "logit_slicing")

    def test_batch_of_three_at_real_scale(self):
        config, emb = self._make_config_explicit()
        labels_to_test = ["book", "cancel", "status"]
        hidden = torch.stack([
            F.normalize(emb[INTENT_TOKEN_IDS[lbl]], dim=0)
            for lbl in labels_to_test
        ])                                   # [3, 896]
        logits = torch.zeros(3, QWEN_VOCAB_SIZE)
        reqs   = [_make_req() for _ in range(3)]
        cpl    = [{"__req__": reqs[i], **config} for i in range(3)]
        SMIELPWithHiddenStates()(logits, cpl, hidden_states=hidden)
        for i, expected in enumerate(labels_to_test):
            self.assertEqual(
                reqs[i].customized_info["smielp"][0]["intent"]["label"], expected,
                msg=f"Batch index {i}: expected '{expected}'",
            )


class TestPhaseBValidation(unittest.TestCase):
    """Input validation guards: wrong shape, NaN, bad label_embeddings."""

    def _config(self):
        return {
            "schema": {
                "intent_labels": ["book", "cancel", "status"],
                "intent_token_ids": [2190, 18515, 2829],
                "entity_slots": [
                    {"name": "city", "bio_labels": ["O", "B", "I"], "bio_token_ids": [46, 33, 40]},
                ],
            },
            "eos_token_id": QWEN_IM_END_ID,
            "label_embeddings": {
                "intent": torch.randn(3, QWEN_HIDDEN_DIM),
                "city":   torch.randn(3, QWEN_HIDDEN_DIM),
            },
        }

    def test_hidden_states_shape_mismatch_falls_back(self):
        """hidden_states with wrong batch dim → logit slicing fallback (no crash)."""
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, 2190] = 10.0   # book wins in logit space
        req = _make_req()
        # Pass 2-row hidden_states for a 1-request batch
        SMIELPWithHiddenStates()(
            logits, [{"__req__": req, **self._config()}],
            hidden_states=torch.randn(2, QWEN_HIDDEN_DIM),   # wrong dim
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["mode"], "logit_slicing")   # fell back
        self.assertEqual(result["intent"]["label"], "book")

    def test_hidden_states_1d_falls_back(self):
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, 18515] = 10.0   # cancel
        req = _make_req()
        SMIELPWithHiddenStates()(
            logits, [{"__req__": req, **self._config()}],
            hidden_states=torch.randn(QWEN_HIDDEN_DIM),   # 1-D, not 2-D
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["mode"], "logit_slicing")

    def test_nan_hidden_state_falls_back(self):
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, 2829] = 10.0   # status
        req = _make_req()
        h_nan = torch.full((1, QWEN_HIDDEN_DIM), float("nan"))
        SMIELPWithHiddenStates()(
            logits, [{"__req__": req, **self._config()}],
            hidden_states=h_nan,
        )
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["mode"], "logit_slicing")
        self.assertEqual(result["intent"]["label"], "status")
        self.assertIsNone(result["hidden_state_l2"])

    def test_wrong_label_embedding_shape_skips_head(self):
        """Wrong embedding dim → that head is skipped, result has no intent."""
        cfg = self._config()
        cfg["label_embeddings"]["intent"] = torch.randn(3, 128)   # wrong hidden_dim
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        req = _make_req()
        SMIELPWithHiddenStates()(
            logits, [{"__req__": req, **cfg}],
            hidden_states=torch.randn(1, QWEN_HIDDEN_DIM),
        )
        result = req.customized_info["smielp"][0]
        # Intent embedding had wrong shape → intent result is None
        self.assertIsNone(result["intent"])
        # City embedding was correct → city entity should be present
        self.assertEqual(len(result["entities"]), 1)


class TestPhaseBAndACoexistence(unittest.TestCase):
    """
    Phase A and Phase B processors can run simultaneously on the same schema config.
    Phase A reads logits; Phase B reads hidden_states. Results should agree on
    classification when both signals are consistent.
    """

    def test_both_processors_agree_on_dominant_intent(self):
        intent_tid = INTENT_TOKEN_IDS["cancel"]
        schema_cfg = {
            "intent_labels":    INTENT_LABELS,
            "intent_token_ids": [INTENT_TOKEN_IDS[l] for l in INTENT_LABELS],
            "entity_slots": [],
        }
        logits = torch.zeros(1, QWEN_VOCAB_SIZE)
        logits[0, intent_tid] = 20.0   # cancel dominates logits

        emb = _make_embedding_matrix(QWEN_BASE_VOCAB, QWEN_HIDDEN_DIM)
        # Make hidden state aligned with cancel's embedding
        h = F.normalize(emb[intent_tid], dim=0).unsqueeze(0)

        req_a  = _make_req()
        req_b  = _make_req()

        params_a = {"__req__": req_a,  "schema": schema_cfg, "eos_token_id": 2}
        params_b = {
            "__req__": req_b,
            "schema": schema_cfg,
            "eos_token_id": 2,
            "label_embeddings": {"intent": emb[[INTENT_TOKEN_IDS[l] for l in INTENT_LABELS]]},
        }

        SimultaneousMultiIntentEntityLogitProcessor()(logits.clone(), [params_a])
        SMIELPWithHiddenStates()(logits.clone(), [params_b], hidden_states=h)

        intent_a = req_a.customized_info["smielp"][0]["intent"]["label"]
        intent_b = req_b.customized_info["smielp"][0]["intent"]["label"]
        self.assertEqual(intent_a, "cancel")
        self.assertEqual(intent_b, "cancel")
        self.assertEqual(intent_a, intent_b)


if __name__ == "__main__":
    unittest.main()
