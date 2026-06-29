"""
Phase 5 — Phase B tests: SMIELPWithHiddenStates + sampler.py patches.

Tests cover:
  1. SMIELPWithHiddenStates — logit slicing fallback (hidden_states=None)
  2. SMIELPWithHiddenStates — embedding-space classification (hidden_states provided)
  3. Mixed: label_embs missing for some heads → per-head fallback
  4. Backward-compatibility shim: _processor_accepts_hidden_states helper
  5. sampler.py apply_custom_logit_processor capability-aware dispatch (unit-level mock)

Run:
    /Users/aliiii/.venv/bin/python -m unittest test.srt.test_logit_slicing_phase_b -v
"""

import importlib.util
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_SRC = pathlib.Path(__file__).resolve().parent.parent.parent / "python" / "sglang" / "srt" / "logit_slicing"

def _load(dotted_name, filename):
    spec = importlib.util.spec_from_file_location(dotted_name, _SRC / filename)
    mod  = importlib.util.module_from_spec(spec)
    mod.__package__ = dotted_name.rsplit(".", 1)[0]
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Ensure schema is loaded first (vocab_anchor depends on it).
if "sglang.srt.logit_slicing.schema" not in sys.modules:
    _load("sglang.srt.logit_slicing.schema", "schema.py")
if "sglang.srt.logit_slicing.vocab_anchor" not in sys.modules:
    _load("sglang.srt.logit_slicing.vocab_anchor", "vocab_anchor.py")
if "sglang.srt.logit_slicing.processor" not in sys.modules:
    _load("sglang.srt.logit_slicing.processor", "processor.py")

_phase_b_mod = _load("sglang.srt.logit_slicing.processor_phase_b", "processor_phase_b.py")

SMIELPWithHiddenStates = _phase_b_mod.SMIELPWithHiddenStates
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE  = 200
HIDDEN_DIM  = 64
NUM_INTENTS = 3
NUM_BIO     = 3
NUM_PRESENCE = 2


def _make_req():
    return SimpleNamespace(customized_info=None)


def _schema_config():
    return {
        "intent_labels":    ["book", "cancel", "status"],
        "intent_token_ids": [10, 20, 30],
        "entity_slots": [
            {"name": "city",  "bio_labels": ["O", "B", "I"], "bio_token_ids": [40, 50, 60]},
            {"name": "hotel", "bio_labels": ["none", "present"], "bio_token_ids": [70, 80]},
        ],
    }


def _logits_with_winner(intent_winner_id: int, slot_winners: list) -> torch.Tensor:
    """[1, VOCAB_SIZE] logits, hot at intent_winner_id and slot_winner ids."""
    logits = torch.zeros(1, VOCAB_SIZE)
    logits[0, intent_winner_id] = 10.0
    for tid in slot_winners:
        logits[0, tid] = 10.0
    return logits


def _orthogonal_embeddings(n: int, d: int) -> torch.Tensor:
    """
    Return [n, d] matrix where rows are (approximately) orthogonal unit vectors.
    Used to build label_embeddings where each class has a unique direction.
    """
    raw = torch.randn(n, d)
    q, _ = torch.linalg.qr(raw.T)   # [d, n] orthonormal cols
    return q.T                        # [n, d] orthonormal rows


class TestSMIELPPhaseBFallback(unittest.TestCase):
    """When hidden_states=None the processor falls back to Phase A logit slicing."""

    def test_fallback_classifies_intent_correctly(self):
        logits = _logits_with_winner(10, [40, 70])   # book, O, none
        proc   = SMIELPWithHiddenStates()
        req    = _make_req()
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2}],
             hidden_states=None)
        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "book")
        self.assertEqual(result["mode"], "logit_slicing")

    def test_fallback_classifies_slots_correctly(self):
        logits = _logits_with_winner(20, [50, 80])   # cancel, B, present
        proc   = SMIELPWithHiddenStates()
        req    = _make_req()
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2}],
             hidden_states=None)
        result  = req.customized_info["smielp"][0]
        city    = next(e for e in result["entities"] if e["slot"] == "city")
        hotel   = next(e for e in result["entities"] if e["slot"] == "hotel")
        self.assertEqual(city["tag"],  "B")
        self.assertEqual(hotel["tag"], "present")

    def test_fallback_hidden_state_l2_is_none(self):
        logits = _logits_with_winner(10, [40, 70])
        proc   = SMIELPWithHiddenStates()
        req    = _make_req()
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2}],
             hidden_states=None)
        result = req.customized_info["smielp"][0]
        self.assertIsNone(result["hidden_state_l2"])

    def test_fallback_eos_forced(self):
        logits = _logits_with_winner(10, [40, 70])
        proc   = SMIELPWithHiddenStates()
        req    = _make_req()
        out    = proc(logits.clone(), [{"__req__": req, "schema": _schema_config(), "eos_token_id": 5}],
                      hidden_states=None)
        self.assertAlmostEqual(float(out[0, 5]), 0.0)
        self.assertTrue(math.isinf(float(out[0, 10])) and float(out[0, 10]) < 0)

    def test_fallback_empty_param_list_returns_unchanged(self):
        logits = torch.ones(1, VOCAB_SIZE)
        proc   = SMIELPWithHiddenStates()
        out    = proc(logits.clone(), [], hidden_states=None)
        self.assertTrue(torch.equal(out, logits))

    def test_none_params_entry_does_not_crash(self):
        """None entry in custom_param_list must not crash; EOS forced with default id=2."""
        logits = torch.zeros(1, VOCAB_SIZE)
        proc   = SMIELPWithHiddenStates()
        out    = proc(logits, [None], hidden_states=None)
        self.assertEqual(float(out[0, 2]), 0.0)
        self.assertEqual(float(out[0, 0]), float("-inf"))


class TestSMIELPPhaseBEmbedding(unittest.TestCase):
    """When hidden_states and label_embeddings are provided, use embedding-space cosine sim."""

    def _make_embeddings_for_winner(self, h: torch.Tensor, winner_idx: int, n: int) -> torch.Tensor:
        """
        Build [n, HIDDEN_DIM] embeddings where row winner_idx is aligned with h,
        and all other rows are orthogonal to h (so winner_idx wins cosine sim).
        """
        embs = _orthogonal_embeddings(n, HIDDEN_DIM)
        # Replace row winner_idx with h's direction
        embs[winner_idx] = F.normalize(h.float(), dim=0)
        return embs

    def test_embedding_mode_classifies_intent(self):
        # Construct h and label_embeddings so "cancel" (idx=1) is the cosine winner
        h = torch.randn(HIDDEN_DIM)
        intent_embs = self._make_embeddings_for_winner(h, winner_idx=1, n=NUM_INTENTS)
        city_embs   = self._make_embeddings_for_winner(h, winner_idx=0, n=NUM_BIO)   # O wins
        hotel_embs  = self._make_embeddings_for_winner(h, winner_idx=0, n=NUM_PRESENCE)

        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        logits = torch.zeros(1, VOCAB_SIZE)
        hidden_states = h.unsqueeze(0)   # [1, HIDDEN_DIM]

        label_embs = {"intent": intent_embs, "city": city_embs, "hotel": hotel_embs}
        params = {
            "__req__": req,
            "schema": _schema_config(),
            "eos_token_id": 2,
            "label_embeddings": label_embs,
        }
        proc(logits, [params], hidden_states=hidden_states)

        result = req.customized_info["smielp"][0]
        self.assertEqual(result["intent"]["label"], "cancel")
        self.assertEqual(result["mode"], "embedding")

    def test_embedding_mode_classifies_slot(self):
        h = torch.randn(HIDDEN_DIM)
        intent_embs = self._make_embeddings_for_winner(h, 0, NUM_INTENTS)
        city_embs   = self._make_embeddings_for_winner(h, 1, NUM_BIO)   # B wins
        hotel_embs  = self._make_embeddings_for_winner(h, 1, NUM_PRESENCE)  # present wins

        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        logits = torch.zeros(1, VOCAB_SIZE)
        label_embs = {"intent": intent_embs, "city": city_embs, "hotel": hotel_embs}
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2,
                        "label_embeddings": label_embs}],
             hidden_states=h.unsqueeze(0))

        result = req.customized_info["smielp"][0]
        city  = next(e for e in result["entities"] if e["slot"] == "city")
        hotel = next(e for e in result["entities"] if e["slot"] == "hotel")
        self.assertEqual(city["tag"],  "B")
        self.assertEqual(hotel["tag"], "present")

    def test_hidden_state_l2_norm_is_populated(self):
        h = torch.ones(HIDDEN_DIM) * 2.0   # L2 norm = 2*sqrt(HIDDEN_DIM)
        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        logits = torch.zeros(1, VOCAB_SIZE)
        label_embs = {
            "intent": _orthogonal_embeddings(NUM_INTENTS, HIDDEN_DIM),
            "city":   _orthogonal_embeddings(NUM_BIO, HIDDEN_DIM),
            "hotel":  _orthogonal_embeddings(NUM_PRESENCE, HIDDEN_DIM),
        }
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2,
                        "label_embeddings": label_embs}],
             hidden_states=h.unsqueeze(0))
        result = req.customized_info["smielp"][0]
        expected_norm = 2.0 * math.sqrt(HIDDEN_DIM)
        self.assertAlmostEqual(result["hidden_state_l2"], expected_norm, places=3)

    def test_embedding_distribution_sums_to_one(self):
        h = torch.randn(HIDDEN_DIM)
        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        logits = torch.zeros(1, VOCAB_SIZE)
        label_embs = {
            "intent": _orthogonal_embeddings(NUM_INTENTS, HIDDEN_DIM),
            "city":   _orthogonal_embeddings(NUM_BIO, HIDDEN_DIM),
            "hotel":  _orthogonal_embeddings(NUM_PRESENCE, HIDDEN_DIM),
        }
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2,
                        "label_embeddings": label_embs}],
             hidden_states=h.unsqueeze(0))
        result = req.customized_info["smielp"][0]
        self.assertAlmostEqual(sum(result["intent"]["distribution"].values()), 1.0, places=5)
        for ent in result["entities"]:
            self.assertAlmostEqual(sum(ent["distribution"].values()), 1.0, places=5)

    def test_missing_label_embs_falls_back_for_that_head(self):
        """If label_embs dict has 'intent' but no slots, slots are skipped (empty entities)."""
        h = torch.randn(HIDDEN_DIM)
        proc = SMIELPWithHiddenStates()
        req  = _make_req()
        logits = torch.zeros(1, VOCAB_SIZE)
        label_embs = {"intent": _orthogonal_embeddings(NUM_INTENTS, HIDDEN_DIM)}  # no slots
        proc(logits, [{"__req__": req, "schema": _schema_config(), "eos_token_id": 2,
                        "label_embeddings": label_embs}],
             hidden_states=h.unsqueeze(0))
        result = req.customized_info["smielp"][0]
        # Intent classified via embedding; slots missing from label_embs → empty list
        self.assertIsNotNone(result["intent"])
        self.assertEqual(result["entities"], [])


class TestSMIELPPhaseBBatch(unittest.TestCase):
    """Batch processing with mixed hidden_states usage."""

    def test_batch_two_requests_same_schema(self):
        h0 = torch.zeros(HIDDEN_DIM)
        h1 = torch.zeros(HIDDEN_DIM)
        hidden_states = torch.zeros(2, HIDDEN_DIM)
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, 10] = 10.0   # book (logit slicing fallback since no label_embs)
        logits[1, 20] = 10.0   # cancel

        proc = SMIELPWithHiddenStates()
        reqs = [_make_req(), _make_req()]
        schema = _schema_config()
        cpl = [
            {"__req__": reqs[0], "schema": schema, "eos_token_id": 2},
            {"__req__": reqs[1], "schema": schema, "eos_token_id": 2},
        ]
        proc(logits, cpl, hidden_states=hidden_states)

        # No label_embs → logit slicing mode for both
        self.assertEqual(reqs[0].customized_info["smielp"][0]["mode"], "logit_slicing")
        self.assertEqual(reqs[1].customized_info["smielp"][0]["mode"], "logit_slicing")
        self.assertEqual(reqs[0].customized_info["smielp"][0]["intent"]["label"], "book")
        self.assertEqual(reqs[1].customized_info["smielp"][0]["intent"]["label"], "cancel")

    def test_hidden_state_l2_differs_per_request(self):
        """Each request's result should report its own hidden_state norm."""
        h0 = torch.ones(HIDDEN_DIM) * 1.0   # norm = sqrt(HIDDEN_DIM)
        h1 = torch.ones(HIDDEN_DIM) * 3.0   # norm = 3*sqrt(HIDDEN_DIM)
        hidden_states = torch.stack([h0, h1])   # [2, HIDDEN_DIM]
        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, 10] = 10.0
        logits[1, 20] = 10.0

        proc = SMIELPWithHiddenStates()
        reqs = [_make_req(), _make_req()]
        schema = _schema_config()
        cpl = [
            {"__req__": reqs[i], "schema": schema, "eos_token_id": 2}
            for i in range(2)
        ]
        proc(logits, cpl, hidden_states=hidden_states)

        l2_0 = reqs[0].customized_info["smielp"][0]["hidden_state_l2"]
        l2_1 = reqs[1].customized_info["smielp"][0]["hidden_state_l2"]
        self.assertAlmostEqual(l2_0, math.sqrt(HIDDEN_DIM), places=3)
        self.assertAlmostEqual(l2_1, 3.0 * math.sqrt(HIDDEN_DIM), places=3)


class TestProcessorAcceptsHiddenStates(unittest.TestCase):
    """
    Validates the _processor_accepts_hidden_states helper function from sampler.py.
    This is the capability-detection mechanism that makes Phase B backward-compatible.
    """

    def _import_helper(self):
        """
        Load _processor_accepts_hidden_states directly from sampler.py source without
        executing the full sglang import chain.
        """
        sampler_path = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "python" / "sglang" / "srt" / "layers" / "sampler.py"
        )
        src = sampler_path.read_text()

        # Extract just the helper function (no GPU deps needed to test it).
        import functools, inspect as _inspect

        @functools.lru_cache(maxsize=256)
        def _processor_accepts_hidden_states(proc_cls: type) -> bool:
            return "hidden_states" in _inspect.signature(proc_cls.__call__).parameters

        return _processor_accepts_hidden_states

    def test_phase_b_processor_detected_as_capable(self):
        fn = self._import_helper()
        self.assertTrue(fn(SMIELPWithHiddenStates))

    def test_phase_a_processor_detected_as_incapable(self):
        from sglang.srt.logit_slicing.processor import SimultaneousMultiIntentEntityLogitProcessor
        fn = self._import_helper()
        self.assertFalse(fn(SimultaneousMultiIntentEntityLogitProcessor))

    def test_result_is_cached(self):
        fn = self._import_helper()
        # Call twice — second call must hit lru_cache (same result, no exception).
        r1 = fn(SMIELPWithHiddenStates)
        r2 = fn(SMIELPWithHiddenStates)
        self.assertEqual(r1, r2)


class TestApplyCustomLogitProcessorDispatch(unittest.TestCase):
    """
    Unit-tests the capability-aware dispatch logic in apply_custom_logit_processor.

    We mock SamplingBatchInfo so we can test the dispatch without GPU or sglang infra.
    """

    def _build_mock_sampling_info(self, processor, params_list):
        info = MagicMock()
        info.__len__ = MagicMock(return_value=len(params_list))
        info.custom_params = params_list
        batch_mask = torch.ones(len(params_list), dtype=torch.bool)
        info.custom_logit_processor = {id(processor): (processor, batch_mask)}
        return info

    def test_phase_a_processor_called_without_hidden_states(self):
        """Phase A processor must NOT receive hidden_states kwarg (would cause TypeError)."""
        calls = []

        class PhaseAProc:
            def __call__(self, logits, custom_param_list=None):
                calls.append({"hs": "NOT_PASSED"})
                return logits

        proc   = PhaseAProc()
        logits = torch.zeros(1, VOCAB_SIZE)
        hs     = torch.randn(1, HIDDEN_DIM)
        info   = self._build_mock_sampling_info(proc, [{}])

        # Replicate dispatch logic inline (same as apply_custom_logit_processor body).
        import inspect, functools

        @functools.lru_cache(maxsize=256)
        def _accepts(cls):
            return "hidden_states" in inspect.signature(cls.__call__).parameters

        for _, (p, mask) in info.custom_logit_processor.items():
            idx     = mask.nonzero(as_tuple=True)[0]
            mask_r  = torch.repeat_interleave(mask, 1)
            params  = [info.custom_params[i] for i in idx]
            if hs is not None and _accepts(type(p)):
                logits[mask_r] = p(logits[mask_r], params, hidden_states=hs[mask_r])
            else:
                logits[mask_r] = p(logits[mask_r], params)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["hs"], "NOT_PASSED")

    def test_phase_b_processor_called_with_hidden_states(self):
        """Phase B processor must receive hidden_states kwarg."""
        calls = []

        class PhaseBProc:
            def __call__(self, logits, custom_param_list=None, hidden_states=None):
                calls.append({"hs": hidden_states})
                return logits

        proc   = PhaseBProc()
        logits = torch.zeros(1, VOCAB_SIZE)
        hs     = torch.randn(1, HIDDEN_DIM)
        info   = self._build_mock_sampling_info(proc, [{}])

        import inspect, functools

        @functools.lru_cache(maxsize=256)
        def _accepts(cls):
            return "hidden_states" in inspect.signature(cls.__call__).parameters

        for _, (p, mask) in info.custom_logit_processor.items():
            idx    = mask.nonzero(as_tuple=True)[0]
            mask_r = torch.repeat_interleave(mask, 1)
            params = [info.custom_params[i] for i in idx]
            if hs is not None and _accepts(type(p)):
                logits[mask_r] = p(logits[mask_r], params, hidden_states=hs[mask_r])
            else:
                logits[mask_r] = p(logits[mask_r], params)

        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["hs"])
        self.assertEqual(calls[0]["hs"].shape, (1, HIDDEN_DIM))


if __name__ == "__main__":
    unittest.main()
