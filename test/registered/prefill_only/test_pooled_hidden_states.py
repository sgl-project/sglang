"""Tests for the return_pooled_hidden_states feature on the scoring API.

Covers both Engine-level (Python API) and HTTP-level (/v1/score) integration:

  TestPooledHiddenStatesEngine     — SeqCls model, single-item scoring
  TestPooledHiddenStatesMISEngine  — SeqCls model, MIS delimiter mode
  TestPooledHiddenStatesHTTP       — HTTP layer serialization round-trip
  TestPooledHiddenStatesCausalLMRejection — CausalLM must reject the flag

Each test class spins up its own Engine or server so GPU memory is isolated.
"""

import json
import unittest

import requests
import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=96, suite="stage-b-test-1-gpu-small")

_SEQCLS_MODEL = "Qwen/Qwen3-0.6B"
_QWEN3_EOT_TOKEN_ID = 151643
_CAUSAL_LM_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
_NUM_LABELS = 4

# Local overrides for offline testing (no network).  Set to None to use HF hub.
_LOCAL_SEQCLS_MODEL = (
    "/shared/public/elr-models/Qwen/Qwen3-0.6B/e6de91484c29aa9480d55605af694f39b081c455"
)
_LOCAL_CAUSAL_LM_MODEL = "/shared/public/elr-models/meta-llama/Llama-3.2-1B-Instruct/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"

import os

if _LOCAL_SEQCLS_MODEL and os.path.isdir(_LOCAL_SEQCLS_MODEL):
    _SEQCLS_MODEL = _LOCAL_SEQCLS_MODEL
if _LOCAL_CAUSAL_LM_MODEL and os.path.isdir(_LOCAL_CAUSAL_LM_MODEL):
    _CAUSAL_LM_MODEL = _LOCAL_CAUSAL_LM_MODEL


# ---------------------------------------------------------------------------
# Engine — single-item scoring (no MIS)
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesEngine(CustomTestCase):
    """Validates return_pooled_hidden_states through the Engine Python API.

    Uses Qwen3ForSequenceClassification with a random head so we only care
    about shape and pipeline plumbing, not numerical accuracy.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": _NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_phs_returned_when_requested(self):
        """Pooled hidden states are present and shaped correctly."""
        result = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad"],
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), 2)
        for phs in result.pooled_hidden_states:
            self.assertIsInstance(phs, torch.Tensor)
            self.assertEqual(phs.dim(), 1)
            self.assertGreater(phs.shape[0], 0)

    def test_phs_none_when_not_requested(self):
        """Without the flag, pooled_hidden_states must be None."""
        result = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad"],
            return_pooled_hidden_states=False,
        )
        self.assertIsNone(result.pooled_hidden_states)

    def test_phs_shape_is_consistent(self):
        """PHS tensors for different items share the same hidden dimension."""
        result = self.engine.score(
            query="Evaluate:",
            items=["Alpha", "Beta", "Gamma"],
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        dims = {phs.shape[0] for phs in result.pooled_hidden_states}
        self.assertEqual(len(dims), 1, "All PHS vectors must share the same hidden dim")
        self.assertGreater(dims.pop(), 0)

    def test_phs_count_matches_items(self):
        """Number of PHS tensors equals number of items for various batch sizes."""
        for n in [1, 3, 5]:
            with self.subTest(n=n):
                result = self.engine.score(
                    query="Classify:",
                    items=[f"Item {i}" for i in range(n)],
                    return_pooled_hidden_states=True,
                )
                self.assertIsNotNone(result.pooled_hidden_states)
                self.assertEqual(len(result.pooled_hidden_states), n)

    def test_phs_on_cpu(self):
        """Returned tensors live on CPU (no GPU references leak to caller)."""
        result = self.engine.score(
            query="Check device:",
            items=["Test"],
            return_pooled_hidden_states=True,
        )
        for phs in result.pooled_hidden_states:
            self.assertEqual(str(phs.device), "cpu")

    def test_phs_deterministic(self):
        """Identical requests produce identical PHS tensors."""
        kwargs = dict(
            query="Evaluate:", items=["A", "B"], return_pooled_hidden_states=True
        )
        phs1 = self.engine.score(**kwargs).pooled_hidden_states
        phs2 = self.engine.score(**kwargs).pooled_hidden_states
        for t1, t2 in zip(phs1, phs2):
            self.assertTrue(
                torch.allclose(t1, t2, atol=1e-5),
                "Pooled hidden states differ across identical requests",
            )

    def test_scores_unaffected_by_phs_flag(self):
        """The phs flag must not change the scores themselves (fp16 tolerance)."""
        kwargs = dict(query="Rate:", items=["X", "Y", "Z"], apply_softmax=True)
        scores_without = self.engine.score(
            **kwargs, return_pooled_hidden_states=False
        ).scores
        scores_with = self.engine.score(
            **kwargs, return_pooled_hidden_states=True
        ).scores
        self.assertEqual(len(scores_without), len(scores_with))
        for row_a, row_b in zip(scores_without, scores_with):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=2)

    def test_phs_with_tokenized_inputs(self):
        """Pre-tokenized inputs also return PHS correctly."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        query, items = "Evaluate:", ["Alpha", "Beta"]
        result = self.engine.score(
            query=tok.encode(query),
            items=[tok.encode(i) for i in items],
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), 2)


# ---------------------------------------------------------------------------
# Engine — MIS delimiter mode
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesMISEngine(CustomTestCase):
    """Validates return_pooled_hidden_states in MIS (delimiter) scoring mode.

    MIS packs all items into one sequence; the PHS at each delimiter position
    should be returned per-item.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            multi_item_scoring_delimiter=_QWEN3_EOT_TOKEN_ID,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": _NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_mis_phs_count_matches_items(self):
        """MIS must return one PHS tensor per item."""
        items = ["Option A", "Option B", "Option C"]
        result = self.engine.score(
            query="Rate each:", items=items, return_pooled_hidden_states=True
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), len(items))

    def test_mis_phs_none_when_not_requested(self):
        result = self.engine.score(
            query="Rate each:",
            items=["A", "B"],
            return_pooled_hidden_states=False,
        )
        self.assertIsNone(result.pooled_hidden_states)

    def test_mis_phs_are_tensors_on_cpu(self):
        result = self.engine.score(
            query="Classify:", items=["X", "Y"], return_pooled_hidden_states=True
        )
        for phs in result.pooled_hidden_states:
            self.assertIsInstance(phs, torch.Tensor)
            self.assertEqual(str(phs.device), "cpu")

    def test_mis_phs_different_items_different_hidden_states(self):
        """Different items should produce distinct PHS vectors."""
        items = [
            "Option A is about cats",
            "Option B is about dogs",
            "Option C is about fish",
        ]
        result = self.engine.score(
            query="Classify:", items=items, return_pooled_hidden_states=True
        )
        phs = result.pooled_hidden_states
        self.assertFalse(
            all(torch.allclose(phs[0], p, atol=1e-6) for p in phs[1:]),
            "All MIS items returned identical hidden states",
        )

    def test_mis_single_item(self):
        """Single item through MIS path still returns one PHS tensor."""
        result = self.engine.score(
            query="Evaluate:", items=["Only one"], return_pooled_hidden_states=True
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), 1)

    def test_mis_many_items(self):
        """10 items all produce PHS tensors of consistent shape."""
        items = [f"Item {i}" for i in range(10)]
        result = self.engine.score(
            query="Classify:", items=items, return_pooled_hidden_states=True
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), len(items))
        shapes = {phs.shape for phs in result.pooled_hidden_states}
        self.assertEqual(len(shapes), 1, "MIS PHS shapes should be uniform")

    def test_mis_scores_unaffected_by_phs_flag(self):
        """Enabling PHS does not alter the returned scores (fp16 tolerance)."""
        kwargs = dict(
            query="Rate:", items=["Alpha", "Beta", "Gamma"], apply_softmax=True
        )
        scores_without = self.engine.score(
            **kwargs, return_pooled_hidden_states=False
        ).scores
        scores_with = self.engine.score(
            **kwargs, return_pooled_hidden_states=True
        ).scores
        for row_a, row_b in zip(scores_without, scores_with):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=2)


# ---------------------------------------------------------------------------
# CausalLM rejection
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesCausalLMRejection(CustomTestCase):
    """CausalLM models must reject return_pooled_hidden_states=True."""

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(model_path=_CAUSAL_LM_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_causal_lm_rejects_phs(self):
        """ValueError raised when requesting PHS from a CausalLM."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.score(
                query="Test",
                items=["Item"],
                label_token_ids=[1, 2],
                return_pooled_hidden_states=True,
            )
        self.assertIn("CausalLM", str(ctx.exception))

    def test_causal_lm_without_phs_still_works(self):
        """Baseline: CausalLM scoring without the flag works fine."""
        result = self.engine.score(
            query="Test",
            items=["Item"],
            label_token_ids=[1, 2],
            apply_softmax=True,
            return_pooled_hidden_states=False,
        )
        self.assertEqual(len(result.scores), 1)
        self.assertIsNone(result.pooled_hidden_states)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesHTTP(CustomTestCase):
    """HTTP integration: /v1/score with return_pooled_hidden_states.

    Validates that the Pydantic schema, JSON serialization, and ORJSONResponse
    round-trip preserves the pooled hidden states as nested lists.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _SEQCLS_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--json-model-override-args",
                json.dumps(
                    {
                        "architectures": ["Qwen3ForSequenceClassification"],
                        "num_labels": _NUM_LABELS,
                    }
                ),
                "--mem-fraction-static",
                "0.15",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(self.base_url + "/v1/score", json=payload)

    def test_phs_in_response_json(self):
        """Response includes pooled_hidden_states as nested float lists."""
        resp = self._post(
            {
                "query": "Rate each:",
                "items": ["Good", "Bad"],
                "return_pooled_hidden_states": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        phs = body.get("pooled_hidden_states")
        self.assertIsNotNone(phs)
        self.assertEqual(len(phs), 2)
        for item_phs in phs:
            self.assertIsInstance(item_phs, list)
            self.assertGreater(len(item_phs), 0)
            for v in item_phs:
                self.assertIsInstance(v, float)

    def test_phs_absent_when_not_requested(self):
        """Without the flag, pooled_hidden_states is null in JSON."""
        resp = self._post(
            {
                "query": "Rate each:",
                "items": ["Good"],
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body.get("pooled_hidden_states"))

    def test_phs_matches_item_count(self):
        """Number of PHS vectors equals number of items."""
        items = ["A", "B", "C", "D"]
        resp = self._post(
            {
                "query": "Classify:",
                "items": items,
                "return_pooled_hidden_states": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        phs = resp.json()["pooled_hidden_states"]
        self.assertEqual(len(phs), len(items))


if __name__ == "__main__":
    unittest.main(verbosity=3)
