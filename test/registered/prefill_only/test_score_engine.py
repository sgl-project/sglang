"""Engine API tests for the /v1/score scoring pipeline.

Two model types, two scoring modes:

  TestCausalLMScoring        — CausalLM, single-item and batched multi-item
  TestSeqClsScoring          — SequenceClassification, single-item mode
  TestSeqClsMISScoring       — SequenceClassification, MIS mode (--enable-mis)
  TestSeqClsMISAdvancedScoring — SeqCls MIS with 12 labels (tensor shape stress)

The Engine (Python API) is the right layer for correctness testing: it
exercises tokenization, forward pass, pooling, and score extraction without
the HTTP serialization overhead.  HTTP-layer tests live in test_score_api.py.
Thorough MIS tests (parity, concurrency, generation models) live in
test_multi_item_scoring.py.
"""

import json
import os
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=85, suite="stage-b-test-1-gpu-small")

_CAUSAL_LM_MODEL = os.environ.get("TEST_MODEL_NAME", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
_SEQCLS_MODEL = os.environ.get("TEST_CLASSIFICATION_BASE_MODEL", "Qwen/Qwen3-0.6B")


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class TestCausalLMScoring(CustomTestCase):
    """CausalLM scoring via Engine — correctness, batching, and edge cases.

    A single Engine instance is shared across all test methods (class-level
    setup) so model loading happens once, not once per test.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(model_path=_CAUSAL_LM_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hf_scores(self, query, items, label_token_ids, item_first=False):
        """Reference scores computed directly with HuggingFace (CPU inference)."""
        tokenizer = AutoTokenizer.from_pretrained(
            _CAUSAL_LM_MODEL, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            _CAUSAL_LM_MODEL, trust_remote_code=True
        )
        try:
            scores = []
            for item in items:
                text = f"{item}{query}" if item_first else f"{query}{item}"
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    last_logits = model(**inputs).logits[0, -1]
                target_probs = torch.softmax(last_logits[label_token_ids], dim=-1)
                scores.append([p.item() for p in target_probs])
            return scores
        finally:
            model.cpu()
            del model, tokenizer
            torch.cuda.empty_cache()

    def _assert_scores_close(self, hf, sgl, tol=0.01):
        self.assertEqual(len(hf), len(sgl))
        for hf_row, sgl_row in zip(hf, sgl):
            self.assertEqual(len(hf_row), len(sgl_row))
            for h, s in zip(hf_row, sgl_row):
                self.assertLessEqual(abs(h - s), tol, f"HF={h:.6f} SGLang={s:.6f}")
            self.assertAlmostEqual(sum(sgl_row), 1.0, places=6)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_scores_match_hf_reference(self):
        """SGLang scores agree with HuggingFace within 1% tolerance."""
        label_token_ids = []
        tokenizer = AutoTokenizer.from_pretrained(
            _CAUSAL_LM_MODEL, trust_remote_code=True
        )
        for token in [" to", " the"]:
            label_token_ids.append(
                tokenizer(token, add_special_tokens=False)["input_ids"][0]
            )
        del tokenizer

        for query, items, item_first in [
            ("I pledge allegiance", ["", " to"], False),
            (" is a city", ["Tokyo", "Japan"], True),
        ]:
            with self.subTest(query=query):
                sgl = self.engine.score(
                    query=query,
                    items=items,
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                    item_first=item_first,
                ).scores
                hf = self._hf_scores(query, items, label_token_ids, item_first)
                self._assert_scores_close(hf, sgl)

    def test_request_avoids_decode_phase(self):
        """Internal request must have max_new_tokens=0, logprob=True, stream=False."""
        captured = []
        original = self.engine.tokenizer_manager.generate_request

        async def capturing_gen(req, request=None):
            captured.append(req)
            async for result in original(req, request):
                yield result

        with patch.object(
            self.engine.tokenizer_manager,
            "generate_request",
            side_effect=capturing_gen,
        ):
            self.engine.score(
                query="What is the capital of",
                items=["France", "Germany"],
                label_token_ids=[1, 2, 3],
                apply_softmax=True,
            )

        self.assertEqual(len(captured), 1)
        req = captured[0]

        if isinstance(req.sampling_params, list):
            max_new_tokens = req.sampling_params[0].get("max_new_tokens", 0)
        elif isinstance(req.sampling_params, dict):
            max_new_tokens = req.sampling_params.get("max_new_tokens", 0)
        else:
            max_new_tokens = getattr(req.sampling_params, "max_new_tokens", 0)

        self.assertEqual(max_new_tokens, 0)
        self.assertTrue(req.return_logprob)
        self.assertFalse(req.stream)

    # ------------------------------------------------------------------
    # Multi-item / batching
    # ------------------------------------------------------------------

    def test_score_batch_sizes(self):
        """Correct output count and shape for batch sizes 1, 2, 4, 8."""
        label_token_ids = [1, 2, 3]
        for n in [1, 2, 4, 8]:
            with self.subTest(n=n):
                scores = self.engine.score(
                    query="The test was",
                    items=[f"test {i}" for i in range(n)],
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), n)
                for row in scores:
                    self.assertEqual(len(row), len(label_token_ids))
                    self.assertTrue(all(isinstance(v, float) for v in row))
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_empty_items(self):
        """Empty items list → empty scores and zero prompt_tokens."""
        result = self.engine.score(
            query="Test query", items=[], label_token_ids=[1, 2], apply_softmax=True
        )
        self.assertEqual(len(result.scores), 0)
        self.assertEqual(result.prompt_tokens, 0)

    def test_score_without_softmax(self):
        """apply_softmax=False returns raw logits (not probability-constrained)."""
        scores = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad", "Neutral"],
            label_token_ids=[1, 2, 3],
            apply_softmax=False,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertEqual(len(row), 3)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_varying_label_token_sets(self):
        """Different label_token_ids lengths all produce correct-shaped output."""
        for n_labels in [1, 2, 4, 8]:
            with self.subTest(n_labels=n_labels):
                scores = self.engine.score(
                    query="Choose:",
                    items=["Option A", "Option B"],
                    label_token_ids=list(range(1, n_labels + 1)),
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), 2)
                for row in scores:
                    self.assertEqual(len(row), n_labels)
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_unicode(self):
        """Unicode query and items do not crash and produce valid scores."""
        scores = self.engine.score(
            query="选择最佳选项：",
            items=["选项A", "选项B", "选项C"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_deterministic(self):
        """Identical calls return numerically equivalent scores (within GPU float tolerance)."""
        kwargs = dict(query="Choose:", items=["A", "B", "C"], label_token_ids=[1, 2, 3])
        scores_a = self.engine.score(**kwargs).scores
        scores_b = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores_a), len(scores_b))
        for row_a, row_b in zip(scores_a, scores_b):
            self.assertEqual(len(row_a), len(row_b))
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=5)

    def test_score_error_handling(self):
        """Invalid argument types raise ValueError or TypeError."""
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query="Q", items=["X"], label_token_ids="bad", apply_softmax=True
            )
        with self.assertRaises((ValueError, TypeError)):
            self.engine.score(
                query="Q", items=None, label_token_ids=[1, 2], apply_softmax=True
            )


# ---------------------------------------------------------------------------
# SequenceClassification — single-item mode
# ---------------------------------------------------------------------------


class TestSeqClsScoring(CustomTestCase):
    """SequenceClassification scoring via Engine — no MIS delimiter.

    Uses json_model_override_args to load Qwen3-0.6B backbone weights into
    Qwen3ForSequenceClassification.  The classification head is randomly
    initialised; shape/pipeline correctness is what matters here.
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_score_shape(self):
        """Each item gets a score vector of length num_labels."""
        scores = self.engine.score(
            query="Rate each option:",
            items=["Option A", "Option B"],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 2)
        for i, row in enumerate(scores):
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_score_single_item_edge_case(self):
        """Single item in the list."""
        scores = self.engine.score(
            query="Evaluate:", items=["Only item"], apply_softmax=True
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_score_without_softmax(self):
        """Without softmax, returns raw logits (no probability constraints)."""
        scores = self.engine.score(
            query="Evaluate:", items=["Alpha", "Beta"], apply_softmax=False
        ).scores
        self.assertEqual(len(scores), 2)
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_deterministic(self):
        """Identical inputs yield near-identical scores (fp16 tolerance)."""
        kwargs = dict(query="Evaluate:", items=["alpha", "beta", "gamma"])
        scores1 = self.engine.score(**kwargs).scores
        scores2 = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores1), len(scores2))
        for s1, s2 in zip(scores1, scores2):
            for v1, v2 in zip(s1, s2):
                self.assertAlmostEqual(v1, v2, places=1)

    def test_score_tokenized_inputs(self):
        """Pre-tokenized query/items match text input scores."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_SEQCLS_MODEL)
        query, items = "Rate this:", ["Good", "Bad"]

        text_scores = self.engine.score(
            query=query, items=items, apply_softmax=True
        ).scores
        token_scores = self.engine.score(
            query=tok.encode(query),
            items=[tok.encode(i) for i in items],
            apply_softmax=True,
        ).scores

        self.assertEqual(len(text_scores), len(token_scores))
        for ts, ks in zip(text_scores, token_scores):
            for t, k in zip(ts, ks):
                self.assertAlmostEqual(t, k, places=4)

    def test_label_token_ids_ignored(self):
        """SeqCls models ignore label_token_ids — output width is always num_labels."""
        scores = self.engine.score(
            query="Evaluate:",
            items=["Test item"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)


# ---------------------------------------------------------------------------
# SequenceClassification — MIS (delimiter) mode
# ---------------------------------------------------------------------------


class TestSeqClsMISScoring(CustomTestCase):
    """SeqCls MIS: all items packed into one sequence separated by delimiter token.

    Uses --enable-mis which hardcodes delimiter token ID 9999.
    Basic pipeline correctness only — thorough MIS tests (parity,
    concurrency, advanced) live in test_multi_item_scoring.py.
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_mis_one_vector_per_item(self):
        """MIS produces exactly one score vector per item."""
        items = ["Option A", "Option B", "Option C"]
        scores = self.engine.score(
            query="Rate each option:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for i, row in enumerate(scores):
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_mis_single_item_edge_case(self):
        """Single item through MIS path."""
        scores = self.engine.score(
            query="Evaluate:", items=["Single item"], apply_softmax=True
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_mis_many_items(self):
        """10 items all return valid probability vectors."""
        items = [f"Item {i}" for i in range(10)]
        scores = self.engine.score(
            query="Classify each:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)


# ---------------------------------------------------------------------------
# SequenceClassification — MIS with many labels (tensor shape stress test)
# ---------------------------------------------------------------------------


class TestSeqClsMISAdvancedScoring(CustomTestCase):
    """SeqCls MIS with 12 labels — stresses the 2-D tensor path in score_and_pool.

    Kept in a separate class (own Engine instance) so it doesn't fight the
    2-label class-level engine for GPU memory.
    """

    NUM_LABELS = 12

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_mis=True,
            attention_backend="flashinfer",
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_many_labels_correct_shape(self):
        """5 items × 12 labels — each score vector has the right length."""
        items = [f"Item {i}" for i in range(5)]
        scores = self.engine.score(
            query="Classify:", items=items, apply_softmax=True
        ).scores
        self.assertEqual(len(scores), len(items))
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=3)
