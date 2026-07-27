import json
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=85, suite="full-1-npu-a3", nightly=True)

_CAUSAL_LM_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
_SEQCLS_MODEL = QWEN3_0_6B_WEIGHTS_PATH


class TestCausalLMScoring(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_CAUSAL_LM_MODEL,
            attention_backend="ascend",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def _hf_scores(self, query, items, label_token_ids, item_first=False):
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

    def test_scores_match_hf_reference(self):
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


class TestSeqClsScoring(CustomTestCase):
    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            attention_backend="ascend",
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


if __name__ == "__main__":
    unittest.main(verbosity=3)
