"""Tests for the return_pooled_hidden_states feature on the scoring API.

Covers both Engine-level (Python API) and HTTP-level (/v1/score) integration:

  TestPooledHiddenStatesEngine     — SeqCls model, single-item scoring
  TestPooledHiddenStatesHTTP       — HTTP layer serialization round-trip
  TestPooledHiddenStatesCausalLMRejection — CausalLM must reject the flag

Each test class spins up its own Engine or server so NPU memory is isolated.
"""

import json
import unittest

import requests
import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

_SEQCLS_MODEL = QWEN3_0_6B_WEIGHTS_PATH
_CAUSAL_LM_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
_NUM_LABELS = 4


# ---------------------------------------------------------------------------
# Engine — SeqCls pooled hidden states on NPU
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesEngine(CustomTestCase):
    """Testcase: Pooled hidden states via Engine Python API on NPU.
    Validates return_pooled_hidden_states on SeqCls models using Ascend NPU.
    Covers presence, shape, count, device, determinism, and score consistency.

    [Test Category] Functionality
    [Test Target] return_pooled_hidden_states on SeqCls (Engine API, NPU)
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
        """PHS returned and shaped correctly when flag is True."""
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
        """PHS is None when return_pooled_hidden_states=False."""
        result = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad"],
            return_pooled_hidden_states=False,
        )
        self.assertIsNone(result.pooled_hidden_states)

    def test_phs_shape_is_consistent(self):
        """All PHS tensors share the same hidden dimension."""
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
        """Returned tensors live on CPU (no NPU references leak to caller)."""
        result = self.engine.score(
            query="Check device:",
            items=["Test"],
            return_pooled_hidden_states=True,
        )
        for phs in result.pooled_hidden_states:
            self.assertEqual(str(phs.device), "cpu")

    def test_phs_deterministic(self):
        """Identical requests produce identical PHS tensors on NPU."""
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
        """PHS flag does not alter the classification scores."""
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
        """Pre-tokenized inputs also return PHS correctly on NPU."""
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
# CausalLM rejection on NPU
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesCausalLMRejection(CustomTestCase):
    """Testcase: CausalLM rejects return_pooled_hidden_states on NPU.
    Validates that CausalLM models raise ValueError when the flag is set.
    Covers rejection behavior and baseline scoring without the flag.

    [Test Category] Functionality
    [Test Target] return_pooled_hidden_states rejection on CausalLM (NPU)
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(model_path=_CAUSAL_LM_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_causal_lm_rejects_phs(self):
        """ValueError raised when requesting PHS from a CausalLM on NPU."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.score(
                query="Test",
                items=["Item"],
                label_token_ids=[1, 2],
                return_pooled_hidden_states=True,
            )
        self.assertIn("CausalLM", str(ctx.exception))

    def test_causal_lm_without_phs_still_works(self):
        """CausalLM scoring without PHS flag works fine on NPU."""
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
# HTTP layer on NPU
# ---------------------------------------------------------------------------


class TestPooledHiddenStatesHTTP(CustomTestCase):
    """Testcase: Pooled hidden states via HTTP /v1/score endpoint on NPU.
    Validates JSON serialization round-trip for return_pooled_hidden_states.
    Covers response structure, null handling, and item count matching.

    [Test Category] Integration
    [Test Target] return_pooled_hidden_states via HTTP /v1/score (NPU)
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
