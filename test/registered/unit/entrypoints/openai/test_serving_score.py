"""Unit tests for srt/entrypoints/openai/serving_score.py — no server.

Covers the /v1/score handler contracts:
- embedding-override payloads are converted to float32 tensors with the
  nested None entries (skip-this-item markers) preserved,
- absent overrides stay None instead of degrading to empty lists,
- pooled hidden states serialize back to lists with None entries intact,
- only ValueError is translated into an HTTP error response; anything else
  must propagate so real bugs are not masked as 400s.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch

from sglang.srt.entrypoints.openai.protocol import ScoringRequest
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_serving(score_result=None, score_side_effect=None) -> OpenAIServingScore:
    tokenizer_manager = Mock()
    tokenizer_manager.server_args = SimpleNamespace(
        tokenizer_metrics_allowed_custom_labels=None
    )
    tokenizer_manager.score_request = AsyncMock(
        return_value=score_result, side_effect=score_side_effect
    )
    return OpenAIServingScore(tokenizer_manager)


def _score_result(
    scores=None, prompt_tokens=7, pooled_hidden_states=None
) -> SimpleNamespace:
    return SimpleNamespace(
        scores=scores if scores is not None else [[0.25, 0.75]],
        prompt_tokens=prompt_tokens,
        pooled_hidden_states=pooled_hidden_states,
    )


def _handle(serving: OpenAIServingScore, request: ScoringRequest):
    return asyncio.run(
        serving._handle_non_streaming_request(request, request, raw_request=None)
    )


class TestEmbedOverrideConversion(CustomTestCase):
    def test_overrides_become_float32_tensors_with_none_items_preserved(self):
        """Derived property: the wire format is nested float lists, the
        internal contract is float32 tensors; a None entry in
        item_embed_overrides means "no override for this item" and must
        survive the conversion instead of crashing or being dropped."""
        serving = _make_serving(score_result=_score_result())
        request = ScoringRequest(
            model="m",
            query="q",
            items=["a", "b"],
            label_token_ids=[5],
            query_embed_overrides=[[1.0, 2.0]],
            item_embed_overrides=[[[3.0, 4.0]], None],
        )
        _handle(serving, request)

        kwargs = serving.tokenizer_manager.score_request.call_args.kwargs
        (query_tensor,) = kwargs["query_embed_overrides"]
        self.assertEqual(query_tensor.dtype, torch.float32)
        self.assertTrue(
            torch.equal(query_tensor, torch.tensor([1.0, 2.0], dtype=torch.float32))
        )

        item_overrides = kwargs["item_embed_overrides"]
        self.assertEqual(len(item_overrides), 2)
        self.assertIsNone(item_overrides[1])
        (item_tensor,) = item_overrides[0]
        self.assertEqual(item_tensor.dtype, torch.float32)
        self.assertTrue(
            torch.equal(item_tensor, torch.tensor([3.0, 4.0], dtype=torch.float32))
        )

    def test_absent_overrides_stay_none(self):
        """Negative branch: no overrides in the request must reach
        score_request as None (the "feature off" signal), not as []."""
        serving = _make_serving(score_result=_score_result())
        request = ScoringRequest(model="m", query="q", items=["a"], label_token_ids=[5])
        _handle(serving, request)

        kwargs = serving.tokenizer_manager.score_request.call_args.kwargs
        self.assertIsNone(kwargs["query_embed_overrides"])
        self.assertIsNone(kwargs["item_embed_overrides"])


class TestResponseSerialization(CustomTestCase):
    def test_pooled_hidden_states_round_trip_with_none_entries(self):
        """Derived property: pooled hidden states come back as tensors mixed
        with None (items that produced no pooled state); the JSON response
        must keep positional None entries so item alignment survives."""
        serving = _make_serving(
            score_result=_score_result(
                pooled_hidden_states=[torch.tensor([0.5, 1.5]), None]
            )
        )
        request = ScoringRequest(
            model="m",
            query="q",
            items=["a", "b"],
            label_token_ids=[5],
            return_pooled_hidden_states=True,
        )
        response = _handle(serving, request)
        body = json.loads(response.body)
        self.assertEqual(body["pooled_hidden_states"], [[0.5, 1.5], None])

    def test_scores_and_usage_fields(self):
        """Bookkeeping: clients parse scores/usage from the scoring response;
        prompt_tokens and total_tokens both mirror the internal token count
        (scoring has no completion tokens)."""
        serving = _make_serving(
            score_result=_score_result(scores=[[0.1, 0.9]], prompt_tokens=11)
        )
        request = ScoringRequest(model="m", query="q", items=["a"], label_token_ids=[5])
        response = _handle(serving, request)
        body = json.loads(response.body)
        self.assertEqual(body["scores"], [[0.1, 0.9]])
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["usage"]["prompt_tokens"], 11)
        self.assertEqual(body["usage"]["total_tokens"], 11)
        self.assertIsNone(body["pooled_hidden_states"])


class TestErrorHandling(CustomTestCase):
    def test_value_error_becomes_400_error_response(self):
        """Contract: user-input problems surface from score_request as
        ValueError and must map to a structured 400 error response."""
        serving = _make_serving(score_side_effect=ValueError("bad label ids"))
        request = ScoringRequest(model="m", query="q", items=["a"], label_token_ids=[5])
        response = _handle(serving, request)
        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertEqual(body["message"], "bad label ids")

    def test_unexpected_errors_propagate(self):
        """Negative branch: only ValueError is translated; an internal bug
        (e.g. RuntimeError) must propagate instead of masquerading as a 400
        client error."""
        serving = _make_serving(score_side_effect=RuntimeError("boom"))
        request = ScoringRequest(model="m", query="q", items=["a"], label_token_ids=[5])
        with self.assertRaises(RuntimeError):
            _handle(serving, request)


if __name__ == "__main__":
    unittest.main()
