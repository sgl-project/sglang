"""Compact PPLX serving keeps trained candidate ordering, media and API guards."""

import asyncio
import json
import math
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest
from sglang.srt.entrypoints.systemone.serving import SystemOneServing, _view
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tokenizer_manager_score_mixin import TokenizerManagerScoreMixin
from sglang.srt.utils import ImageData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class ReadoutManager:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []
        self.served_model_name = "pplx"

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        yield [
            {"embedding": row, "meta_info": {"prompt_tokens": 20 + i}}
            for i, row in enumerate(self.rows)
        ]


def handler(rows):
    result = SystemOneServing.__new__(SystemOneServing)
    result.tokenizer_manager = ReadoutManager(rows)
    result.decision_config = {"token_ids": list(range(1000, 1255)), "temperature": 2.0}
    result.compact_decision_readout = True
    result.joint_head_config = None
    result.pointer_head = None
    return result


class TestCompactDecider(CustomTestCase):
    def test_candidates_selected_before_temperature_and_softmax(self):
        row = [0.0, 2.0, -2.0] + [1000.0] * 252
        serving = handler([row, row])
        encoded = iter([([10, 11], [1000, 1001, 1002], []), ([12], [1001, 1000], [])])
        scores, tokens = asyncio.run(serving._score_compact_decider(encoded, None))
        expected = [math.exp(x) for x in (0.0, 1.0, -1.0)]
        total = sum(expected)
        for value, reference in zip(scores[0], expected):
            self.assertAlmostEqual(value, reference / total)
        self.assertAlmostEqual(scores[1][0], math.exp(1) / (1 + math.exp(1)))
        self.assertEqual(tokens, 41)
        request = serving.tokenizer_manager.requests[0]
        self.assertIsInstance(request, EmbeddingReqInput)
        self.assertEqual(request.input_ids, [[10, 11], [12]])
        self.assertIsNone(request.image_data)

    def test_images_remain_attached_to_each_rendered_prompt(self):
        image_a = ImageData(url="data:image/png;base64,first")
        image_b = ImageData(url="data:image/png;base64,second")
        images = [[image_a, image_b], [image_b]]
        serving = handler([[0.0] * 255, [0.0] * 255])
        encoded = iter(
            [([10, 11], [1000, 1001], images[0]), ([12], [1001, 1000], images[1])]
        )
        asyncio.run(serving._score_compact_decider(encoded, None))
        request = serving.tokenizer_manager.requests[0]
        self.assertEqual(request.image_data, images)
        request.normalize_batch_and_arguments()
        self.assertEqual(request[0].image_data, images[0])
        self.assertEqual(request[1].image_data, images[1])

    def test_systemone_response_uses_existing_answer_shapes_and_confidence(self):
        row = [0.0, 2.0, -2.0] + [1000.0] * 252
        serving = handler([row, row, row])
        request = SystemOneRequest(
            model="pplx",
            state="ticket",
            questions={
                "team": {
                    "type": "choice",
                    "criteria": {"billing": None, "support": None},
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "severity": {"type": "score", "criteria": ["low", "medium", "high"]},
            },
        )
        encoded = iter(
            [
                ([1], [1000, 1001], []),
                ([2], [1001, 1000], []),
                ([3], [1000, 1001, 1002], []),
            ]
        )
        views = [_view(q) for q in request.questions.values()]
        response = asyncio.run(
            serving._handle_non_streaming_request(encoded, (request, views), None)
        )
        body = json.loads(response.body)
        self.assertEqual(body["answers"]["team"]["choice"], "support")
        self.assertAlmostEqual(
            body["answers"]["urgent"]["noul"], math.exp(1) / (1 + math.exp(1))
        )
        self.assertEqual(body["usage"]["input_tokens"], 63)
        self.assertEqual(body["usage"]["output_tokens"], 0)
        self.assertIn("confidence", body["answers"]["severity"])
        for answer in body["answers"].values():
            self.assertNotIn("x_label_mass", answer)

    def test_disabled_mode_still_calls_the_existing_scorer(self):
        serving = handler([])
        serving.compact_decision_readout = False
        serving._score = mock.AsyncMock(
            return_value=(
                [],
                [],
                SimpleNamespace(
                    scores=[[0.25, 0.75]],
                    token_logprobs=[[-2.0, -1.0]],
                    prompt_tokens=12,
                ),
            )
        )
        serving._score_compact_decider = mock.AsyncMock(
            side_effect=AssertionError("compact path used")
        )
        request = SystemOneRequest(
            model="pplx",
            state="ticket",
            questions={"ok": {"type": "noul", "instructions": "OK?"}},
        )
        response = asyncio.run(
            serving._handle_non_streaming_request(
                iter([]), (request, [_view(request.questions["ok"])]), None
            )
        )
        self.assertEqual(json.loads(response.body)["answers"]["ok"]["noul"], 0.25)
        serving._score.assert_awaited_once()
        serving._score_compact_decider.assert_not_awaited()

    def test_generation_refused_before_model_execution_but_health_embedding_allowed(
        self,
    ):
        manager = SimpleNamespace(
            model_config=SimpleNamespace(
                compact_decision_readout=True,
                joint_head_config=None,
                pointer_head=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "do not generate"):
            TokenizerManager._validate_joint_schema_request(
                manager, GenerateReqInput(input_ids=[1]), [1]
            )
        TokenizerManager._validate_joint_schema_request(
            manager, EmbeddingReqInput(input_ids=[1]), [1]
        )
        with self.assertRaisesRegex(ValueError, "last-token"):
            TokenizerManager._validate_joint_schema_request(
                manager,
                EmbeddingReqInput(input_ids=[1], token_indices_to_pool=[0]),
                [1],
            )

    def test_raw_score_cannot_silently_ignore_vocabulary_labels(self):
        manager = SimpleNamespace(
            is_generation=False,
            model_config=SimpleNamespace(compact_decision_readout=True),
        )
        for kwargs in (
            {"label_token_ids": [1000, 1001]},
            {"apply_softmax": True},
            {"score_extraction_token_id": 7},
            {"return_token_logprobs": True},
        ):
            with self.subTest(kwargs), self.assertRaisesRegex(ValueError, "raw logits"):
                asyncio.run(
                    TokenizerManagerScoreMixin.score_request(
                        manager, items=[[1]], **kwargs
                    )
                )

    def test_invalid_calibration_and_wrong_readout_shape_are_rejected(self):
        serving = handler([[0.0] * 254])
        with self.assertRaisesRegex(RuntimeError, "number of codes"):
            asyncio.run(
                serving._score_compact_decider(iter([([1], [1000, 1001], [])]), None)
            )
        serving.decision_config["temperature"] = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            asyncio.run(
                serving._score_compact_decider(iter([([1], [1000, 1001], [])]), None)
            )


if __name__ == "__main__":
    unittest.main()
