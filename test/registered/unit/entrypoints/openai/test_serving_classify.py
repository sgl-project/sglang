"""Unit tests for srt/entrypoints/openai/serving_classify.py"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.entrypoints.openai.protocol import ClassifyRequest
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SEQUENCE_ARCHITECTURE = "Qwen3ForSequenceClassification"
TOKEN_ARCHITECTURE = "BertForTokenClassification"


def _serving(*, architecture, problem_type=None, num_labels=2):
    tokenizer_manager = Mock(
        model_config=SimpleNamespace(
            hf_config=PretrainedConfig(
                architectures=[architecture],
                problem_type=problem_type,
                id2label={i: f"class_{i}" for i in range(num_labels)},
            ),
            embedding_model_spec=resolve_embedding_model_spec(
                [architecture], is_embedding_requested=False, is_embedding_gemma=False
            ),
        ),
        server_args=None,
        served_model_name="test-classifier",
        request_logger=SimpleNamespace(log_requests=False),
    )
    return OpenAIServingClassify(
        tokenizer_manager=tokenizer_manager, template_manager=None
    )


def _classify(serving, *, request, outputs):
    async def generate_request(adapted_request, raw_request):
        serving.tokenizer_manager.last_request = adapted_request
        yield outputs

    serving.tokenizer_manager.generate_request = generate_request
    return asyncio.run(serving.handle_request(request=request, raw_request=None))


def _output(scores, *, num_tokens):
    return {"embedding": scores, "meta_info": {"prompt_tokens": num_tokens}}


class TestServingClassify(CustomTestCase):
    def _assert_probs(self, actual, expected):
        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(expected), rtol=0, atol=1e-6
        )

    def test_sequence_probabilities_follow_problem_type(self):
        """Multi-label models score classes independently; others keep softmax."""
        for problem_type, scores, expected_probs, expected_label in (
            (None, [2.0, 2.0], [0.5, 0.5], "class_0"),
            (
                "single_label_classification",
                [0.0, 1.0],
                [0.268941, 0.731059],
                "class_1",
            ),
            (
                "multi_label_classification",
                [2.0, 2.0, -2.0],
                [0.880797, 0.880797, 0.119203],
                "class_0",
            ),
        ):
            with self.subTest(problem_type=problem_type, scores=scores):
                serving = _serving(
                    architecture=SEQUENCE_ARCHITECTURE,
                    problem_type=problem_type,
                    num_labels=3,
                )
                response = _classify(
                    serving,
                    request=ClassifyRequest(input="some text"),
                    outputs=_output(scores, num_tokens=4),
                )
                item = response.data[0]
                self.assertEqual(item.label, expected_label)
                self.assertEqual(item.num_classes, len(scores))
                self._assert_probs(item.probs, expected_probs)

    def test_sequence_head_may_be_narrower_or_wider_than_id2label(self):
        """Sequence heads only need a label for their winning class."""
        for architecture, scores, expected_probs in (
            # A reward head returns one score against the two default labels.
            ("Qwen2ForRewardModel", [0.3], [1.0]),
            # This head is sized by its own config field, not by id2label.
            (
                "LlamaForClassification",
                [2.0, 1.0, 0.0],
                [0.665241, 0.244728, 0.090031],
            ),
        ):
            with self.subTest(architecture=architecture):
                response = _classify(
                    _serving(architecture=architecture),
                    request=ClassifyRequest(input="some text"),
                    outputs=_output(scores, num_tokens=4),
                )
                item = response.data[0]
                self.assertEqual(item.label, "class_0")
                self.assertEqual(item.num_classes, len(scores))
                self._assert_probs(item.probs, expected_probs)

    def test_token_classifier_returns_one_row_and_label_per_token(self):
        scores = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]
        for problem_type, expected_probs in (
            (
                "single_label_classification",
                [[0.5, 0.5], [0.731059, 0.268941], [0.119203, 0.880797]],
            ),
            (
                "multi_label_classification",
                [[0.5, 0.5], [0.731059, 0.5], [0.5, 0.880797]],
            ),
        ):
            with self.subTest(problem_type=problem_type):
                serving = _serving(
                    architecture=TOKEN_ARCHITECTURE, problem_type=problem_type
                )
                response = _classify(
                    serving,
                    request=ClassifyRequest(input=[101, 7592, 102]),
                    outputs=_output(scores, num_tokens=3),
                )
                self.assertEqual(
                    serving.tokenizer_manager.last_request.input_ids, [101, 7592, 102]
                )
                item = response.data[0]
                self.assertEqual(item.label, ["class_0", "class_0", "class_1"])
                self.assertEqual(item.num_classes, 2)
                self._assert_probs(item.probs, expected_probs)

        with self.subTest("single token, single class"):
            response = _classify(
                _serving(architecture=TOKEN_ARCHITECTURE, num_labels=1),
                request=ClassifyRequest(input=[7592]),
                outputs=_output([[2.0]], num_tokens=1),
            )
            item = response.data[0]
            self.assertEqual(item.label, ["class_0"])
            self.assertEqual(item.probs, [[1.0]])
            self.assertEqual(item.num_classes, 1)

    def test_batched_requests_keep_each_input_token_rows(self):
        serving = _serving(architecture=TOKEN_ARCHITECTURE)
        response = _classify(
            serving,
            request=ClassifyRequest(input=["first input", "second"]),
            outputs=[
                _output([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], num_tokens=3),
                _output([[2.0, 0.0]], num_tokens=1),
            ],
        )
        self.assertEqual(
            serving.tokenizer_manager.last_request.text, ["first input", "second"]
        )
        payload = response.model_dump()
        self.assertEqual([item["index"] for item in payload["data"]], [0, 1])
        self.assertEqual(
            [item["label"] for item in payload["data"]],
            [["class_1", "class_0", "class_0"], ["class_0"]],
        )
        self.assertEqual([item["num_classes"] for item in payload["data"]], [2, 2])
        self._assert_probs(
            payload["data"][0]["probs"],
            [[0.268941, 0.731059], [0.731059, 0.268941], [0.5, 0.5]],
        )
        self._assert_probs(payload["data"][1]["probs"], [[0.880797, 0.119203]])
        self.assertEqual(payload["usage"]["prompt_tokens"], 4)

    def test_scores_that_disagree_with_the_task_fail_closed(self):
        """Mis-shaped or unmapped scores return 500, not a 200 "Default" label."""
        for architecture, num_labels, output in (
            (SEQUENCE_ARCHITECTURE, 2, _output([], num_tokens=2)),
            (SEQUENCE_ARCHITECTURE, 2, _output([[0.0, 1.0]], num_tokens=2)),
            # The winning class has no id2label entry.
            (SEQUENCE_ARCHITECTURE, 2, _output([0.0, 1.0, 2.0], num_tokens=2)),
            # An aborted request carries no scores.
            (SEQUENCE_ARCHITECTURE, 2, {"meta_info": {"prompt_tokens": 2}}),
            # One pooled vector, as a backend that pools the sequence returns.
            (TOKEN_ARCHITECTURE, 2, _output([0.1, 0.2], num_tokens=2)),
            (TOKEN_ARCHITECTURE, 2, _output([[0.0, 1.0]], num_tokens=2)),
            (TOKEN_ARCHITECTURE, 2, _output([[0.0, 1.0], [0.0]], num_tokens=2)),
            # Two class scores per token against three labels.
            (TOKEN_ARCHITECTURE, 3, _output([[2.0, 0.0], [0.0, 2.0]], num_tokens=2)),
        ):
            with self.subTest(architecture=architecture, output=output):
                response = _classify(
                    _serving(architecture=architecture, num_labels=num_labels),
                    request=ClassifyRequest(input="two tokens"),
                    outputs=output,
                )
                self.assertEqual(response.status_code, 500)
                self.assertIn(
                    "classification scores", json.loads(response.body)["message"]
                )


if __name__ == "__main__":
    unittest.main()
