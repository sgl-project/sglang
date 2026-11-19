"""Classification probabilities from model logits, without model loading."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from transformers import PreTrainedConfig

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import (
    ClassifyData,
    ClassifyRequest,
    ClassifyResponse,
)
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestServingClassify(CustomTestCase):
    def _serving(self, problem_type=None, num_labels=2, hf_config=None):
        if hf_config is None:
            hf_config = PreTrainedConfig(
                problem_type=problem_type,
                id2label={i: f"category_{i}" for i in range(num_labels)},
            )
        manager = Mock(
            model_config=SimpleNamespace(hf_config=hf_config),
            server_args=None,
            served_model_name="test-classifier",
            model_path="test-model-path",
            request_logger=SimpleNamespace(log_requests=False),
        )
        return OpenAIServingClassify(manager, None)

    def test_multi_label_probabilities(self):
        serving = self._serving("multi_label_classification", num_labels=3)
        result = serving._build_classify_response([{"embedding": [2.0, 2.0, -2.0]}])
        item = result.data[0]
        for actual, expected in zip(item.probs, [0.880797, 0.880797, 0.119203]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertGreater(sum(item.probs), 1.0)
        self.assertEqual(item.label, "category_0")
        self.assertEqual(item.labels, ["category_0", "category_1"])
        self.assertEqual(item.num_classes, 3)

    def test_single_label_and_unset_problem_types_use_softmax(self):
        for problem_type in (None, "single_label_classification"):
            with self.subTest(problem_type=problem_type):
                serving = self._serving(problem_type)
                item = serving._build_classify_response(
                    [{"embedding": [2.0, 2.0]}]
                ).data[0]
                self.assertEqual(item.probs, [0.5, 0.5])
                self.assertNotIn("labels", item.model_dump(mode="json"))

    def test_regression_returns_raw_scores(self):
        for embedding, scores in (
            ([-2.0, 3.0], [-2.0, 3.0]),
            ([-2.0], [-2.0]),
            (0.0, [0.0]),
        ):
            with self.subTest(embedding=embedding):
                serving = self._serving("regression", num_labels=len(scores))
                item = serving._build_classify_response(
                    [{"embedding": embedding}]
                ).data[0]
                self.assertEqual(item.probs, scores)
                self.assertEqual(item.num_classes, len(scores))
                self.assertNotIn("labels", item.model_dump(mode="json"))

    def test_missing_problem_type_keeps_softmax(self):
        serving = self._serving(
            hf_config=SimpleNamespace(id2label={0: "negative", 1: "positive"})
        )
        item = serving._build_classify_response([{"embedding": [0.0, 1.0]}]).data[0]
        self.assertAlmostEqual(item.probs[0], 0.268941, places=6)
        self.assertAlmostEqual(item.probs[1], 0.731059, places=6)
        self.assertEqual(item.label, "positive")

    def test_single_label_sigmoid(self):
        serving = self._serving("multi_label_classification", num_labels=1)
        item = serving._build_classify_response([{"embedding": [2.0]}]).data[0]
        self.assertAlmostEqual(item.probs[0], 0.880797, places=6)
        self.assertEqual(item.label, "category_0")
        self.assertEqual(item.num_classes, 1)

    def test_sigmoid_extreme_logits(self):
        serving = self._serving("multi_label_classification", num_labels=3)
        item = serving._build_classify_response(
            [{"embedding": [-1000.0, 0.0, 1000.0]}]
        ).data[0]
        self.assertEqual(item.probs, [0.0, 0.5, 1.0])
        self.assertEqual(item.label, "category_2")
        self.assertEqual(item.labels, ["category_1", "category_2"])

    def test_threshold_must_be_between_zero_and_one(self):
        for threshold in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValidationError):
                ClassifyRequest(input="example", threshold=threshold)

    def test_single_and_batched_requests(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                serving = self._serving("multi_label_classification")
                outputs = [
                    {"embedding": [-2.0, 2.0], "meta_info": {"prompt_tokens": 3}},
                    {"embedding": [2.0, -2.0], "meta_info": {"prompt_tokens": 5}},
                ]

                async def generate_request(adapted_request, raw_request):
                    self.assertEqual(adapted_request.text, request.input)
                    yield outputs if batched else outputs[0]

                serving.tokenizer_manager.generate_request = generate_request
                request = (
                    ClassifyRequest(input=["first", "second"], threshold=0.9)
                    if batched
                    else ClassifyRequest(input="first")
                )
                response = asyncio.run(serving.handle_request(request, None))
                self.assertIsInstance(response, ClassifyResponse)
                payload = response.model_dump()
                self.assertEqual(payload["model"], "test-classifier")
                self.assertEqual(payload["usage"]["prompt_tokens"], 8 if batched else 3)
                self.assertEqual(payload["usage"]["total_tokens"], 8 if batched else 3)
                self.assertEqual(payload["usage"]["completion_tokens"], 0)
                self.assertEqual(len(payload["data"]), 2 if batched else 1)
                for i, item in enumerate(payload["data"]):
                    self.assertEqual(item["index"], i)
                    self.assertEqual(item["label"], f"category_{1 - i}")
                    self.assertEqual(item["labels"], [] if batched else ["category_1"])
                    self.assertEqual(item["num_classes"], 2)
                    self.assertAlmostEqual(item["probs"][1 - i], 0.880797, places=6)
                    self.assertAlmostEqual(item["probs"][i], 0.119203, places=6)

    def test_token_multi_label_probabilities(self):
        serving = self._serving("multi_label_classification")
        result = serving._build_classify_response(
            [{"embedding": [[2.0, 2.0], [-2.0, 2.0], [0.0, -1000.0]]}]
        )
        item = result.data[0]
        self.assertEqual(item.label, ["category_0", "category_1", "category_0"])
        self.assertEqual(
            item.labels, [["category_0", "category_1"], ["category_1"], ["category_0"]]
        )
        self.assertEqual(item.num_classes, 2)
        expected = [[0.880797, 0.880797], [0.119203, 0.880797], [0.5, 0.0]]
        for row, expected_row in zip(item.probs, expected, strict=True):
            for actual, probability in zip(row, expected_row, strict=True):
                self.assertAlmostEqual(actual, probability, places=6)

    def test_token_softmax_over_labels(self):
        for problem_type in (None, "single_label_classification"):
            with self.subTest(problem_type=problem_type):
                serving = self._serving(problem_type)
                item = serving._build_classify_response(
                    [{"embedding": [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]}]
                ).data[0]
                self.assertEqual(item.label, ["category_0", "category_0", "category_1"])
                self.assertNotIn("labels", item.model_dump(mode="json"))
                self.assertEqual(item.num_classes, 2)
                expected = [[0.5, 0.5], [0.731059, 0.268941], [0.119203, 0.880797]]
                for row, expected_row in zip(item.probs, expected, strict=True):
                    self.assertAlmostEqual(sum(row), 1.0, places=6)
                    for actual, probability in zip(row, expected_row, strict=True):
                        self.assertAlmostEqual(actual, probability, places=6)

    def test_single_token_single_class_preserves_matrix(self):
        serving = self._serving("multi_label_classification", num_labels=1)
        item = serving._build_classify_response([{"embedding": [[2.0]]}]).data[0]
        self.assertEqual(item.label, ["category_0"])
        self.assertEqual(item.num_classes, 1)
        self.assertEqual(len(item.probs), 1)
        self.assertEqual(len(item.probs[0]), 1)
        self.assertAlmostEqual(item.probs[0][0], 0.880797, places=6)

    def test_single_and_batched_token_requests(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                serving = self._serving("multi_label_classification")
                outputs = [
                    {
                        "embedding": [[-2.0, 2.0], [2.0, -2.0], [0.0, 0.0]],
                        "meta_info": {"prompt_tokens": 3},
                    },
                    {"embedding": [[0.0, 1000.0]], "meta_info": {"prompt_tokens": 1}},
                ]

                async def generate_request(adapted_request, raw_request):
                    if batched:
                        self.assertEqual(adapted_request.text, request.input)
                    else:
                        self.assertEqual(adapted_request.input_ids, request.input)
                    yield outputs if batched else outputs[0]

                serving.tokenizer_manager.generate_request = generate_request
                request = (
                    ClassifyRequest(input=["first input", "second"], threshold=0.9)
                    if batched
                    else ClassifyRequest(input=[1, 2, 3])
                )
                app = FastAPI()
                app.state.openai_serving_classify = serving
                # Exercise the production route without starting model workers.
                app.router.routes.append(
                    next(
                        route
                        for route in http_server.app.routes
                        if getattr(route, "path", None) == "/v1/classify"
                    )
                )
                with TestClient(app) as client:
                    response = client.post("/v1/classify", json=request.model_dump())
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["usage"]["prompt_tokens"], 4 if batched else 3)
                self.assertEqual(payload["usage"]["total_tokens"], 4 if batched else 3)
                self.assertEqual(payload["usage"]["completion_tokens"], 0)
                self.assertEqual(len(payload["data"]), 2 if batched else 1)
                self.assertEqual(
                    payload["data"][0]["label"],
                    ["category_1", "category_0", "category_0"],
                )
                self.assertEqual(payload["data"][0]["probs"][2], [0.5, 0.5])
                self.assertEqual(
                    payload["data"][0]["labels"],
                    [[], [], []]
                    if batched
                    else [["category_1"], ["category_0"], ["category_0", "category_1"]],
                )
                self.assertAlmostEqual(
                    payload["data"][0]["probs"][0][1], 0.880797, places=6
                )
                if batched:
                    self.assertEqual(payload["data"][1]["index"], 1)
                    self.assertEqual(payload["data"][1]["label"], ["category_1"])
                    self.assertEqual(payload["data"][1]["labels"], [["category_1"]])
                    self.assertEqual(payload["data"][1]["probs"], [[0.5, 1.0]])
                self.assertTrue(
                    all(item["num_classes"] == 2 for item in payload["data"])
                )

    def test_response_schema_keeps_labels_and_probabilities_consistent(self):
        for label, labels, probs in (
            ("a", ["a", "b"], [0.5, 0.5]),
            (["a", "b"], [["a", "b"], ["b"]], [[0.5, 0.5], [0.25, 0.75]]),
        ):
            item = {
                "index": 0,
                "label": label,
                "labels": labels,
                "probs": probs,
                "num_classes": 2,
            }
            with self.subTest(label=label):
                self.assertEqual(ClassifyData.model_validate(item).model_dump(), item)
                wrong_label = ["a", "b"] if isinstance(label, str) else "a"
                with self.assertRaisesRegex(ValidationError, "probs must match"):
                    ClassifyData.model_validate({**item, "label": wrong_label})
                wrong_labels = [["a"], ["b"]] if isinstance(label, str) else ["a", "b"]
                with self.assertRaisesRegex(ValidationError, "labels must match"):
                    ClassifyData.model_validate({**item, "labels": wrong_labels})

    def test_empty_logits_keep_existing_fallback(self):
        for embedding in ([], [[], []]):
            with self.subTest(embedding=embedding):
                serving = self._serving("multi_label_classification")
                item = serving._build_classify_response(
                    [{"embedding": embedding}]
                ).data[0]
                self.assertEqual(item.label, "Default")
                self.assertEqual(item.probs, [1.0])
                self.assertEqual(item.num_classes, 1)


if __name__ == "__main__":
    unittest.main()
