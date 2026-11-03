"""CPU tests for the Cohere /v2/embed request and error contract."""

import json
import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.cohere.protocol import (
    CohereEmbedRequest,
    create_error_response,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCohereEmbedRequest(unittest.TestCase):
    def test_defaults_and_explicit_null_optionals(self):
        for values in (
            {},
            dict(
                model=None,
                input_type=None,
                output_dimension=None,
                max_tokens=None,
                embedding_types=None,
                images=None,
                inputs=None,
            ),
        ):
            with self.subTest(values=values):
                request = CohereEmbedRequest(texts=["hello", ""], **values)
                self.assertEqual(request.texts, ["hello", ""])
                self.assertIsNone(request.model)
                self.assertIsNone(request.embedding_types)
                self.assertEqual(request.truncate, "END")
                self.assertEqual(request.priority, 0)
                self.assertFalse(request.has_images)

    def test_exactly_one_nonempty_input_collection(self):
        invalid = [
            {},
            {"texts": None, "images": None, "inputs": None},
            {"texts": []},
            {"images": []},
            {"inputs": []},
            {"texts": ["a"], "images": ["data:image/png;base64,YQ=="]},
            {"texts": ["a"], "images": []},
            {"images": ["x"], "inputs": [{"content": [{"type": "text", "text": "a"}]}]},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                CohereEmbedRequest(**values)

    def test_input_types_and_positive_numeric_fields(self):
        for field, value in (
            ("texts", "not a list"),
            ("texts", [1]),
            ("images", [""]),
            ("inputs", ["not an object"]),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ValidationError),
            ):
                CohereEmbedRequest(**{field: value})
        for field, invalid_values in (
            ("max_tokens", (-2, 1.5, "wrong")),
            ("output_dimension", (0, -1, 1.5, "wrong")),
        ):
            for value in invalid_values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValidationError),
                ):
                    CohereEmbedRequest(texts=["a"], **{field: value})
        request = CohereEmbedRequest(texts=["a"], max_tokens=1, output_dimension=8)
        self.assertEqual((request.max_tokens, request.output_dimension), (1, 8))

    def test_max_tokens_accepts_vllm_sentinel_and_zero(self):
        for policy in ("END", "START", "NONE"):
            for value in (-1, 0):
                with self.subTest(policy=policy, value=value):
                    request = CohereEmbedRequest(
                        texts=["a"], truncate=policy, max_tokens=value
                    )
                    self.assertEqual(request.max_tokens, value)

    def test_truncation_and_output_enums(self):
        for direction in ("END", "START", "NONE"):
            request = CohereEmbedRequest(texts=["a"], truncate=direction)
            self.assertEqual(request.truncate, direction)
        for values in (
            {"truncate": "end"},
            {"truncate": None},
            {"embedding_types": ["int8"]},
            {"embedding_types": "float"},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                CohereEmbedRequest(texts=["a"], **values)
        request = CohereEmbedRequest(
            texts=["a"], embedding_types=["float", "base64", "binary", "ubinary"]
        )
        self.assertEqual(len(request.embedding_types), 4)

    def test_priority_signed_64_bit_bounds(self):
        for priority in (-(2**63), 0, 2**63 - 1):
            self.assertEqual(
                CohereEmbedRequest(texts=["a"], priority=priority).priority, priority
            )
        for priority in (-(2**63) - 1, 2**63):
            with self.subTest(priority=priority), self.assertRaises(ValidationError):
                CohereEmbedRequest(texts=["a"], priority=priority)

    def test_model_defined_input_type_is_not_a_fixed_enum(self):
        request = CohereEmbedRequest(texts=["a"], input_type="custom-task-name")
        self.assertEqual(request.input_type, "custom-task-name")

    def test_ordered_content_and_batch_boundaries(self):
        request = CohereEmbedRequest(
            inputs=[
                {
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": "image-a"}},
                        {"type": "text", "text": "between"},
                        {"type": "image_url", "image_url": {"url": "image-b"}},
                    ]
                },
                {"content": [{"type": "text", "text": "second sample"}]},
            ]
        )
        self.assertTrue(request.has_images)
        self.assertEqual(len(request.inputs), 2)
        self.assertEqual(
            [part.type for part in request.inputs[0].content],
            ["text", "image_url", "text", "image_url"],
        )
        self.assertEqual(request.inputs[0].content[3].image_url["url"], "image-b")
        self.assertEqual(request.inputs[1].content[0].text, "second sample")

    def test_content_payload_validation(self):
        for content in (
            [],
            [{"type": "text"}],
            [{"type": "image_url"}],
            [{"type": "image_url", "image_url": {}}],
            [{"type": "image_url", "image_url": {"url": ""}}],
            [{"type": "image_url", "image_url": "image-a"}],
            [{"type": "video", "video": "video-a"}],
        ):
            with self.subTest(content=content), self.assertRaises(ValidationError):
                CohereEmbedRequest(inputs=[{"content": content}])

    def test_images_and_text_only_inputs_detection(self):
        self.assertTrue(CohereEmbedRequest(images=["image-a", "image-b"]).has_images)
        self.assertFalse(
            CohereEmbedRequest(
                inputs=[{"content": [{"type": "text", "text": "a"}]}]
            ).has_images
        )

    def test_no_artificial_batch_size_limit(self):
        request = CohereEmbedRequest(texts=["a"] * 129)
        self.assertEqual(len(request.texts), 129)


class TestCohereErrorResponse(unittest.TestCase):
    def test_nested_error_and_null_param(self):
        response = create_error_response("invalid input")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            json.loads(response.body),
            {
                "error": {
                    "message": "invalid input",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
        )

    def test_explicit_status_type_and_param(self):
        response = create_error_response(
            "model unavailable",
            status_code=404,
            error_type="NotFoundError",
            param="model",
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            json.loads(response.body)["error"],
            {
                "message": "model unavailable",
                "type": "NotFoundError",
                "param": "model",
                "code": 404,
            },
        )


if __name__ == "__main__":
    unittest.main()
