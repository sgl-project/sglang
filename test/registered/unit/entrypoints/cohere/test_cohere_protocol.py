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
        for value in (-1, 0):
            with self.subTest(value=value):
                request = CohereEmbedRequest(texts=["a"], max_tokens=value)
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

    def test_status_selects_type_and_server_errors_hide_detail(self):
        for status, expected_type, expected_message in (
            (404, "NotFoundError", "detail"),
            (429, "BadRequestError", "detail"),
            (500, "InternalServerError", "Internal server error"),
            (503, "InternalServerError", "Internal server error"),
        ):
            with self.subTest(status=status):
                response = create_error_response(
                    "detail", status_code=status, param="model"
                )
                self.assertEqual(response.status_code, status)
                self.assertEqual(
                    json.loads(response.body)["error"],
                    {
                        "message": expected_message,
                        "type": expected_type,
                        "param": "model",
                        "code": status,
                    },
                )


if __name__ == "__main__":
    unittest.main()
