"""CPU tests using known IEEE-754 bytes and explicit bit-packing fixtures."""

import base64
import unittest

from sglang.srt.entrypoints.cohere.encoding import (
    build_embed_response,
    build_typed_embeddings,
)
from sglang.srt.entrypoints.cohere.protocol import CohereEmbedRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCohereEncoding(unittest.TestCase):
    def test_base64_is_little_endian_float32(self):
        result = build_typed_embeddings([[1.0, -2.5, 0.0]], ["base64"])
        # 1.0 = 0x3f800000; -2.5 = 0xc0200000; 0.0 = 0x00000000.
        self.assertEqual(
            base64.b64decode(result.base64[0]),
            bytes.fromhex("0000803f000020c000000000"),
        )
        self.assertEqual(set(result.model_dump(exclude_none=True)), {"base64"})

    def test_binary_sign_zero_and_msb_order(self):
        result = build_typed_embeddings(
            [
                [-1.0, 0.0, -0.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                [-1.0] * 8,
                [1.0] * 8,
            ],
            ["binary", "ubinary"],
        )
        # 01110010 = 114; signed representation subtracts 128 (not int8 cast).
        self.assertEqual(result.ubinary, [[114], [0], [255]])
        self.assertEqual(result.binary, [[-14], [-128], [127]])

    def test_binary_preserves_byte_and_batch_order(self):
        rows = [[1.0] + [-1.0] * 14 + [1.0], [-1.0] * 16]
        result = build_typed_embeddings(rows, ["ubinary", "binary"])
        self.assertEqual(result.ubinary, [[128, 1], [0, 0]])
        self.assertEqual(result.binary, [[0, -127], [-128, -128]])

    def test_binary_requires_divisible_dimension(self):
        for encoding in ("binary", "ubinary"):
            with (
                self.subTest(encoding=encoding),
                self.assertRaisesRegex(ValueError, "multiple of 8"),
            ):
                build_typed_embeddings([[0.1, 0.2, 0.3]], [encoding])
        self.assertEqual(build_typed_embeddings([[0.1]], ["float"]).float, [[0.1]])

    def test_all_formats_share_the_supplied_result(self):
        rows = [[1.0] + [-1.0] * 7, [-1.0] * 8]
        result = build_typed_embeddings(rows, ["base64", "float", "ubinary", "binary"])
        self.assertEqual(result.float, rows)
        self.assertEqual(result.ubinary, [[128], [0]])
        self.assertEqual(result.binary, [[0], [-128]])
        self.assertEqual(len(result.base64), 2)
        self.assertEqual(
            set(result.model_dump(exclude_none=True)),
            {"base64", "float", "ubinary", "binary"},
        )

    def test_empty_batch_and_unknown_encoding(self):
        result = build_typed_embeddings([], ["float", "base64", "binary", "ubinary"])
        self.assertEqual(
            result.model_dump(),
            {"float": [], "base64": [], "binary": [], "ubinary": []},
        )
        with self.assertRaisesRegex(ValueError, "Unsupported embedding type"):
            build_typed_embeddings([[1.0]], ["int8"])


class TestCohereResponse(unittest.TestCase):
    @staticmethod
    def results():
        return [
            {"embedding": [1.0] * 8, "meta_info": {"prompt_tokens": 3}},
            {"embedding": [-1.0] * 8, "meta_info": {"prompt_tokens": 7}},
        ]

    def test_texts_echo_original_input_and_keep_result_order(self):
        request = CohereEmbedRequest(texts=["first", "second"])
        result = build_embed_response(request, self.results(), request_id="embd-test")
        self.assertEqual(
            result.model_dump(exclude_none=True),
            {
                "id": "embd-test",
                "embeddings": {"float": [[1.0] * 8, [-1.0] * 8]},
                "texts": ["first", "second"],
                "meta": {
                    "api_version": {"version": "2"},
                    "billed_units": {"input_tokens": 10, "image_tokens": 0},
                },
                "response_type": "embeddings_by_type",
            },
        )

    def test_null_and_empty_embedding_types_use_float_default(self):
        for types in (None, []):
            with self.subTest(types=types):
                request = CohereEmbedRequest(texts=["a", "b"], embedding_types=types)
                result = build_embed_response(request, self.results())
                self.assertEqual(
                    set(result.embeddings.model_dump(exclude_none=True)), {"float"}
                )
                self.assertTrue(result.id.startswith("embd-"))

    def test_image_batch_omits_texts_and_uses_image_accounting(self):
        request = CohereEmbedRequest(
            images=["image-a", "image-b"], embedding_types=["ubinary"]
        )
        payload = build_embed_response(request, self.results()).model_dump(
            exclude_none=True
        )
        self.assertNotIn("texts", payload)
        self.assertEqual(payload["embeddings"], {"ubinary": [[255], [0]]})
        self.assertEqual(
            payload["meta"]["billed_units"], {"input_tokens": 0, "image_tokens": 10}
        )

    def test_mixed_batch_uses_request_level_image_accounting(self):
        request = CohereEmbedRequest(
            inputs=[
                {"content": [{"type": "text", "text": "only text"}]},
                {"content": [{"type": "image_url", "image_url": {"url": "image-a"}}]},
            ]
        )
        payload = build_embed_response(request, self.results()).model_dump(
            exclude_none=True
        )
        self.assertNotIn("texts", payload)
        self.assertEqual(
            payload["meta"]["billed_units"], {"input_tokens": 0, "image_tokens": 10}
        )

    def test_text_only_inputs_do_not_echo_texts(self):
        request = CohereEmbedRequest(
            inputs=[
                {"content": [{"type": "text", "text": "first"}]},
                {"content": [{"type": "text", "text": "second"}]},
            ]
        )
        payload = build_embed_response(request, self.results()).model_dump(
            exclude_none=True
        )
        self.assertNotIn("texts", payload)
        self.assertEqual(
            payload["meta"]["billed_units"], {"input_tokens": 10, "image_tokens": 0}
        )


if __name__ == "__main__":
    unittest.main()
