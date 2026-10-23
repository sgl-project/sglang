import unittest

import numpy as np
import orjson
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    JsonSchemaResponseFormat,
    TopLogprob,
    UsageInfo,
)
from sglang.srt.utils.json_response import (
    SGLangORJSONResponse,
    dumps_json,
    model_json_response,
    orjson_response,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


class TestJSONResponseUtils(unittest.TestCase):
    def test_dumps_json_maps_non_finite_values_to_null(self):
        payload = {
            "neg_inf": float("-inf"),
            "pos_inf": float("inf"),
            "nan": float("nan"),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertIsNone(parsed["neg_inf"])
        self.assertIsNone(parsed["pos_inf"])
        self.assertIsNone(parsed["nan"])

    def test_dumps_json_supports_numpy_and_non_string_keys(self):
        payload = {
            1: np.array([1, 2, 3], dtype=np.int64),
            "scalar": np.float32(1.5),
        }
        parsed = orjson.loads(dumps_json(payload))

        self.assertEqual(parsed["1"], [1, 2, 3])
        self.assertAlmostEqual(parsed["scalar"], 1.5)

    def test_orjson_response_uses_expected_media_type(self):
        response = orjson_response({"value": float("-inf")}, status_code=201)
        parsed = orjson.loads(response.body)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.media_type, "application/json")
        self.assertIsNone(parsed["value"])

    def test_sglang_orjson_response_serializes_with_shared_options(self):
        response = SGLangORJSONResponse(content={"value": float("-inf")})
        parsed = orjson.loads(response.body)

        self.assertIsNone(parsed["value"])


def _chat_response_with_top_logprobs(logprob: float) -> ChatCompletionResponse:
    top = [
        TopLogprob(token="é", bytes=[195, 169], logprob=logprob),
        TopLogprob(token=" 中", bytes=[32, 228, 184, 173], logprob=-1e-5),
    ]
    return ChatCompletionResponse(
        id="req-0",
        created=0,
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="é 中"),
                logprobs=ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token="é",
                            bytes=[195, 169],
                            logprob=-0.25,
                            top_logprobs=top,
                        )
                    ]
                ),
                finish_reason="length",
                meta_info={
                    "output_token_logprobs": [(-0.25, 7, "é")],
                    "output_top_logprobs": [[(logprob, 7, "é"), (-1e16, 8, None)]],
                },
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )


class TestModelJSONResponse(unittest.TestCase):
    def test_matches_fastapi_encoding(self):
        response = _chat_response_with_top_logprobs(-3.5)
        expected = JSONResponse(jsonable_encoder(response)).body

        rendered = model_json_response(response)

        self.assertEqual(rendered.status_code, 200)
        self.assertEqual(rendered.media_type, "application/json")
        self.assertEqual(orjson.loads(rendered.body), orjson.loads(expected))

    def test_keeps_the_response_model_serializer(self):
        """The response's wrap serializer must still drop an unset `sglext`."""
        rendered = model_json_response(_chat_response_with_top_logprobs(-3.5))

        self.assertNotIn("sglext", orjson.loads(rendered.body))

    def test_applies_field_aliases_like_fastapi(self):
        schema = JsonSchemaResponseFormat(name="s", schema={"type": "object"})

        parsed = orjson.loads(model_json_response(schema).body)

        self.assertEqual(parsed, jsonable_encoder(schema))
        self.assertEqual(parsed["schema"], {"type": "object"})

    def test_maps_non_finite_logprobs_to_null(self):
        response = _chat_response_with_top_logprobs(float("-inf"))

        parsed = orjson.loads(model_json_response(response).body)

        choice = parsed["choices"][0]
        self.assertIsNone(
            choice["logprobs"]["content"][0]["top_logprobs"][0]["logprob"]
        )
        self.assertIsNone(choice["meta_info"]["output_top_logprobs"][0][0][0])

    def test_passes_non_models_through(self):
        error = orjson_response({"error": "bad"}, status_code=400)

        self.assertIs(model_json_response(error), error)


if __name__ == "__main__":
    unittest.main()
