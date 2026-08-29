import json
import unittest
from types import SimpleNamespace

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tokenizer:
    # Hugging Face uses int(1e30) when a tokenizer has no defined maximum.
    model_max_length = int(1e30)

    def encode(self, text, add_special_tokens=True):
        return [101, 7592, 102] if add_special_tokens else [7592]


class OpenAIServingTokenizeTest(unittest.IsolatedAsyncioTestCase):
    async def test_uses_effective_context_length_in_serializable_response(self):
        tokenizer_manager = SimpleNamespace(
            tokenizer=_Tokenizer(),
            model_config=SimpleNamespace(context_len=65_536),
            server_args=SimpleNamespace(),
        )
        serving = OpenAIServingTokenize(tokenizer_manager)
        request = TokenizeRequest(prompt="hello")

        response = await serving._handle_non_streaming_request(
            request, request, raw_request=None
        )
        body = ORJSONResponse(content=response.model_dump()).body

        self.assertEqual(
            json.loads(body),
            {
                "tokens": [101, 7592, 102],
                "count": 3,
                "max_model_len": 65_536,
            },
        )


if __name__ == "__main__":
    unittest.main()
