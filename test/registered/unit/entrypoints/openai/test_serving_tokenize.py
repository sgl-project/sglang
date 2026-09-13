import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MASK_ID = 156895


class _Tokenizer:
    model_max_length = 128

    def encode(self, text, *, add_special_tokens):
        return [len(text), MASK_ID]


class _TokenizerManager:
    def __init__(self):
        self.server_args = SimpleNamespace()
        self.tokenizer = _Tokenizer()

    def normalize_dllm_prompt_token_ids(self, input_ids):
        return [
            replacement_id
            for token_id in input_ids
            for replacement_id in ([31, 32] if token_id == MASK_ID else [token_id])
        ]


class TestOpenAIServingTokenize(unittest.TestCase):
    def test_text_prompts_return_normalized_reusable_ids(self):
        serving = OpenAIServingTokenize(_TokenizerManager())
        cases = (
            ("one", [3, 31, 32], 3),
            (["one", "four"], [[3, 31, 32], [4, 31, 32]], [3, 3]),
        )

        for prompt, expected_tokens, expected_count in cases:
            with self.subTest(prompt=prompt):
                request = TokenizeRequest(prompt=prompt)
                response = asyncio.run(
                    serving._handle_non_streaming_request(request, request, None)
                )

                self.assertEqual(response.tokens, expected_tokens)
                self.assertEqual(response.count, expected_count)


if __name__ == "__main__":
    unittest.main()
