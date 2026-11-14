import asyncio
import unittest
from types import SimpleNamespace

import orjson

from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, chunks):
        self.served_model_name = "test-model"
        self._chunks = chunks

    async def generate_request(self, obj, request=None):
        for chunk in self._chunks:
            yield chunk


async def _collect_stream(streaming_response):
    items = []
    async for raw in streaming_response.body_iterator:
        items.append(orjson.loads(raw))
    return items


class TestOllamaServing(CustomTestCase):
    def test_convert_options_sentinel_values_pass_sampling_params_verify(self):
        serving = OllamaServing(tokenizer_manager=SimpleNamespace())

        for num_predict in (-1, -2, None):
            with self.subTest(num_predict=num_predict):
                sp_dict = serving._convert_options_to_sampling_params(
                    {"num_predict": num_predict, "top_k": 0}
                )
                self.assertEqual(sp_dict["max_new_tokens"], 2048)
                self.assertEqual(sp_dict["top_k"], -1)
                sp = SamplingParams(**sp_dict)
                sp.verify(vocab_size=32000)

    def test_done_reason_reflects_length_finish_reason(self):
        for finish_type, expected_reason in (("length", "length"), ("stop", "stop")):
            chunks = [
                {
                    "text": "hello world",
                    "meta_info": {
                        "finish_reason": {"type": finish_type},
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                    },
                }
            ]
            serving = OllamaServing(tokenizer_manager=_FakeTokenizerManager(chunks))
            req = GenerateReqInput(text="hi", sampling_params={"max_new_tokens": 2})

            chat_resp = asyncio.run(
                serving._generate_chat_response(req, None, "test-model")
            )
            self.assertEqual(chat_resp.done_reason, expected_reason)

            gen_resp = asyncio.run(
                serving._generate_generate_response(req, None, "test-model")
            )
            self.assertEqual(gen_resp.done_reason, expected_reason)

            stream_chat = asyncio.run(
                serving._stream_chat_response(req, None, "test-model")
            )
            chat_chunks = asyncio.run(_collect_stream(stream_chat))
            self.assertEqual(chat_chunks[-1]["done_reason"], expected_reason)

            stream_gen = asyncio.run(
                serving._stream_generate_response(req, None, "test-model")
            )
            gen_chunks = asyncio.run(_collect_stream(stream_gen))
            self.assertEqual(gen_chunks[-1]["done_reason"], expected_reason)


if __name__ == "__main__":
    unittest.main()
