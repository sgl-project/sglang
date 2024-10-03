import json
import unittest

import openai
import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestCacheReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(cls.model, cls.base_url, timeout=300, other_args=["--disable-cuda-graph",])
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")


    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(self, return_logprob=False, top_logprobs_num=0, n=1):
        response = requests.post(
            self.base_url + "/generate",
            # we use an uncommon start to minimise the chance that the cache is hit by chance
            json={
                "text": "_ The capital of France is",
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 128,
                    "n": n,
                    "stop_token_ids": [119690],
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        return response

    def run_openai(self, message):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=128,
        )
        return response

    def test_generate(self):
        response = self.run_decode()
        # print(response.json())
        # print("=" * 100)
        cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
        print(f"sglang first request cached_tokens: {cached_tokens}")
        print(f"sglang first request prompt_tokens: {int(response.json()['meta_info']['prompt_tokens'])}")
        # can't assure to be 0: depends on the initialisation request / if a template is used with the model
        assert cached_tokens < 5


    def cache_report_openai(self, message):
        response = self.run_openai(message)
        print(f"openai first request cached_tokens: {int(response.usage.cached_tokens)}")
        first_cached_tokens = int(response.usage.cached_tokens)
        # assert int(response.usage.cached_tokens) == 0
        response = self.run_openai(message)
        cached_tokens = int(response.usage.cached_tokens)
        print(f"openai second request usage: {response.usage}")
        print(f"openai second request cached_tokens: {cached_tokens}")
        assert cached_tokens > 0
        assert cached_tokens == int(response.usage.prompt_tokens)-1
        return first_cached_tokens

    def test_cache_report_openai(self):
        self.cache_report_openai("Introduce the capital of France.")

        first_cached_tokens_1 = self.cache_report_openai("How many sparrow do you need to lift a coconut?")

        first_cached_tokens_2 = self.cache_report_openai("* sing something about cats")
        # first request may not have 0 cached tokens, but if they only have the template in common they
        # should be the same once the cache is warmed up
        assert first_cached_tokens_1 == first_cached_tokens_2


if __name__ == "__main__":
    unittest.main()
