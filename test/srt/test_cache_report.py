import asyncio
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestCacheReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.min_cached = 5
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=[
                "--chunked-prefill-size=40",
                "--enable-cache-report",
            ],
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")
        cls.aclient = openai.AsyncClient(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

        usage = cls.run_openai(cls, "1").usage
        # we can assume that our request is of size 1, plus the total template size
        # ideally we would like to know the begin size / end size of the template to be more precise
        total_template_size = usage.prompt_tokens - 1
        print(f"template size: {total_template_size}")
        usage2 = cls.run_openai(cls, "2").usage
        assert usage2.prompt_tokens_details.cached_tokens <= total_template_size
        cls.min_cached = max(
            usage2.prompt_tokens_details.cached_tokens,
            total_template_size - usage2.prompt_tokens_details.cached_tokens,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
            max_tokens=100,
        )
        return response

    async def run_openai_async(self, message):
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=100,
        )
        return response

    def cache_report_openai(self, message):
        response = self.run_openai(message)
        print(
            f"openai first request cached_tokens: {int(response.usage.prompt_tokens_details.cached_tokens)}"
        )
        first_cached_tokens = int(response.usage.prompt_tokens_details.cached_tokens)
        # assert int(response.usage.cached_tokens) == 0
        assert first_cached_tokens < self.min_cached
        response = self.run_openai(message)
        cached_tokens = int(response.usage.prompt_tokens_details.cached_tokens)
        print(f"openai second request cached_tokens: {cached_tokens}")
        assert cached_tokens > 0
        assert cached_tokens == int(response.usage.prompt_tokens) - 1
        return first_cached_tokens

    async def cache_report_openai_async(self, message):
        response = await self.run_openai_async(message)
        cached_tokens = int(response.usage.prompt_tokens_details.cached_tokens)
        prompt_tokens = int(response.usage.prompt_tokens)
        return cached_tokens, prompt_tokens

    def test_generate(self):
        print("=" * 100)
        response = self.run_decode()
        # print(response.json())
        cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
        print(f"sglang first request cached_tokens: {cached_tokens}")
        print(
            f"sglang first request prompt_tokens: {int(response.json()['meta_info']['prompt_tokens'])}"
        )
        # can't assure to be 0: depends on the initialisation request / if a template is used with the model
        assert cached_tokens < self.min_cached
        response = self.run_decode()
        cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
        print(f"sglang second request cached_tokens: {cached_tokens}")
        print(
            f"sglang second request prompt_tokens: {int(response.json()['meta_info']['prompt_tokens'])}"
        )
        assert cached_tokens == int(response.json()["meta_info"]["prompt_tokens"]) - 1

    def test_cache_split_prefill_openai(self):
        print("=" * 100)
        self.cache_report_openai(
            "€ This is a very long and unique text that should not be already cached, the twist is"
            " that it should be longer than the chunked-prefill-size, so it should be split among"
            " several prefill requests. Still, it shouldn't be cached"
        )

    def test_cache_report_openai(self):
        print("=" * 100)
        # warm up the cache, for the template
        self.run_openai("Introduce the capital of France.")

        first_cached_tokens_1 = self.run_openai(
            "How many sparrow do you need to lift a coconut?"
        ).usage.prompt_tokens_details.cached_tokens

        usage_2 = self.run_openai("* sing something about cats").usage
        first_cached_tokens_2 = usage_2.prompt_tokens_details.cached_tokens
        # first request may not have 0 cached tokens, but if they only have the template in common they
        # should be the same once the cache is warmed up
        assert first_cached_tokens_1 == first_cached_tokens_2

        resp = self.run_openai("* sing something about cats and dogs")
        print(resp.usage)

        resp = self.run_openai("* sing something about cats, please")
        print(resp.usage)
        assert (
            resp.usage.prompt_tokens_details.cached_tokens
            >= usage_2.prompt_tokens - self.min_cached
        )

    def test_cache_report_openai_async(self):
        print("=" * 100)

        async def run_test():
            task0 = asyncio.create_task(
                self.cache_report_openai_async(
                    "first request, to start the inference and let the next two request be started in the same batch"
                )
            )
            await asyncio.sleep(0.05)  # to force the first request to be started first
            task1 = asyncio.create_task(
                self.cache_report_openai_async(
                    "> can the same batch parallel request use the cache?"
                )
            )
            task2 = asyncio.create_task(
                self.cache_report_openai_async(
                    "> can the same batch parallel request use the cache?"
                )
            )
            result0, result1, result2 = await asyncio.gather(task0, task1, task2)

            cached_tokens0, prompt_tokens0 = result0
            cached_tokens1, prompt_tokens1 = result1
            cached_tokens2, prompt_tokens2 = result2

            print(
                f"Async request 0 - Cached tokens: {cached_tokens0}, Prompt tokens: {prompt_tokens0}"
            )
            print(
                f"Async request 1 - Cached tokens: {cached_tokens1}, Prompt tokens: {prompt_tokens1}"
            )
            print(
                f"Async request 2 - Cached tokens: {cached_tokens2}, Prompt tokens: {prompt_tokens2}"
            )

            # Assert that no requests used the cache (becausefirst is alone, and the next two are in the same batch)
            # If a new optimisation limiting starting request with same prefix at the same time was added
            # to maximise the cache hit, this would not be true
            assert cached_tokens1 == cached_tokens2 == cached_tokens0

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
