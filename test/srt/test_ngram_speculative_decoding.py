import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List

import aiohttp
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    0.8,
]


class TestNgramSpeculativeDecodingBase(CustomTestCase):

    model = DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.79  # derived tests need to override this
    spec_decode_threshold = 1.8  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=4,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestNgramSpeculativeDecodingTriton(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestNgramSpeculativeDecodingFlashinfer(TestNgramSpeculativeDecodingBase):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


class TestNgramSpeculativeDecodingPaged(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "64",
        ]


class TestNgramSpeculativeBatchGeneration(TestNgramSpeculativeDecodingBase):
    model = DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "4",
            "--speculative-algorithm",
            "NGRAM",
            "--speculative-num-draft-tokens",
            "8",
            "--mem-fraction-static",
            "0.7",
            "--skip-server-warmup",
            "--dtype",
            "float16",
            "--speculative-batch-size-threshold",
            "2",
        ]

    def test_batch_generation(self):
        """
        We gradually send over requests every 10 tokens per request to mimic the increase
        in the running batch size.
        """
        requests.get(self.base_url + "/flush_cache")

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = {"temperature": 0, "max_new_tokens": 100}
        tokenizer = get_tokenizer(self.model)

        outputs = asyncio.run(
            async_stream_ramp_up_http(
                self.base_url,
                prompts,
                sampling_params,
                tokens_until_next_request=10,
                tokenizer=tokenizer,
            )
        )

        self.assertEqual(len(outputs), len(prompts))
        for output in outputs:
            self.assertTrue(output["text"].strip())
            self.assertIsNotNone(output["meta_info"])
            self.assertIn("spec_verify_ct", output["meta_info"])

        spec_verify_cts = [item["meta_info"]["spec_verify_ct"] for item in outputs]
        self.assertEqual(spec_verify_cts, [18, 9, 10, 18])


async def async_stream_ramp_up_http(
    base_url: str,
    prompts: List[str],
    sampling_params: Dict,
    tokens_until_next_request: int,
    tokenizer,
) -> List[Dict[str, Any]]:
    outputs = [{"text": "", "meta_info": None} for _ in prompts]
    token_counts = [0] * len(prompts)
    started = [False] * len(prompts)
    tasks: Dict[int, asyncio.Task] = {}
    queue: asyncio.Queue = asyncio.Queue()
    last_started = 0
    next_to_launch = 1

    async def stream_one(idx: int):
        pos = 0
        try:
            async for payload in _stream_server_events(
                base_url, prompts[idx], sampling_params
            ):
                chunk_text = payload.get("text", "")
                if not chunk_text:
                    continue
                cleaned_chunk = chunk_text[pos:]
                if not cleaned_chunk:
                    continue
                pos = len(chunk_text)
                outputs[idx]["text"] += cleaned_chunk
                token_counts[idx] = len(
                    tokenizer.encode(outputs[idx]["text"], truncation=False)
                )
                meta_info = payload.get("meta_info")
                if meta_info:
                    outputs[idx]["meta_info"] = meta_info
                await queue.put(("chunk", idx))
        except Exception as exc:  # pragma: no cover - surfaced via queue
            await queue.put(("error", idx, exc))
            return
        await queue.put(("done", idx))

    def launch_next():
        nonlocal next_to_launch, last_started
        if next_to_launch < len(prompts) and not started[next_to_launch]:
            started[next_to_launch] = True
            last_started = next_to_launch
            tasks[next_to_launch] = asyncio.create_task(stream_one(next_to_launch))
            next_to_launch += 1

    started[0] = True
    tasks[0] = asyncio.create_task(stream_one(0))
    finished = 0

    while finished < len(prompts):
        item = await queue.get()
        typ = item[0]
        if typ == "chunk":
            _, idx = item
            if idx == last_started and token_counts[idx] >= tokens_until_next_request:
                launch_next()
        elif typ == "done":
            _, idx = item
            finished += 1
            if idx == last_started:
                launch_next()
        elif typ == "error":
            _, _, exc = item
            for task in tasks.values():
                task.cancel()
            raise exc

    await asyncio.gather(*tasks.values())
    return outputs


async def _stream_server_events(base_url: str, prompt: str, sampling_params: Dict):
    url = base_url + "/generate"
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "stream": True,
    }
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk, _ in response.content.iter_chunks():
                buffer += chunk.decode("utf-8")
                while True:
                    start = buffer.find("data:")
                    if start == -1:
                        break
                    end = buffer.find("\n\n", start)
                    if end == -1:
                        break
                    line = buffer[start:end].strip()
                    buffer = buffer[end + 2 :]
                    if line == "data: [DONE]":
                        return
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    yield json.loads(data_str)


if __name__ == "__main__":
    unittest.main()
