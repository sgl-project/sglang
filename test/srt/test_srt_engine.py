"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_3_sync_streaming_combination
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

import torch

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.test.few_shot_gsm8k_engine import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)


class TestSRTEngine(unittest.TestCase):

    def test_1_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(model_path=model_path, random_seed=42)
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(model_path=model_path, random_seed=42)
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_2_engine_multiple_generate(self):
        # just to ensure there is no issue running multiple generate calls
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(model_path=model_path, random_seed=42)
        engine.generate(prompt, sampling_params)
        engine.generate(prompt, sampling_params)
        engine.shutdown()

    def test_3_sync_streaming_combination(self):

        prompt = "AI safety is..."
        sampling_params = {"temperature": 0.8, "top_p": 0.95}

        async def async_streaming(engine):

            generator = await engine.async_generate(
                prompt, sampling_params, stream=True
            )

            async for output in generator:
                print(output["text"], end="", flush=True)
            print()

        # Create an LLM.
        llm = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        )

        # 1. sync + non streaming
        print("\n\n==== 1. sync + non streaming ====")
        output = llm.generate(prompt, sampling_params)

        print(output["text"])

        # 2. sync + streaming
        print("\n\n==== 2. sync + streaming ====")
        output_generator = llm.generate(prompt, sampling_params, stream=True)
        for output in output_generator:
            print(output["text"], end="", flush=True)
        print()

        loop = asyncio.get_event_loop()
        # 3. async + non_streaming
        print("\n\n==== 3. async + non streaming ====")
        output = loop.run_until_complete(llm.async_generate(prompt, sampling_params))
        print(output["text"])

        # 4. async + streaming
        print("\n\n==== 4. async + streaming ====")
        loop.run_until_complete(async_streaming(llm))

        llm.shutdown()

    def test_4_gsm8k(self):

        args = SimpleNamespace(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            local_data_path=None,
            num_shots=5,
            num_questions=200,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.3)

    def test_5_prompt_input_ids_consistency(self):
        prompt = "The capital of UK is"

        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        engine = sgl.Engine(
            model_path=model_path, random_seed=42, disable_radix_cache=True
        )
        sampling_params = {"temperature": 0, "max_new_tokens": 8}
        out1 = engine.generate(prompt, sampling_params)["text"]

        tokenizer = get_tokenizer(model_path)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
            "text"
        ]

        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_6_engine_runtime_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST

        engine = sgl.Engine(model_path=model_path, is_embedding=True, random_seed=42)
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(model_path=model_path, is_embedding=True, random_seed=42)
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_7_engine_cpu_offload(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
            cpu_offload_gb=3,
        )
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_8_engine_offline_throughput(self):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        self.assertGreater(result["total_throughput"], 3000)


if __name__ == "__main__":
    unittest.main()
