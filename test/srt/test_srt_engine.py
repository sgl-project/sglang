"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_4_sync_async_stream_combination
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
    CustomTestCase,
)


class TestSRTEngine(CustomTestCase):

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

    def test_2_engine_runtime_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST

        engine = sgl.Engine(model_path=model_path, is_embedding=True, random_seed=42)
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(model_path=model_path, is_embedding=True, random_seed=42)
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_3_engine_token_ids_consistency(self):
        # just to ensure there is no issue running multiple generate calls
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path, random_seed=42, disable_radix_cache=True
        )
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

    def test_4_sync_async_stream_combination(self):
        prompt = "AI safety is"
        sampling_params = {"temperature": 0.8, "top_p": 0.95}

        # Create an LLM.
        llm = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        )

        if True:
            # 1. sync + non streaming
            print("\n\n==== 1. sync + non streaming ====")
            output = llm.generate(prompt, sampling_params)
            print(output["text"])

            # 2. sync + streaming
            print("\n\n==== 2. sync + streaming ====")
            output_generator = llm.generate(prompt, sampling_params, stream=True)
            offset = 0
            for output in output_generator:
                print(output["text"][offset:], end="", flush=True)
                offset = len(output["text"])
            print()

        if True:
            loop = asyncio.get_event_loop()
            # 3. async + non_streaming
            print("\n\n==== 3. async + non streaming ====")
            output = loop.run_until_complete(
                llm.async_generate(prompt, sampling_params)
            )
            print(output["text"])

            # 4. async + streaming
            async def async_streaming(engine):
                generator = await engine.async_generate(
                    prompt, sampling_params, stream=True
                )

                offset = 0
                async for output in generator:
                    print(output["text"][offset:], end="", flush=True)
                    offset = len(output["text"])
                print()

            print("\n\n==== 4. async + streaming ====")
            loop.run_until_complete(async_streaming(llm))

        llm.shutdown()

    def test_5_gsm8k(self):

        args = SimpleNamespace(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            local_data_path=None,
            num_shots=5,
            num_questions=1400,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.33)

    def test_6_engine_cpu_offload(self):
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

    def test_7_engine_offline_throughput(self):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        self.assertGreater(result["total_throughput"], 3000)

    def test_8_engine_async_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST

        engine = sgl.Engine(
            model_path=model_path,
            is_embedding=True,
            random_seed=42,
            disable_radix_cache=True,
        )

        # Get sync and async embeddings
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        loop = asyncio.get_event_loop()
        out2 = torch.tensor(
            loop.run_until_complete(engine.async_encode(prompt))["embedding"]
        )

        engine.shutdown()

        print("\n==== Shapes ====")
        print(f"sync shape: {out1.shape}")
        print(f"async shape: {out2.shape}")

        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-5, rtol=1e-3),
            "Sync and async embeddings are not equal within tolerance",
        )


if __name__ == "__main__":
    unittest.main()
