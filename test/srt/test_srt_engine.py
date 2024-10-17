"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_3_sync_streaming_combination
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

import sglang as sgl
from sglang.test.few_shot_gsm8k_engine import run_eval
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestSRTEngine(unittest.TestCase):

    def test_1_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(model_path=model_path, random_seed=42, log_level="error")
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(model_path=model_path, random_seed=42)
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        assert out1 == out2, f"{out1} != {out2}"

    def test_2_engine_multiple_generate(self):
        # just to ensure there is no issue running multiple generate calls
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(model_path=model_path, random_seed=42, log_level="error")
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
            model_path=DEFAULT_MODEL_NAME_FOR_TEST,
            log_level="error",
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
            model_path=DEFAULT_MODEL_NAME_FOR_TEST,
            local_data_path=None,
            num_shots=5,
            num_questions=200,
        )

        metrics = run_eval(args)
        assert metrics["accuracy"] > 0.7


if __name__ == "__main__":
    unittest.main()
