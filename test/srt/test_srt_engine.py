"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_4_sync_async_stream_combination
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            num_questions=200,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.3)

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

    def test_8_engine_return_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print("Model Path:", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            return_hidden_states=True,
            skip_tokenizer_init=True,
        )
        outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        engine.shutdown()

        for output in outputs:
            self.assertEqual(len(output["meta_info"]["hidden_states"]), 8)
            for hidden_state in output["meta_info"]["hidden_states"]:
                self.assertIsInstance(hidden_state, torch.Tensor)
        # Checks that splicing of the batch was done correctly
        self.assertGreater(
            outputs[1]["meta_info"]["hidden_states"][0].shape[0],
            outputs[0]["meta_info"]["hidden_states"][0].shape[0],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )

        for input_id, output in zip(input_ids, outputs):
            print(input_id, output["token_ids"])
            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["token_ids"][:-1]], device=model.device
                    ),
                    output_hidden_states=True,
                )
            print("=== HF Hiddens ===")
            print(hf_out["hidden_states"][-1][0])
            sg_hidden_states = torch.cat(
                [
                    i.unsqueeze(0) if len(i.shape) == 1 else i
                    for i in output["meta_info"]["hidden_states"]
                ]
            ).to("cuda")
            print("=== SRT Hiddens ===")
            print(sg_hidden_states)

            self.assertTrue(
                torch.allclose(
                    hf_out["hidden_states"][-1][0],
                    sg_hidden_states.to("cuda"),
                    atol=4e-1,
                    rtol=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
