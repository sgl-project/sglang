"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_4_sync_async_stream_combination
"""

import asyncio
import json
import unittest

import torch
from transformers import AutoModelForCausalLM

import sglang as sgl
from sglang.benchmark.offline_throughput import BenchArgs, throughput_test
from sglang.srt.hf_transformers import get_tokenizer
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=293, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=261, suite="stage-b-test-1-gpu-small-amd")


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

    def test_9_engine_positional_embed_overrides(self):
        # The embeddings decide the output: another prompt's embeddings over these
        # ids give its tokens.
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        sampling_params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
        tokenizer = get_tokenizer(model_path)
        input_ids = tokenizer.encode("The capital of France is")
        other_ids = tokenizer.encode("Soup needs salt and also")
        self.assertEqual(len(input_ids), len(other_ids))
        embed_tokens = AutoModelForCausalLM.from_pretrained(
            model_path
        ).get_input_embeddings()
        with torch.no_grad():
            other_embeds = embed_tokens(torch.tensor(other_ids)).float()
        overrides = PositionalEmbeds(
            embeds=other_embeds, positions=list(range(len(other_ids)))
        )

        # Overrides always prefill eagerly, so the ids must too for equal outputs.
        engine = sgl.Engine(
            model_path=model_path, random_seed=42, disable_prefill_cuda_graph=True
        )
        plain = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        expected = engine.generate(input_ids=other_ids, sampling_params=sampling_params)
        out1 = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            positional_embed_overrides=overrides,
        )
        out2 = asyncio.get_event_loop().run_until_complete(
            engine.async_generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                positional_embed_overrides=overrides,
            )
        )
        engine.shutdown()

        self.assertNotEqual(expected["output_ids"], plain["output_ids"])
        self.assertEqual(out1["output_ids"], expected["output_ids"])
        self.assertEqual(out2["output_ids"], expected["output_ids"])


if __name__ == "__main__":
    unittest.main()
