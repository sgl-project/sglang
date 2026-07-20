"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_4_sync_async_stream_combination
"""

import asyncio
import json
import unittest

import torch

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=387, stage="base-b", runner_config="1-gpu-large")
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

        self.assertEqual(out1, out2)

    def test_6_engine_cpu_offload(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=512,
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=512,
            cpu_offload_gb=3,
            disable_cuda_graph=True,
        )
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(out1, out2)

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
