import asyncio
import json
import unittest

import torch

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPUSRTEngine(CustomTestCase):

    def test_1_engine_runtime_consistency(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        self.assertEqual(out1, out2)

    def test_2_engine_runtime_encode_consistency(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        engine = sgl.Engine(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_3_engine_token_ids_consistency(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]

        tokenizer = get_tokenizer(LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
            "text"
        ]
        engine.shutdown()
        self.assertEqual(out1, out2)

    def test_6_engine_cpu_offload(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=512,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=512,
            cpu_offload_gb=3,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()
        self.assertEqual(out1, out2)

    def test_8_engine_async_encode_consistency(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        engine = sgl.Engine(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
        )

        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        loop = asyncio.get_event_loop()
        out2 = torch.tensor(
            loop.run_until_complete(engine.async_encode(prompt))["embedding"]
        )

        engine.shutdown()
        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-5, rtol=1e-3),
            "Sync and async embeddings are not equal within tolerance",
        )


if __name__ == "__main__":
    unittest.main()
