import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestLoraBasicFunction(CustomTestCase):
    """Testcase：Verify the use different lora, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --enable-lora, --lora-path,
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--max-loras-per-batch",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_use_different_lora(self):
        base_params = {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        }

        # Get base model output
        response = requests.post(f"{DEFAULT_URL_FOR_TEST}/generate", json=base_params)
        self.assertEqual(response.status_code, 200)
        text_no_lora = response.json()["text"]

        # Test different LoRA adapters
        texts = []
        for lora_path in ["lora_a", "lora_b"]:
            params = base_params.copy()
            params["lora_path"] = lora_path
            response = requests.post(f"{DEFAULT_URL_FOR_TEST}/generate", json=params)
            texts.append(response.json()["text"])

        text_lora_a, text_lora_b = texts

        # Verify all outputs are different
        self.assertNotEqual(
            text_no_lora, text_lora_a, "Base model and LoRA A produced same text"
        )
        self.assertNotEqual(
            text_no_lora, text_lora_b, "Base model and LoRA B produced same text"
        )
        self.assertNotEqual(
            text_lora_a, text_lora_b, "LoRA A and LoRA B produced same text"
        )

    def test_lora_with_stream(self):
        """Compare streaming and non-streaming consistency"""
        base_request = {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            "lora_path": "lora_a",
        }

        # Non-streaming
        disable_stream_text = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate", json=base_request
        ).json()["text"]

        # Streaming
        response_stream = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={**base_request, "stream": True},
            stream=True,
        )

        stream_text = ""
        for chunk in response_stream.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:") and chunk != "data: [DONE]":
                data = json.loads(chunk[5:].strip("\n"))
                stream_text += data.get("text", "")

        self.assertIn(disable_stream_text, stream_text)

    def test_lora_lora_target_modules(self):
        # Verify lora_target_modules parameter is correctly
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        expected_modules = [
            "k_proj",
            "down_proj",
            "gate_up_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "v_proj",
            "q_proj",
            "up_proj",
        ]
        actual_modules = response.json()["lora_target_modules"]
        self.assertCountEqual(expected_modules, actual_modules)
        # Verify max_loras_per_batch parameter is correctly set in server info
        self.assertEqual(response.json()["max_loras_per_batch"], 2)

    def test_lora_with_sampling_parameters(self):
        # test loras with temperature
        response_texts = []
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0.8,
                        "max_new_tokens": 32,
                    },
                    "lora_path": "lora_a",
                },
            )
            self.assertEqual(response.status_code, 200)
            response_text = response.json()["text"]
            response_texts.append(response_text)
        self.assertNotEqual(response_texts[0], response_texts[1], f"same response_text")

    def test_lora_kv_cache(self):
        # test kv cache reuse
        input_ids_first = [1] * 200
        input_ids_second = input_ids_first + [2] * 70

        def make_request(lora_path, input_ids, expected_cached_tokens):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                    "lora_path": lora_path,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["meta_info"]["cached_tokens"], expected_cached_tokens
            )

        # For the first request, using lora_a, the expected cache size is 0.
        make_request("lora_a", input_ids_first, 0)

        # The second request uses lora_b, expecting a cache of 0 (different lora types do not share cache).
        make_request("lora_b", input_ids_first, 0)

        # The third request uses lora_a again, but the input is longer, same lora share cache.
        make_request("lora_a", input_ids_second, 128)

    def test_batch_with_lora(self):
        # test use lora in batch requests can work properly
        prompts = [
            "What is AI",
            "Explain neural network",
            "How does deep learning differ from machine learning",
            "What is reinforcement learning",
            "Explain natural language processing",
            "What are neural network layers",
            "How do activation functions work",
            "Explain backpropagation",
            "What is computer vision",
            "How do LLMs work",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                },
                "lora_path": "lora_a",
            },
        )
        results = response.json()
        for i, result in enumerate(results):
            self.assertGreater(len(result["text"]), 0)

    def test_lora_session(self):
        # test the correct collaboration of lora with session management functionality
        # Create two sessions
        s1, s2 = [
            requests.post(
                f"{DEFAULT_URL_FOR_TEST}/open_session",
                json={"capacity_of_str_len": 1000},
            ).json()
            for _ in range(2)
        ]
        self.assertNotEqual(s1, s2, "Session IDs should be different")

        # Common params
        base = {
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            "lora_path": "lora_a",
        }

        # First conversation
        r1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                **base,
                "text": "My pet is a cat named Mimi.",
                "session_params": {"id": s1},
            },
        )
        rid = r1.json()["meta_info"]["id"]

        # Test memory in both sessions
        r2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                **base,
                "text": "What is my pet's name?",
                "session_params": {"id": s1, "rid": rid},
            },
        )
        # Second conversation
        r3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                **base,
                "text": "What is my pet's name?",
                "session_params": {"id": s2},
            },
        )

        self.assertIn("Mimi", r2.text, f"Session should remember, got: {r2.text}")
        self.assertNotIn(
            "Mimi", r3.text, f"New session shouldn't remember, got: {r3.text}"
        )

    def test_lora_with_json_schema(self):
        # test lora and json schema can work properly
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
            }
        )
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "Generate person information",
                "sampling_params": {
                    "temperature": 0.3,
                    "max_new_tokens": 128,
                    "json_schema": json_schema,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        parsed_json = json.loads(result["text"])
        self.assertIn("name", parsed_json)
        self.assertIn("age", parsed_json)
        self.assertIn("city", parsed_json)


if __name__ == "__main__":
    unittest.main()
