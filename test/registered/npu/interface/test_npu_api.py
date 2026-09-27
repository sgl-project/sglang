import json
import logging
import os
import shutil
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Global variables: Manage server process and initialization status
GLOBAL_SERVER_PROCESS = None
GLOBAL_SERVER_INITIALIZED = False
OUTPUT_DIR = "./profiler_dir"

register_npu_ci(est_time=1600, suite="nightly-npu-a3-merged", nightly=True)


class TestNpuApi(CustomTestCase):
    """Testcase: Verify that the basic functions of the API interfaces work properly and the returned parameters are consistent with the configurations.

    [Test Category] Interface
    [Test Target] /health; /health_generate; /ping; /model_info; /server_info; /v1/loads; /v1/models; /v1/models/{model:path}; /generate
    """

    @classmethod
    def setUpClass(cls):
        global GLOBAL_SERVER_PROCESS, GLOBAL_SERVER_INITIALIZED
        # Start server only if not initialized
        if not GLOBAL_SERVER_INITIALIZED:
            cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
            other_args = [
                "--attention-backend",
                "ascend",
                "--enable-return-hidden-states",
            ]
            # Start server and save to global variable
            GLOBAL_SERVER_PROCESS = popen_launch_server(
                cls.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )
            GLOBAL_SERVER_INITIALIZED = True
            cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        # First class does not terminate server
        pass

    def test_api_health(self):
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)

    def test_api_health_generate(self):
        response = requests.get(f"{self.base_url}/health_generate")
        self.assertEqual(response.status_code, 200)

    def test_api_ping(self):
        response = requests.get(f"{self.base_url}/ping")
        self.assertEqual(response.status_code, 200)

    def test_api_model_info(self):
        response = requests.get(f"{self.base_url}/model_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_path"], self.model)
        self.assertEqual(response.json()["tokenizer_path"], self.model)
        self.assertTrue(response.json()["is_generation"])
        self.assertIsNone(response.json()["preferred_sampling_params"])
        self.assertEqual(response.json()["weight_version"], "default")
        self.assertFalse(response.json()["has_image_understanding"])
        self.assertFalse(response.json()["has_audio_understanding"])
        self.assertEqual(response.json()["model_type"], "llama")
        self.assertEqual(response.json()["architectures"][0], "LlamaForCausalLM")

    def test_api_server_info(self):
        response = requests.get(f"{self.base_url}/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_path"], self.model)
        self.assertEqual(response.json()["tokenizer_path"], self.model)

    def test_api_v1_loads(self):
        response = requests.get(f"{self.base_url}/v1/loads")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("loads", body)
        self.assertIn("aggregate", body)
        self.assertGreaterEqual(len(body["loads"]), 1)
        load = body["loads"][0]
        self.assertGreaterEqual(load["num_running_reqs"], 0)
        self.assertGreaterEqual(load["num_waiting_reqs"], 0)
        self.assertGreaterEqual(load["num_used_tokens"], 0)
        self.assertGreaterEqual(load["num_total_tokens"], 0)

    def test_api_v1_models(self):
        response = requests.get(f"{self.base_url}/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], self.model)
        self.assertEqual(response.json()["data"][0]["object"], "model")
        self.assertEqual(response.json()["data"][0]["owned_by"], "sglang")
        self.assertEqual(response.json()["data"][0]["root"], self.model)
        self.assertEqual(response.json()["data"][0]["max_model_len"], 131072)

    def test_api_v1_models_path(self):
        response = requests.get(f"{self.base_url}/v1/models/{self.model}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], self.model)
        self.assertEqual(response.json()["object"], "model")
        self.assertEqual(response.json()["owned_by"], "sglang")
        self.assertEqual(response.json()["root"], self.model)
        self.assertEqual(response.json()["max_model_len"], 131072)

    def test_api_generate_single_text(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "rid": "req_001",
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 20,
                },
                "return_logprob": True,
                "stream": False,
                "return_hidden_states": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        meta_info_keys = response.json()["meta_info"].keys()
        self.assertEqual("req_001", response.json()["meta_info"]["id"])
        self.assertIn("Paris", response.json()["text"])
        self.assertEqual(20, response.json()["meta_info"]["completion_tokens"])
        self.assertIn("input_token_logprobs", meta_info_keys)
        self.assertIn("output_token_logprobs", meta_info_keys)
        self.assertIn("hidden_states", meta_info_keys)

    def test_api_generate_batch_texts(self):
        rids = ["req_1", "req_2"]
        texts = [
            "The capital of France is",
            "What is the best time of year to visit Japan for cherry blossoms?",
        ]
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "rid": rids,
                "text": texts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 20,
                },
                "return_logprob": False,
                "stream": False,
                "return_hidden_states": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual("req_1", response.json()[0]["meta_info"]["id"])
        self.assertIn("Paris", response.json()[0]["text"])
        self.assertEqual("req_2", response.json()[1]["meta_info"]["id"])
        self.assertIn("Japan", response.json()[1]["text"])

    def test_api_generate_temperature(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 5,
                    "max_new_tokens": 20,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        text1 = response.json()["text"]
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 5,
                    "max_new_tokens": 20,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        text2 = response.json()["text"]
        self.assertNotEqual(text2, text1)

    def test_api_generate_input_ids(self):
        text = "The capital of France is"
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "rid": "req_002",
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
                "return_logprob": False,
                "stream": True,
                "return_hidden_states": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        lines = response.text.strip().split("\n")
        self.assertGreaterEqual(len(lines), 10)
        json_data = lines[-3][6:]
        data = json.loads(json_data)
        meta_info_keys = data["meta_info"].keys()
        self.assertEqual("req_002", data["meta_info"]["id"])
        self.assertIn("Paris", data["text"])
        self.assertEqual(10, data["meta_info"]["completion_tokens"])
        self.assertNotIn("input_token_logprobs", meta_info_keys)
        self.assertNotIn("output_token_logprobs", meta_info_keys)
        self.assertNotIn("hidden_states", meta_info_keys)


class TestChatCompletionsInterface(CustomTestCase):
    """Testcase: The test is to verify whether the functions of each parameter of the v1/chat/completions interface are normal.

    [Test Category] Interface
    [Test Target] v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        # Skip initialization, directly reuse global server
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        # Do not terminate server
        pass

    def test_model_and_messages(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], self.model)
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], "default")
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_max_completion_tokens(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_completion_tokens": 1,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_stream(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        has_reasoning = False
        has_content = False

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True
                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertTrue(
            has_reasoning, "Reasoning content not included in stream response"
        )
        self.assertTrue(has_content, "Normal content not included in stream response")

    def test_temperature(self):
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "temperature": 0,
            },
        )
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "temperature": 0,
            },
        )
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertEqual(content1, content2)

        response3 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "temperature": 2,
            },
        )
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")
        content3 = response3.json()["choices"][0]["message"]["content"]

        response4 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "temperature": 2,
            },
        )
        self.assertEqual(response4.status_code, 200, f"Failed with: {response4.text}")
        content4 = response4.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content3, content4)

    def test_return_hidden_states(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "return_hidden_states": True,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertIn("hidden_states", response.json()["choices"][0])

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertNotIn("hidden_states", response.json()["choices"][0])

    def test_top_k(self):
        response1 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "top_k": 20,
            },
        )
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        content1 = response1.json()["choices"][0]["message"]["content"]

        response2 = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a five-character quatrain for me.",
                    }
                ],
                "top_k": 20,
            },
        )
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")
        content2 = response2.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content1, content2)

    def test_stop_token_ids(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stop_token_ids": [1, 13],
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["matched_stop"], 13)

    def test_rid(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "rid": "sssss",
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["id"], "sssss")


class TestEnableThinking(CustomTestCase):
    """Testcase: The test is to verify whether the functions of each parameter of the v1/completions interface are normal.

    [Test Category] Interface
    [Test Target] v1/completions
    """

    @classmethod
    def setUpClass(cls):
        # Skip initialization, directly reuse global server
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.additional_chat_kwargs = {}
        logging.basicConfig(level=logging.INFO)  # Initialize logging

    @classmethod
    def tearDownClass(cls):
        # Do not terminate server
        pass

    def test_model_parameters_model(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"model": self.model, "prompt": "who are you?"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assertEqual(data["model"], self.model)

    def test_model_parameters_prompt(self):
        # str format
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        # list[int] format
        list_int = [1, 2, 3, 4]
        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_int},
        )
        logging.info(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")

        # list[str] format
        list_str = ["who is you", "hello world", "ABChello"]
        response2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_str},
        )
        logging.info(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")

        # list[list[int]] format
        list_list_int = [[14990], [1350, 445, 14990, 1879, 899], [14623, 525, 498, 30]]
        response3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": list_list_int},
        )
        logging.info(f"response3.json:{response3.json()}")
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")

    def test_model_parameters_max_tokens(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "max_tokens": 1},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        logging.info(f"finish_reason:{response.json()['choices'][0]['finish_reason']}")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_model_parameters_stream(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "stream": True},
        )
        logging.info(f"response.text:{response.text}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        has_text = False
        logging.info("\n=== Stream With Reasoning ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        if "text" in data["choices"][0]:
                            has_text = True
        self.assertTrue(has_text, "Text content not included in stream response")

    def test_model_parameters_temperature(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 0},
        )
        logging.info(f"response1.json:{response1.json()}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        self.assertEqual(
            response.json()["choices"][0]["text"],
            response1.json()["choices"][0]["text"],
        )

        response2 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 2},
        )
        logging.info(f"response2.json:{response2.json()}")
        self.assertEqual(response2.status_code, 200, f"Failed with: {response2.text}")

        response3 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "temperature": 2},
        )
        logging.info(f"response3.json:{response3.json()}")
        self.assertEqual(response3.status_code, 200, f"Failed with: {response3.text}")
        self.assertNotEqual(
            response2.json()["choices"][0]["text"],
            response3.json()["choices"][0]["text"],
        )

    def test_model_parameters_hidden_states(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "return_hidden_states": True},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertIn("hidden_states", response.json()["choices"][0])

    def test_model_parameters_top_k(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "top_k": 20},
        )
        logging.info(f"response.json:{response.json()}")
        logging.info(f"response.text:{response.text}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        response1 = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "top_k": 20},
        )
        logging.info(f"response1.json:{response1.json()}")
        logging.info(f"response1.text:{response1.text}")
        self.assertEqual(response1.status_code, 200, f"Failed with: {response1.text}")
        self.assertNotEqual(
            response.json()["choices"][0]["text"],
            response1.json()["choices"][0]["text"],
        )

    def test_model_parameters_stop_token_ids(self):
        list_ids = [13]
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "prompt": "who are you?",
                "stop_token_ids": list_ids,
                "max_tokens": 1024,
            },
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["choices"][0]["matched_stop"], 13)

    def test_model_parameters_rid(self):
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": "who are you?", "rid": "10086"},
        )
        logging.info(f"response.json:{response.json()}")
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        self.assertEqual(response.json()["id"], "10086")


class TestStartProfile(CustomTestCase):
    """Testcase: Verify the correctness of /start_profile API with different parameter combinations (start_step/num_steps) on Ascend NPU backend.

    [Test Category] Interface
    [Test Target] /start_profile
    """

    @classmethod
    def setUpClass(cls):
        # Skip initialization, reuse global server + configure profiler directory
        envs.SGLANG_TORCH_PROFILER_DIR.set(OUTPUT_DIR)
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        # Terminate server in last class
        global GLOBAL_SERVER_PROCESS
        if GLOBAL_SERVER_PROCESS:
            kill_process_tree(GLOBAL_SERVER_PROCESS.pid)
            GLOBAL_SERVER_PROCESS = None

    def setUp(self):
        self._clear_profile_dir()

    def test_start_profile_1(self):
        self._start_profile(start_step="15", num_steps=5)
        self._post_request()
        self._check_non_empty_profile_dir()

    def test_start_profile_2(self):
        self._clear_profile_dir()
        self._check_empty_profile_dir()
        self._start_profile()
        self._post_request()
        requests.post(f"{self.base_url}/stop_profile")
        self._check_non_empty_profile_dir()

    def test_start_profile_3(self):
        self._start_profile(num_steps=5)
        self._post_request()
        self._check_non_empty_profile_dir()

    def _start_profile(self, **kwargs):
        response = requests.post(
            f"{self.base_url}/start_profile",
            json=kwargs if kwargs else None,
        )
        self.assertEqual(response.status_code, 200)
        return response

    def _post_request(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)

    def _clear_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def _check_non_empty_profile_dir(self):
        self.assertTrue(os.path.isdir(OUTPUT_DIR), "Profiler directory does not exist")
        self.assertNotEqual(
            len(os.listdir(OUTPUT_DIR)), 0, "Profiler directory is empty"
        )

    def _check_empty_profile_dir(self):
        if os.path.isdir(OUTPUT_DIR):
            self.assertEqual(
                len(os.listdir(OUTPUT_DIR)), 0, "Profiler directory is not empty"
            )


if __name__ == "__main__":
    unittest.main()
