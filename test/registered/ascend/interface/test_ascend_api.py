import json
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendApi(CustomTestCase):
    """Testcase: Verify that the basic functions of the API interfaces work properly and the returned parameters are consistent with the configurations.

    [Test Category] Interface
    [Test Target] /health; /health_generate; /ping; /model_info; /server_info; /get_load; /v1/models; /v1/models/{model:path}; /generate
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        other_args = [
            "--attention-backend",
            "ascend",
            "--enable-return-hidden-states",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_api_health(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health")
        self.assertEqual(response.status_code, 200)

    def test_api_health_generate(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

    def test_api_ping(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/ping")
        self.assertEqual(response.status_code, 200)

    def test_api_model_info(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/model_info")
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
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_path"], self.model)
        self.assertEqual(response.json()["tokenizer_path"], self.model)

    def test_api_get_load(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_load")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()[0]["rid"])
        self.assertIsNone(response.json()[0]["http_worker_ipc"])
        self.assertIsNone(response.json()[0]["dp_rank"])
        self.assertGreaterEqual(response.json()[0]["num_reqs"], 0)
        self.assertGreaterEqual(response.json()[0]["num_waiting_reqs"], 0)
        self.assertGreaterEqual(response.json()[0]["num_tokens"], 0)

    def test_api_v1_models(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], self.model)
        self.assertEqual(response.json()["data"][0]["object"], "model")
        self.assertEqual(response.json()["data"][0]["owned_by"], "sglang")
        self.assertEqual(response.json()["data"][0]["root"], self.model)
        self.assertEqual(response.json()["data"][0]["max_model_len"], 131072)

    def test_api_v1_models_path(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/v1/models/{self.model}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], self.model)
        self.assertEqual(response.json()["object"], "model")
        self.assertEqual(response.json()["owned_by"], "sglang")
        self.assertEqual(response.json()["root"], self.model)
        self.assertEqual(response.json()["max_model_len"], 131072)

    def test_api_generate_single_text(self):
        # Verify that inference succeeds with single text input.
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
        # Verify that inference succeeds with batch text inputs.
        rids = ["req_1", "req_2"]
        texts = [
            "The capital of France is",
            "What is the best time of year to visit Japan for cherry blossoms?",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
        # Verify the randomness of inference results at high temperature.
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
        # Verify that inference succeeds when using input_ids instead of text input with streaming output enabled.
        text = "The capital of France is"
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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


if __name__ == "__main__":

    unittest.main()
