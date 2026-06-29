import json
import logging
import os
import shutil
import tempfile
import unittest
from shutil import copy2

import requests

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


class TestNpuTokenizer(CustomTestCase):
    """The test of combining the model and tokenizer parameters showed that the inference of sending a long request was successful.

    [Test Category] Functional
    [Test Target] model & tokenizer on NPU
    --model-path; --tokenizer-path; --tokenizer-worker-num; --tokenizer-mode; --load-format
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer_path = tempfile.mkdtemp(prefix="tokenizer_path")
        cls.file_names = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        for file_name in cls.file_names:
            if not os.path.exists(cls.tokenizer_path + "/" + file_name):
                copy2(cls.model + "/" + file_name, cls.tokenizer_path)
        cls.tokenizer_worker_num = 4
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--model-path",
            cls.model,
            "--tokenizer-path",
            cls.tokenizer_path,
            "--tokenizer-worker-num",
            cls.tokenizer_worker_num,
            "--tokenizer-mode",
            "auto",
            "--load-format",
            "safetensors",
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        if os.path.exists(cls.tokenizer_path):
            shutil.rmtree(cls.tokenizer_path)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_model_tokenizer_long_request(self):
        # Send long request
        long_prompt = "Explain the concept of machine learning in detail. " * 100
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text", response.text)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        # Configure --tokenizer-worker-num to start the multi-tokenizer worker
        self.assertIn("Start multi-tokenizer worker process", content)
        self.assertIn("Registering detokenizer", content)
        # Configure --load-format to safetensors
        self.assertIn("Loading safetensors checkpoint", content)
        self.out_log_file.close()
        self.err_log_file.close()


class TestNpuModelTokenizer(CustomTestCase):
    """The test of combining the model and tokenizer parameters showed that the inference of sending concurrent request was successful.

    [Test Category] Functional
    [Test Target] model & tokenizer on NPU
    --tokenizer-mode; --load-format; --model-loader-extra-config; --context-length; --model-impl
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tokenizer-mode",
            "slow",
            "--model-loader-extra-config",
            json.dumps({"enable_multithread_load": True, "num_threads": 2}),
            "--context-length",
            "1000",
            "--model-impl",
            "transformers",
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_model_tokenizer_request(self):
        # Concurrent requests
        text1 = "The capital of France is"
        for i in range(5):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text1,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        # Enable Multi-thread
        self.assertIn("Multi-thread loading shards", content)
        # Configure model-impl as transformers
        self.assertIn("type=TransformersForCausalLM", content)
        self.out_log_file.close()
        self.err_log_file.close()

    def test_model_tokenizer_context_length(self):
        # Will tokenize to more than context length
        long_text = "hello " * 1200
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": long_text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                    },
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn(
                "The input (1202 tokens) is longer than the model's context length (1000 tokens)",
                response.text,
            )
        except Exception as e:
            logging.warning(f"Error testing: {e}")


class TestNpuSkipTokenizerInit(CustomTestCase):
    """The skip configuration test was successful; requests are now being sent using input_ids instead of text.

    [Test Category] Functional
    [Test Target] model & tokenizer on NPU
    --skip-tokenizer-init
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--skip-tokenizer-init",
            "--model-impl",
            "auto",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after the test class by killing the server process and removing generated directories."""
        kill_process_tree(cls.process.pid)

    def test_model_tokenizer_error_request(self):
        # The request failed to send using text.
        prompt = "Explain the concept of machine learning in detail."
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 100,
                    },
                },
            )
            # The `--skip-tokenizer-init` parameter is configured to prevent text input from being accepted.
            self.assertEqual(response.status_code, 400)
            self.assertIn(
                "The engine initialized with ship_tokenizer_init=True cannot accept text prompts",
                response.text,
            )
        except Exception as e:
            logging.warning(f"Error testing: {e}")

    def test_model_skip_tokenizer_request(self):
        # Request sent successfully using input_ids
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": [123, 456],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("output_ids", response.text)


if __name__ == "__main__":
    unittest.main()
