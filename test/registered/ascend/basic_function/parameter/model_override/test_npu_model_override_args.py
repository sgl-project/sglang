import logging
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestModelOverrideArgs(CustomTestCase):
    """Test model override functionality on NPU environment.

    [Test Category] Functional
    [Test Target] model override on NPU
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _launch_server_with_overrides(
        self,
        model_override_args='{"num_hidden_layers": 2}',
        preferred_sampling_params='{"temperature": 0.7,  "max_new_tokens": 128}',
    ):
        """Launch server with model override args parameters."""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--json-model-override-args",
            model_override_args,
            "--preferred-sampling-params",
            preferred_sampling_params,
        ]

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def _test_basic_inference(self):
        """Test basic inference functionality."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_001_batch_processing_requests(self):
        """Test multiple configuration parameters simultaneously overriding, batch processing requests."""
        logging.warning("\n=== Test 001: Override multiple parameters ===")
        self.process = self._launch_server_with_overrides(
            model_override_args='{"num_hidden_layers": 3, "num_key_value_heads": 4}',
            preferred_sampling_params='{"temperature": 0.7,  "max_new_tokens": 127, "min_p": 1}',
        )

        try:
            response = requests.get(f"{self.base_url}/model_info")
            result = response.json()
            self.assertEqual(result["preferred_sampling_params"]["temperature"], 0.7)
            self.assertEqual(result["preferred_sampling_params"]["min_p"], 1)
            self.assertEqual(result["preferred_sampling_params"]["max_new_tokens"], 127)

            result1 = self._test_basic_inference()
            self.assertIn("text", result1)
            self.assertGreater(len(result1["text"]), 0)
            # Test request parameters take precedence, in the request parameters, max_new_tokens is set to 32.
            self.assertIn("length", result1["meta_info"]["finish_reason"])
            self.assertEqual(result1["meta_info"]["completion_tokens"], 32)
            logging.warning(
                f"Inference with multiple overrides: {result1['text'][:50]}..."
            )

            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            run_eval(args)
            logging.warning(f"Batch processing requests successful.")

        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_002_multiple_sampling_parameters(self):
        """Test configuration with multiple sampling parameters."""
        logging.warning("\n=== Test 002: multiple sampling parameters ===")
        self.process = self._launch_server_with_overrides(
            model_override_args='{"num_hidden_layers": 3, "max_position_embeddings": 50, "num_key_value_heads": 4}',
            preferred_sampling_params='{"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_new_tokens": 256, "min_new_tokens": 1, "logit_bias": {"123": 100}}',
        )

        try:
            response = requests.get(f"{self.base_url}/model_info")
            result = response.json()
            self.assertEqual(result["preferred_sampling_params"]["temperature"], 0.7)
            self.assertEqual(result["preferred_sampling_params"]["top_p"], 0.9)
            self.assertEqual(result["preferred_sampling_params"]["top_k"], 40)
            self.assertEqual(result["preferred_sampling_params"]["max_new_tokens"], 256)
            self.assertEqual(
                result["preferred_sampling_params"]["logit_bias"], {"123": 100}
            )

            result1 = self._test_basic_inference()
            self.assertIn("text", result1)
            self.assertGreater(len(result1["text"]), 0)
            logging.warning(
                f"Inference with multiple sampling: {result1['text'][:50]}..."
            )

            # If `max_position_embeddings` is set to 50, an error will occur if the input length exceeds 50.
            response2 = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The ancient Romans made significant contributions to various fields, "
                    "including law, philosophy, science, and literature. They were known "
                    "for their engineering achievements, such as the construction of the Colosseum and the Pantheon. "
                    "Their art and architecture were also highly esteemed, with the Colosseum being a symbol of "
                    "their power and influence. In science, they made important contributions to astronomy "
                    "and mathematics. Literature was also a major part of their culture, ",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )

            self.assertEqual(response2.status_code, 400)
            self.assertIn(
                "longer than the model's context length (50 tokens)", response2.text
            )

        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_003_long_sequence_request(self):
        """Test configuration with multiple sampling penalty parameters, long sequence request."""
        logging.warning("\n=== Test 003: multiple sampling penalty parameters ===")
        self.process = self._launch_server_with_overrides(
            model_override_args='{"num_hidden_layers": 3, "num_key_value_heads": 4}',
            preferred_sampling_params='{"temperature": 0.7, "max_new_tokens": 64, "frequency_penalty": 0.5, "presence_penalty": 0.3, "repetition_penalty": 1.2}',
        )

        try:
            response = requests.get(f"{self.base_url}/model_info")
            result = response.json()
            self.assertEqual(result["preferred_sampling_params"]["temperature"], 0.7)
            self.assertEqual(result["preferred_sampling_params"]["max_new_tokens"], 64)
            self.assertEqual(
                result["preferred_sampling_params"]["frequency_penalty"], 0.5
            )
            self.assertEqual(
                result["preferred_sampling_params"]["presence_penalty"], 0.3
            )
            self.assertEqual(
                result["preferred_sampling_params"]["repetition_penalty"], 1.2
            )

            long_prompt = "Explain the concept of machine learning in detail. " * 100
            response1 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": long_prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 128,
                    },
                },
                timeout=180,
            )
            self.assertEqual(response1.status_code, 200)
            self.assertGreater(len(response1.text), 50)
            logging.warning(
                f"Long sequence test passed, result length: {len(response1.text)}"
            )
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


if __name__ == "__main__":
    unittest.main()
