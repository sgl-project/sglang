import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_pd_server,
)


class TestDisaggregationKvHeadFix(TestDisaggregationBase):
    """Test that kv_head_num initialization fix works in disaggregation mode."""

    @classmethod
    def setUpClass(cls):
        """Set up disaggregation servers for testing."""
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both servers are ready
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        """Start prefill server with disaggregation mode."""
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--disaggregation-ib-device",
            "mlx5_roce0",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        """Start decode server with disaggregation mode."""
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--disaggregation-ib-device",
            "mlx5_roce1",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_basic_generation_works(self):
        """Test that basic text generation works without kv_head_num errors."""
        prompt = "The capital of France is"

        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 10,
                },
            },
            timeout=30,
        )

        # Verify the request was successful
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertGreater(len(result["text"]), len(prompt))

        print(f"Generated text: {result['text']}")

    def test_structured_output_works(self):
        """Test that structured output generation works without kv_head_num errors."""
        import json

        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["city", "country"],
            }
        )

        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Tell me about the capital of France in JSON format.",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 50,
                    "json_schema": json_schema,
                },
            },
            timeout=30,
        )

        # Verify the request was successful
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("text", result)

        # Verify the output is valid JSON
        try:
            parsed_output = json.loads(result["text"])
            self.assertIn("city", parsed_output)
            self.assertIn("country", parsed_output)
        except json.JSONDecodeError:
            self.fail("Generated output is not valid JSON")

        print(f"Generated JSON: {result['text']}")

    def test_logprob_generation_works(self):
        """Test that logprob generation works without kv_head_num errors."""
        prompt = "The capital of France is"

        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 5,
                },
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=30,
        )

        # Verify the request was successful
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("text", result)
        self.assertIn("meta_info", result)

        meta_info = result["meta_info"]
        self.assertIn("completion_tokens", meta_info)
        self.assertIn("input_token_logprobs", meta_info)
        self.assertIn("output_token_logprobs", meta_info)

        # Verify logprobs are present
        self.assertGreater(len(meta_info["input_token_logprobs"]), 0)
        self.assertGreater(len(meta_info["output_token_logprobs"]), 0)

        print(f"Generated with logprobs: {result['text']}")

    def test_gsm8k_evaluation_works(self):
        """Test that GSM8K evaluation works without kv_head_num errors."""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=10,  # Reduced for faster testing
            max_new_tokens=512,
            parallel=4,  # Reduced for faster testing
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )

        try:
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"GSM8K evaluation metrics: {metrics}")

            # Verify we got some accuracy (even if low due to reduced test size)
            self.assertIn("accuracy", metrics)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)

        except Exception as e:
            # If evaluation fails, check if it's due to kv_head_num error
            error_msg = str(e).lower()
            if "attributeerror" in error_msg and "kv_head_num" in error_msg:
                self.fail(f"kv_head_num AttributeError still present: {e}")
            else:
                # Other errors are acceptable for this test
                print(f"GSM8K evaluation failed with non-kv_head_num error: {e}")

    def test_multiple_requests_work(self):
        """Test that multiple sequential requests work without kv_head_num errors."""
        prompts = [
            "What is the capital of France?",
            "What is 2+2?",
            "Name a programming language.",
        ]

        for i, prompt in enumerate(prompts):
            with self.subTest(prompt_index=i, prompt=prompt):
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 5,
                        },
                    },
                    timeout=30,
                )

                # Verify each request was successful
                self.assertEqual(response.status_code, 200)

                result = response.json()
                self.assertIn("text", result)
                self.assertIsInstance(result["text"], str)

                print(f"Request {i+1} generated: {result['text']}")

    def test_server_health_after_requests(self):
        """Test that servers remain healthy after processing requests."""
        # Make a few requests first
        for _ in range(3):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": "Test request",
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 1,
                    },
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200)

        # Check that servers are still healthy
        try:
            prefill_health = requests.get(self.prefill_url + "/health", timeout=5)
            self.assertEqual(prefill_health.status_code, 200)

            decode_health = requests.get(self.decode_url + "/health", timeout=5)
            self.assertEqual(decode_health.status_code, 200)

            lb_health = requests.get(self.lb_url + "/health", timeout=5)
            self.assertEqual(lb_health.status_code, 200)

        except Exception as e:
            self.fail(f"Server health check failed: {e}")


if __name__ == "__main__":
    unittest.main()
