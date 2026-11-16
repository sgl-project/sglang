import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")


class TestRequestLengthValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=("--max-total-tokens", "1000", "--context-length", "1000"),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_input_length_longer_than_context_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 1200  # Will tokenize to more than context length

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_maximum_allowed_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 999  # the maximum allowed length is 994 tokens

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_context_length_streaming(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 1200

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                stream=True,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_max_tokens_validation(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello "

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1200,
            )

        self.assertIn(
            "max_completion_tokens is too large",
            str(cm.exception),
        )

    def test_token_ids_logprob_out_of_vocabulary(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for token_ids_logprob in ([-1], [2_000_000_000]):
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json={
                    "text": "hi",
                    "sampling_params": {"max_new_tokens": 1},
                    "return_logprob": True,
                    "token_ids_logprob": token_ids_logprob,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("out-of-vocabulary", response.text)

    def test_token_ids_logprob_rejects_nested_list(self):
        # Nested lists are a batch-level wire format; a single request must
        # pass a flat list of ints. A ragged nested list with in-vocab ids
        # would otherwise crash the scheduler in the sampler gather.
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for token_ids_logprob in ([[0]], [[0], [1, 2]]):
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json={
                    "text": "hi",
                    "sampling_params": {"max_new_tokens": 1},
                    "return_logprob": True,
                    "token_ids_logprob": token_ids_logprob,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("flat list of integers", response.text)

    def test_token_ids_logprob_batch_with_one_oov(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json={
                "text": ["hi", "hi"],
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "token_ids_logprob": [[0], [2_000_000_000]],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("out-of-vocabulary", response.text)

    def test_token_ids_logprob_valid(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json={
                "text": "hi",
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "token_ids_logprob": [0],
            },
        )
        self.assertEqual(response.status_code, 200)


class TestKVCapacityValidation(CustomTestCase):
    process = None
    rust_frontend = False

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=("--max-total-tokens", "512", "--context-length", "4096"),
            env={"SGLANG_RUST_SERVER": "1" if cls.rust_frontend else "0"},
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_default_budget_is_not_an_explicit_request(self):
        for params, expected_status in (({}, 200), ({"max_new_tokens": 128}, 400)):
            with self.subTest(params=params):
                response = requests.post(
                    self.base_url + "/generate",
                    json={"input_ids": [100] * 400, "sampling_params": params},
                    timeout=30,
                )
                self.assertEqual(response.status_code, expected_status, response.text)
                if expected_status == 400:
                    self.assertIn("max_new_tokens=128", response.text)

    def test_explicit_kv_budget_is_rejected_and_server_recovers(self):
        """A KV-limited output budget is a client error, never a shortened success."""
        payloads = (
            (
                "/generate",
                {"text": "Hello", "sampling_params": {"max_new_tokens": 1024}},
            ),
            (
                "/v1/completions",
                {
                    "model": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                    "prompt": "Hello",
                    "max_tokens": 1024,
                },
            ),
            (
                "/v1/chat/completions",
                {
                    "model": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                },
            ),
        )
        for path, payload in payloads:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    response = requests.post(
                        self.base_url + path,
                        json={**payload, "stream": stream},
                        timeout=30,
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertIn("max_new_tokens=1024", response.text)
                    self.assertIn("KV cache capacity", response.text)
                    self.assertIn("Maximum supported max_new_tokens", response.text)
                    control = requests.post(
                        self.base_url + "/generate",
                        json={
                            "text": "Hello",
                            "sampling_params": {"max_new_tokens": 1},
                        },
                        timeout=30,
                    )
                    self.assertEqual(control.status_code, 200, control.text)


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustKVCapacityValidation(TestKVCapacityValidation):
    rust_frontend = True


if __name__ == "__main__":
    unittest.main()
