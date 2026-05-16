import unittest

import requests

from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="stage-b", runner_config="2-gpu-large")


class TestDPAttentionDP2TP2(
    CustomTestCase,
    GSM8KMixin,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    gsm8k_accuracy_thres = 0.6

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._env_override = envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(
            True
        )
        cls._env_override.__enter__()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_override.__exit__(None, None, None)


class TestDPAttentionMixedChunk(
    CustomTestCase,
    GSM8KMixin,
):
    gsm8k_accuracy_thres = 0.6

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDPRetract(
    CustomTestCase,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--max-total-tokens",
                "4500",
                "--max-running-requests",
                "128",
                "--chunked-prefill-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        with envs.SGLANG_TEST_RETRACT.override(True):
            run_radix_attention_test(self.base_url)
            self.assertIsNone(self.process.poll())


class TestDPAttentionDP2TP2VLM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-VL-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_url = DEFAULT_IMAGE_URL
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_vlm_generate(self):
        chat_template = get_chat_template_by_model_path(self.model)
        prompt = f"{chat_template.image_token}What is in this image?"
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "image_data": [self.image_url],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        )
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        self.assertIn("output_ids", response_json)
        self.assertGreater(len(response_json["output_ids"]), 0)


if __name__ == "__main__":
    unittest.main()
