import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_32B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="stage-b-test-4-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestDPAttentionDP2TP2(
    CustomTestCase,
    NPUGSM8KMixin,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    gsm8k_accuracy_thres = 0.6

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH
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
                "4",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
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
    NPUGSM8KMixin,
):
    gsm8k_accuracy_thres = 0.35

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
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
                "--attention-backend",
                "ascend",
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
        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
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
                "--attention-backend",
                "ascend",
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
        cls.model = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
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
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_vlm_generate(self):
        # Go through /v1/chat/completions so the server inserts the model's own
        # image placeholder instead of the test guessing one.
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": self.image_url},
                            },
                            {"type": "text", "text": "What is in this image?"},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
        )
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        self.assertTrue(response_json["choices"][0]["message"]["content"])

        # image_tokens comes from the prefill's multimodal item offsets, so a
        # non-zero count is what proves the image reached the vision tower.
        usage_details = response_json["usage"].get("prompt_tokens_details")
        self.assertIsNotNone(usage_details, "prompt carried no multimodal tokens")
        self.assertGreater(usage_details.get("image_tokens", 0), 0)


if __name__ == "__main__":
    unittest.main()
