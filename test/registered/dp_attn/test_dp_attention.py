import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=500, suite="stage-b-test-2-gpu-large-amd")


@unittest.skipIf(is_in_amd_ci(), "This test case cannot run on ROCm.")
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
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_override.__exit__(None, None, None)


class TestDPAttentionGatherv(
    CustomTestCase,
    GSM8KMixin,
):
    """Exercise the variable-length all_gatherv + reduce_scatterv DP-MoE path
    (SGLANG_DP_USE_GATHERV=1). The path only activates for the
    attn_tp_size == 1, tp_size == dp_size layout, which tp2 + dp2 satisfies.
    Without this test the gatherv/reduce_scatterv code is never exercised by CI
    (it is gated behind the env var, default off). gsm8k must stay correct since
    the change is a pure communication reorg, not a numerics change."""

    gsm8k_accuracy_thres = 0.6

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_DP_USE_GATHERV": "1"},
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--chunked-prefill-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


@unittest.skipIf(is_in_amd_ci(), "This test case cannot run on ROCm.")
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


@unittest.skipIf(is_in_amd_ci(), "This test case cannot run on ROCm.")
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


@unittest.skipIf(is_in_amd_ci(), "This test case cannot run on ROCm.")
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
