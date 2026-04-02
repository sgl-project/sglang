import unittest
from types import SimpleNamespace

import requests

from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=350, suite="stage-c-test-4-gpu-h100")
register_amd_ci(est_time=350, suite="stage-c-test-4-gpu-amd")


@unittest.skipIf(
    is_in_amd_ci(),
    "DeepSeek MLA forward_mla NameError on AMD (batched_gemm not defined)",
)
class TestDPAttentionDP2TP4(
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
                "--tp=4",
                "--enable-dp-attention",
                "--dp=2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


@unittest.skipIf(
    is_in_amd_ci(),
    "DeepSeek MTP forward_mla NameError on AMD + needs 8 GPUs",
)
class TestDPAttentionDP2TP2DeepseekV3MTP(
    CustomTestCase,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix",
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=2",
            "--speculative-eagle-topk=4",
            "--speculative-num-draft-tokens=4",
            "--speculative-draft-model-path",
            DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
            "--tp-size=4",
            "--enable-dp-attention",
            "--dp-size=2",
        ]
        if not is_in_amd_ci():
            other_args += ["--mem-frac", "0.7"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k (deepseek-v3 mtp + dp):\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 2.5)


@unittest.skipIf(
    is_in_amd_ci(),
    "Qwen3-VL-30B-A3B-Instruct OOMs at TP=4 DP=2 on MI325 4-GPU runners",
)
class TestDPAttentionDP2TP4VLM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_url = DEFAULT_IMAGE_URL
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
