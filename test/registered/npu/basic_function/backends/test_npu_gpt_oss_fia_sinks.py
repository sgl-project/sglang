"""Real pretrained GPT-OSS prefill and eager decode with the Ascend FIA backend."""

import hashlib
import math
import os
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GPT_OSS_20B_BF16_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")


class TestGPTOSSFIAPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Genuine BF16 weights from unsloth/gpt-oss-20b-BF16, HF revision
        # 5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e. The ModelScope mirror
        # has identical config, index and safetensors files.
        cls.model = os.environ.get(
            "SGLANG_TEST_GPT_OSS_20B_MODEL_PATH", GPT_OSS_20B_BF16_WEIGHTS_PATH
        )
        expected_metadata = {
            "config.json": "1f2cdaf00d4c53dac6250cd435e31b3dafb7ce60b7dc5e7622ee292d57861975",
            "model.safetensors.index.json": "bb72789eabdd4253264b146ad5a1f98d3ac215e214c942e563e756357e044f74",
        }
        for name, expected in expected_metadata.items():
            actual = hashlib.sha256((Path(cls.model) / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Unexpected GPT-OSS checkpoint metadata: {name}")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, local_files_only=True, trust_remote_code=False
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"ASCEND_USE_FIA": "1", "TRANSFORMERS_OFFLINE": "1"},
            other_args=[
                "--device",
                "npu",
                "--tp-size",
                "1",
                "--load-format",
                "safetensors",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "ascend",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--mem-fraction-static",
                "0.90",
                "--page-size",
                "128",
                "--max-total-tokens",
                "2048",
                "--max-running-requests",
                "1",
                "--context-length",
                "512",
                "--chunked-prefill-size",
                "512",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_prefill_below_and_above_sliding_window(self):
        # GPT-OSS alternates SWA128 and full-attention layers, both with D64
        # learned sinks. Previously full prefill reached BSND with pre_tokens=-1
        # and failed FIA parameter validation. Keep graphs off to isolate prefill.
        # Generate multiple tokens to exercise the eager TND sink decode path too.
        prompts = {
            "short": "What is 17 plus 25? Answer in one sentence.",
            "long": "Read this story carefully. "
            + "Mira is a botanist. Each morning she checks the blue greenhouse, "
            "waters three young trees, and records the weather in a notebook. "
            "Her colleague Arun checks the red greenhouse and cleans the tools. "
            * 5
            + "Which greenhouse does Mira check, and how many young trees does she water?",
        }
        for case, prompt in prompts.items():
            with self.subTest(case=case):
                input_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    reasoning_effort="low",
                )
                self.assertGreater(len(input_ids), 0)
                self.assertLessEqual(len(input_ids), 504)
                if case == "long":
                    self.assertGreater(len(input_ids), 128)
                else:
                    self.assertLess(len(input_ids), 128)
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": input_ids,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        "return_logprob": True,
                    },
                    timeout=240,
                )
                response.raise_for_status()
                result = response.json()
                self.assertIsInstance(result["text"], str)
                meta = result["meta_info"]
                self.assertEqual(meta["prompt_tokens"], len(input_ids))
                self.assertGreater(meta["completion_tokens"], 1)
                output_logprobs = meta["output_token_logprobs"]
                self.assertEqual(len(output_logprobs), meta["completion_tokens"])
                self.assertTrue(all(math.isfinite(item[0]) for item in output_logprobs))


if __name__ == "__main__":
    unittest.main()
