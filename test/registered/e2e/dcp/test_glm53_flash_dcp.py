import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=2400, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM53FlashDCP(GSM8KMixin, CustomTestCase):
    spec_args = ()
    gsm8k_num_examples = 200
    gsm8k_score_threshold = 0.90

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.model = try_cached_model("zai-org/GLM-5.3-Flash")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--tp-size",
                "8",
                "--ep-size",
                "8",
                "--dcp-size",
                "8",
                "--dcp-comm-backend",
                "ag_rs",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--kv-cache-dtype",
                "bfloat16",
                "--moe-runner-backend",
                "deep_gemm",
                "--max-total-tokens",
                "8192",
                "--context-length",
                "32768",
                "--chunked-prefill-size",
                "2048",
                "--max-running-requests",
                "8",
                "--max-mamba-cache-size",
                "128",
                "--cuda-graph-max-bs-decode",
                "8",
                *cls.spec_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _needle(self, wave, request):
        code = f"{wave + 1}{request + 2}7319"
        text = f"Document {wave}/{request}. The secret passcode is {code}.\n"
        text += "This document contains ordinary background information.\n" * 2000
        text += "\nWhat is the secret passcode? Reply with only the six digits."
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        self.assertGreater(len(ids), 8192)
        self.assertLess(len(ids), 20000)
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 2048},
            },
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        self.assertIn(code, result["text"])
        return result

    def test_allocation_watermark_and_prefix_reuse(self):
        for wave in range(2):
            with ThreadPoolExecutor(max_workers=3) as executor:
                list(
                    executor.map(
                        lambda request, wave=wave: self._needle(wave, request), range(3)
                    )
                )
            cached = self._needle(wave, 2)
            self.assertGreater(cached["meta_info"]["cached_tokens"], 0)


class TestGLM53FlashDCPEagle(TestGLM53FlashDCP):
    spec_args = (
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    )


if __name__ == "__main__":
    unittest.main()
