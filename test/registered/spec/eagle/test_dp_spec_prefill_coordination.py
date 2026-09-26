import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-h100")


class TestDPSpecPrefillCoordination(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "6",
            "--speculative-eagle-topk",
            "10",
            "--speculative-num-draft-tokens",
            "32",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.75",
            "--cuda-graph-max-bs-decode",
            "64",
        ]
        with (
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, rank, tokens=256):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "routed_dp_rank": rank,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                    "sampling_seed": 42,
                },
            },
            timeout=240,
        )
        response.raise_for_status()
        result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], tokens)
        output = result.get("output_ids") or result["text"]
        self.assertTrue(output)
        return output

    def test_decode_matches_isolated_output_during_prefill(self):
        prompt = "Continue the positive integers separated by commas: 1, 2, 3,"
        expected = {rank: self._generate(prompt, rank) for rank in range(2)}
        for decode_rank in range(2):
            long_prompt = (
                f"Fresh context {uuid.uuid4().hex}:\n"
                + "\n".join(
                    f"Entry {i}: apples are fruit; water is liquid; the sky appears blue."
                    for i in range(1536)
                )
                + "\nSummarize these entries briefly."
            )
            with self.subTest(decode_rank=decode_rank), ThreadPoolExecutor(2) as pool:
                decode = pool.submit(self._generate, prompt, decode_rank)
                time.sleep(0.5)
                prefill = pool.submit(self._generate, long_prompt, 1 - decode_rank, 32)
                self.assertEqual(decode.result(), expected[decode_rank])
                prefill.result()
            self.assertEqual(self._generate(prompt, decode_rank), expected[decode_rank])


if __name__ == "__main__":
    unittest.main()
