"""End-to-end CUDA-graph coverage for linear and tree UNO decoding.

The test runs both modes on the same prompts. Linear UNO alternates
LoRA-draft and clean-target variants in one graph runner. Tree UNO uses a
private LoRA-draft runner before native EAGLE tree verification. Besides the
generation contract, the comparison guards that tree search improves TPF over
the linear proposal on a small, fixed GSM8K sample.
"""

import os
import unittest
from typing import NamedTuple

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=480,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="The UNO draft LoRA checkpoint is not published yet.",
)

MODEL = "Qwen/Qwen3-8B"
LORA_PATH_ENV = "SGLANG_TEST_UNO_LORA_PATH"
MAX_NEW_TOKENS = 128
# One LoRA draft forward plus one clean verification forward. The draft's
# clean-root token is useful output and therefore remains in the numerator.
FORWARDS_PER_UNO_CYCLE = 2
PROMPTS = (
    (
        "Question: Janet's ducks lay 16 eggs per day. She eats three for "
        "breakfast every morning and bakes muffins for her friends every day "
        "with four. She sells the remainder at the farmers' market daily for "
        "$2 per fresh duck egg. How much in dollars does she make every day "
        "at the farmers' market?\nAnswer:"
    ),
    (
        "Question: A robe takes 2 bolts of blue fiber and half that much "
        "white fiber. How many bolts in total does it take?\nAnswer:"
    ),
    (
        "Question: Josh decides to try flipping a house. He buys a house for "
        "$80,000 and then puts in $50,000 in repairs. This increased the value "
        "of the house by 150%. How much profit did he make?\nAnswer:"
    ),
)


class _UnoConfig(NamedTuple):
    name: str
    speculative_num_steps: int
    speculative_eagle_topk: int
    speculative_num_draft_tokens: int


LINEAR_CONFIG = _UnoConfig(
    name="linear",
    speculative_num_steps=1,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=8,  # F = 8
)
TREE_CONFIG = _UnoConfig(
    name="tree",
    speculative_num_steps=7,  # F = 8
    speculative_eagle_topk=16,
    speculative_num_draft_tokens=8,  # Q = 8
)


class TestUnoCudaGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.adapter_path = os.environ.get(LORA_PATH_ENV)
        if cls.adapter_path is None:
            raise unittest.SkipTest(
                f"Set {LORA_PATH_ENV} to the trained UNO draft LoRA checkpoint."
            )

    def _run_config(self, config: _UnoConfig) -> float:
        process = None
        try:
            process = popen_launch_server(
                MODEL,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--dtype",
                    "bfloat16",
                    "--attention-backend",
                    "fa3",
                    "--speculative-algorithm",
                    "UNO",
                    "--uno-lora-path",
                    self.adapter_path,
                    "--speculative-num-steps",
                    str(config.speculative_num_steps),
                    "--speculative-eagle-topk",
                    str(config.speculative_eagle_topk),
                    "--speculative-num-draft-tokens",
                    str(config.speculative_num_draft_tokens),
                    "--max-running-requests",
                    str(len(PROMPTS)),
                    "--cuda-graph-max-bs-decode",
                    str(len(PROMPTS)),
                    "--mem-fraction-static",
                    "0.7",
                    "--page-size",
                    "1",
                    "--disable-radix-cache",
                    "--random-seed",
                    "17",
                ],
            )
            return self._run_generation_contract(config)
        finally:
            if process is not None:
                kill_process_tree(process.pid)

    def _run_generation_contract(self, config: _UnoConfig) -> float:
        server_info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(
            server_info["speculative_eagle_topk"], config.speculative_eagle_topk
        )
        self.assertEqual(
            server_info["speculative_num_steps"], config.speculative_num_steps
        )
        self.assertEqual(
            server_info["speculative_num_draft_tokens"],
            config.speculative_num_draft_tokens,
        )

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": PROMPTS,
                "sampling_params": {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)

        results = response.json()
        self.assertEqual(len(results), len(PROMPTS))
        total_completion_tokens = 0
        total_verify_ct = 0
        for result in results:
            self.assertTrue(result["text"].strip())
            meta_info = result["meta_info"]
            self.assertEqual(meta_info["completion_tokens"], MAX_NEW_TOKENS)
            total_completion_tokens += meta_info["completion_tokens"]
            total_verify_ct += meta_info.get("spec_verify_ct", 0)

        self.assertGreater(
            total_verify_ct, 0, f"{config.name} performed no verify steps"
        )
        total_forwards = FORWARDS_PER_UNO_CYCLE * total_verify_ct
        tpf = total_completion_tokens / total_forwards
        self.assertGreater(
            tpf,
            1.5,
            f"{config.name} did not advance beyond autoregressive decoding: {tpf=}",
        )
        return tpf

    def test_tree_tpf_exceeds_linear_tpf(self):
        linear_tpf = self._run_config(LINEAR_CONFIG)
        tree_tpf = self._run_config(TREE_CONFIG)
        print(f"UNO GSM8K sample: {linear_tpf=:.3f}, {tree_tpf=:.3f}")
        self.assertGreater(
            tree_tpf,
            linear_tpf,
            f"Tree UNO did not improve TPF: {linear_tpf=:.3f}, {tree_tpf=:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
