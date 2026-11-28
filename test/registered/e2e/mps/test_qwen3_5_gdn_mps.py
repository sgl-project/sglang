"""Serve hybrid GDN Qwen3.5 on eager Torch MPS and match HF greedy decoding."""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_mps_ci(est_time=300, suite="extra-e2e-test-mps")

MODEL = "Qwen/Qwen3.5-0.8B"
NUM_NEW_TOKENS = 24
PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    # Longer than the 64-token prefill chunk, so the GDN state crosses chunks.
    "Explain the theory of relativity in simple terms. " * 25,
]


def _hf_greedy(model_path: str, prompts: list[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    prompt_ids, output_ids = [], []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            out = model.generate(ids, max_new_tokens=NUM_NEW_TOKENS, do_sample=False)
            prompt_ids.append(ids[0].tolist())
            output_ids.append(out[0, ids.shape[1] :].tolist())
    return prompt_ids, output_ids


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires Torch MPS")
class TestQwen35GdnMps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.prompt_ids, cls.hf_output_ids = _hf_greedy(cls.model, PROMPTS)
        # fp32 so greedy tokens match the HF fp32 oracle; bf16 flips near-ties.
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="mps",
            other_args=[
                "--device",
                "mps",
                "--json-model-override-args",
                '{"language_model_only":true}',
                "--dtype",
                "float32",
                "--chunked-prefill-size",
                "64",
                "--context-length",
                "2048",
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def _generate(self, input_ids):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": NUM_NEW_TOKENS,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def test_batched_greedy_matches_hf(self):
        outputs = self._generate(self.prompt_ids)
        self.assertEqual([o["output_ids"] for o in outputs], self.hf_output_ids)

    def test_prefix_reuse_matches_recompute(self):
        long_ids = self.prompt_ids[-1]
        first = self._generate(long_ids)
        self.assertEqual(first["output_ids"], self.hf_output_ids[-1])
        continuation = long_ids + first["output_ids"] + self.prompt_ids[0]
        warm = self._generate(continuation)
        self.assertGreater(warm["meta_info"]["cached_tokens"], len(long_ids))
        requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=60)
        cold = self._generate(continuation)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
        self.assertEqual(warm["output_ids"], cold["output_ids"])


if __name__ == "__main__":
    unittest.main()
