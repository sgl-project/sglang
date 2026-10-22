"""PPU Flash Attention 3 wrapper verification.

Launches a server with FA3 attention backend on PPU and verifies
generation correctness via GSM8K evaluation.
"""

import tempfile
import unittest
from pathlib import Path

import httpx
import requests
from openai import OpenAI

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import request, server, small_model
from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8K_URL, GSM8KEval
from sglang.test.test_utils import CustomTestCase, write_github_step_summary

register_ppu_ci(est_time=106, suite="stage-b-test-1-gpu-ppu")


class _StrictCompletionSampler:
    """Same completion payload as the shared sampler, with no failure retries."""

    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.errors = []
        self.completed = []

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, messages):
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt="\n".join(
                    m["content"] for m in messages if isinstance(m.get("content"), str)
                ),
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                stop=["Question", "Assistant:", "<|separator|>"],
            )
            text = response.choices[0].text or ""
            # Append is thread-safe; the scorer invokes 64 request threads.
            self.completed.append(True)
            return text
        except Exception as error:
            # The shared scorer converts request exceptions to wrong answers.
            # Retain them so even one transport/server error fails this PR test.
            self.errors.append(str(error))
            raise


class TestPPUFlashAttention(CustomTestCase):
    gsm8k_accuracy_thres = 0.28
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64

    @classmethod
    def setUpClass(cls):
        cls.model = small_model()
        # Keep the existing GSM8K server defaults, including concurrency.
        cls._server = server(bounded=False)
        cls.base_url = cls._server.__enter__()
        cls.addClassCleanup(cls._server.__exit__, None, None, None)

    def test_gsm8k(self):
        request(self.base_url, "GET", "/flush_cache").raise_for_status()
        # Reuse the exact five-shot dataset/scorer without its unbounded download
        # or retrying sampler. Neither shared CUDA nor nightly helpers are changed.
        with tempfile.TemporaryDirectory() as directory:
            with requests.get(GSM8K_URL, timeout=(5, 45)) as response:
                response.raise_for_status()
                data = Path(directory) / "gsm8k.jsonl"
                data.write_bytes(response.content)
            evaluate = GSM8KEval(
                num_examples=self.gsm8k_num_questions,
                num_threads=self.gsm8k_num_threads,
                num_shots=5,
                data_path=str(data),
            )
            with OpenAI(
                base_url=self.base_url + "/v1",
                api_key="EMPTY",
                timeout=httpx.Timeout(120, connect=5),
                max_retries=0,
            ) as client:
                sampler = _StrictCompletionSampler(client, self.model)
                result = evaluate(sampler)
            self.assertFalse(sampler.errors, sampler.errors[:3])
            self.assertEqual(len(sampler.completed), self.gsm8k_num_questions)
            print(f"gsm8k_score={result.score:.4f}")
            write_github_step_summary(f"### PPU GSM8K\nscore={result.score:.4f}\n")
            self.assertGreaterEqual(result.score, self.gsm8k_accuracy_thres)


if __name__ == "__main__":
    unittest.main(verbosity=3)
