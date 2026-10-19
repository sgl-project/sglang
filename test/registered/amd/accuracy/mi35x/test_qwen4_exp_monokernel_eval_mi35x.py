"""MI35x Qwen4-Exp (Qwen3.8-Flash-Next AttnFP8) FlyDSL monokernel accuracy test (1 GPU).

Serves amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 with NEXTN speculative decoding (k = 3, up to 4
running requests) and the monokernel (SGLANG_ENABLE_QWEN4_MONOKERNEL: target verify,
SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL: MTP draft), which replaces the whole decoder stack for batches of <= 8
requests, and checks that

  * both monokernels were built and every verify batch ran through them (no fallback to the non-fused path),
  * 5-shot GSM8K accuracy (chat API) at client concurrency 1, 2 and 4 and the speculative accept length stay at the
    non-fused path's level,
  * a passkey hidden in a > 100k-token context is retrieved, which exercises the kernel's two-level top-512
    sparse attention selection and long-context KV addressing.

The kernel compiles once per CUDA-graph batch size on first launch (cached on disk afterwards), hence the long
launch timeout. MXFP4 / FP8 MFMA need gfx950: MI35x only.

Registry: nightly-amd-1-gpu-mi35x suite
"""

import os
import random
import re
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=3600, suite="nightly-amd-1-gpu-mi35x", nightly=True)

MODEL_PATH = os.environ.get(
    "QWEN4_EXP_ATTNFP8_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
)
# First launch JIT-compiles the kernel for every CUDA-graph batch size (~2.5 min each).
SERVER_LAUNCH_TIMEOUT = 3600

SERVER_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FLYDSL_FORCE": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
    "SGLANG_ENABLE_QWEN4_MONOKERNEL": "1",
    "SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL": "1",
}

PASSKEY_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    "A quiet river runs through the old valley, and the farmers count their sheep at dusk. "
)


class _Qwen4ExpMonokernelBase:
    num_steps: int
    max_running_requests: int
    # client concurrencies of the GSM8K runs (verify batches of 1..max_running_requests requests)
    gsm8k_concurrencies: tuple
    # 5-shot GSM8K (chat template: the completion API ends the turn right after "Answer:" for this reasoning
    # model) on the first 200 questions; the non-fused path scores ~0.98 here.
    gsm8k_threshold = 0.93
    accept_length_threshold: float

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_file = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = {**os.environ, **SERVER_ENV}
        n = cls.max_running_requests
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp",
                "1",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                str(cls.num_steps),
                "--speculative-num-draft-tokens",
                str(cls.num_steps + 1),
                "--speculative-eagle-topk",
                "1",
                "--max-running-requests",
                str(n),
                "--cuda-graph-max-bs-decode",
                str(n),
            ],
            env=env,
            return_stdout_stderr=(cls.log_file, cls.log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "log_file"):
            cls.log_file.close()
            os.unlink(cls.log_file.name)

    def _server_log(self) -> str:
        self.log_file.flush()
        with open(self.log_file.name) as f:
            return f.read()

    def test_a_gsm8k(self):
        for concurrency in self.gsm8k_concurrencies:
            with self.subTest(concurrency=concurrency):
                metrics = run_eval(
                    SimpleNamespace(
                        base_url=self.base_url,
                        model=MODEL_PATH,
                        eval_name="gsm8k",
                        api="chat",
                        num_examples=200,
                        num_threads=concurrency,
                        max_tokens=2048,
                    )
                )
                print(f"{concurrency=} {metrics=}")
                self.assertGreater(metrics["score"], self.gsm8k_threshold)
        server_info = requests.get(self.base_url + "/server_info").json()
        accept_length = server_info["internal_states"][0]["avg_spec_accept_length"]
        print(f"{accept_length=}")
        self.assertGreater(accept_length, self.accept_length_threshold)

    def test_z_monokernel_used(self):
        """Runs last: both monokernels were built, every captured verify batch size runs through the monokernel
        and no eager verify batch fell back to the non-fused path."""
        log = self._server_log()
        self.assertIn("Qwen4-Exp monokernel: L=", log)
        self.assertIn("Qwen4-Exp MTP monokernel:", log)
        verified = {
            int(r)
            for r in re.findall(r"Qwen4-Exp monokernel first verify at R=(\d+):", log)
        }
        self.assertEqual(verified, set(range(1, self.max_running_requests + 1)))
        fallbacks = [
            int(x)
            for x in re.findall(
                r"Qwen4-Exp monokernel: eager verify forwards \d+, fallbacks (\d+)", log
            )
        ]
        self.assertTrue(
            fallbacks, "no verify forward went through the monokernel runner"
        )
        self.assertEqual(fallbacks[-1], 0)


class TestQwen4ExpMonokernelK3(_Qwen4ExpMonokernelBase, CustomTestCase):
    num_steps = 3
    max_running_requests = 4
    gsm8k_concurrencies = (1, 2, 4)
    accept_length_threshold = 2.8

    def test_b_long_context_passkey(self):
        rng = random.Random(0)
        key = rng.randint(10000, 99999)
        reps = 3000  # 42 tokens per filler unit: ~126k-token context
        before = reps // 2
        haystack = (
            PASSKEY_FILLER * before
            + f" The pass key is {key}. Remember it. {key} is the pass key. "
            + PASSKEY_FILLER * (reps - before)
        )
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": [
                    {
                        "role": "user",
                        "content": haystack
                        + "\n\nWhat is the pass key? Answer with the number only.",
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0,
            },
            timeout=1800,
        ).json()
        self.assertGreater(resp["usage"]["prompt_tokens"], 100_000)
        answer = resp["choices"][0]["message"]["content"].split("</think>")[-1]
        self.assertIn(str(key), answer)


if __name__ == "__main__":
    unittest.main()
