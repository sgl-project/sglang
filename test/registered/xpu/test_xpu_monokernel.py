"""Load test of --enable-xpu-monokernel (sgl-kernel-xpu's Qwen3.5 / Qwen3.6 MoE
monokernels): a 4-layer Qwen3.6-35B-A3B-FP8 (3 GatedDeltaNet + 1 full-attention
layer, dummy weights) at TP2 on two Xe2 GPUs with 32 Xe cores. The server must
start with the monokernels, route its decode (and MTP draft decode) steps to
them, and give the stock path's greedy tokens and logprobs.

Usage:
python3 -m unittest test_xpu_monokernel
"""

import json
import os
import tempfile
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=900, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

MODEL = os.environ.get("SGLANG_TEST_XPU_MONO_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
LAYERS = ["linear_attention"] * 3 + ["full_attention"]
PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4,",
    "def fib(n):",
    "Hello, my name is",
]
MTP = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "2",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "3",
]


def _two_mono_gpus() -> bool:
    if not torch.xpu.is_available() or torch.xpu.device_count() < 2:
        return False
    try:
        from sgl_kernel import mono_decode
    except ImportError:
        return False
    return all(mono_decode.supports(torch.device("xpu", i)) for i in range(2))


def _generate(base_url: str) -> list:
    out = []
    for prompt in PROMPTS:
        r = requests.post(
            base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=300,
        )
        r.raise_for_status()
        out.append(
            [(lp, tok) for lp, tok, _ in r.json()["meta_info"]["output_token_logprobs"]]
        )
    return out


def _serve(mono: bool, extra: list) -> tuple:
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    args = [
        "--load-format",
        "dummy",
        "--json-model-override-args",
        json.dumps(
            {"text_config": {"num_hidden_layers": len(LAYERS), "layer_types": LAYERS}}
        ),
        "--tp-size",
        "2",
        "--attention-backend",
        "intel_xpu",
        "--page-size",
        "64",
        "--disable-radix-cache",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "8",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.6",
        # a fixed pool: sized from the memory fraction, the 4-layer target's
        # pool (and the MTP drafter's copy of it) would fill the cards
        "--max-total-tokens",
        "131072",
    ] + extra
    if mono:
        args.append("--enable-xpu-monokernel")
    process = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=args,
        return_stdout_stderr=(log, log),
    )
    try:
        outputs = _generate(DEFAULT_URL_FOR_TEST)
    finally:
        kill_process_tree(process.pid)
    log.seek(0)
    return outputs, log.read()


@unittest.skipUnless(
    _two_mono_gpus(), "needs two Xe2 GPUs with 32 Xe cores and the mono decode kernels"
)
class TestXPUMonokernel(CustomTestCase):
    def _compare(self, extra: list, routed: list):
        stock, _ = _serve(False, extra)
        mono, log = _serve(True, extra)
        self.assertIn("XPU monokernel: 4 layers (1 full attention)", log)
        for pattern in routed:
            self.assertRegex(log, pattern)
        agree = 0
        for s, m in zip(stock, mono):
            for (ls, ts), (lm, tm) in zip(s, m):
                if ts != tm:  # a near tie may go either way; the rest diverges
                    break
                agree += 1
                self.assertLess(abs(ls - lm), 0.02, f"logprob {lm} vs stock {ls}")
        self.assertGreaterEqual(agree, 0.75 * sum(len(s) for s in stock))

    def test_decode(self):
        self._compare(
            [], [r"monokernel qwen3_5_moe_35b_a3b_fp8 on xe2 runs decode steps"]
        )

    def test_mtp(self):
        self._compare(
            MTP,
            [
                r"monokernel qwen3_5_moe_35b_a3b_fp8 on xe2 runs target_verify steps",
                r"monokernel qwen3_5_moe_35b_a3b_fp8_mtp on xe2: widths",
            ],
        )


if __name__ == "__main__":
    unittest.main()
