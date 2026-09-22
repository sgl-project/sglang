"""Qwen3-30B-A3B served with DWDP at TP=4, DWDP=4.

The only test that exercises the composite virtual address space against a real
MoE model: each rank physically holds 32 of the 128 experts per layer and reads
the other 96 out of peer-exported physical memory, so a wrong page size, offset
or prefetch ordering shows up as garbage tokens rather than as a crash.
``test_dwdp_weight_prefetch.py`` pins the byte-exactness of one layer; this pins
that 48 of them in sequence still decode.

Generations are deliberately short: DWDP re-fetches every peer expert on each
decode step outside ``--disaggregation-mode prefill``, so decode runs at well
under one token per second on 4 ranks.

Runs on CUDA or Intel XPU, whichever the host has.
"""

from __future__ import annotations

import unittest

import requests

from sglang.srt.utils import is_cuda, is_xpu, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.dwdp_test_utils import launch_dwdp_server
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

register_xpu_ci(est_time=1200, suite="nightly-xpu-4-gpu", nightly=True)
register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")

MODEL = "Qwen/Qwen3-30B-A3B"
DWDP_SIZE = 4

# weights come off disk uncached in CI, and DWDP copies each local shard into
# driver-owned physical memory afterwards
LAUNCH_TIMEOUT = 1800

SERVER_ARGS = [
    "--dwdp-size",
    str(DWDP_SIZE),
    "--mem-fraction-static",
    "0.85",
    "--max-total-tokens",
    "8192",
    "--context-length",
    "8192",
]

# more than one chunked-prefill chunk, so a layer's pool pages are rebound
# several times per request instead of once
_LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. " * 260
) + "\nSummarize the sentence above in five words:"


@unittest.skipUnless(is_cuda() or is_xpu(), "requires a CUDA or Intel XPU device")
class TestDwdpQwen3_30BA3B(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_dwdp_server(
            MODEL,
            cls.base_url,
            extra_args=SERVER_ARGS,
            timeout=LAUNCH_TIMEOUT,
            tp_size=DWDP_SIZE,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()["text"]

    def test_greedy_answer_is_correct(self):
        """A peer expert read through the wrong page still produces fluent-looking
        tokens, so this asserts the answer itself rather than just non-emptiness."""
        text = self._generate("The capital of France is", 16)
        self.assertIn("paris", text.lower(), f"got: {text!r}")

    def test_chunked_prefill_over_many_layers(self):
        """A prompt spanning several prefill chunks: every chunk walks all 48 MoE
        layers, so each buffer slot is handed back and forth between two layers
        many times. Rebinding a slot too early tore down the pages under a live
        read and killed the rank."""
        text = self._generate(_LONG_PROMPT, 16)
        self.assertGreater(len(text.strip()), 0, "empty reply after chunked prefill")


if __name__ == "__main__":
    unittest.main()
