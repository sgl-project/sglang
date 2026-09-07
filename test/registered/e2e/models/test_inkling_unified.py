"""Inkling under ``--enable-unified-memory`` -- the first TRI-pool model.

Boots the shrunken ``thinkingmachines/Inkling`` checkpoint (``test`` revision)
with the unified memory pool: one byte buffer, chain
``[mamba/conv (up END) | swa (FLOAT) | full (down END)]``. Inkling is the only
in-tree model that is BOTH mambaish (conv-only SConv state riding the mamba
machinery) and hybrid-SWA, so booting AT ALL proves the tri routing branch
(the 2-pool branches would either mis-store SWA KV at full lifetime — the
pre-tri hazard — or fail loud).

Guards (undertrained checkpoint — code-path correctness, not answer quality):
  - tri boot + generation through the Triton backend (unified forces triton;
    Inkling's fa4 default is NOT unified-compatible);
  - decode/prefill KV consistency via the input-vs-output logprobs match
    (catches wrong-slot reads through the v2p translate on any of the three
    pools);
  - multi-turn prefix reuse: a repeated prefix must reproduce identical
    logprobs (radix reuse + SWA tombstone recycling + conv COW);
  - a long-generation turn that slides past the SWA window (exercises
    out-of-window free_swa -> float holes -> in-place reuse during decode).

An optional env-gated parity class re-runs fixed prompts on the STATIC pools
and compares logprobs (``INKLING_UNIFIED_PARITY=1``) — the eval-host lane;
kept out of per-commit CI to bound cost.

    python -m pytest test/registered/models/test_inkling_unified.py -v
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci

# Aliased so pytest does not collect the imported `test_`-prefixed helper.
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=115, stage="base-b", runner_config="1-gpu-large")

_MODEL_PATH = os.environ.get("INKLING_TEST_MODEL_PATH", "thinkingmachines/Inkling")
_MODEL_REVISION = os.environ.get("INKLING_TEST_MODEL_REVISION", "test")


def _unified_args():
    """Server args for the tri-pool boot. Mirrors test_inkling.py's fixture
    minus the multimodal/parser surface (KV-path focus), plus the unified
    flags. The ratios still feed boot sizing until the byte configurator
    lands; the runtime split floats regardless."""
    args = [
        "--trust-remote-code",
        "--enable-unified-memory",
        # Unified requires the Triton strided page-major read/write paths.
        "--attention-backend",
        "triton",
        "--page-size",
        "128",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        # Inkling defaults to a FULL prefill graph, which unified rejects at
        # boot: the prefill graph runner bypasses the virtual->physical rebind.
        "--cuda-graph-backend-prefill",
        "disabled",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--mem-fraction-static",
        "0.5",
    ]
    if _MODEL_REVISION:
        args += ["--revision", _MODEL_REVISION]
    return args


def _static_args():
    args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--page-size",
        "128",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        # Inkling defaults to a FULL prefill graph, which unified rejects at
        # boot: the prefill graph runner bypasses the virtual->physical rebind.
        "--cuda-graph-backend-prefill",
        "disabled",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--mem-fraction-static",
        "0.5",
    ]
    if _MODEL_REVISION:
        args += ["--revision", _MODEL_REVISION]
    return args


_PARITY_PROMPTS = [
    "The capital of France is",
    "1 + 2 + 3 + 4 + 5 =",
    "List three prime numbers:",
]


def _greedy_generate(base_url, text, max_new_tokens=32, logprobs=False):
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    if logprobs:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestInklingUnifiedTriPool(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_unified_args(),
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def test_generation_basic(self):
        """Booting IS the tri-routing gate; every prompt must complete."""
        for prompt in _PARITY_PROMPTS:
            data = _greedy_generate(self.base_url, prompt, max_new_tokens=16)
            self.assertIn("text", data, data)
            self.assertGreater(len(data["text"].strip()), 0, data)

    def test_input_output_logprobs_match(self):
        """Prefill-vs-decode KV consistency through all three sub-pools'
        translates (wrong-slot reads surface as logprob mismatches)."""
        assert_logprobs_match(
            self.base_url,
            {self.model: {"kl_div": 1e-2}},
            self.model,
            max_samples=4,
            max_new_tokens=256,
            trust_remote_code=True,
        )

    def test_repeated_prefix_reproduces_logprobs(self):
        """Multi-turn prefix reuse: radix hit + conv COW + swa recycling must
        not change the numerics of a greedy re-run."""
        prompt = (
            "In a quiet village by the sea, a clockmaker kept a ledger of "
            "every tide. One morning the ledger read:"
        )
        first = _greedy_generate(
            self.base_url, prompt, max_new_tokens=24, logprobs=True
        )
        second = _greedy_generate(
            self.base_url, prompt, max_new_tokens=24, logprobs=True
        )
        self.assertEqual(first["text"], second["text"])
        lp1 = [t[0] for t in first["meta_info"]["output_token_logprobs"]]
        lp2 = [t[0] for t in second["meta_info"]["output_token_logprobs"]]
        for a, b in zip(lp1, lp2):
            self.assertAlmostEqual(a, b, places=3)

    def test_long_decode_slides_past_swa_window(self):
        """A generation long enough to age tokens out of the SWA window
        exercises free_swa -> float holes -> in-place reuse mid-decode."""
        data = _greedy_generate(
            self.base_url,
            "Write an unbroken story about a lighthouse: ",
            max_new_tokens=512,
        )
        self.assertGreater(len(data["text"].strip()), 0, data)


@unittest.skipUnless(
    os.environ.get("INKLING_UNIFIED_PARITY") == "1",
    "eval-host lane: set INKLING_UNIFIED_PARITY=1 (two sequential server boots)",
)
class TestInklingUnifiedVsStaticParity(CustomTestCase):
    """Greedy logprob parity: unified tri-pool vs static pools, same prompts.
    Two sequential boots -- the strongest wrong-slot tripwire short of GSM8K."""

    @classmethod
    def _collect(cls, other_args):
        proc = popen_launch_server(
            _MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        try:
            out = []
            for p in _PARITY_PROMPTS:
                data = _greedy_generate(
                    DEFAULT_URL_FOR_TEST, p, max_new_tokens=32, logprobs=True
                )
                out.append(
                    (
                        data["text"],
                        [t[0] for t in data["meta_info"]["output_token_logprobs"]],
                    )
                )
            return out
        finally:
            kill_process_tree(proc.pid)

    def test_parity(self):
        static = self._collect(_static_args())
        unified = self._collect(_unified_args())
        for (s_text, s_lp), (u_text, u_lp) in zip(static, unified):
            self.assertEqual(s_text, u_text)
            for a, b in zip(s_lp, u_lp):
                self.assertAlmostEqual(a, b, places=2)


if __name__ == "__main__":
    unittest.main()
