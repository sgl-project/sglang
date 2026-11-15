"""Greedy logprob parity for Inkling: unified tri-pool vs static pools.

Not registered in CI: it boots two servers back to back. The tri-pool guards
that run per commit live in test/registered/e2e/models/test_inkling_unified.py,
which this file imports its server args and prompts from.

Run: python3 test/manual/models/test_inkling_unified_parity.py
"""

import sys
import unittest
from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "registered" / "e2e" / "models")
)
from test_inkling_unified import (  # noqa: E402
    _MODEL_PATH,
    _MODEL_REVISION,
    _PARITY_PROMPTS,
    _greedy_generate,
    _unified_args,
)


def _static_args():
    args = [
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--page-size",
        "128",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        # Match the unified cell's Triton pin, which cannot serve Inkling's
        # default FULL prefill graph.
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
