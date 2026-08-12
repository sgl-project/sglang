"""DP>1 readback of routed experts over DeepEP-class a2a backends.

With DP attention + a DeepEP-class a2a backend, the MoE layer sees only the
attention rank's DP-local tokens, so RoutedExpertsCapturer must gather at
capture time and read back from the buffer head. If the backend is not
recognized, requests owned by dp_rank > 0 read unwritten buffer rows and
silently return garbage expert ids (dp_rank 0 sits at offset 0 and looks
correct, which is why a DP>1 test is required).

Oracle: solo-vs-concurrent consistency. A request served alone is correct
even on a misclassifying tree (with the other rank empty, the global offset
degenerates to 0), so its per-token expert sets form a valid baseline. The
same prompts served concurrently must reproduce those sets; a misclassified
backend instead reads whatever the offset region holds (often well-formed
rows belonging to other tokens or graph warmup, which per-row validity
checks cannot catch). Radix cache is disabled so the concurrent phase cannot
serve cached prefix rows written by the solo phase.

Uses a dummy-weight single-layer 24-expert DeepSeek-V3 so each server boots
in seconds (same pattern as test_deepseek_v3_cutedsl_4gpu.py); generation
quality is irrelevant — only the capture/readback plumbing is under test.
"""

import concurrent.futures
import json
import os
import unittest

import numpy as np
import pybase64
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="deepep-8-gpu-h200")

_MODEL = os.environ.get("SGLANG_ROUTED_EXPERTS_TEST_MODEL", "deepseek-ai/DeepSeek-V3")
_NUM_EXPERTS = 24
_NUM_LAYERS = 1
_TOPK = 8  # DeepSeek-V3 num_experts_per_tok

_DUMMY_WEIGHT_ENV = {
    # Dummy random weights legitimately produce NaN logits; sanitize instead
    # of crashing (same rationale as test_deepseek_v3_cutedsl_4gpu.py).
    "SGLANG_ENABLE_ASYNC_ASSERT": "0",
    "SGLANG_SANITIZE_NAN_LOGITS": "1",
    "SGLANG_CUDA_COREDUMP": "0",
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
}


def _deep_ep_has(attr: str) -> bool:
    try:
        import deep_ep  # noqa: F401
    except ImportError:
        return False
    return hasattr(deep_ep, attr)


class _ReadbackMixin:
    backend_args: list

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--load-format",
            "dummy",
            "--json-model-override-args",
            json.dumps(
                {
                    "num_hidden_layers": _NUM_LAYERS,
                    "first_k_dense_replace": 0,
                    "n_routed_experts": _NUM_EXPERTS,
                }
            ),
            "--tp",
            "2",
            "--dp",
            "2",
            "--ep",
            "2",
            "--enable-dp-attention",
            "--enable-return-routed-experts",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.5",
            *cls.backend_args,
        ]
        cls.process = popen_launch_server(
            _MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                **os.environ,
                **_DUMMY_WEIGHT_ENV,
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
                "SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _one_request(self, i: int):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": f"{self._WORDS[i]} is item number {i}. Describe it in detail.",
                "sampling_params": {"max_new_tokens": 24, "temperature": 0},
                "return_routed_experts": True,
            },
            timeout=300,
        )
        self.assertEqual(resp.status_code, 200)
        meta = resp.json()["meta_info"]
        self.assertIn("routed_experts", meta)
        arr = np.frombuffer(pybase64.b64decode(meta["routed_experts"]), dtype=np.int32)
        self.assertEqual(
            arr.size % (_NUM_LAYERS * _TOPK),
            0,
            f"req{i}: payload size {arr.size} not a multiple of layers*topk",
        )
        rows = arr.reshape(-1, _NUM_LAYERS, _TOPK)
        self.assertGreater(rows.shape[0], 0)
        self.assertTrue(
            bool(((rows >= 0) & (rows < _NUM_EXPERTS)).all()),
            f"req{i}: expert id out of range [{rows.min()}, {rows.max()}]",
        )
        return rows

    _WORDS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    _N_REQ = 6

    def test_dp2_readback(self):
        # Phase 1 — solo baselines: sequential requests leave the other DP
        # rank empty, the global offset degenerates to 0, and the readback is
        # correct even when the backend is misclassified.
        solo = [self._one_request(i) for i in range(self._N_REQ)]

        # Phase 2 — the same prompts concurrently: joint forward batches give
        # dp_rank > 0 requests a non-zero global offset, which is exactly the
        # path a misclassified backend gets wrong.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._N_REQ) as ex:
            conc = list(ex.map(self._one_request, range(self._N_REQ)))

        for i in range(self._N_REQ):
            a, b = solo[i], conc[i]
            n = min(a.shape[0], b.shape[0])
            total = match = 0
            for t in range(n):
                for layer in range(_NUM_LAYERS):
                    total += 1
                    if set(a[t, layer].tolist()) == set(b[t, layer].tolist()):
                        match += 1
            frac = match / max(1, total)
            self.assertGreaterEqual(
                frac,
                0.9,
                f"req{i}: only {frac:.1%} of per-token expert sets match the "
                "solo baseline — the capturer is reading rows that belong to "
                "other tokens (DeepEP-class backend misclassification)",
            )


@unittest.skipUnless(_deep_ep_has("Buffer"), "DeepEP (v1 Buffer) not installed")
class TestRoutedExpertsReadbackDeepEP(_ReadbackMixin, CustomTestCase):
    backend_args = [
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--deepep-dispatcher-output-dtype",
        "fp8",
        "--moe-runner-backend",
        "deep_gemm",
    ]


@unittest.skipUnless(
    _deep_ep_has("ElasticBuffer"), "DeepEP v2 (ElasticBuffer) not installed"
)
class TestRoutedExpertsReadbackDeepEPv2(_ReadbackMixin, CustomTestCase):
    backend_args = [
        "--moe-a2a-backend",
        "deepep_v2",
        "--deepep-v2-mode",
        "direct",
        "--deepep-v2-dispatcher-output-dtype",
        "fp8",
        "--moe-runner-backend",
        "deep_gemm",
    ]


if __name__ == "__main__":
    unittest.main()
