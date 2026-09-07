"""DP>1 routed-expert readback parity for DeepEP-family A2A backends."""

import concurrent.futures
import json
import os
import unittest

import numpy as np
import pybase64
import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=72, stage="base-c", runner_config="4-gpu-h100")

_MODEL = os.environ.get("SGLANG_ROUTED_EXPERTS_TEST_MODEL", "deepseek-ai/DeepSeek-V3")
_NUM_EXPERTS = 24
_NUM_LAYERS = 1
_TOPK = 8

_DUMMY_WEIGHT_ENV = {
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


def _deep_ep_nccl_compatible() -> bool:
    try:
        version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        return False
    return version is not None and version >= (2, 30, 7)


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
            # Keep the startup budget within the test's 256-token buffer.
            "--chunked-prefill-size",
            "256",
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
        if getattr(cls, "process", None):
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
        solo = [self._one_request(i) for i in range(self._N_REQ)]

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
@unittest.skipUnless(
    _deep_ep_nccl_compatible(), "DeepEP v2 requires NCCL runtime >= 2.30.7"
)
class TestRoutedExpertsReadbackDeepEPv2(_ReadbackMixin, CustomTestCase):
    backend_args = [
        "--moe-a2a-backend",
        "deepep_v2",
        "--deepep-v2-mode",
        "direct",
        "--moe-runner-backend",
        "deep_gemm",
    ]


if __name__ == "__main__":
    unittest.main()
