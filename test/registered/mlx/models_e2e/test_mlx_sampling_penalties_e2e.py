"""Black-box E2E coverage for sampling penalties on the MLX backend.

This test deliberately complements the model-free formula and lifecycle tests:
it starts the real HTTP server, keeps radix caching and overlap scheduling on,
and sends concurrent requests through the public ``/generate`` contract.  The
fixture uses a small model and token-level assertions; it does not treat prose
quality or throughput on a thermally constrained machine as correctness.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_mlx_ci(est_time=1, suite="stage-b-e2e-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx on Apple Silicon"

MODEL_PATH = os.environ.get("SGLANG_MLX_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit")
MEM_FRACTION_STATIC = os.environ.get("SGLANG_MLX_TEST_MEM_FRACTION", "0.45")
MIN_FREE_GB = float(os.environ.get("SGLANG_MLX_TEST_MIN_FREE_GB", "4"))

_PROMPT = "Repeat the number 1 forever. 1 1 1 1 1"
_MAX_NEW_TOKENS = 24
_CASES = {
    "baseline": {},
    "frequency": {"frequency_penalty": 2.0},
    "presence": {"presence_penalty": 2.0},
    "repetition": {"repetition_penalty": 2.0},
    "combined": {
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0,
        "repetition_penalty": 1.2,
    },
}


def _available_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxSamplingPenaltiesE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        available_gb = _available_gb()
        if available_gb is not None and available_gb < MIN_FREE_GB:
            raise unittest.SkipTest(
                f"insufficient free memory: {available_gb:.1f} GB < "
                f"{MIN_FREE_GB:.1f} GB needed to serve {MODEL_PATH}"
            )

        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mlx-enable-sampling",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--mem-fraction-static",
                MEM_FRACTION_STATIC,
                "--max-running-requests",
                str(len(_CASES)),
                "--context-length",
                "512",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            process.wait(timeout=30)

    def _generate(self, penalties):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": _PROMPT,
                "sampling_params": {
                    "temperature": 0,
                    "top_k": 1,
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "ignore_eos": True,
                    **penalties,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertEqual(output["meta_info"]["completion_tokens"], _MAX_NEW_TOKENS)
        self.assertEqual(len(output["output_ids"]), _MAX_NEW_TOKENS)
        return output["output_ids"]

    def test_concurrent_public_requests_apply_each_penalty_without_row_leakage(self):
        references = {
            name: self._generate(penalties) for name, penalties in _CASES.items()
        }
        barrier = threading.Barrier(len(_CASES))

        def run_case(penalties):
            barrier.wait(timeout=10)
            return self._generate(penalties)

        with ThreadPoolExecutor(max_workers=len(_CASES)) as executor:
            futures = {
                name: executor.submit(run_case, penalties)
                for name, penalties in _CASES.items()
            }
            outputs = {name: future.result() for name, future in futures.items()}

        baseline = references["baseline"]
        self.assertEqual(len(set(baseline)), 2, baseline)
        self.assertEqual(len(set(baseline[::2])), 1, baseline)
        self.assertEqual(len(set(baseline[1::2])), 1, baseline)

        for name in _CASES:
            with self.subTest(name=name, mode="concurrent_matches_solo"):
                self.assertEqual(outputs[name], references[name])

        for name in ("frequency", "presence", "repetition", "combined"):
            with self.subTest(name=name):
                penalized = references[name]
                self.assertEqual(penalized[0], baseline[0])
                self.assertNotEqual(penalized, baseline)
                self.assertGreater(len(set(penalized)), len(set(baseline)))

    def test_penalty_state_does_not_leak_to_a_follow_up_request(self):
        baseline_before = self._generate({})
        penalized = self._generate(_CASES["combined"])
        baseline_after = self._generate({})

        self.assertNotEqual(penalized, baseline_before)
        self.assertEqual(baseline_after, baseline_before)


if __name__ == "__main__":
    unittest.main()
