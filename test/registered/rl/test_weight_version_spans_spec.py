import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=360, stage="extra-a", runner_config="1-gpu-large")

_REQUEST_TIMEOUT = 180


def _assert_spans_contiguous(test, meta_info):
    spans = meta_info["weight_versions"]
    test.assertGreater(len(spans), 0)
    test.assertEqual(spans[0]["start"], 0)
    for prev, cur in zip(spans, spans[1:]):
        test.assertEqual(prev["end"], cur["start"])
        test.assertNotEqual(prev["version"], cur["version"])
    test.assertEqual(meta_info["weight_version"], spans[-1]["version"])
    return spans


class TestWeightVersionSpansSpecOverlap(CustomTestCase):
    """Spans under the overlap scheduler combined with EAGLE speculative decoding."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--weight-version",
                "spec-v0",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model",
                DEFAULT_DRAFT_MODEL_EAGLE,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "8",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, max_new_tokens: int, prompt: str = "The capital of France is"):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _current_version(self):
        response = requests.get(f"{self.base_url}/get_model_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()["weight_version"]

    def _pause(self, mode: str):
        requests.post(
            f"{self.base_url}/pause_generation",
            json={"mode": mode},
            timeout=30,
        ).raise_for_status()

    def _continue(self):
        requests.post(
            f"{self.base_url}/continue_generation",
            json={},
            timeout=30,
        ).raise_for_status()

    def _set_weight_version(self, new_version: str):
        response = requests.post(
            f"{self.base_url}/update_weight_version",
            json={"new_version": new_version, "abort_all_requests": False},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

    def test_01_single_span_covers_all_accepted_tokens(self):
        """Draft-token overshoot must not leak past the reported completion length."""
        data = self._generate(max_new_tokens=32)

        meta_info = data["meta_info"]
        spans = _assert_spans_contiguous(self, meta_info)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["version"], self._current_version())
        self.assertEqual(spans[0]["end"], meta_info["completion_tokens"])

    def test_02_retract_update_continue_keeps_exact_boundaries(self):
        """Spans stay contiguous and clamped when a retract-paused update lands mid-speculation."""
        previous_version = self._current_version()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._generate,
                    max_new_tokens=1024 - 16 * i,
                    prompt=f"Write a long story about the number {i}.",
                )
                for i in range(4)
            ]

            time.sleep(2)
            self._pause("retract")
            try:
                self._set_weight_version("spec-v1")
            finally:
                self._continue()

            results = [future.result() for future in futures]

        split_count = 0
        for data in results:
            meta_info = data["meta_info"]
            spans = _assert_spans_contiguous(self, meta_info)
            self.assertEqual(spans[-1]["end"], meta_info["completion_tokens"])
            self.assertEqual(spans[0]["version"], previous_version)
            if len(spans) > 1:
                split_count += 1
                self.assertEqual(
                    [span["version"] for span in spans],
                    [previous_version, "spec-v1"],
                )
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            split_count,
            0,
            "No request spanned the update -- the retract boundary under "
            "speculative decoding was not recorded.",
        )


if __name__ == "__main__":
    unittest.main()
