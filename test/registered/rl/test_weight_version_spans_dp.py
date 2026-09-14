import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="2-gpu-large")

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


class TestWeightVersionSpansDataParallel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--weight-version", "dp-v0", "--dp-size", "2"],
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

    def _set_weight_version(self, new_version: str):
        response = requests.post(
            f"{self.base_url}/update_weight_version",
            json={"new_version": new_version, "abort_all_requests": False},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

    def test_01_new_requests_on_every_rank_see_the_new_version(self):
        """Once the endpoint returns, requests round-robined to either DP rank must stamp the new version."""
        self._set_weight_version("dp-v1")
        self.assertEqual(self._current_version(), "dp-v1")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._generate, max_new_tokens=8) for _ in range(8)
            ]
            results = [future.result() for future in futures]

        for data in results:
            spans = _assert_spans_contiguous(self, data["meta_info"])
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["version"], "dp-v1")

    def test_02_inflight_requests_on_every_rank_are_swept(self):
        """A version change must split in-flight requests regardless of which DP rank runs them."""
        previous_version = self._current_version()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    self._generate,
                    max_new_tokens=512 - 16 * i,
                    prompt=f"Write a long story about the number {i}.",
                )
                for i in range(8)
            ]
            time.sleep(2)
            self._set_weight_version("dp-v2")
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
                    [previous_version, "dp-v2"],
                )
                self.assertGreater(spans[0]["end"], 0)

        self.assertGreater(
            split_count,
            1,
            "At most one in-flight request was split -- a DP rank was likely "
            "missed by the weight version sweep.",
        )


if __name__ == "__main__":
    unittest.main()
