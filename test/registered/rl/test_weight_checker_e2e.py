# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""End-to-end test for the /weights_checker HTTP endpoint.

Exercises the full HTTP -> tokenizer_manager -> scheduler -> model_runner ->
WeightChecker chain on a real engine. Unit tests in
test/registered/unit/utils/test_weight_checker.py cover the in-module logic
with stubs; this file is the thin integration cover."""

import unittest

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

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class TestWeightCheckerE2E(CustomTestCase):
    """All cases share one launched server (setUpClass).

    The reset case mutates weights to random; it is named to sort last so any
    case that needs intact weights runs first. The server is torn down right
    after, so leaving the engine in a corrupted state is harmless."""

    @classmethod
    def setUpClass(cls):
        cls.url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, action: str) -> requests.Response:
        return requests.post(
            f"{self.url}/weights_checker", json={"action": action}, timeout=120
        )

    def test_a_snapshot_then_compare_unchanged_succeeds(self):
        resp = self._post("snapshot")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_b_unknown_action_returns_400(self):
        resp = self._post("nonsense_action")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported", resp.json()["message"])

    def test_z_snapshot_reset_compare_detects_diff(self):
        """Destructive: leaves weights randomized. Named test_z_* so it runs last."""
        self.assertEqual(self._post("snapshot").status_code, 200)
        self.assertEqual(self._post("reset_tensors").status_code, 200)

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("max_abs_err", body["message"])


if __name__ == "__main__":
    unittest.main()
