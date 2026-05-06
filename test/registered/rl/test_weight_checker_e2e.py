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
test/registered/unit/utils/test_weight_checker.py cover the in-module
logic; this file is the thin integration cover plus interaction with
update_weights_from_tensor."""

import unittest
from typing import List, Tuple

import requests
import torch

from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=150, suite="stage-b-test-1-gpu-large")

_MODEL_NAME = "Qwen/Qwen3-0.6B"
# We address the up half via the HF-style unfused name "up_proj.weight". sglang's
# stacked_params_mapping rewrites this to "gate_up_proj.weight" with shard_id=1,
# so the upload writes only the up half of the fused tensor. Sending the fused
# name directly hits a name.replace() collision (gate_up_proj contains up_proj),
# producing a malformed key like "gate_gate_up_proj.weight" and crashing load.
_UP_PROJ_SHAPE = (3072, 1024)  # intermediate_size, hidden_size for Qwen3-0.6B


class TestWeightCheckerE2E(CustomTestCase):
    """All cases share one launched server (setUpClass).

    The reset case mutates weights to random; it is named to sort last so any
    case that needs intact weights runs first. The server is torn down right
    after, so leaving the engine in a corrupted state is harmless."""

    @classmethod
    def setUpClass(cls):
        cls.url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            _MODEL_NAME,
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

    def _update_weights(
        self, named_tensors: List[Tuple[str, torch.Tensor]]
    ) -> requests.Response:
        return requests.post(
            f"{self.url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": True,
            },
            timeout=120,
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

    def test_c_update_with_diff_tensor_makes_compare_fail(self):
        """A snapshot then an update with new bytes must make compare fail."""
        self.assertEqual(self._post("snapshot").status_code, 200)

        # The unfused HF name "up_proj" is what update_weights_from_tensor accepts;
        # sglang's loader rewrites it onto the fused gate_up_proj tensor.
        upload_name = "model.layers.5.mlp.up_proj.weight"
        new_tensor = torch.full(_UP_PROJ_SHAPE, 1.5, device="cuda")
        update_resp = self._update_weights([(upload_name, new_tensor)])
        self.assertEqual(update_resp.status_code, 200)
        self.assertTrue(update_resp.json()["success"])

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertFalse(body["success"])
        # The error references the fused on-device parameter name, not the upload alias.
        self.assertIn("model.layers.5.mlp.gate_up_proj.weight", body["message"])
        self.assertIn("max_abs_err", body["message"])

    def test_d_update_with_same_tensor_keeps_compare_passing(self):
        """Prime a param, snapshot, push the same bytes again, compare must pass."""
        param_name = "model.layers.6.mlp.up_proj.weight"
        same_tensor = torch.full(_UP_PROJ_SHAPE, 0.25, device="cuda")

        # Step 1: prime the param to a known value.
        self.assertTrue(
            self._update_weights([(param_name, same_tensor)]).json()["success"]
        )
        # Step 2: snapshot the now-primed state.
        self.assertEqual(self._post("snapshot").status_code, 200)
        # Step 3: push the exact same bytes again — should be a byte-perfect no-op.
        self.assertTrue(
            self._update_weights([(param_name, same_tensor)]).json()["success"]
        )
        # Step 4: compare passes.
        resp = self._post("compare")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_e_checksum_returns_ranks_with_hashes(self):
        """checksum action must yield a ranks list with hex hashes per rank."""
        resp = self._post("checksum")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("ranks", body)
        ranks = body["ranks"]
        self.assertIsInstance(ranks, list)
        self.assertGreaterEqual(len(ranks), 1)

        first = ranks[0]
        self.assertIn("checksums", first)
        self.assertIn("parallelism_info", first)

        info = first["parallelism_info"]
        for key in (
            "tp_rank",
            "tp_size",
            "dp_rank",
            "dp_size",
            "pp_rank",
            "pp_size",
            "rank",
            "size",
        ):
            self.assertIn(key, info)

        checksums = first["checksums"]
        self.assertGreater(len(checksums), 0)
        for name, h in checksums.items():
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 16, f"unexpected hash length for {name!r}: {h!r}")
            int(h, 16)

    def test_e_checksum_is_stable_across_calls(self):
        """Two consecutive checksum calls with no weight update must match."""
        first = self._post("checksum").json()["ranks"]
        second = self._post("checksum").json()["ranks"]
        self.assertEqual(first, second)

    def test_e_checksum_changes_after_weight_update(self):
        """Updating a tensor must change its corresponding hash."""
        param_name = "model.layers.7.mlp.up_proj.weight"
        fused_name = "model.layers.7.mlp.gate_up_proj.weight"

        before = self._post("checksum").json()["ranks"][0]["checksums"]
        before_hash = before.get(fused_name)
        self.assertIsNotNone(before_hash, f"missing {fused_name!r} in checksum keys")

        new_tensor = torch.full(_UP_PROJ_SHAPE, 0.5, device="cuda")
        self.assertTrue(
            self._update_weights([(param_name, new_tensor)]).json()["success"]
        )

        after = self._post("checksum").json()["ranks"][0]["checksums"]
        self.assertNotEqual(after[fused_name], before_hash)

    def test_e_checksum_skips_non_persistent_buffers(self):
        """No checksum entry should contain a non-persistent-buffer substring."""
        ranks = self._post("checksum").json()["ranks"]
        for rank in ranks:
            for name in rank["checksums"]:
                self.assertNotIn("cos_sin_cache", name)
                self.assertNotIn("inv_freq", name)
                self.assertNotIn("freqs_cis", name)
                self.assertNotIn("_weight_fp32", name)

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
