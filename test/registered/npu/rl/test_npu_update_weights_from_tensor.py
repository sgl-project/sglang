"""P0 NPU tests: online weight refit from tensors (same-chip RL).

The oracle is a normal startup load of the same payload: a dummy-started server
must reproduce the signature of a server that loaded the checkpoint at startup.
Comparing two live updates with each other would only prove determinism.
"""

import os
import unittest

import requests
import torch

from sglang.test.ascend.npu_rl_utils import (
    begin_weight_update,
    checkpoint_tensor_index,
    end_weight_update,
    generation_signature,
    health,
    load_checkpoint_tensor,
    npu_available,
    npu_server,
    read_weights_by_name,
    serialize_named_tensors,
    update_full_checkpoint,
    update_weights_from_flattened_bucket,
    update_weights_from_tensor,
    warmup_npu,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1800, suite="nightly-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
BASE_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.5",
    "--max-model-len",
    "512",
    "--max-running-requests",
    "4",
)
LANE_ARGS = BASE_ARGS + ("--load-format", "dummy")
NORM_WEIGHT = "model.norm.weight"


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuUpdateWeightsFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        warmup_npu()
        with npu_server(MODEL, extra_args=BASE_ARGS) as base_url:
            cls.reference = generation_signature(base_url, PROMPTS)
        cls.tensor_index = checkpoint_tensor_index(MODEL)

    def _lane(self):
        return npu_server(MODEL, extra_args=LANE_ARGS)

    def _assert_serves_reference(self, base_url, message):
        actual = generation_signature(base_url, PROMPTS)
        self.assertEqual(actual, self.reference, message)

    def test_update_matches_startup_load(self):
        with self._lane() as base_url:
            dummy = generation_signature(base_url, PROMPTS)
            self.assertNotEqual(
                dummy,
                self.reference,
                "dummy weights produced the reference signature; the oracle cannot "
                "distinguish a successful refit from a no-op",
            )

            update_full_checkpoint(base_url, MODEL)
            self._assert_serves_reference(
                base_url, "tensor refit did not reproduce startup loading"
            )

    def test_update_twice_reloads(self):
        with self._lane() as base_url:
            update_full_checkpoint(base_url, MODEL)
            first = generation_signature(base_url, PROMPTS)
            self.assertEqual(first, self.reference)

            update_full_checkpoint(base_url, MODEL)
            second = generation_signature(base_url, PROMPTS)
            self.assertEqual(second, self.reference, "second refit changed the output")

    def test_update_flattened_bucket(self):
        original = load_checkpoint_tensor(self.tensor_index, NORM_WEIGHT)
        updated = original.new_full(original.shape, 0.5)

        with self._lane() as base_url:
            begin_weight_update(base_url)
            update_weights_from_flattened_bucket(
                base_url, [(NORM_WEIGHT, updated.to("npu"))]
            )
            end_weight_update(base_url)

            values = read_weights_by_name(base_url, NORM_WEIGHT, truncate_size=4)
            self.assertTrue(
                all(abs(value - 0.5) < 1e-2 for value in values),
                f"flattened bucket update did not land: {values}",
            )

            begin_weight_update(base_url)
            update_weights_from_tensor(
                base_url,
                [(NORM_WEIGHT, original.to("npu"))],
                flush_cache_after=True,
            )
            end_weight_update(base_url)

            values = read_weights_by_name(base_url, NORM_WEIGHT, truncate_size=4)
            expected = original.flatten()[:4].float().tolist()
            self.assertTrue(
                all(abs(a - b) < 1e-2 for a, b in zip(values, expected)),
                f"per-tensor restore did not land: {values} vs {expected}",
            )

    def test_update_subset_changes_output(self):
        original = load_checkpoint_tensor(self.tensor_index, NORM_WEIGHT)
        updated = original.new_full(original.shape, 0.0)

        with self._lane() as base_url:
            update_full_checkpoint(base_url, MODEL)
            self._assert_serves_reference(base_url, "precondition failed")

            begin_weight_update(base_url)
            update_weights_from_tensor(
                base_url, [(NORM_WEIGHT, updated.to("npu"))], flush_cache_after=True
            )
            end_weight_update(base_url)
            changed = generation_signature(base_url, PROMPTS)
            self.assertNotEqual(
                changed, self.reference, "a subset update did not change the output"
            )

            begin_weight_update(base_url)
            update_weights_from_tensor(
                base_url,
                [(NORM_WEIGHT, original.to("npu"))],
                flush_cache_after=True,
            )
            end_weight_update(base_url)
            self._assert_serves_reference(
                base_url, "restoring the subset did not restore the output"
            )

    def test_update_failure_is_reported(self):
        with self._lane() as base_url:
            update_full_checkpoint(base_url, MODEL)

            bad = torch.zeros(1, dtype=torch.bfloat16, device="npu")
            begin_weight_update(base_url)
            response = requests.post(
                base_url + "/update_weights_from_tensor",
                json={
                    "serialized_named_tensors": [
                        serialize_named_tensors([(NORM_WEIGHT, bad)])
                    ],
                    "load_format": None,
                    "flush_cache": True,
                },
                timeout=300,
            )
            try:
                end_weight_update(base_url)
            except Exception:
                # A rejected payload may also fail the finalize step; the server
                # must stay usable either way.
                pass

            self.assertNotEqual(
                response.status_code, 200, "malformed tensor payload was accepted"
            )
            self.assertTrue(health(base_url))
            self._assert_serves_reference(
                base_url, "failed update left the served weights corrupted"
            )


if __name__ == "__main__":
    unittest.main()
