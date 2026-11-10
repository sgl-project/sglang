"""P0 NPU tests: online weight refit from tensors (same-chip RL).

The oracle is a normal startup load of the same payload: a dummy-started server
must reproduce the signature of a server that loaded the checkpoint at startup.
Comparing two live updates with each other would only prove determinism.
"""

import os
import unittest

import requests
import torch

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    begin_weight_update,
    checkpoint_tensor_index,
    end_weight_update,
    generation_signature,
    health,
    load_checkpoint_tensor,
    npu_available,
    npu_server,
    serialize_named_tensors,
    update_full_checkpoint,
    update_weights_from_flattened_bucket,
    update_weights_from_tensor,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1800, suite="base-b-test-1-npu-a3")
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
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "512",
    "--max-running-requests",
    "4",
)
LANE_ARGS = BASE_ARGS + ("--load-format", "dummy")
TARGET_WEIGHT = "model.layers.0.mlp.down_proj.weight"


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuUpdateWeightsFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with npu_server(MODEL, extra_args=BASE_ARGS) as base_url:
            cls.reference = generation_signature(base_url, PROMPTS)
        cls.tensor_index = checkpoint_tensor_index(MODEL)
        # Keep one lane server for the whole class: creating NPU tensors in the
        # test process and then launching another server resets the device.
        cls._lane_ctx = npu_server(MODEL, extra_args=LANE_ARGS)
        cls.lane_url = cls._lane_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._lane_ctx.__exit__(None, None, None)

    def _assert_serves_reference(self, message):
        actual = generation_signature(self.lane_url, PROMPTS)
        self.assertEqual(actual, self.reference, message)

    def _ensure_reference(self):
        if generation_signature(self.lane_url, PROMPTS) != self.reference:
            update_full_checkpoint(self.lane_url, MODEL)

    def test_01_update_matches_startup_load(self):
        dummy = generation_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            dummy,
            self.reference,
            "dummy weights produced the reference signature; the oracle cannot "
            "distinguish a successful refit from a no-op",
        )

        update_full_checkpoint(self.lane_url, MODEL)
        self._assert_serves_reference("tensor refit did not reproduce startup loading")

    def test_02_update_twice_reloads(self):
        update_full_checkpoint(self.lane_url, MODEL)
        first = generation_signature(self.lane_url, PROMPTS)
        self.assertEqual(first, self.reference)

        update_full_checkpoint(self.lane_url, MODEL)
        second = generation_signature(self.lane_url, PROMPTS)
        self.assertEqual(second, self.reference, "second refit changed the output")

    def test_03_update_flattened_bucket(self):
        self._ensure_reference()
        original = load_checkpoint_tensor(self.tensor_index, TARGET_WEIGHT)
        updated = original.new_full(original.shape, 0.0)

        begin_weight_update(self.lane_url)
        update_weights_from_flattened_bucket(
            self.lane_url, [(TARGET_WEIGHT, updated.to("npu"))]
        )
        end_weight_update(self.lane_url)
        bucket_signature = generation_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            bucket_signature, self.reference, "flattened bucket update had no effect"
        )

        begin_weight_update(self.lane_url)
        update_weights_from_tensor(
            self.lane_url,
            [(TARGET_WEIGHT, original.to("npu"))],
            flush_cache_after=True,
        )
        end_weight_update(self.lane_url)
        self._assert_serves_reference("per-tensor restore did not restore the output")

        begin_weight_update(self.lane_url)
        update_weights_from_tensor(
            self.lane_url,
            [(TARGET_WEIGHT, updated.to("npu"))],
            flush_cache_after=True,
        )
        end_weight_update(self.lane_url)
        per_tensor_signature = generation_signature(self.lane_url, PROMPTS)
        self.assertEqual(
            bucket_signature,
            per_tensor_signature,
            "flattened bucket and per-tensor updates produced different outputs",
        )

        begin_weight_update(self.lane_url)
        update_weights_from_tensor(
            self.lane_url,
            [(TARGET_WEIGHT, original.to("npu"))],
            flush_cache_after=True,
        )
        end_weight_update(self.lane_url)
        self._assert_serves_reference("second restore did not restore the output")

    def test_04_update_subset_changes_output(self):
        self._ensure_reference()
        original = load_checkpoint_tensor(self.tensor_index, TARGET_WEIGHT)
        updated = original.new_full(original.shape, 0.0)

        self._assert_serves_reference("precondition failed")

        begin_weight_update(self.lane_url)
        update_weights_from_tensor(
            self.lane_url, [(TARGET_WEIGHT, updated.to("npu"))], flush_cache_after=True
        )
        end_weight_update(self.lane_url)
        changed = generation_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            changed, self.reference, "a subset update did not change the output"
        )

        begin_weight_update(self.lane_url)
        update_weights_from_tensor(
            self.lane_url,
            [(TARGET_WEIGHT, original.to("npu"))],
            flush_cache_after=True,
        )
        end_weight_update(self.lane_url)
        self._assert_serves_reference("restoring the subset did not restore the output")

    def test_05_update_failure_is_reported(self):
        self._ensure_reference()
        bad = torch.zeros(1, dtype=torch.bfloat16, device="npu")
        begin_weight_update(self.lane_url)
        response = requests.post(
            self.lane_url + "/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    serialize_named_tensors([(TARGET_WEIGHT, bad)])
                ],
                "load_format": None,
                "flush_cache": True,
            },
            timeout=300,
        )
        try:
            end_weight_update(self.lane_url)
        except Exception:
            # A rejected payload may also fail the finalize step; the server
            # must stay usable either way.
            pass

        self.assertNotEqual(
            response.status_code, 200, "malformed tensor payload was accepted"
        )
        self.assertTrue(health(self.lane_url))
        self._assert_serves_reference("failed update left the served weights corrupted")


if __name__ == "__main__":
    unittest.main()
