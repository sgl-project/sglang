"""P0 NPU tests: online weight refit from tensors with TP=2 (per-rank IPC).

The tensor-update request carries one serialized payload per TP rank; each
rank's payload holds full (unsharded) tensors created on that rank's device, so
every worker rebuilds its own NPU IPC handles and shards the weights inside
``load_weights``. This is the multi-rank IPC path the vLLM-Ascend weight
transfer tests exercise; the single-rank payload branch is a strict subset.

The oracle is a TP=2 server that loads the same generated checkpoint at
startup; every update round must reproduce its signature (text + token
logprobs). Updates run inside a begin/end session and are repeated twice.

GLM-5.2 is served as a 6-layer, 8-expert model generated from config only (the
published checkpoint is ~1.4 TB), with the real vocabulary and tokenizer.

The ``flattened_bucket`` (``load_format="flattened_bucket"``) full refit is
currently blocked: routed-expert tensors loaded from views of one flattened
buffer come out wrong (reproduced on both this IPC path and the HCCL path).
See ``test_02_flattened_bucket_refit``.
"""

import os
import time
import unittest

import requests
import torch

from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.test_npu_rl_utils import (
    assert_logprob_signature_equal,
    begin_weight_update,
    checkpoint_tensor_index,
    end_weight_update,
    health,
    load_checkpoint_tensor,
    logprob_signature,
    npu_device_count,
    npu_rl_env,
    npu_server,
    serialize_named_tensors,
    update_full_checkpoint_per_rank,
    update_weights_from_tensor_per_rank,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1200, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=1200, suite="nightly-2-npu-a3", nightly=True)

CASE = weights.GLM52_CASE
MODEL = weights.model_path(CASE)
PROMPTS = weights.SIGNATURE_PROMPTS
TP_SIZE = 2
UPDATE_ROUNDS = 2
TARGET_WEIGHT = "model.layers.0.mlp.down_proj.weight"
SUBSET_WEIGHT = "model.layers.1.mlp.down_proj.weight"
MOE_LAYER = 3
MOE_EXPERTS = CASE.hf_overrides["n_routed_experts"]
FLATTENED_BUCKET_SKIP_REASON = (
    "blocked: flattened_bucket updates (load_format='flattened_bucket') corrupt "
    "routed-expert weights; the experts are loaded from views of one shared "
    "buffer and come out wrong on both the IPC and HCCL paths. Per-tensor "
    "refits reproduce startup loading."
)
# The default HCCL socket port is machine-wide; TP collectives need their own
# range on shared hosts.
HCCL_PORT_RANGE = os.environ.setdefault(
    "HCCL_NPU_SOCKET_PORT_RANGE",
    f"{30000 + (os.getpid() * 7) % 20000}-{30000 + (os.getpid() * 7) % 20000 + 99}",
)
GLM_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--context-length",
    "4096",
    "--max-running-requests",
    "4",
    "--tp-size",
    str(TP_SIZE),
    # The trainer process holds one full payload per rank on the NPUs.
    "--mem-fraction-static",
    "0.35",
    *weights.server_override_args(CASE),
)
REFERENCE_ARGS = GLM_ARGS
LANE_ARGS = GLM_ARGS + ("--load-format", "dummy")


def _ids(signature):
    return [item[0] for item in signature]


@unittest.skipUnless(npu_device_count() >= TP_SIZE, "requires at least 2 Ascend NPUs")
class TestNpuUpdateWeightsFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = weights.ensure_checkpoint(CASE)
        cls.tensor_index = checkpoint_tensor_index(cls.checkpoint)
        env = {**npu_rl_env(), "HCCL_NPU_SOCKET_PORT_RANGE": HCCL_PORT_RANGE}
        with npu_server(cls.checkpoint, extra_args=REFERENCE_ARGS, env=env) as base_url:
            cls.reference = logprob_signature(base_url, PROMPTS)
        cls._lane_ctx = npu_server(MODEL, extra_args=LANE_ARGS, env=env)
        cls.lane_url = cls._lane_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._lane_ctx.__exit__(None, None, None)

    def tearDown(self):
        try:
            end_weight_update(self.lane_url)
        except Exception:
            pass

    def _assert_serves_reference(self, message):
        assert_logprob_signature_equal(
            logprob_signature(self.lane_url, PROMPTS), self.reference, message
        )

    def _refit(self, *, flattened_bucket):
        for round_idx in range(UPDATE_ROUNDS):
            update_full_checkpoint_per_rank(
                self.lane_url,
                self.checkpoint,
                TP_SIZE,
                flattened_bucket=flattened_bucket,
            )
            mode = "flattened_bucket" if flattened_bucket else "per-tensor"
            self._assert_serves_reference(
                f"{mode} TP2 refit round {round_idx + 1} did not reproduce "
                "startup loading"
            )

    def test_01_per_tensor_per_rank_refit(self):
        self._refit(flattened_bucket=False)

    @unittest.skip(FLATTENED_BUCKET_SKIP_REASON)
    def test_02_flattened_bucket_refit(self):
        self._refit(flattened_bucket=True)

    def test_03_flattened_bucket_matches_per_tensor_update(self):
        original = load_checkpoint_tensor(self.tensor_index, TARGET_WEIGHT)
        updated = original.new_full(original.shape, 0.0)

        update_weights_from_tensor_per_rank(
            self.lane_url,
            [(TARGET_WEIGHT, updated)],
            TP_SIZE,
            flattened_bucket=True,
        )
        flattened_bucket_signature = logprob_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            _ids(flattened_bucket_signature),
            _ids(self.reference),
            "flattened_bucket update had no effect",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(TARGET_WEIGHT, original)], TP_SIZE
        )
        self._assert_serves_reference("per-tensor restore did not restore the output")

        update_weights_from_tensor_per_rank(
            self.lane_url, [(TARGET_WEIGHT, updated)], TP_SIZE
        )
        per_tensor_signature = logprob_signature(self.lane_url, PROMPTS)
        self.assertEqual(
            _ids(flattened_bucket_signature),
            _ids(per_tensor_signature),
            "flattened_bucket and per-tensor updates produced different outputs",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(TARGET_WEIGHT, original)], TP_SIZE
        )
        self._assert_serves_reference("second restore did not restore the output")

    def test_04_update_moe_experts(self):
        names = [
            f"model.layers.{MOE_LAYER}.mlp.experts.{expert}.down_proj.weight"
            for expert in range(MOE_EXPERTS)
        ]
        originals = [load_checkpoint_tensor(self.tensor_index, name) for name in names]

        update_weights_from_tensor_per_rank(
            self.lane_url,
            [
                (name, original.new_full(original.shape, 0.0))
                for name, original in zip(names, originals)
            ],
            TP_SIZE,
        )
        changed = logprob_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            _ids(changed),
            _ids(self.reference),
            "zeroing every routed expert of a MoE layer had no effect",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url,
            [(name, original) for name, original in zip(names, originals)],
            TP_SIZE,
        )
        self._assert_serves_reference(
            "restoring the routed experts did not restore the output"
        )

    def test_05_update_subset_changes_output(self):
        original = load_checkpoint_tensor(self.tensor_index, SUBSET_WEIGHT)
        updated = original.new_full(original.shape, 0.0)

        update_weights_from_tensor_per_rank(
            self.lane_url, [(SUBSET_WEIGHT, updated)], TP_SIZE
        )
        changed = logprob_signature(self.lane_url, PROMPTS)
        self.assertNotEqual(
            _ids(changed),
            _ids(self.reference),
            "a subset update did not change the output",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(SUBSET_WEIGHT, original)], TP_SIZE
        )
        self._assert_serves_reference("restoring the subset did not restore the output")

    def test_06_update_failure_is_fail_stop(self):
        bad = torch.zeros(1, dtype=torch.bfloat16, device="npu")
        payloads = [
            serialize_named_tensors([(TARGET_WEIGHT, bad)]) for _ in range(TP_SIZE)
        ]
        begin_weight_update(self.lane_url)
        try:
            response = requests.post(
                self.lane_url + "/update_weights_from_tensor",
                json={
                    "serialized_named_tensors": payloads,
                    "load_format": None,
                    "flush_cache": True,
                },
                timeout=300,
            )
        except requests.RequestException:
            pass
        else:
            self.assertNotEqual(
                response.status_code, 200, "malformed tensor payload was accepted"
            )
        finally:
            try:
                end_weight_update(self.lane_url)
            except Exception:
                pass

        # An in-place load can fail after mutating part of the weights, so the
        # engine must not keep serving: no rollback is possible here.
        deadline = time.time() + 30
        while health(self.lane_url) and time.time() < deadline:
            time.sleep(0.5)
        self.assertFalse(
            health(self.lane_url),
            "engine stayed alive after a failed in-place weight update",
        )


if __name__ == "__main__":
    unittest.main()
