"""P0 NPU tests: tensor weight refit with TP=2 (per-rank NPU IPC).

One payload is sent per TP rank and compared (text + token logprobs) against a
server that loaded the same generated checkpoint at startup. Each model case
becomes its own test class, so new models only add a ``WeightUpdateModelCase``.
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
    checkpoint_manifest,
    checkpoint_tensor_index,
    end_weight_update,
    health,
    load_checkpoint_tensor,
    logprob_signature,
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

TP_SIZE = 2
UPDATE_ROUNDS = 2
FLATTENED_BUCKET_SKIP_REASON = (
    "blocked: flattened_bucket updates corrupt routed-expert weights on both "
    "the IPC and HCCL paths; per-tensor updates reproduce startup loading."
)
# TP collectives need their own socket range on shared hosts.
HCCL_PORT_RANGE = os.environ.setdefault(
    "HCCL_NPU_SOCKET_PORT_RANGE",
    f"{30000 + (os.getpid() * 7) % 20000}-{30000 + (os.getpid() * 7) % 20000 + 99}",
)
BASE_ARGS = (
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
    # Leave room for the trainer's full payload on the same NPU.
    "--mem-fraction-static",
    "0.35",
)


def _server_args(case, *, dummy):
    args = (*BASE_ARGS, "--tp-size", str(TP_SIZE), *case.extra_server_args)
    return args + (("--load-format", "dummy") if dummy else ())


def _pick_tensor(manifest, suffixes, *, exclude=()):
    for suffix in suffixes:
        for name, _, _ in manifest:
            if name.endswith(suffix) and name not in exclude:
                return name
    raise AssertionError(f"no tensor ending with {suffixes}")


def _ids(signature):
    return [item[0] for item in signature]


class _TensorSuite:
    CASE = None

    @classmethod
    def setUpClass(cls):
        case = cls.CASE
        cls.checkpoint = weights.ensure_checkpoint(case)
        cls.manifest = checkpoint_manifest(cls.checkpoint)
        cls.tensor_index = checkpoint_tensor_index(cls.checkpoint)
        cls.target = _pick_tensor(
            cls.manifest, [".self_attn.o_proj.weight", ".linear_attn.out_proj.weight"]
        )
        cls.subset = _pick_tensor(
            cls.manifest,
            [".self_attn.o_proj.weight", ".linear_attn.out_proj.weight"],
            exclude={cls.target},
        )
        cls.expert_tensors = [
            name
            for name, _, _ in cls.manifest
            if ".mlp.experts." in name and "down_proj" in name
        ]
        env = {**npu_rl_env(), "HCCL_NPU_SOCKET_PORT_RANGE": HCCL_PORT_RANGE}
        with npu_server(
            cls.checkpoint, extra_args=_server_args(case, dummy=False), env=env
        ) as base_url:
            cls.reference = logprob_signature(base_url, weights.SIGNATURE_PROMPTS)
        cls._lane = npu_server(
            cls.checkpoint, extra_args=_server_args(case, dummy=True), env=env
        )
        cls.lane_url = cls._lane.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._lane.__exit__(None, None, None)

    def tearDown(self):
        try:
            end_weight_update(self.lane_url)
        except Exception:
            pass

    def _assert_reference(self, message):
        assert_logprob_signature_equal(
            logprob_signature(self.lane_url, weights.SIGNATURE_PROMPTS),
            self.reference,
            message,
        )

    def test_01_per_tensor_refit(self):
        for round_idx in range(UPDATE_ROUNDS):
            update_full_checkpoint_per_rank(
                self.lane_url, self.checkpoint, TP_SIZE, flattened_bucket=False
            )
            self._assert_reference(
                f"per-tensor refit round {round_idx + 1} did not reproduce "
                "startup loading"
            )

    @unittest.skip(FLATTENED_BUCKET_SKIP_REASON)
    def test_02_flattened_bucket_refit(self):
        update_full_checkpoint_per_rank(
            self.lane_url, self.checkpoint, TP_SIZE, flattened_bucket=True
        )
        self._assert_reference(
            "flattened_bucket refit did not reproduce startup loading"
        )

    def test_03_flattened_bucket_matches_per_tensor_update(self):
        original = load_checkpoint_tensor(self.tensor_index, self.target)
        updated = original.new_full(original.shape, 0.0)

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.target, updated)], TP_SIZE, flattened_bucket=True
        )
        flattened = logprob_signature(self.lane_url, weights.SIGNATURE_PROMPTS)
        self.assertNotEqual(
            _ids(flattened),
            _ids(self.reference),
            "flattened_bucket update had no effect",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.target, original)], TP_SIZE
        )
        self._assert_reference("per-tensor restore did not restore the output")

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.target, updated)], TP_SIZE
        )
        per_tensor = logprob_signature(self.lane_url, weights.SIGNATURE_PROMPTS)
        self.assertEqual(
            _ids(flattened),
            _ids(per_tensor),
            "flattened_bucket and per-tensor updates produced different outputs",
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.target, original)], TP_SIZE
        )
        self._assert_reference("second restore did not restore the output")

    def test_04_update_moe_experts(self):
        originals = [
            load_checkpoint_tensor(self.tensor_index, name)
            for name in self.expert_tensors
        ]
        update_weights_from_tensor_per_rank(
            self.lane_url,
            [
                (name, original.new_full(original.shape, 0.0))
                for name, original in zip(self.expert_tensors, originals)
            ],
            TP_SIZE,
        )
        changed = logprob_signature(self.lane_url, weights.SIGNATURE_PROMPTS)
        self.assertNotEqual(
            _ids(changed), _ids(self.reference), "routed-expert update had no effect"
        )

        update_weights_from_tensor_per_rank(
            self.lane_url,
            [
                (name, original)
                for name, original in zip(self.expert_tensors, originals)
            ],
            TP_SIZE,
        )
        self._assert_reference(
            "restoring the routed experts did not restore the output"
        )

    def test_05_update_subset_changes_output(self):
        original = load_checkpoint_tensor(self.tensor_index, self.subset)
        updated = original.new_full(original.shape, 0.0)

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.subset, updated)], TP_SIZE
        )
        changed = logprob_signature(self.lane_url, weights.SIGNATURE_PROMPTS)
        self.assertNotEqual(
            _ids(changed), _ids(self.reference), "subset update had no effect"
        )

        update_weights_from_tensor_per_rank(
            self.lane_url, [(self.subset, original)], TP_SIZE
        )
        self._assert_reference("restoring the subset did not restore the output")

    def test_06_update_failure_is_fail_stop(self):
        bad = torch.zeros(1, dtype=torch.bfloat16, device="npu")
        payloads = [
            serialize_named_tensors([(self.target, bad)]) for _ in range(TP_SIZE)
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

        # A failed in-place load cannot be rolled back.
        deadline = time.time() + 30
        while health(self.lane_url) and time.time() < deadline:
            time.sleep(0.5)
        self.assertFalse(
            health(self.lane_url),
            "engine stayed alive after a failed in-place weight update",
        )


def _make_suite(case):
    name = "TestNpuUpdateWeightsFromTensor_" + case.id.replace("-", "_").replace(
        ".", "_"
    )
    return type(
        name, (_TensorSuite, CustomTestCase), {"CASE": case, "__module__": __name__}
    )


for _case in weights.MODEL_CASES:
    _name = "TestNpuUpdateWeightsFromTensor_" + _case.id.replace("-", "_").replace(
        ".", "_"
    )
    globals()[_name] = _make_suite(_case)


if __name__ == "__main__":
    unittest.main()
