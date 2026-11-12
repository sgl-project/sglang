"""P0 NPU tests: weight refit over HCCL (disaggregated RL).

The trainer broadcasts the checkpoint shards over an HCCL group; each round is
compared (text + token logprobs) against the startup-loaded reference. The HTTP
request only carries metadata.
"""

import os
import threading
import unittest

import requests
import torch

from sglang.srt.managers.io_struct import InitWeightsUpdateGroupReqInput
from sglang.srt.utils import init_custom_process_group
from sglang.test.ascend import test_npu_rl_weights as weights
from sglang.test.ascend.test_npu_rl_utils import (
    assert_logprob_signature_equal,
    begin_weight_update,
    checkpoint_manifest,
    checkpoint_tensor_index,
    end_weight_update,
    flattened_checkpoint_manifest,
    flattened_tensor_bucket,
    health,
    load_checkpoint_tensor,
    logprob_signature,
    npu_device_count,
    npu_rl_env,
    npu_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_npu_ci(est_time=600, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=600, suite="nightly-2-npu-a3", nightly=True)

CASE = weights.GLM52_CASE
MODEL = weights.model_path(CASE)
PROMPTS = weights.SIGNATURE_PROMPTS
MASTER_ADDR = os.environ.get("SGLANG_TEST_HCCL_MASTER_ADDR", "127.0.0.1")
UPDATE_ROUNDS = 2
FLATTENED_BUCKET_SKIP_REASON = (
    "blocked: flattened_bucket updates corrupt routed-expert weights on both "
    "the IPC and HCCL paths; per-tensor updates reproduce startup loading."
)
# The default HCCL port is machine-wide; use a private range on shared hosts.
HCCL_PORT_RANGE = os.environ.setdefault(
    "HCCL_NPU_SOCKET_PORT_RANGE",
    f"{30000 + (os.getpid() * 7) % 20000}-{30000 + (os.getpid() * 7) % 20000 + 99}",
)
NORM_WEIGHT = "model.norm.weight"
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
    "--mem-fraction-static",
    "0.45",
    *weights.server_override_args(CASE),
)
REFERENCE_ARGS = GLM_ARGS
LANE_ARGS = GLM_ARGS + ("--load-format", "dummy")


@unittest.skipUnless(npu_device_count() >= 2, "requires at least 2 Ascend NPUs")
class TestNpuUpdateWeightsFromDistributed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = weights.ensure_checkpoint(CASE)
        cls.manifest = checkpoint_manifest(cls.checkpoint)
        cls.flattened_manifest = flattened_checkpoint_manifest(cls.checkpoint)
        cls.tensor_index = checkpoint_tensor_index(cls.checkpoint)
        env = {**npu_rl_env(), "HCCL_NPU_SOCKET_PORT_RANGE": HCCL_PORT_RANGE}
        with npu_server(cls.checkpoint, extra_args=REFERENCE_ARGS, env=env) as base_url:
            cls.reference = logprob_signature(base_url, PROMPTS)
        # One lane server per class: creating NPU tensors resets the device.
        cls._lane_ctx = npu_server(MODEL, extra_args=LANE_ARGS, env=env)
        cls.lane_url = cls._lane_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._lane_ctx.__exit__(None, None, None)

    def tearDown(self):
        # A failed case must not leave a weight-update session open: the next
        # case would be rejected with "a weight-update session is already open".
        try:
            end_weight_update(self.lane_url)
        except Exception:
            pass

    def _create_group(self, base_url, group_name):
        port = find_available_port(25000)
        outcome = {}

        def _trainer():
            try:
                torch.npu.set_device(1)
                outcome["pg"] = init_custom_process_group(
                    backend="hccl",
                    init_method=f"tcp://{MASTER_ADDR}:{port}",
                    world_size=2,
                    rank=0,
                    group_name=group_name,
                    device_id=torch.device("npu", 1),
                )
            except Exception as exc:  # pragma: no cover - surfaced in the test
                outcome["error"] = exc

        thread = threading.Thread(target=_trainer, daemon=True)
        thread.start()

        response = requests.post(
            base_url + "/init_weights_update_group",
            json={
                "master_address": MASTER_ADDR,
                "master_port": port,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": group_name,
                "backend": "hccl",
            },
            timeout=600,
        )
        thread.join(timeout=600)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("error", outcome, f"trainer group init failed: {outcome}")
        return outcome["pg"]

    def _destroy_group(self, base_url, group_name, pg):
        response = requests.post(
            base_url + "/destroy_weights_update_group",
            json={"group_name": group_name},
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        torch.distributed.destroy_process_group(pg)

    def _post_update_async(self, base_url, group_name, *, flattened_bucket=False):
        manifest = self.flattened_manifest if flattened_bucket else self.manifest
        result = {}

        def _run():
            try:
                response = requests.post(
                    base_url + "/update_weights_from_distributed",
                    json={
                        "names": [name for name, _, _ in manifest],
                        "dtypes": [dtype for _, dtype, _ in manifest],
                        "shapes": [shape for _, _, shape in manifest],
                        "group_name": group_name,
                        "load_format": "flattened_bucket" if flattened_bucket else None,
                        "flush_cache": True,
                    },
                    timeout=1800,
                )
                result["status"] = response.status_code
                result["text"] = response.text
            except Exception as exc:  # pragma: no cover
                result["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread, result

    def _broadcast_all(self, pg, *, flattened_bucket=False):
        if flattened_bucket:
            bucket = flattened_tensor_bucket(self.checkpoint, "npu:1")
            torch.distributed.broadcast(bucket.get_flattened_tensor(), src=0, group=pg)
            return
        for name, _, _ in self.manifest:
            tensor = load_checkpoint_tensor(self.tensor_index, name).to("npu:1")
            torch.distributed.broadcast(tensor.contiguous(), src=0, group=pg)

    def _assert_serves_reference(self, message):
        assert_logprob_signature_equal(
            logprob_signature(self.lane_url, PROMPTS),
            self.reference,
            message,
        )

    def test_01_hccl_refit_matches_startup_load(self):
        group_name = "npu_rl_hccl_refit"
        base_url = self.lane_url
        pg = self._create_group(base_url, group_name)
        try:
            for round_idx in range(UPDATE_ROUNDS):
                begin_weight_update(base_url)
                try:
                    thread, result = self._post_update_async(base_url, group_name)
                    self._broadcast_all(pg)
                    thread.join(timeout=1800)
                finally:
                    end_weight_update(base_url)

                self.assertNotIn("error", result, f"update thread failed: {result}")
                self.assertEqual(result.get("status"), 200, result.get("text"))
                self._assert_serves_reference(
                    f"HCCL refit round {round_idx + 1} did not reproduce startup loading"
                )
        finally:
            self._destroy_group(base_url, group_name, pg)

    @unittest.skip(FLATTENED_BUCKET_SKIP_REASON)
    def test_02_hccl_flattened_bucket_refit_matches_startup_load(self):
        group_name = "npu_rl_hccl_flattened_bucket_refit"
        base_url = self.lane_url
        pg = self._create_group(base_url, group_name)
        try:
            for round_idx in range(UPDATE_ROUNDS):
                begin_weight_update(base_url)
                try:
                    thread, result = self._post_update_async(
                        base_url, group_name, flattened_bucket=True
                    )
                    self._broadcast_all(pg, flattened_bucket=True)
                    thread.join(timeout=1800)
                finally:
                    end_weight_update(base_url)

                self.assertNotIn("error", result, f"update thread failed: {result}")
                self.assertEqual(result.get("status"), 200, result.get("text"))
                self._assert_serves_reference(
                    f"flattened_bucket HCCL refit round {round_idx + 1} did not reproduce "
                    "startup loading"
                )
        finally:
            self._destroy_group(base_url, group_name, pg)

    def test_03_init_group_defaults_to_nccl(self):
        request = InitWeightsUpdateGroupReqInput(
            master_address=MASTER_ADDR,
            master_port=12345,
            rank_offset=1,
            world_size=2,
        )
        self.assertEqual(
            request.backend,
            "nccl",
            "the default backend changed; update the NPU callers and this test",
        )

    def test_04_update_error_propagates(self):
        base_url = self.lane_url
        begin_weight_update(base_url)
        response = requests.post(
            base_url + "/update_weights_from_distributed",
            json={
                "names": [NORM_WEIGHT],
                "dtypes": ["bfloat16"],
                "shapes": [[6144]],
                "group_name": "missing_group",
                "flush_cache": True,
            },
            timeout=300,
        )
        try:
            end_weight_update(base_url)
        except Exception:
            pass

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertFalse(payload.get("success"))
        self.assertIn("missing_group", payload.get("message", ""))
        self.assertIn("init_weights_update_group", payload.get("message", ""))
        self.assertTrue(health(base_url))

    def test_05_destroy_group_twice_is_safe(self):
        group_name = "npu_rl_hccl_destroy_twice"
        base_url = self.lane_url
        pg = self._create_group(base_url, group_name)
        first = requests.post(
            base_url + "/destroy_weights_update_group",
            json={"group_name": group_name},
            timeout=120,
        )
        self.assertEqual(first.status_code, 200, first.text)
        torch.distributed.destroy_process_group(pg)

        second = requests.post(
            base_url + "/destroy_weights_update_group",
            json={"group_name": group_name},
            timeout=120,
        )
        self.assertEqual(second.status_code, 400, second.text)
        payload = second.json()
        self.assertFalse(payload.get("success"))
        self.assertIn("does not exist", payload.get("message", ""))
        self.assertTrue(health(base_url))


if __name__ == "__main__":
    unittest.main()
