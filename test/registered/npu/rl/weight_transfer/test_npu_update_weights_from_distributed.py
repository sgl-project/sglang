"""P0 NPU tests: online weight refit over HCCL (disaggregated RL).

NPU 0 hosts the inference engine, NPU 1 hosts the trainer. The trainer creates
an HCCL group through ``/init_weights_update_group`` and broadcasts the
checkpoint shards in the order requested by ``/update_weights_from_distributed``.
"""

import os
import threading
import unittest

import requests
import torch

from sglang.srt.managers.io_struct import InitWeightsUpdateGroupReqInput
from sglang.srt.utils import init_custom_process_group
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    begin_weight_update,
    checkpoint_manifest,
    checkpoint_tensor_index,
    end_weight_update,
    generation_signature,
    health,
    load_checkpoint_tensor,
    npu_available,
    npu_device_count,
    npu_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_npu_ci(est_time=1800, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=1800, suite="nightly-2-npu-a3", nightly=True)

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
MASTER_ADDR = os.environ.get("SGLANG_TEST_HCCL_MASTER_ADDR", "127.0.0.1")
NORM_WEIGHT = "model.norm.weight"


@unittest.skipUnless(
    npu_available() and npu_device_count() >= 2, "requires at least 2 Ascend NPUs"
)
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuUpdateWeightsFromDistributed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with npu_server(MODEL, extra_args=BASE_ARGS) as base_url:
            cls.reference = generation_signature(base_url, PROMPTS)
        cls.manifest = checkpoint_manifest(MODEL)
        cls.tensor_index = checkpoint_tensor_index(MODEL)
        # Keep one lane server for the class: the trainer creates NPU tensors
        # in this process, and launching another server afterwards resets the
        # device on some CANN/driver combinations.
        cls._lane_ctx = npu_server(MODEL, extra_args=LANE_ARGS)
        cls.lane_url = cls._lane_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._lane_ctx.__exit__(None, None, None)

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

    def _post_update_async(self, base_url, group_name):
        result = {}

        def _run():
            try:
                response = requests.post(
                    base_url + "/update_weights_from_distributed",
                    json={
                        "names": [name for name, _, _ in self.manifest],
                        "dtypes": [dtype for _, dtype, _ in self.manifest],
                        "shapes": [shape for _, _, shape in self.manifest],
                        "group_name": group_name,
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

    def _broadcast_all(self, pg):
        for name, _, _ in self.manifest:
            tensor = load_checkpoint_tensor(self.tensor_index, name).to("npu:1")
            torch.distributed.broadcast(tensor.contiguous(), src=0, group=pg)

    def test_01_hccl_refit_matches_startup_load(self):
        group_name = "npu_rl_hccl_refit"
        base_url = self.lane_url
        pg = self._create_group(base_url, group_name)
        try:
            begin_weight_update(base_url)
            thread, result = self._post_update_async(base_url, group_name)
            self._broadcast_all(pg)
            thread.join(timeout=1800)
            end_weight_update(base_url)

            self.assertNotIn("error", result, f"update thread failed: {result}")
            self.assertEqual(result.get("status"), 200, result.get("text"))
            self.assertEqual(
                generation_signature(base_url, PROMPTS),
                self.reference,
                "HCCL refit did not reproduce startup loading",
            )
        finally:
            self._destroy_group(base_url, group_name, pg)

    def test_02_init_group_defaults_to_nccl(self):
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

    def test_03_update_error_propagates(self):
        base_url = self.lane_url
        begin_weight_update(base_url)
        response = requests.post(
            base_url + "/update_weights_from_distributed",
            json={
                "names": [NORM_WEIGHT],
                "dtypes": ["bfloat16"],
                "shapes": [[1024]],
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

    def test_04_destroy_group_twice_is_safe(self):
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
