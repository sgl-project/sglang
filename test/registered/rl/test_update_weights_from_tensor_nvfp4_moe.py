"""Partial weight updates must not disturb the NVFP4 MoE experts they skip.

end_weight_update post-processes every MoE layer, and NVFP4 MoE post-processing
permutes FP4 weights and block scales in place. Pushing the checkpoint's own
bytes for a few tensors must therefore leave every expert parameter
bit-identical, including the MoE layers and experts the update did not touch.
The /weights_checker raw_checksum action hashes each parameter's raw bytes.
"""

import json
import os
import unittest
from typing import List, Tuple

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang.srt.utils import MultiprocessingSerializer, get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="nightly", runner_config="1-gpu-small")

_MODEL_NAME = "nvidia/Qwen3-30B-A3B-NVFP4"
_EXPERT_PREFIX = "model.layers.0.mlp.experts.0"
_EXPERT_TENSORS = [
    f"{_EXPERT_PREFIX}.{proj}.{suffix}"
    for proj in ("gate_proj", "up_proj", "down_proj")
    for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
]


def _checkpoint_tensors(names: List[str]) -> List[Tuple[str, torch.Tensor]]:
    path = snapshot_download(_MODEL_NAME, allow_patterns=["*.json", "*.safetensors"])
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    tensors = []
    for name in names:
        with safe_open(os.path.join(path, weight_map[name]), framework="pt") as f:
            tensors.append((name, f.get_tensor(name).cuda()))
    return tensors


@unittest.skipIf(get_device_sm() < 100, "NVFP4 MoE requires SM100+")
class TestUpdateWeightsFromTensorNvFp4Moe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = DEFAULT_URL_FOR_TEST
        other_args = ["--mem-fraction-static", "0.6"]
        if get_device_sm() >= 120:
            # SM100 resolves auto to the TRT-LLM layout; SM120 serves CUTLASS.
            other_args += ["--moe-runner-backend", "flashinfer_cutlass"]
        cls.process = popen_launch_server(
            _MODEL_NAME,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _raw_checksums(self) -> dict:
        resp = requests.post(
            f"{self.url}/weights_checker", json={"action": "raw_checksum"}, timeout=600
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        checksums = resp.json()["ranks"][0]["checksums"]
        return {n: h for n, h in checksums.items() if ".mlp.experts." in n}

    def _update_weights(self, named_tensors: List[Tuple[str, torch.Tensor]]) -> dict:
        begin = requests.post(f"{self.url}/begin_weight_update", json={}, timeout=120)
        self.assertEqual(begin.status_code, 200, begin.text)
        resp = requests.post(
            f"{self.url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": True,
            },
            timeout=600,
        )
        end = requests.post(f"{self.url}/end_weight_update", json={}, timeout=600)
        self.assertEqual(end.status_code, 200, end.text)
        return resp.json()

    def test_same_bytes_partial_updates_keep_every_expert(self):
        before = self._raw_checksums()
        self.assertGreater(len(before), 0)
        cases = (
            ("dense tensor only", ["model.norm.weight"]),
            ("one expert of one layer", _EXPERT_TENSORS),
        )
        for name, tensor_names in cases:
            with self.subTest(name=name):
                result = self._update_weights(_checkpoint_tensors(tensor_names))
                self.assertTrue(result["success"], result)
                after = self._raw_checksums()
                changed = sorted(n for n in before if before[n] != after.get(n))
                self.assertEqual(changed, [])


if __name__ == "__main__":
    unittest.main()
