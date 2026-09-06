"""Publish A@2 before live A@1/B/C streams finish, without changing their outputs."""

import concurrent.futures
import json
import os
import threading
import time
import unittest

import numpy as np
import requests
import torch

from sglang.srt.managers.scheduler_components.weight_updater import _sha256_tensor
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-large")


class TestLoRAPublicationOverlap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get(
            "SGLANG_LORA_PUBLICATION_TEST_URL", DEFAULT_URL_FOR_TEST
        )
        cls.process = popen_launch_server(
            os.environ.get("SGLANG_LORA_PUBLICATION_TEST_MODEL", "Qwen/Qwen3-0.6B"),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--max-loras-per-batch",
                "4",
                "--lora-target-modules",
                "q_proj",
                "k_proj",
                "v_proj",
                "--lora-backend",
                "triton",
                "--mem-fraction-static",
                "0.15",
                "--context-length",
                "4096",
                "--cuda-graph-max-bs-decode",
                "4",
                "--max-running-requests",
                "8",
                "--incremental-streaming-output",
                "--random-seed",
                "11",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, endpoint, payload):
        response = requests.post(
            f"{self.base_url}/{endpoint}", json=payload, timeout=120
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIsNot(result.get("success"), False, result)
        return result

    def _weights(self, seed):
        rng = torch.Generator().manual_seed(seed)
        tensors = {}
        for layer in range(28):
            for module, out_features in (
                ("q_proj", 2048),
                ("k_proj", 1024),
                ("v_proj", 1024),
            ):
                prefix = f"model.layers.{layer}.self_attn.{module}"
                tensors[f"{prefix}.lora_A.weight"] = (
                    torch.randn(8, 1024, generator=rng, dtype=torch.bfloat16) * 0.04
                )
                tensors[f"{prefix}.lora_B.weight"] = (
                    torch.randn(out_features, 8, generator=rng, dtype=torch.bfloat16)
                    * 0.04
                )
        return {name: tensor.cuda() for name, tensor in tensors.items()}

    def _assert_pending(self, name):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello",
                "lora_path": name,
                "sampling_params": {"max_new_tokens": 1},
            },
            timeout=10,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("not ready", response.text)

    def _publish(self, name, tensors):
        result = self._post(
            "register_lora_adapter",
            {
                "lora_name": name,
                "config_dict": {
                    "peft_type": "LORA",
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": ["q_proj", "k_proj", "v_proj"],
                },
                "defer_publish": True,
            },
        )
        self.assertTrue(result["pending"])
        self._assert_pending(name)
        session_id = f"publish-{name}"
        self._post(
            "begin_weight_update",
            {"sync_base": False, "new_lora_names": [name], "session_id": session_id},
        )
        items = [(f"{name}:{key}", tensor) for key, tensor in tensors.items()]
        midpoint = len(items) // 2
        for bucket in (items[:midpoint], items[midpoint:]):
            transfer = [(key, tensor.clone()) for key, tensor in bucket]
            self._post(
                "update_weights_from_tensor",
                {
                    "serialized_named_tensors": [
                        MultiprocessingSerializer.serialize(transfer, output_str=True)
                    ],
                    "flush_cache": False,
                    "session_id": session_id,
                },
            )
            for _, tensor in transfer:
                tensor.zero_()
            torch.cuda.synchronize()
            self._assert_pending(name)
        self._post(
            "end_weight_update",
            {
                "session_id": session_id,
                "expected_lora_checksums": {
                    name: {
                        key: _sha256_tensor(tensor) for key, tensor in tensors.items()
                    }
                },
            },
        )

    def _stream(self, name, started, progress):
        payload = {
            "text": "Continue counting integers, separated by commas: 1, 2, 3,",
            "lora_path": name,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1536,
                "ignore_eos": True,
            },
        }
        tokens = []
        with requests.post(
            f"{self.base_url}/generate", json=payload, stream=True, timeout=120
        ) as response:
            self.assertEqual(
                response.status_code,
                200,
                response.text if response.status_code != 200 else "",
            )
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                self.assertNotIn("error", chunk, chunk)
                tokens.extend(chunk.get("output_ids", []))
                progress[name] = len(tokens)
                if len(tokens) >= 8:
                    started.set()
        self.assertEqual(len(tokens), 1536)
        return tokens, time.monotonic()

    def _probe(self, name):
        result = self._post(
            "generate",
            {
                "text": "The quick brown fox jumps over the lazy dog. Explain what happened.",
                "lora_path": name,
                "return_logprob": True,
                "logprob_start_len": 0,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )
        return np.array(
            [
                item[0]
                for item in result["meta_info"]["input_token_logprobs"]
                if item[0] is not None
            ]
        )

    def test_publish_completes_before_existing_streams_and_preserves_old_versions(self):
        names = ("A@1", "B", "C")
        for seed, name in enumerate(names, start=1):
            self._publish(name, self._weights(seed))
        before = {name: self._probe(name) for name in names}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            baseline = {
                name: pool.submit(self._stream, name, threading.Event(), {})
                for name in names
            }
            expected = {
                name: future.result(timeout=120)[0] for name, future in baseline.items()
            }
            started = {name: threading.Event() for name in names}
            progress = {}
            active = {
                name: pool.submit(self._stream, name, started[name], progress)
                for name in names
            }
            for event in started.values():
                self.assertTrue(event.wait(30), "generation did not start")
            self.assertTrue(all(not future.done() for future in active.values()))
            publish_start = time.monotonic()
            self._publish("A@2", self._weights(4))
            published = time.monotonic()
            self.assertTrue(
                all(not future.done() for future in active.values()), progress
            )
            progress_at_publish = dict(progress)
            results = {
                name: future.result(timeout=120) for name, future in active.items()
            }
        for name, (tokens, finished) in results.items():
            self.assertLess(published, finished)
            self.assertEqual(
                tokens, expected[name], f"{name} changed during A@2 publication"
            )
            np.testing.assert_allclose(
                self._probe(name), before[name], atol=0.01, rtol=0.001
            )
        self.assertGreater(np.max(np.abs(self._probe("A@2") - before["A@1"])), 0.001)
        print(
            json.dumps(
                {
                    "publish_seconds": published - publish_start,
                    "tokens_at_publish": progress_at_publish,
                    "remaining_generation_seconds": {
                        name: finish - published
                        for name, (_, finish) in results.items()
                    },
                    "old_version_tokens_unchanged": True,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
