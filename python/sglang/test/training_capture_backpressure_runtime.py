"""A slow auxiliary publisher must throttle every rank without blocking serving."""

import hashlib
import json
import os
import sys
import time
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_pressure_runtime import ARPressureCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot


class CohortBackpressureRuntimeBase(ARPressureCaptureRuntimeBase):
    def exercise_backpressure(self):
        root = self.root / f"backpressure-{self.tp_size}-{self.pp_size}"
        root.mkdir()
        path = root / "capture.json"
        path.write_text(
            json.dumps(
                {
                    "dataset_id": "runtime-cohort-pressure",
                    "model_id": self.model_id,
                    "producer_revision": "checkout-under-test",
                    "selected_layer_ids": [0, 14, 27],
                    "catalog_endpoint": self.catalog.endpoint,
                    "journal_directory": str(root / "journal"),
                    "store": self.store_setup,
                    "sample_ratio": 1.0,
                    "max_sample_tokens": 64,
                    "max_inflight_samples": 2,
                    "max_host_bytes": 32 << 20,
                    "kv_d2h_batch_tokens": 16,
                    "teacher_d2h_batch_tokens": 16,
                    "max_device_bytes": 8 << 20,
                    "storage_chunk_tokens": 64,
                    "adaptive": {
                        "interval_seconds": 0.1,
                        "writer_stall_seconds": 0.5,
                        "cooldown_seconds": 0.2,
                    },
                }
            )
        )
        url = f"http://127.0.0.1:{self.new_bootstrap_port()}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [
                    sys.executable,
                    "-m",
                    "sglang.test.training_capture_backpressure_server",
                ]
                + command[2:],
                *args,
            )

        with patch.object(test_utils, "_launch_server_process", observed_server):
            process = test_utils.popen_launch_server(
                self.model,
                url,
                timeout=600,
                env={**os.environ, "TRAINING_CAPTURE_TEST_OUTPUT": str(root)},
                other_args=[
                    "--tp-size",
                    str(self.tp_size),
                    "--pp-size",
                    str(self.pp_size),
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.25",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--enable-metrics",
                    "--enable-metrics-for-all-schedulers",
                    "--cuda-graph-config",
                    json.dumps(
                        {
                            "prefill": {"backend": "disabled"},
                            "decode": {"backend": "full", "bs": [1, 2, 4], "max_bs": 4},
                        }
                    ),
                    *(["--disable-overlap-schedule"] if self.pp_size > 1 else []),
                ],
            )
        self.addCleanup(self.stop_process, process)
        gate = root / "manifest.pause"
        first = len(self.catalog.publications)
        expected = {}

        def wait_for(predicate):
            deadline = time.monotonic() + 30
            while True:
                states = self.all_states(root, url)
                if predicate(states):
                    return states
                self.assertLess(time.monotonic(), deadline, states)
                time.sleep(0.03)

        def healthy(states):
            return all(
                s["states"].get("available", 0) == 2
                and s["admission"]["effective_ratio"] == 1
                for s in states.values()
            )

        def generate(rid, prompt):
            response = requests.post(
                url + "/generate",
                json={
                    "rid": rid,
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4,
                        "ignore_eos": True,
                        "logit_bias": {"100": 100.0},
                    },
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["output_ids"], [100] * 4)
            self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
            return prompt + result["output_ids"]

        try:
            wait_for(healthy)
            rid = "healthy-before"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(
                rid, [17, 18] * 8
            )
            self.catalog.wait_publications(first + 1)
            before = wait_for(healthy)
            gate.touch()
            rid = "stalled-manifest"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(
                rid, [19, 20] * 8
            )
            blocked = wait_for(
                lambda states: all(
                    s["admission"]["effective_ratio"] == 0
                    and s["states"].get("available", 0) == 0
                    for s in states.values()
                )
            )
            started = json.loads((root / "manifest.started").read_text())
            aux = f"pp{self.pp_size - 1}-tp0"
            self.assertEqual(started["rank"], aux)
            self.assertEqual(blocked[aux]["admission"]["reason"], "writer_stall")
            for rank, state in blocked.items():
                self.assertEqual(state["admission"]["cohort_effective_ratio"], 0)
                self.assertEqual(state["counters"]["admitted"], 2)
                if rank != aux:
                    self.assertEqual(state["cohort_writer"]["pending"], 0, rank)
                    self.assertEqual(
                        state["admission"]["local_effective_ratio"], 1, rank
                    )
                self.assertIsNone(state["disabled_reason"], rank)
            self.assertEqual(len(self.catalog.publications), first + 1)

            sampled_before = blocked["pp0-tp0"]["request_router"].get("sampled_out", 0)
            started_at = time.monotonic()
            for index in range(8):
                generate(f"during-stall-{index}", [23, 24] * 8)
            inference_seconds = time.monotonic() - started_at
            still_blocked = self.all_states(root, url)
            self.assertEqual(
                still_blocked["pp0-tp0"]["request_router"]["sampled_out"]
                - sampled_before,
                8,
            )
            for rank, state in still_blocked.items():
                self.assertEqual(state["counters"]["admitted"], 2, rank)
                self.assertEqual(state["admission"]["effective_ratio"], 0, rank)
                self.assertEqual(state["host_pool"]["quarantined"], 0, rank)
                self.assertEqual(
                    state["host_pool"]["allocated_bytes"],
                    before[rank]["host_pool"]["allocated_bytes"],
                    rank,
                )
            deadline = time.monotonic() + 10
            while True:
                response = requests.get(url + "/metrics", timeout=10)
                response.raise_for_status()
                ratios = {}
                for family in text_string_to_metric_families(response.text):
                    for sample in family.samples:
                        if (
                            sample.name == "sglang:training_capture_sample_ratio"
                            and sample.labels.get("kind") == "effective"
                        ):
                            ratios[
                                f"pp{sample.labels['pp_rank']}-tp{sample.labels['tp_rank']}"
                            ] = sample.value
                if ratios == dict.fromkeys(self.rank_names(), 0):
                    break
                # Gauges refresh independently of request and control threads.
                self.assertLess(time.monotonic(), deadline, ratios)
                time.sleep(0.05)
            self.assertEqual(len(self.catalog.publications), first + 1)
            gate.unlink()
            self.catalog.wait_publications(first + 2)
            wait_for(healthy)
            rid = "healthy-after"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(
                rid, [25, 26] * 8
            )
            publications = self.catalog.wait_publications(first + 3)[first:]
            final = wait_for(healthy)
            for rank, state in final.items():
                self.assertEqual(state["counters"]["admitted"], 3, rank)
                self.assertFalse(
                    any(
                        k.startswith("failed_") and v
                        for k, v in state["counters"].items()
                    ),
                    rank,
                )
                self.assertEqual(state["host_pool"]["quarantined"], 0, rank)
                self.assertEqual(state["cohort_writer"]["pending"], 0, rank)
            self.assertEqual(
                sum(s["counters"].get("ready", 0) for s in final.values()), 3
            )
            self.assertFalse(self.catalog.errors)
        finally:
            gate.unlink(missing_ok=True)
            self.stop_process(process)

        references = [torch.load(p, weights_only=True) for p in root.rglob("*.pt")]
        self.assertTrue(any(f["cuda_graph"] for f in references))
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        objects = tensor_bytes = 0
        for publication in publications:
            manifest, tensors = read_snapshot(reader, publication)
            self.assertEqual(
                tensors["token_ids"].tolist(),
                expected.pop(manifest.provenance.trace_id),
            )
            check_capture_snapshot(
                self, manifest, tensors, references, capture_mode="autoregressive"
            )
            objects += len(manifest.objects)
            tensor_bytes += sum(obj.nbytes for obj in manifest.objects)
        self.assertFalse(expected)
        print(
            json.dumps(
                {
                    "cohort_backpressure": {
                        "tp_size": self.tp_size,
                        "pp_size": self.pp_size,
                        "blocked_publisher": aux,
                        "requests_completed": 11,
                        "requests_while_stalled": 8,
                        "stalled_inference_seconds": inference_seconds,
                        "post_exit_snapshots": 3,
                        "tensor_objects": objects,
                        "tensor_bytes": tensor_bytes,
                        "producer_exited": process.poll() is not None,
                        "blocked": blocked,
                        "final": final,
                    }
                }
            ),
            flush=True,
        )
