"""AR requests survive real KV exhaustion while incomplete captures retire."""

import hashlib
import json
import os
import sys
import time
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.environ import envs
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.training_capture_utils import read_snapshot


class ARPressureCaptureRuntimeBase(PDCaptureRuntimeBase):
    """Use the real Store/Catalog fixture with a single ordinary AR server."""

    def exercise_pressure(self, *, cuda_graph, overlap):
        root = self.root / f"ar-pressure-{cuda_graph}-{overlap}"
        root.mkdir()
        config = {
            "dataset_id": "runtime-ar-pressure",
            "model_id": "Qwen/Qwen3-0.6B",
            "producer_revision": "checkout-under-test",
            "selected_layer_ids": [0, 14, 27],
            "catalog_endpoint": self.catalog.endpoint,
            "journal_directory": str(root / "journal"),
            "store": self.store_setup,
            "sample_ratio": 1.0,
            "max_sample_tokens": 128,
            "max_inflight_samples": 4,
            "max_host_bytes": 64 << 20,
            "kv_d2h_batch_tokens": 16,
            "max_device_bytes": 8 << 20,
            "storage_chunk_tokens": 64,
            "http_timeout_seconds": 2.0,
        }
        path = root / "capture.json"
        path.write_text(json.dumps(config))
        url = f"http://127.0.0.1:{free_port()}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [sys.executable, "-m", "sglang.test.training_capture_pressure_server"]
                + command[2:],
                *args,
            )

        with (
            envs.SGLANG_TEST_RETRACT.override(False),
            patch.object(test_utils, "_launch_server_process", observed_server),
        ):
            process = test_utils.popen_launch_server(
                self.model,
                url,
                timeout=240,
                env={**os.environ, "TRAINING_CAPTURE_TEST_OUTPUT": str(root)},
                other_args=[
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--enable-metrics",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.25",
                    "--max-total-tokens",
                    "256",
                    "--schedule-conservativeness",
                    "0.05",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-backend-decode",
                    "full" if cuda_graph else "disabled",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "4",
                    *([] if overlap else ["--disable-overlap-schedule"]),
                ],
            )
        self.addCleanup(self.stop_process, process)
        first_publication = len(self.catalog.publications)

        def wait_available():
            deadline = time.monotonic() + 30
            while True:
                response = requests.get(url + "/server_info", timeout=10)
                response.raise_for_status()
                current = response.json()["internal_states"][0]["training_capture"]
                if current["states"].get("available", 0) == 4:
                    return current
                self.assertLess(time.monotonic(), deadline, current)
                time.sleep(0.03)

        def retraction_metric():
            response = requests.get(url + "/metrics", timeout=10)
            response.raise_for_status()
            return sum(
                sample.value
                for family in text_string_to_metric_families(response.text)
                for sample in family.samples
                if sample.name == "sglang:num_retracted_requests_total"
            )

        prefix = f"ar-pressure-{cuda_graph}-{overlap}"
        rids = [f"{prefix}-{index}" for index in range(4)]
        prompts = [[17 + i, 900 + i] * 8 for i in range(4)]
        params = {
            "temperature": 0,
            "max_new_tokens": 80,
            "ignore_eos": True,
            "logit_bias": {"100": 100.0},
        }
        expected, retired = {}, set()
        try:
            before = wait_available()
            metric_before = retraction_metric()
            response = requests.post(
                url + "/generate",
                json={"rid": rids, "input_ids": prompts, "sampling_params": params},
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            results = response.json()
            self.assertEqual(len(results), 4)
            retractions = 0
            for rid, prompt, result in zip(rids, prompts, results, strict=True):
                self.assertEqual(result["meta_info"]["id"], rid)
                self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
                self.assertEqual(result["output_ids"], [100] * 80)
                count = result["meta_info"]["num_retractions"]
                retractions += count
                if count:
                    retired.add(rid)
                else:
                    expected[hashlib.sha256(rid.encode()).hexdigest()] = (
                        prompt + result["output_ids"]
                    )
            self.assertTrue(retired, "no automatic retraction under KV pressure")
            self.assertTrue(expected, "no capture survived the pressure workload")
            after = wait_available()
            self.assertEqual(retraction_metric() - metric_before, retractions)
            for counter, count in (
                ("admitted", 4),
                ("ready", len(expected)),
                ("failed_request_aborted_or_retracted", len(retired)),
            ):
                self.assertEqual(
                    after["counters"].get(counter, 0)
                    - before["counters"].get(counter, 0),
                    count,
                    after,
                )
            self.assertEqual(after["host_pool"]["quarantined"], 0)
            self.assertEqual(
                len(self.catalog.publications), first_publication + len(expected)
            )

            fresh_rid = prefix + "-fresh"
            response = requests.post(
                url + "/generate",
                json={
                    "rid": fresh_rid,
                    "input_ids": prompts[0],
                    "sampling_params": params | {"max_new_tokens": 4},
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            fresh = response.json()
            self.assertEqual(fresh["output_ids"], [100] * 4)
            self.assertEqual(fresh["meta_info"]["num_retractions"], 0)
            expected[hashlib.sha256(fresh_rid.encode()).hexdigest()] = (
                prompts[0] + fresh["output_ids"]
            )
            publications = self.catalog.wait_publications(
                first_publication + len(expected)
            )[first_publication:]
            final = wait_available()
            self.assertEqual(
                final["counters"]["admitted"] - before["counters"].get("admitted", 0), 5
            )
            self.assertEqual(final["host_pool"]["quarantined"], 0)
            self.assertEqual(final["enable_overlap"], overlap)
            self.assertEqual(
                final["counters"].get("cuda_graph_forwards", 0) > 0, cuda_graph
            )
            self.assertFalse(self.catalog.errors)
        finally:
            self.stop_process(process)

        events = [
            json.loads(line)
            for line in (root / "retractions.jsonl").read_text().splitlines()
        ]
        observed_retired, capture_ids = set(), set()
        for event in events:
            self.assertFalse(event["debug_retract"])
            self.assertLess(event["available_before"], event["required_next_decode"])
            self.assertGreater(event["available_after"], event["available_before"])
            self.assertFalse(event["aborted"])
            for req in event["retracted"]:
                observed_retired.add(req["rid"])
                self.assertTrue(req["context_detached"])
                self.assertTrue(req["finalizer_detached"])
                self.assertTrue(req["capture_attempted"])
                self.assertTrue(req["is_retracted"])
                if req["capture_id"] is not None:
                    self.assertNotIn(req["capture_id"], capture_ids)
                    capture_ids.add(req["capture_id"])
                    record = self.catalog.captures[req["capture_id"]]
                    self.assertEqual(record["state"], "FAILED")
                    self.assertEqual(record["reason"], "request_aborted_or_retracted")
                    self.assertNotIn(req["capture_id"], self.catalog.publications)
        self.assertEqual(observed_retired, retired)
        self.assertEqual(len(capture_ids), len(retired))
        self.assertEqual(sum(len(e["retracted"]) for e in events), retractions)

        frames = [
            (int(path.stem), torch.load(path, weights_only=True))
            for path in sorted((root / "capture-reference").glob("*.pt"))
        ]
        reused_slots = set()
        for event in events:
            released = {slot for req in event["retracted"] for slot in req["slots"]}
            for sequence, reference in frames:
                if (
                    sequence > event["reference_boundary"]
                    and reference["trace_id"] in expected
                ):
                    reused_slots.update(
                        released.intersection(reference["kv_slots"].tolist())
                    )
        self.assertTrue(
            reused_slots, "released KV slots were not reused by captured requests"
        )
        references = [frame for _, frame in frames]
        self.assertTrue(any(r["batch_size"] == 4 for r in references))
        if cuda_graph:
            self.assertTrue(
                any(r["cuda_graph"] and r["batch_size"] == 3 for r in references)
            )
        if overlap:
            self.assertTrue(any(r["result_lag"] == 1 for r in references))
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        objects, tensor_bytes = 0, 0
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
                    "ar_memory_pressure": {
                        "cuda_graph": cuda_graph,
                        "overlap": overlap,
                        "kv_pool_tokens": 256,
                        "requested_path_tokens": 384,
                        "completed_requests": len(results),
                        "output_tokens": 320,
                        "num_retractions": retractions,
                        "retired_requests": sorted(retired),
                        "reused_kv_slots": len(reused_slots),
                        "source_frames": len(references),
                        "ready_samples": len(publications),
                        "tensor_objects": objects,
                        "tensor_bytes": tensor_bytes,
                        "producer_exited": process.poll() is not None,
                        "events": [
                            {k: v for k, v in e.items() if k != "retracted"}
                            | {
                                "retracted": [
                                    {
                                        k: v
                                        for k, v in r.items()
                                        if k not in ("slots", "capture_id")
                                    }
                                    for r in e["retracted"]
                                ]
                            }
                            for e in events
                        ],
                        "capture_after": final,
                    }
                }
            ),
            flush=True,
        )
