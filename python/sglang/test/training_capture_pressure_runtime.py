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
    """Use the real Store/Catalog fixture with one ordinary AR endpoint."""

    tp_size = 1
    pp_size = 1

    def rank_names(self):
        return [
            f"pp{pp}-tp{tp}" for pp in range(self.pp_size) for tp in range(self.tp_size)
        ]

    def all_states(self, root, url):
        probe = root / "rank-states"
        probe.mkdir(exist_ok=True)
        nonce = str(time.monotonic_ns())
        (probe / "request").write_text(nonce)
        response = requests.get(url + "/server_info", timeout=10)
        response.raise_for_status()
        deadline = time.monotonic() + 10
        states = {}
        while len(states) != len(self.rank_names()):
            for rank in self.rank_names():
                path = probe / (rank + ".json")
                if path.exists():
                    value = json.loads(path.read_text())
                    if value["nonce"] == nonce:
                        states[rank] = value["state"]
            self.assertLess(time.monotonic(), deadline, (nonce, states))
            time.sleep(0.01)
        return states

    def exercise_pressure(self, *, cuda_graph, overlap):
        prefix = f"ar-pressure-tp{self.tp_size}-pp{self.pp_size}-{cuda_graph}-{overlap}"
        distributed = self.tp_size * self.pp_size > 1
        root = self.root / prefix
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
            "teacher_d2h_batch_tokens": 16,
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
                timeout=600,
                env={**os.environ, "TRAINING_CAPTURE_TEST_OUTPUT": str(root)},
                other_args=[
                    "--tp-size",
                    str(self.tp_size),
                    "--pp-size",
                    str(self.pp_size),
                    "--pp-max-micro-batch-size",
                    str(4 // self.pp_size),
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--enable-metrics",
                    "--enable-metrics-for-all-schedulers",
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
                current = self.all_states(root, url)
                if all(
                    state["states"].get("available", 0) == 4
                    and state.get("test_service_admission_ready", True)
                    for state in current.values()
                ):
                    return current
                self.assertLess(time.monotonic(), deadline, current)
                time.sleep(0.03)

        def retraction_metric():
            response = requests.get(url + "/metrics", timeout=10)
            response.raise_for_status()
            values = dict.fromkeys(self.rank_names(), 0)
            for family in text_string_to_metric_families(response.text):
                for sample in family.samples:
                    if sample.name == "sglang:num_retracted_requests_total":
                        rank = f"pp{sample.labels.get('pp_rank', 0)}-tp{sample.labels.get('tp_rank', 0)}"
                        values[rank] += sample.value
            return values

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
            metric_after = retraction_metric()
            for rank in self.rank_names():
                self.assertEqual(
                    metric_after[rank] - metric_before[rank], retractions, rank
                )
                old, current = before[rank], after[rank]
                self.assertEqual(
                    current["counters"].get("admitted", 0)
                    - old["counters"].get("admitted", 0),
                    4,
                    rank,
                )
                failures = {
                    k: v - old["counters"].get(k, 0)
                    for k, v in current["counters"].items()
                    if k.startswith("failed_") and v > old["counters"].get(k, 0)
                }
                allowed = {"failed_request_aborted_or_retracted"}
                if distributed:
                    allowed.add("failed_peer_capture_failed")
                self.assertLessEqual(failures.keys(), allowed, (rank, failures))
                self.assertEqual(sum(failures.values()), len(retired), (rank, failures))
                self.assertEqual(current["host_pool"]["quarantined"], 0, rank)
            self.assertEqual(
                sum(s["counters"].get("ready", 0) for s in after.values())
                - sum(s["counters"].get("ready", 0) for s in before.values()),
                len(expected),
            )
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
            for rank, state in final.items():
                self.assertEqual(
                    state["counters"]["admitted"]
                    - before[rank]["counters"].get("admitted", 0),
                    5,
                    rank,
                )
                self.assertEqual(state["host_pool"]["quarantined"], 0, rank)
                self.assertEqual(state["enable_overlap"], overlap, rank)
                self.assertEqual(
                    state["counters"].get("cuda_graph_forwards", 0) > 0,
                    cuda_graph,
                    rank,
                )
                self.assertIsNone(state["disabled_reason"], rank)
                self.assertFalse(state["admission_paused"], rank)
                self.assertEqual(state["queued"], 0, rank)
            self.assertEqual(
                sum(s["counters"].get("ready", 0) for s in final.values())
                - sum(s["counters"].get("ready", 0) for s in before.values()),
                len(expected),
            )
            self.assertFalse(self.catalog.errors)
        finally:
            self.stop_process(process)

        references, rank_observations = [], {}
        first_admissions = None
        for pp in range(self.pp_size):
            for tp in range(self.tp_size):
                rank = f"pp{pp}-tp{tp}"
                admissions = [
                    json.loads(line)
                    for line in (root / f"admissions-{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                by_request = {a["rid"]: a["capture_id"] for a in admissions}
                self.assertEqual(len(admissions), 5, rank)
                self.assertTrue(
                    any(set(a["active_capture_rids"]) == set(rids) for a in admissions),
                    (rank, "four captures never active together"),
                )
                self.assertEqual(set(by_request), {*rids, fresh_rid}, rank)
                self.assertEqual(len(set(by_request.values())), 5, rank)
                if first_admissions is None:
                    first_admissions = by_request
                self.assertEqual(by_request, first_admissions, rank)
                for rid in retired:
                    capture_id = by_request[rid]
                    record = self.catalog.captures[capture_id]
                    self.assertEqual(record["state"], "FAILED", (rank, rid))
                    self.assertEqual(
                        record["reason"],
                        "cohort_failed"
                        if distributed
                        else "request_aborted_or_retracted",
                        (rank, rid),
                    )
                    self.assertNotIn(capture_id, self.catalog.publications)
                self.assertTrue(
                    all(
                        by_request[rid] in self.catalog.publications
                        for rid in set(by_request) - retired
                    ),
                    rank,
                )
                events = [
                    json.loads(line)
                    for line in (root / f"retractions-{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                observed_retired, capture_ids = set(), set()
                for event in events:
                    self.assertFalse(event["debug_retract"], rank)
                    self.assertLess(
                        event["available_before"], event["required_next_decode"], rank
                    )
                    self.assertGreater(
                        event["available_after"], event["available_before"], rank
                    )
                    self.assertFalse(event["aborted"], rank)
                    for req in event["retracted"]:
                        observed_retired.add(req["rid"])
                        self.assertTrue(req["context_detached"], rank)
                        self.assertTrue(req["finalizer_detached"], rank)
                        self.assertTrue(req["capture_attempted"], rank)
                        self.assertTrue(req["is_retracted"], rank)
                        if req["capture_id"] is not None:
                            self.assertEqual(
                                req["capture_id"], by_request[req["rid"]], rank
                            )
                            self.assertNotIn(req["capture_id"], capture_ids, rank)
                            capture_ids.add(req["capture_id"])
                self.assertEqual(observed_retired, retired, rank)
                self.assertEqual(
                    sum(len(e["retracted"]) for e in events), retractions, rank
                )
                directory = root / "capture-reference"
                if self.pp_size > 1:
                    directory /= f"pp{pp}"
                if self.tp_size > 1:
                    directory /= f"tp{tp}"
                frames = [
                    (int(path.stem), torch.load(path, weights_only=True))
                    for path in sorted(directory.glob("*.pt"))
                ]
                reused_slots = set()
                for event in events:
                    released = {
                        slot for req in event["retracted"] for slot in req["slots"]
                    }
                    for sequence, reference in frames:
                        if (
                            sequence > event["reference_boundary"]
                            and reference["trace_id"] in expected
                        ):
                            reused_slots.update(
                                released.intersection(reference["kv_slots"].tolist())
                            )
                self.assertTrue(
                    reused_slots,
                    (rank, "released KV slots were not reused by captured requests"),
                )
                local = [frame for _, frame in frames]
                self.assertTrue(
                    any(r["batch_size"] == 4 // self.pp_size for r in local), rank
                )
                if cuda_graph:
                    # PP splits the four requests into two concurrent microbatches.
                    for size in (1, 2) if self.pp_size > 1 else (3,):
                        self.assertTrue(
                            any(
                                r["cuda_graph"]
                                and r["forward_mode"] == "DECODE"
                                and r["batch_size"] == size
                                for r in local
                            ),
                            (rank, size),
                        )
                if overlap:
                    self.assertTrue(any(r["result_lag"] == 1 for r in local), rank)
                if pp < self.pp_size - 1:
                    self.assertFalse(any(r["predictions"] for r in local), rank)
                else:
                    self.assertTrue(any(r["predictions"] for r in local), rank)
                references.extend(local)
                rank_observations[rank] = {
                    "admissions": by_request,
                    "peak_active_captures": max(
                        len(a["active_capture_rids"]) for a in admissions
                    ),
                    "observed_batch_sizes": sorted({r["batch_size"] for r in local}),
                    "decode_graph_batch_sizes": sorted(
                        {
                            r["batch_size"]
                            for r in local
                            if r["cuda_graph"] and r["forward_mode"] == "DECODE"
                        }
                    ),
                    "reused_kv_slots": len(reused_slots),
                    "source_frames": len(local),
                    "graph_frames": sum(r["cuda_graph"] for r in local),
                    "metric_retractions": metric_after[rank] - metric_before[rank],
                    "events": [
                        {k: v for k, v in e.items() if k != "retracted"}
                        | {
                            "retracted": [
                                {k: v for k, v in req.items() if k != "slots"}
                                for req in e["retracted"]
                            ]
                        }
                        for e in events
                    ],
                }
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        objects, tensor_bytes = 0, 0
        for publication in publications:
            manifest, tensors = read_snapshot(reader, publication)
            self.assertEqual(
                (manifest.topology.tp_size, manifest.topology.pp_size),
                (self.tp_size, self.pp_size),
            )
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
                        "tp_size": self.tp_size,
                        "pp_size": self.pp_size,
                        "cuda_graph": cuda_graph,
                        "overlap": overlap,
                        "kv_pool_tokens": 256,
                        "requested_path_tokens": 384,
                        "completed_requests": len(results),
                        "output_tokens": 320,
                        "num_retractions": retractions,
                        "retired_requests": sorted(retired),
                        "reused_kv_slots": sum(
                            r["reused_kv_slots"] for r in rank_observations.values()
                        ),
                        "source_frames": len(references),
                        "ready_samples": len(publications),
                        "tensor_objects": objects,
                        "tensor_bytes": tensor_bytes,
                        "producer_exited": process.poll() is not None,
                        "rank_observations": rank_observations,
                        "capture_after": final,
                    }
                }
            ),
            flush=True,
        )
