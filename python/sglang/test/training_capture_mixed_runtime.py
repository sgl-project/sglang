"""Real mixed prefill/decode capture, including an unselected prefill request."""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests
import torch

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test import test_utils
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.training_capture_utils import read_snapshot


class MixedCaptureRuntimeBase(PDCaptureRuntimeBase):
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

    def exercise_mixed(self, *, overlap, prefill="disabled", decode="disabled"):
        name = f"mixed-tp{self.tp_size}-pp{self.pp_size}-{overlap}-{prefill}-{decode}"
        first = len(self.catalog.publications)
        results, summaries = [], []
        captured = None
        for enabled in (False, True):
            root = self.root / f"{name}-{enabled}"
            root.mkdir()
            config = {
                "dataset_id": "runtime-mixed",
                "model_id": self.model_id,
                "producer_revision": "checkout-under-test",
                "selected_layer_ids": [0, 14, 27],
                "catalog_endpoint": self.catalog.endpoint,
                "journal_directory": str(root / "journal"),
                "store": self.store_setup,
                "sample_ratio": 1.0,
                "max_sample_tokens": 512,
                "max_inflight_samples": 4,
                "max_host_bytes": 64 << 20,
                "max_device_bytes": 8 << 20,
                "kv_d2h_batch_tokens": 16,
                "teacher_d2h_batch_tokens": 16,
                "storage_chunk_tokens": 64,
            }
            path = root / "capture.json"
            path.write_text(json.dumps(config))
            graph = {
                "prefill": {"backend": prefill, "bs": [16, 32, 64, 128], "max_bs": 128},
                "decode": {"backend": decode, "bs": [1, 2, 4], "max_bs": 4},
            }
            if prefill == "full":
                graph["prefill"]["full_prefill_max_req"] = 4
            launch = test_utils._launch_server_process

            def observed_server(command, *args, launch=launch):
                return launch(
                    [sys.executable, "-m", "sglang.test.training_capture_mixed_server"]
                    + command[2:],
                    *args,
                )

            url = f"http://127.0.0.1:{free_port()}"
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
                        "--pp-max-micro-batch-size",
                        "4",
                        *(["--training-capture-config", str(path)] if enabled else []),
                        "--enable-mixed-chunk",
                        "--skip-server-warmup",
                        "--skip-tokenizer-init",
                        "--attention-backend",
                        "flashinfer",
                        "--disable-flashinfer-autotune",
                        "--mem-fraction-static",
                        "0.25",
                        "--max-total-tokens",
                        "4096",
                        "--max-running-requests",
                        "4",
                        "--max-prefill-tokens",
                        "128",
                        "--chunked-prefill-size",
                        "128",
                        "--cuda-graph-config",
                        json.dumps(graph),
                        *(
                            ["--enable-torch-compile-debug-mode"]
                            if prefill == "tc_piecewise"
                            else []
                        ),
                        *([] if overlap else ["--disable-overlap-schedule"]),
                    ],
                )
            self.addCleanup(self.stop_process, process)
            prompts = {
                "decode": list(range(900, 917)),
                "partial": [100, 200, 300, 400] * 67 + [501, 502, 503],
                "one": list(range(3000, 3033)),
                "excluded": list(range(4000, 4529)),
                "fresh": list(range(6000, 6040)),
            }
            prompts["cached"] = prompts["partial"]
            lengths = {
                "decode": 24,
                "partial": 6,
                "one": 1,
                "excluded": 1,
                "cached": 4,
                "fresh": 4,
            }
            biases = {"decode": 100, "partial": 101, "excluded": 102, "cached": 103}

            def generate(
                kind, *, url=url, prompts=prompts, lengths=lengths, biases=biases
            ):
                response = requests.post(
                    url + "/generate",
                    json={
                        "rid": f"{name}-{kind}",
                        "input_ids": prompts[kind],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": lengths[kind],
                            "ignore_eos": True,
                            **(
                                {"logit_bias": {str(biases[kind]): 100.0}}
                                if kind in biases
                                else {}
                            ),
                        },
                    },
                    timeout=120,
                )
                self.assertEqual(response.status_code, 200, response.text)
                value = response.json()
                self.assertEqual(len(value["output_ids"]), lengths[kind])
                if kind in biases:
                    self.assertEqual(
                        value["output_ids"], [biases[kind]] * lengths[kind]
                    )
                return value["output_ids"]

            try:
                info = requests.get(url + "/server_info", timeout=10).json()
                self.assertTrue(info["enable_mixed_chunk"])
                self.assertEqual(info["training_capture_config"] is not None, enabled)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    future = pool.submit(generate, "decode")
                    deadline = time.monotonic() + 30
                    while not (root / "decode-ready").exists():
                        self.assertLess(
                            time.monotonic(), deadline, "decode gate not reached"
                        )
                        if future.done():
                            self.fail(
                                f"decode completed before gate: {future.result()}"
                            )
                        time.sleep(0.02)
                    pending = {
                        k: pool.submit(generate, k)
                        for k in ("partial", "one", "excluded")
                    }
                    outputs = {k: f.result() for k, f in pending.items()}
                    outputs["decode"] = future.result()
                for kind in ("cached", "fresh"):
                    if enabled:
                        self.catalog.wait_publications(first + len(outputs) - 1)
                    outputs[kind] = generate(kind)
                results.append(outputs)
                if enabled:
                    publications = self.catalog.wait_publications(first + 5)[first:]
                    captured = {
                        hashlib.sha256(f"{name}-{k}".encode()).hexdigest(): prompts[k]
                        + v
                        for k, v in outputs.items()
                        if k != "excluded"
                    }
                    self.assertEqual(outputs, results[0])
                    deadline = time.monotonic() + 20
                    while True:
                        states = self.all_states(root, url)
                        if all(
                            s["states"].get("available", 0) == 4
                            for s in states.values()
                        ):
                            break
                        self.assertLess(time.monotonic(), deadline, states)
                        time.sleep(0.03)
                    self.assertEqual(
                        sum(s["counters"].get("ready", 0) for s in states.values()), 5
                    )
                    for rank, state in states.items():
                        self.assertEqual(state["counters"]["admitted"], 5, rank)
                        self.assertIsNone(state["disabled_reason"], (rank, state))
                        self.assertFalse(state["admission_paused"], rank)
                        self.assertFalse(
                            {
                                k: v
                                for k, v in state["counters"].items()
                                if k.startswith("failed_") and v
                            },
                            rank,
                        )
                        self.assertEqual(state["host_pool"]["quarantined"], 0, rank)
                        self.assertEqual(state["queued"], 0, rank)
                    ingress = states["pp0-tp0"]
                    if self.tp_size * self.pp_size == 1:
                        self.assertEqual(ingress["counters"]["excluded_length"], 1)
                    else:
                        self.assertEqual(ingress["request_router"]["selected"], 5)
                        self.assertGreaterEqual(
                            ingress["request_router"]["excluded"], 1
                        )
                        for state in states.values():
                            self.assertEqual(state["request_router"]["attached"], 5)
                else:
                    self.assertEqual(len(self.catalog.publications), first)
            finally:
                self.stop_process(process)

            rank_batches = {}
            for rank in self.rank_names():
                batches = [
                    json.loads(line)
                    for line in (root / f"mixed-{rank}.jsonl").read_text().splitlines()
                ]
                self.assertTrue(batches, rank)
                self.assertTrue(
                    any(f"{name}-excluded" in b["rids"] for b in batches), rank
                )
                self.assertTrue(all(b["decode_rids"] for b in batches), rank)
                rank_batches[rank] = batches
            for rank, batches in rank_batches.items():
                self.assertEqual(batches, rank_batches["pp0-tp0"], rank)
            summaries.append(
                {
                    "capture": enabled,
                    "mixed_batches": {k: len(v) for k, v in rank_batches.items()},
                    "outputs": outputs,
                }
            )
            if not enabled:
                continue
            references = [
                torch.load(p, weights_only=True) for p in self.reference_paths(root)
            ]
            mixed = [r for r in references if r["forward_mode"] == "MIXED"]
            self.assertTrue(mixed)
            decode_trace = hashlib.sha256(f"{name}-decode".encode()).hexdigest()
            partial_trace = hashlib.sha256(f"{name}-partial".encode()).hexdigest()
            cached_trace = hashlib.sha256(f"{name}-cached".encode()).hexdigest()
            rank_frames = {}
            for pp in range(self.pp_size):
                for tp in range(self.tp_size):
                    rank = f"pp{pp}-tp{tp}"
                    frames = [
                        r
                        for r in references
                        if (r["pp_rank"], r["tp_rank"]) == (pp, tp)
                    ]
                    mixed_frames = [r for r in frames if r["forward_mode"] == "MIXED"]
                    self.assertTrue(mixed_frames, rank)
                    self.assertTrue(
                        any(r["trace_id"] == decode_trace for r in mixed_frames), rank
                    )
                    partial = [
                        r for r in mixed_frames if r["trace_id"] == partial_trace
                    ]
                    self.assertTrue(
                        any(
                            len(r["tokens"]) < len(prompts["partial"]) for r in partial
                        ),
                        rank,
                    )
                    self.assertTrue(
                        any(
                            len(r["tokens"]) >= len(prompts["partial"]) for r in partial
                        ),
                        rank,
                    )
                    if pp == self.pp_size - 1:
                        self.assertTrue(
                            any(
                                r["trace_id"] == decode_trace and r["predictions"]
                                for r in mixed_frames
                            ),
                            rank,
                        )
                        self.assertTrue(any(r["predictions"] for r in partial), rank)
                    else:
                        self.assertFalse(any(r["predictions"] for r in frames), rank)
                    self.assertEqual(
                        any(r["cuda_graph"] for r in mixed_frames),
                        prefill != "disabled",
                        rank,
                    )
                    self.assertEqual(
                        any(
                            r["cuda_graph"] and r["forward_mode"] == "DECODE"
                            for r in frames
                        ),
                        decode != "disabled",
                        rank,
                    )
                    self.assertTrue(
                        any(
                            r["trace_id"] == cached_trace
                            and r["extend_prefix_length"] >= 270
                            for r in frames
                            if r["extend_prefix_length"] is not None
                        ),
                        rank,
                    )
                    rank_frames[rank] = {
                        "mixed_source_frames": len(mixed_frames),
                        "mixed_replay_frames": sum(
                            r["cuda_graph"] for r in mixed_frames
                        ),
                        "decode_replay_frames": sum(
                            r["cuda_graph"] and r["forward_mode"] == "DECODE"
                            for r in frames
                        ),
                        "teacher_rows": sum(len(r["predictions"]) for r in frames),
                    }
            reader = MooncakeSnapshotStore.connect(self.store_setup)
            self.addCleanup(reader.close)
            seen, objects, nbytes = set(), 0, 0
            for publication in publications:
                manifest, tensors = read_snapshot(reader, publication)
                trace = manifest.provenance.trace_id
                self.assertNotIn(trace, seen)
                seen.add(trace)
                self.assertEqual(tensors["token_ids"].tolist(), captured[trace])
                self.assertEqual(
                    (manifest.topology.tp_size, manifest.topology.pp_size),
                    (self.tp_size, self.pp_size),
                )
                self.check_snapshot(
                    manifest, tensors, references, capture_mode="autoregressive"
                )
                objects += len(manifest.objects)
                nbytes += manifest.total_tensor_bytes
            self.assertEqual(seen, set(captured))
            self.assertFalse(self.catalog.errors)
        print(
            json.dumps(
                {
                    "mixed_capture": name,
                    "overlap": overlap,
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size,
                    "prefill": prefill,
                    "decode": decode,
                    "samples": len(captured),
                    "tensor_objects": objects,
                    "tensor_bytes": nbytes,
                    "mixed_source_frames": len(mixed),
                    "mixed_replay_frames": sum(r["cuda_graph"] for r in mixed),
                    "teacher_fingerprint": manifest.teacher.fingerprint_sha256,
                    "capture": states,
                    "rank_frames": rank_frames,
                    "runs": summaries,
                }
            ),
            flush=True,
        )
