"""Prefill graph replay -> owned raw teacher/KV -> independent Store readback."""

import hashlib
import json
import sys
import time
from unittest.mock import patch

import requests
import torch

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase, free_port
from sglang.test.training_capture_utils import read_snapshot


class PrefillCaptureRuntimeBase(PDCaptureRuntimeBase):
    """Reuse the real Store/HTTP Catalog fixture; serving is not disaggregated."""

    def exercise_prefill(self, backend, *, overlap=True):
        root = self.root / f"prefill-{backend}-{overlap}"
        root.mkdir()
        config = {
            "dataset_id": "runtime-prefill-graph",
            "model_id": "Qwen/Qwen3-0.6B",
            "producer_revision": "checkout-under-test",
            "selected_layer_ids": [0, 14, 27],
            "catalog_endpoint": self.catalog.endpoint,
            "journal_directory": str(root / "journal"),
            "store": self.store_setup,
            "sample_ratio": 1.0,
            "max_sample_tokens": 320,
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
                [sys.executable, "-m", "sglang.test.training_capture_prefill_server"]
                + command[2:],
                *args,
            )

        graph_config = {
            "prefill": {"backend": backend, "bs": [16, 32, 64, 128], "max_bs": 128},
            "decode": {"backend": "full", "bs": [1, 2, 4], "max_bs": 4},
        }
        if backend == "full":
            graph_config["prefill"]["full_prefill_max_req"] = 4
        with patch.object(test_utils, "_launch_server_process", observed_server):
            process = test_utils.popen_launch_server(
                self.model,
                url,
                timeout=600,
                other_args=[
                    "--training-capture-config",
                    str(path),
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
                    json.dumps(graph_config),
                    *([] if overlap else ["--disable-overlap-schedule"]),
                ],
            )
        self.addCleanup(self.stop_process, process)
        first_publication = len(self.catalog.publications)
        expected = {}

        def state():
            response = requests.get(url + "/server_info", timeout=10)
            response.raise_for_status()
            return response.json()["internal_states"][0]["training_capture"]

        def generate(name, prompts, response_lengths):
            deadline = time.monotonic() + 20
            while state()["states"].get("available", 0) < len(prompts):
                self.assertLess(time.monotonic(), deadline, state())
                time.sleep(0.05)
            rids = [f"{backend}-{overlap}-{name}-{i}" for i in range(len(prompts))]
            response = requests.post(
                url + "/generate",
                json={
                    "rid": rids,
                    "input_ids": prompts,
                    "sampling_params": [
                        {
                            "temperature": 0,
                            "max_new_tokens": length,
                            "ignore_eos": True,
                            "logit_bias": {"100": 100.0},
                        }
                        for length in response_lengths
                    ],
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            for rid, prompt, length, result in zip(
                rids, prompts, response_lengths, response.json(), strict=True
            ):
                self.assertEqual(len(result["output_ids"]), length)
                expected[hashlib.sha256(rid.encode()).hexdigest()] = (
                    prompt + result["output_ids"]
                )
            self.catalog.wait_publications(first_publication + len(expected))

        try:
            prompt = [100, 200, 300, 400] * 67 + [501, 502, 503]
            generate("chunked", [prompt], [4])
            generate("cached-one-token", [prompt], [1])
            generate("cached-extension", [prompt + list(range(600, 619))], [4])
            for iteration in range(2):
                generate(
                    f"batch-{iteration}",
                    [
                        list(
                            range(
                                2000 + iteration * 1000 + i * 100,
                                2000 + iteration * 1000 + i * 100 + length,
                            )
                        )
                        for i, length in enumerate([33, 37, 43])
                    ],
                    [4, 3, 1],
                )
            publications = self.catalog.wait_publications(
                first_publication + len(expected)
            )[first_publication:]
            final_state = state()
            self.assertEqual(final_state["counters"]["admitted"], len(expected))
            self.assertEqual(final_state["enable_overlap"], overlap)
        finally:
            self.stop_process(process)

        references = [
            torch.load(path, weights_only=True)
            for path in sorted((root / "capture-reference").glob("*.pt"))
        ]
        prefill = [r for r in references if r["forward_mode"] == "EXTEND"]
        replay = [r for r in prefill if r["cuda_graph"]]
        self.assertTrue(replay, "no actual prefill graph replay")
        self.assertTrue(all(r["prefill_graph"] is not None for r in replay))
        self.assertTrue(
            any(
                r["prefill_graph"]["raw_tokens"] < r["prefill_graph"]["padded_tokens"]
                for r in replay
            ),
            "no token-padded prefill graph replay",
        )
        self.assertTrue(any(r["batch_size"] == 3 for r in replay))
        if backend == "full":
            self.assertTrue(
                any(
                    r["batch_size"] < r["prefill_graph"]["request_slots"]
                    for r in replay
                )
            )
        self.assertTrue(any(r["extend_prefix_length"] >= 271 for r in replay))
        self.assertTrue(any(not r["predictions"] for r in prefill))
        self.assertTrue(any(r["extend_prefix_length"] == 128 for r in prefill))
        self.assertTrue(any(r["extend_prefix_length"] == 256 for r in prefill))
        self.assertTrue(
            any(r["forward_mode"] == "DECODE" and r["cuda_graph"] for r in references)
        )
        if overlap:
            self.assertTrue(any(r["result_lag"] == 1 for r in references))
        buffers = {}
        for r in replay:
            buffers.setdefault(r["prefill_graph"]["input_buffer"], set()).add(
                r["prefill_graph"]["replay_id"]
            )
        self.assertTrue(any(len(replays) > 1 for replays in buffers.values()))

        # Open a new client after the producer has exited and all graph buffers
        # have been destroyed. No target model runs to reconstruct the sample.
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        objects, tensor_bytes = 0, 0
        captured = {}
        for publication in publications:
            manifest, tensors = read_snapshot(reader, publication)
            trace = manifest.provenance.trace_id
            self.assertNotIn(trace, captured)
            captured[trace] = tensors["token_ids"].tolist()
            check_capture_snapshot(
                self, manifest, tensors, references, capture_mode="autoregressive"
            )
            objects += len(manifest.objects)
            tensor_bytes += sum(obj.nbytes for obj in manifest.objects)
        self.assertEqual(captured, expected)
        print(
            json.dumps(
                {
                    "prefill_capture": backend,
                    "overlap": overlap,
                    "samples": len(captured),
                    "tensor_objects": objects,
                    "tensor_bytes": tensor_bytes,
                    "source_frames": len(references),
                    "prefill_replay_frames": len(replay),
                    "graph_shapes": sorted(
                        {
                            (
                                r["batch_size"],
                                r["prefill_graph"]["raw_tokens"],
                                r["prefill_graph"]["padded_tokens"],
                            )
                            for r in replay
                        }
                    ),
                    "graph_prefix_lengths": sorted(
                        {r["extend_prefix_length"] for r in replay}
                    ),
                    "reused_graph_buffers": sum(len(v) > 1 for v in buffers.values()),
                    "producer_exited": process.poll() is not None,
                    "capture_state": final_state,
                }
            ),
            flush=True,
        )
