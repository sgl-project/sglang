"""Real P/D KV transfer -> one complete Store sample, with online source parity.

Independent P/D workers share the available GPUs. TCP exercises Mooncake's actual
transfer and Store APIs; this does not certify RDMA or production Catalog retention.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests
import torch

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.test_utils import CustomTestCase, popen_launch_server
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import read_snapshot


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(
    torch.cuda.is_available()
    and shutil.which("mooncake_master")
    and importlib.util.find_spec("mooncake"),
    "CUDA and Mooncake required",
)
class PDCaptureRuntimeBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.model = os.environ.get("TRAINING_CAPTURE_TEST_MODEL", "Qwen/Qwen3-0.6B")
        if not Path(cls.model).is_dir():
            from huggingface_hub import snapshot_download

            cls.model = snapshot_download(cls.model)
        cls.catalog = TestCaptureCatalog()
        cls.addClassCleanup(cls.catalog.close)
        port = free_port()
        cls.master_log = tempfile.TemporaryFile()  # noqa: SIM115
        cls.addClassCleanup(cls.master_log.close)
        cls.master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={port}",
                f"--metrics_port={free_port()}",
                f"--http_metadata_server_port={free_port()}",
            ],
            stdout=cls.master_log,
            stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.stop_process, cls.master)
        deadline = time.monotonic() + 20
        while True:
            if cls.master.poll() is not None or time.monotonic() > deadline:
                cls.master_log.seek(0)
                raise RuntimeError(cls.master_log.read().decode(errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        cls.store_setup = {
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 0,
            "local_buffer_size": 16 << 20,
            "protocol": "tcp",
            "rdma_devices": "",
            "master_server_addr": f"127.0.0.1:{port}",
        }
        cls.segment = MooncakeSnapshotStore.connect(
            cls.store_setup | {"global_segment_size": 256 << 20}
        )
        cls.addClassCleanup(cls.segment.close)
        cls.reader = MooncakeSnapshotStore.connect(cls.store_setup)
        cls.addClassCleanup(cls.reader.close)

    @staticmethod
    def stop_process(process):
        kill_process_tree(process.pid)
        process.wait(timeout=20)

    def launch(self, role, root, *, replay, tp_size):
        folder = root / role
        folder.mkdir()
        config = {
            "dataset_id": "runtime-pd",
            "model_id": "Qwen/Qwen3-0.6B",
            "producer_revision": "checkout-under-test",
            "selected_layer_ids": [0, 14, 27],
            "catalog_endpoint": self.catalog.endpoint,
            "journal_directory": str(folder / "journal"),
            "store": self.store_setup,
            "sample_ratio": 1.0,
            "max_sample_tokens": 256,
            "max_inflight_samples": 4,
            "max_host_bytes": 64 << 20,
            "kv_d2h_batch_tokens": 16,
            "max_device_bytes": 8 << 20,
            "storage_chunk_tokens": 64,
            "http_timeout_seconds": 2.0,
        }
        path = folder / "capture.json"
        path.write_text(json.dumps(config))
        url = f"http://127.0.0.1:{free_port()}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [sys.executable, "-m", "sglang.test.pd_capture_server"] + command[2:],
                *args,
            )

        with patch.object(test_utils, "_launch_server_process", observed_server):
            process = popen_launch_server(
                self.model,
                url,
                timeout=240,
                env={**os.environ, "MOONCAKE_PROTOCOL": "tcp"},
                other_args=[
                    "--disaggregation-mode",
                    role,
                    "--tp-size",
                    str(tp_size),
                    "--disaggregation-transfer-backend",
                    "mooncake",
                    "--disaggregation-bootstrap-port",
                    str(self.bootstrap_port),
                    "--disaggregation-decode-enable-radix-cache",
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.20",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-backend-decode",
                    "full" if replay else "disabled",
                    "--cuda-graph-bs-decode",
                    "1",
                    "2",
                    "4",
                    *([] if replay else ["--disable-overlap-schedule"]),
                ],
            )
        self.addCleanup(self.stop_process, process)
        return process, url

    def generate(self, rid, prompt, count, *, biased=False):
        batched = isinstance(rid, list)
        rooms = [self.bootstrap_room + i + 1 for i in range(len(rid) if batched else 1)]
        self.bootstrap_room = rooms[-1]
        sampling = {"temperature": 0, "ignore_eos": True}
        if biased:
            sampling["logit_bias"] = {"100": 100.0}
        payload = {
            "rid": rid,
            "input_ids": prompt,
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": rooms if batched else rooms[0],
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = [
                executor.submit(
                    requests.post,
                    url + "/generate",
                    json=payload
                    | {"sampling_params": sampling | {"max_new_tokens": n}},
                    timeout=120,
                )
                for url, n in ((self.prefill_url, 1), (self.decode_url, count))
            ]
            prefill, decode = [future.result() for future in pending]
        self.assertEqual(prefill.status_code, 200, prefill.text)
        self.assertEqual(decode.status_code, 200, decode.text)
        result = decode.json()
        for item in result if batched else [result]:
            self.assertEqual(len(item["output_ids"]), count, item)
            if biased:
                self.assertEqual(item["output_ids"], [100] * count)
        return result

    def capture_state(self):
        return requests.get(self.decode_url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]

    def abort_capture(self, rid, *, decode_tp):
        before = self.capture_state()
        self.bootstrap_room += 1
        payload = {
            "rid": rid,
            "input_ids": [1, 2, 3, 4],
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": self.bootstrap_room,
        }
        sampling = {"temperature": 0, "ignore_eos": True, "logit_bias": {"100": 100.0}}
        with self.catalog.condition:
            failures = {
                key
                for key, value in self.catalog.captures.items()
                if value["state"] == "FAILED"
            }
        with ThreadPoolExecutor(max_workers=1) as executor:
            prefill = executor.submit(
                requests.post,
                self.prefill_url + "/generate",
                json=payload | {"sampling_params": sampling | {"max_new_tokens": 1}},
                timeout=120,
            )
            with requests.post(
                self.decode_url + "/generate",
                json=payload
                | {
                    "stream": True,
                    "sampling_params": sampling | {"max_new_tokens": 200},
                },
                stream=True,
                timeout=120,
            ) as response:
                self.assertEqual(response.status_code, 200)
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: ") and line != b"data: [DONE]":
                        aborted = requests.post(
                            self.decode_url + "/abort_request",
                            json={"rid": rid},
                            timeout=20,
                        )
                        self.assertEqual(aborted.status_code, 200, aborted.text)
                        break
                else:
                    self.fail("PD stream ended before cancellation")
            self.assertEqual(prefill.result().status_code, 200)
        with self.catalog.condition:
            reason = (
                "cohort_failed" if decode_tp > 1 else "request_aborted_or_retracted"
            )
            self.assertTrue(
                self.catalog.condition.wait_for(
                    lambda: any(
                        key not in failures
                        and value["state"] == "FAILED"
                        and value.get("reason") == reason
                        for key, value in self.catalog.captures.items()
                    ),
                    timeout=20,
                ),
                [(v["state"], v.get("reason")) for v in self.catalog.captures.values()],
            )
        deadline = time.monotonic() + 20
        while True:
            state = self.capture_state()
            if state["states"].get("available", 0) == state["reservations"]:
                break
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.05)
        self.assertEqual(
            state["counters"]["admitted"], before["counters"]["admitted"] + 1
        )
        self.assertEqual(state["host_pool"]["quarantined"], 0)
        if decode_tp > 1:
            self.assertGreater(
                state["request_router"].get("cancelled", 0),
                before["request_router"].get("cancelled", 0),
            )
        else:
            self.assertGreater(
                state["counters"].get("failed_request_aborted_or_retracted", 0),
                before["counters"].get("failed_request_aborted_or_retracted", 0),
            )
        return state

    def exercise(self, *, replay, prefill_tp=1, decode_tp=1):
        root = self.root / f"p{prefill_tp}-d{decode_tp}-replay-{replay}"
        root.mkdir()
        self.bootstrap_port, self.bootstrap_room = free_port(), 5000
        prefill, self.prefill_url = self.launch(
            "prefill", root, replay=replay, tp_size=prefill_tp
        )
        decode, self.decode_url = self.launch(
            "decode", root, replay=replay, tp_size=decode_tp
        )
        first = len(self.catalog.publications)
        responses = {}
        prompt = [1] + [16, 17, 18, 19] * 38
        for rid, tokens, count, biased in (
            ("single", [1, 2, 3], 1, False),
            ("chunked", prompt, 7, True),
            ("cached", prompt, 5, True),
        ):
            name = f"{rid}-{replay}"
            result = self.generate(name, tokens, count, biased=biased)
            responses[hashlib.sha256(name.encode()).hexdigest()] = (tokens, result)
            self.catalog.wait_publications(first + len(responses), timeout=45)
        names = [f"batch-{i}-{replay}" for i in range(2)]
        prompts = [[1, 3, 9, 2], [1, 7, 8, 3, 2, 4, 5]]
        for name, tokens, result in zip(
            names, prompts, self.generate(names, prompts, 3, biased=True), strict=True
        ):
            responses[hashlib.sha256(name.encode()).hexdigest()] = (tokens, result)
        expected = len(responses)
        publications = self.catalog.wait_publications(first + expected, timeout=45)
        references = [
            torch.load(path, weights_only=True) for path in sorted(root.rglob("*.pt"))
        ]
        if replay:
            self.assertTrue(any(frame["cuda_graph"] for frame in references))
        self.assertTrue(any(frame["batch_size"] > 1 for frame in references))
        for publication in publications[first:]:
            manifest, tensors = read_snapshot(self.reader, publication)
            self.assertEqual(manifest.topology.tp_size, decode_tp)
            tokens, result = responses.pop(manifest.provenance.trace_id)
            self.assertEqual(
                tensors["token_ids"].tolist(), tokens + result["output_ids"]
            )
            check_capture_snapshot(
                self, manifest, tensors, references, capture_mode="pd_autoregressive"
            )
        self.assertFalse(responses)
        for fault in ("missing", "stale"):
            with self.catalog.condition:
                failures_before = sum(
                    v["state"] == "FAILED" for v in self.catalog.captures.values()
                )
            self.generate(f"{fault}-{replay}", [1, 3, 8, 2], 3, biased=True)
            with self.catalog.condition:
                self.assertTrue(
                    self.catalog.condition.wait_for(
                        lambda failures_before=failures_before: sum(
                            v["state"] == "FAILED"
                            for v in self.catalog.captures.values()
                        )
                        > failures_before,
                        timeout=20,
                    )
                )
            self.assertEqual(len(self.catalog.publications), first + expected)
        state = self.abort_capture(f"abort-{replay}", decode_tp=decode_tp)
        self.assertEqual(len(self.catalog.publications), first + expected)
        self.assertGreaterEqual(state["counters"]["pd_handoff_committed"], expected + 1)
        self.assertGreaterEqual(state["counters"]["failed_pd_handoff_failed"], 1)
        self.assertIsNone(prefill.poll())
        self.assertIsNone(decode.poll())
        print(
            json.dumps(
                {
                    "prefill_tp": prefill_tp,
                    "decode_tp": decode_tp,
                    "replay": replay,
                    "capture": state,
                },
                sort_keys=True,
            )
        )
