"""Real TP/PP inference must preserve global KV heads and pre-sampler scores.

The Store segment outlives all model workers. Online attention observations are
independent of the capture exporter; the HTTP Catalog remains a test double.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import read_snapshot

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(
    torch.cuda.device_count() >= 2
    and shutil.which("mooncake_master")
    and importlib.util.find_spec("mooncake"),
    "two CUDA devices and Mooncake required",
)
class TestDistributedCaptureRuntime(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.server = cls.master = cls.segment = cls.reader = cls.catalog = None
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.files = ExitStack()
        cls.addClassCleanup(cls.files.close)
        cls.master_log = cls.files.enter_context(tempfile.TemporaryFile())  # noqa: SIM115
        cls.model_path = os.environ.get(
            "TRAINING_CAPTURE_TEST_MODEL", "Qwen/Qwen3-0.6B"
        )
        if not Path(cls.model_path).is_dir():
            from huggingface_hub import snapshot_download

            cls.model_path = snapshot_download(cls.model_path)
        cls.catalog = TestCaptureCatalog()
        port = free_port()
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
        # Bare host lets each rank's SDK bind an independent available port.
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
        cls.reader = MooncakeSnapshotStore.connect(cls.store_setup)

    @classmethod
    def stop_server(cls):
        if cls.server is not None:
            kill_process_tree(cls.server.pid)
            cls.server.wait(timeout=20)
            cls.server = None

    @classmethod
    def tearDownClass(cls):
        cls.stop_server()
        for store in (cls.reader, cls.segment):
            if store is not None:
                store.close()
        if cls.catalog is not None:
            cls.catalog.close()
        if cls.master is not None:
            # The SDK launcher can wrap a child binary; reap both processes.
            kill_process_tree(cls.master.pid)
            cls.master.wait(timeout=20)
        cls.temporary.cleanup()

    def launch(self, tp, pp, *, replay=False):
        root = self.root / f"tp{tp}-pp{pp}-replay{replay}"
        root.mkdir()
        config = {
            "dataset_id": f"runtime-tp{tp}-pp{pp}",
            "model_id": "Qwen/Qwen3-0.6B",
            "producer_revision": "checkout-under-test",
            "selected_layer_ids": [0, 14, 27],
            "catalog_endpoint": self.catalog.endpoint,
            "journal_directory": str(root / "journal"),
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
        path = root / "capture.json"
        path.write_text(json.dumps(config))
        self.dump_path = root / "reference"
        self.url = f"http://127.0.0.1:{free_port()}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [
                    sys.executable,
                    "-m",
                    (
                        "sglang.test.training_capture_distributed_server"
                        if replay
                        else "sglang.test.training_capture_server"
                    ),
                ]
                + command[2:],
                *args,
            )

        schedule_args = [] if replay and pp == 1 else ["--disable-overlap-schedule"]
        graph_args = (
            [
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
            ]
            if replay
            else [
                "--cuda-graph-backend-decode",
                "disabled",
                "--debug-tensor-dump-output-folder",
                str(self.dump_path),
                "--debug-tensor-dump-layers",
                "0",
                "14",
                "27",
            ]
        )
        with patch.object(test_utils, "_launch_server_process", observed_server):
            type(self).server = popen_launch_server(
                self.model_path,
                self.url,
                timeout=240,
                env={
                    **os.environ,
                    "TRAINING_CAPTURE_REPLAY_DIRECTORY": str(self.dump_path),
                },
                other_args=[
                    "--tp-size",
                    str(tp),
                    "--pp-size",
                    str(pp),
                    *schedule_args,
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
                    "--training-capture-config",
                    str(path),
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    *graph_args,
                ],
            )

    def check_capture(self, tp, pp):
        self.launch(tp, pp)
        prompt = [100, 200, 300, 400] * 40
        observations = []
        seen = set(self.dump_path.glob("*/Pass*.pt"))
        previous = len(self.catalog.publications)
        try:
            for index, length in enumerate((1, 4, 3)):
                rid = f"tp{tp}-pp{pp}-{index}"
                params = {
                    "temperature": 0,
                    "max_new_tokens": length,
                    "ignore_eos": True,
                }
                if index == 2:
                    params["logit_bias"] = {"100": 100.0}
                response = requests.post(
                    self.url + "/generate",
                    json={"rid": rid, "input_ids": prompt, "sampling_params": params},
                    timeout=120,
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                if index == 2:
                    self.assertEqual(result["output_ids"], [100] * length)
                if index:
                    self.assertGreater(result["meta_info"]["cached_tokens"], 0)
                state = requests.get(self.url + "/server_info", timeout=10)
                state.raise_for_status()
                print(
                    json.dumps(
                        {
                            "distributed_state": [
                                item["training_capture"]
                                for item in state.json()["internal_states"]
                            ]
                        }
                    ),
                    flush=True,
                )
                publications = self.catalog.wait_publications(
                    previous + index + 1, timeout=45
                )
                publication = publications[-1]
                paths = set(self.dump_path.glob("*/Pass*.pt")) - seen
                self.assertTrue(paths)
                observations.append(
                    (rid, publication, prompt + result["output_ids"], sorted(paths))
                )
                seen.update(paths)
        finally:
            self.stop_server()
        reference = {}
        for rid, publication, tokens, paths in observations:
            manifest, packed = read_snapshot(self.reader, publication)
            self.assertEqual(
                manifest.provenance.trace_id, hashlib.sha256(rid.encode()).hexdigest()
            )
            self.assertEqual(manifest.topology.tp_size, tp)
            self.assertEqual(manifest.topology.pp_size, pp)
            self.assertEqual(packed["token_ids"].tolist(), tokens)
            self.assertEqual(
                packed["loss_mask"].tolist(),
                [0] * len(prompt) + [1] * (len(tokens) - len(prompt)),
            )
            self.assertEqual(packed["kv_valid"].tolist(), [1] * (len(tokens) - 1) + [0])
            self.assertEqual(
                packed["logits_positions"].tolist(),
                list(range(len(prompt), len(tokens))),
            )
            self.check_online(manifest, packed, paths, reference)
            print(
                json.dumps(
                    {
                        "distributed_exact": rid,
                        "objects": len(manifest.objects),
                        "tensor_bytes": manifest.total_tensor_bytes,
                    }
                ),
                flush=True,
            )

        self.check_replay(tp, pp, prompt, packed)

    def check_replay(self, tp, pp, prompt, eager):
        self.launch(tp, pp, replay=True)
        seen = set(self.dump_path.glob("*/Pass*.pt"))
        previous = len(self.catalog.publications)
        try:
            response = requests.post(
                self.url + "/generate",
                json={
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 3,
                        "ignore_eos": True,
                        "logit_bias": {"100": 100.0},
                    },
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["output_ids"], [100] * 3)
            publication = self.catalog.wait_publications(previous + 1, timeout=45)[-1]
            response = requests.get(self.url + "/server_info", timeout=10)
            response.raise_for_status()
            state = response.json()["internal_states"][0]["training_capture"]
            self.assertGreaterEqual(state["counters"].get("cuda_graph_forwards", 0), 2)
            self.assertEqual(state["enable_overlap"], pp == 1)
            if pp == 1:
                self.assertGreater(state["counters"].get("overlap_forwards", 0), 0)
        finally:
            self.stop_server()
        manifest, packed = read_snapshot(self.reader, publication)
        self.assertEqual(
            (manifest.topology.tp_size, manifest.topology.pp_size), (tp, pp)
        )
        for name in ("token_ids", "position_ids", "loss_mask", "logits_positions"):
            torch.testing.assert_close(packed[name], eager[name], rtol=0, atol=0)
        paths = sorted(set(self.dump_path.glob("*/Pass*.pt")) - seen)
        self.check_online(manifest, packed, paths, {}, replay=True)
        print(json.dumps({"distributed_replay": [tp, pp], "state": state}), flush=True)

    def check_online(self, manifest, packed, paths, reference, *, replay=False):
        tp, pp = manifest.topology.tp_size, manifest.topology.pp_size
        n = manifest.sequence.total_length
        teacher_positions = {rank: [] for rank in range(tp)}
        graph_counts = {}
        observed_ranks = set()
        for path in paths:
            match = re.fullmatch(r"TP(\d+)_PP(\d+)_Rank\d+_pid\d+", path.parent.name)
            self.assertIsNotNone(match)
            tp_rank, pp_rank = map(int, match.groups())
            observed_ranks.add((tp_rank, pp_rank))
            dump = torch.load(path, weights_only=True)
            positions = dump["model.forward_batch_info.positions"]
            keep = positions < n
            if dump.get("cuda_graph", False):
                graph_counts[tp_rank, pp_rank] = (
                    graph_counts.get((tp_rank, pp_rank), 0) + 1
                )
            torch.testing.assert_close(
                packed["token_ids"][positions[keep]],
                dump["model.forward_batch_info.input_ids"][keep].int(),
                rtol=0,
                atol=0,
            )
            prediction = int(positions[-1]) + 1
            if pp_rank == pp - 1 and manifest.sequence.prompt_length <= prediction < n:
                row = prediction - manifest.sequence.prompt_length
                raw = dump["logits_processor"][0, : manifest.teacher.vocab_size].float()
                ids = packed["teacher_topk_ids"][row].long()
                values = packed["teacher_topk_logits"][row]
                torch.testing.assert_close(values, raw[ids], rtol=0, atol=0)
                self.assertEqual(values.min().item(), raw.topk(128).values[-1].item())
                torch.testing.assert_close(
                    packed["teacher_logsumexp"][row],
                    raw.logsumexp(-1),
                    rtol=0,
                    atol=1e-5,
                )
                teacher_positions[tp_rank].append(prediction)
            for layer in manifest.kv.layers:
                prefix = f"model.layers.{layer.layer_id}.self_attn.attn.input_"
                if prefix + "k" not in dump:
                    continue
                heads = layer.num_kv_heads // tp
                for component, dim in (
                    ("k", layer.key_head_dim),
                    ("v", layer.value_head_dim),
                ):
                    name = f"target_{component}.{layer.layer_id}"
                    values = dump[prefix + component].reshape(-1, heads, dim)
                    if name not in reference:
                        reference[name] = (
                            torch.empty(
                                256, layer.num_kv_heads, dim, dtype=values.dtype
                            ),
                            torch.zeros(256, layer.num_kv_heads, dtype=torch.bool),
                        )
                    data, valid = reference[name]
                    h0, h1 = tp_rank * heads, (tp_rank + 1) * heads
                    data[positions[keep], h0:h1] = values[keep]
                    valid[positions[keep], h0:h1] = True
        self.assertEqual(observed_ranks, {(t, p) for t in range(tp) for p in range(pp)})
        nv = int(packed["kv_valid"].sum())
        for layer in manifest.kv.layers:
            for component in ("k", "v"):
                name = f"target_{component}.{layer.layer_id}"
                data, valid = reference[name]
                self.assertTrue(valid[:nv].all(), name)
                if replay:
                    torch.testing.assert_close(
                        packed["kv_valid"],
                        valid[:n].all(-1).to(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    packed[name], data[:nv], rtol=0, atol=0, msg=name
                )
        if replay:
            self.assertEqual(set(graph_counts), observed_ranks)
            self.assertTrue(all(count >= 2 for count in graph_counts.values()))
        for positions in teacher_positions.values():
            self.assertEqual(positions, list(range(manifest.sequence.prompt_length, n)))

    def test_tensor_parallel_capture(self):
        self.check_capture(tp=2, pp=1)

    def test_pipeline_parallel_capture(self):
        self.check_capture(tp=1, pp=2)


if __name__ == "__main__":
    unittest.main()
