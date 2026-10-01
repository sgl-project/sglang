"""Real AR capture -> Mooncake -> independent reader -> reference parity.

The HTTP Catalog is a test double. This does not certify production retention,
distributed serving, or cross-node RDMA. Reference inference occurs only in
this validation test, after the serving producer has shut down.
"""

import argparse
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    DTYPES,
    decode_manifest,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server
from sglang.test.training_capture_catalog import TestCaptureCatalog

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")

MODEL_PATH = "Qwen/Qwen3-0.6B"
ASSERT_HF_KV = False


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def store_setup(master, segment_bytes=0):
    return dict(
        local_hostname=f"127.0.0.1:{free_port()}",
        metadata_server="P2PHANDSHAKE",
        global_segment_size=segment_bytes,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        master_server_addr=master,
    )


@unittest.skipUnless(
    torch.cuda.is_available()
    and shutil.which("mooncake_master")
    and importlib.util.find_spec("mooncake"),
    "CUDA and Mooncake required",
)
class TestTrainingCaptureRuntime(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.server = cls.master = cls.segment = cls.reader = cls.catalog = None
        cls.temporary = tempfile.TemporaryDirectory()
        cls.master_log = tempfile.TemporaryFile()
        cls.model_path = MODEL_PATH
        if not Path(cls.model_path).is_dir():
            from huggingface_hub import snapshot_download

            cls.model_path = snapshot_download(cls.model_path)
        cls.catalog = TestCaptureCatalog()
        port = free_port()
        address = f"127.0.0.1:{port}"
        cls.master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={port}",
                f"--metrics_port={free_port()}",
                f"--http_metadata_server_port={free_port()}",
                # Use the master's operational read lease. A 100ms lease can
                # expire during process teardown even for hard-pinned objects.
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
        # Storage lifetime is independent of the serving producer's process.
        cls.segment = MooncakeSnapshotStore.connect(store_setup(address, 128 << 20))
        cls.reader = MooncakeSnapshotStore.connect(store_setup(address))
        config = dict(
            dataset_id="runtime-test",
            model_id="Qwen/Qwen3-0.6B",
            producer_revision="checkout-under-test",
            selected_layer_ids=[0, 14, 27],
            catalog_endpoint=cls.catalog.endpoint,
            journal_directory=str(Path(cls.temporary.name) / "journal"),
            store=store_setup(address),
            sample_ratio=1.0,
            max_sample_tokens=256,
            max_inflight_samples=4,
            max_host_bytes=64 << 20,
            kv_d2h_batch_tokens=16,
            max_device_bytes=8 << 20,
            storage_chunk_tokens=64,
            http_timeout_seconds=2.0,
        )
        path = Path(cls.temporary.name) / "capture.json"
        path.write_text(json.dumps(config))
        cls.capture_path = path
        cls.url = f"http://127.0.0.1:{free_port()}"
        cls.dump_path = Path(cls.temporary.name) / "reference"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [sys.executable, "-m", "sglang.test.training_capture_server"]
                + command[2:],
                *args,
            )

        with patch.object(test_utils, "_launch_server_process", observed_server):
            cls.launch_server(path)

    @classmethod
    def launch_server(cls, path, *, cuda_graph=False):
        graph_args = (
            [
                "--cuda-graph-backend-decode",
                "full",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
            ]
            if cuda_graph
            else [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--debug-tensor-dump-output-folder",
                str(cls.dump_path),
                "--debug-tensor-dump-layers",
                "0",
                "14",
                "27",
            ]
        )
        cls.server = popen_launch_server(
            cls.model_path,
            cls.url,
            timeout=240,
            other_args=[
                "--disable-overlap-schedule",
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
                *graph_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.server is not None:
            kill_process_tree(cls.server.pid)
            cls.server.wait(timeout=20)
        for store in (cls.reader, cls.segment):
            if store is not None:
                store.close()
        if cls.catalog is not None:
            cls.catalog.close()
        if cls.master is not None:
            cls.master.terminate()
            try:
                cls.master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.master.kill()
                cls.master.wait()
        cls.master_log.close()
        cls.temporary.cleanup()

    def read_sample(self, publication):
        data = self.reader.get_tensor(
            publication["manifest_key"],
            [publication["manifest_nbytes"]],
            torch.uint8,
            publication["manifest_sha256"],
        )
        manifest = decode_manifest(bytes(tensor_bytes(data)))
        tensors = {
            obj.key: self.reader.get_tensor(
                obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256
            )
            for obj in manifest.objects
        }
        validate_tensors(manifest, tensors)
        names = {obj.name for obj in manifest.objects}
        packed = {
            name: torch.cat(
                [
                    tensors[obj.key]
                    for obj in sorted(
                        (obj for obj in manifest.objects if obj.name == name),
                        key=lambda obj: obj.token_range[0] if obj.kind == "kv" else 0,
                    )
                ]
            )
            for name in names
        }
        return manifest, packed

    def test_chunk_prefix_single_token_and_raw_teacher_reference(self):
        samples, responses = [], []
        seen_dumps = set(self.dump_path.glob("*/Pass*.pt"))
        reference_kv = {}
        prompt = [100, 200, 300, 400] * 40
        for index, length in enumerate((1, 4, 3)):
            params = {"temperature": 0, "max_new_tokens": length, "ignore_eos": True}
            if index == 2:
                params["logit_bias"] = {"100": 100.0}
            response = requests.post(
                self.url + "/generate",
                json={"input_ids": prompt, "sampling_params": params},
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            responses.append(result)
            state_response = requests.get(self.url + "/server_info", timeout=10)
            state_response.raise_for_status()
            capture_state = state_response.json()["internal_states"][0][
                "training_capture"
            ]
            self.assertGreaterEqual(
                capture_state["counters"].get("admitted", 0), index + 1, capture_state
            )
            print(json.dumps({"capture_after_response": capture_state}), flush=True)
            try:
                publications = self.catalog.wait_publications(index + 1, timeout=30)
            except TimeoutError:
                state_response = requests.get(self.url + "/server_info", timeout=10)
                print(
                    json.dumps(
                        {
                            "capture_after_timeout": state_response.json()[
                                "internal_states"
                            ][0]["training_capture"]
                        }
                    ),
                    flush=True,
                )
                raise
            manifest, tensors = self.read_sample(publications[-1])
            self.assertEqual(manifest.sequence.response_length, length)
            self.assertEqual(
                tensors["token_ids"].tolist(), prompt + result["output_ids"]
            )
            self.assertEqual(
                tensors["loss_mask"].tolist(), [0] * len(prompt) + [1] * length
            )
            self.assertEqual(
                tensors["kv_valid"].tolist(), [1] * (len(prompt) + length - 1) + [0]
            )
            self.assertEqual(
                tensors["logits_positions"].tolist(),
                list(range(len(prompt), len(prompt) + length)),
            )
            dump_paths = set(self.dump_path.glob("*/Pass*.pt")) - seen_dumps
            self.check_runtime_kv(manifest, tensors, sorted(dump_paths), reference_kv)
            seen_dumps.update(dump_paths)
            samples.append((manifest, tensors))
        self.assertGreater(responses[1]["meta_info"]["cached_tokens"], 0)
        self.assertEqual(responses[2]["output_ids"], [100] * 3)
        torch.testing.assert_close(
            samples[0][1]["teacher_topk_logits"][0],
            samples[2][1]["teacher_topk_logits"][0],
            rtol=0.02,
            atol=0.1,
        )
        self.assertFalse(self.catalog.errors)
        kill_process_tree(self.server.pid)
        self.server.wait(timeout=20)
        type(self).server = None
        # Verify objects remain readable after the producer has disappeared.
        for publication in self.catalog.wait_publications(3):
            self.read_sample(publication)
        self.check_graph_replay(samples)
        from sglang.test.training_capture_overlap_runtime import (
            exercise_overlap_capture,
        )

        for cuda_graph in (False, True):
            exercise_overlap_capture(
                self,
                model_path=self.model_path,
                directory=self.temporary.name,
                samples=samples,
                responses=responses,
                cuda_graph=cuda_graph,
            )
        from sglang.test.dspark_target_kv_runtime import exercise_target_kv_draft

        for cuda_graph in (False, True):
            exercise_target_kv_draft(
                self,
                model_path=self.model_path,
                directory=self.temporary.name,
                samples=samples,
                responses=responses,
                cuda_graph=cuda_graph,
            )
        for cuda_graph in (False, True):
            exercise_target_kv_draft(
                self,
                model_path=self.model_path,
                directory=self.temporary.name,
                samples=samples,
                responses=responses,
                cuda_graph=cuda_graph,
                enable_overlap=True,
            )
        self.check_reference(samples)
        from sglang.test.training_capture_admission_runtime import (
            exercise_adaptive_capture,
            exercise_latency_capture,
        )

        exercise_adaptive_capture(
            self, model_path=self.model_path, directory=self.temporary.name
        )
        exercise_latency_capture(
            self, model_path=self.model_path, directory=self.temporary.name
        )

    def check_graph_replay(self, samples):
        type(self).url = f"http://127.0.0.1:{free_port()}"
        self.launch_server(self.capture_path, cuda_graph=True)
        for index, (manifest, expected) in enumerate(samples):
            params = {
                "temperature": 0,
                "max_new_tokens": manifest.sequence.response_length,
                "ignore_eos": True,
            }
            if index == 2:
                params["logit_bias"] = {"100": 100.0}
            response = requests.post(
                self.url + "/generate",
                json={
                    "input_ids": expected["token_ids"][
                        : manifest.sequence.prompt_length
                    ].tolist(),
                    "sampling_params": params,
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)
            publication = self.catalog.wait_publications(4 + index, timeout=30)[-1]
            _, actual = self.read_sample(publication)
            for name in expected:
                torch.testing.assert_close(
                    actual[name], expected[name], rtol=0, atol=0, msg=name
                )
        state = requests.get(self.url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]
        self.assertGreater(state["counters"].get("cuda_graph_forwards", 0), 0, state)
        print(json.dumps({"graph_capture": state}), flush=True)
        kill_process_tree(self.server.pid)
        self.server.wait(timeout=20)
        type(self).server = None

    def check_runtime_kv(self, manifest, tensors, paths, reference):
        """Compare against model outputs captured before the KV pool writes."""
        self.assertTrue(paths)
        n = manifest.sequence.total_length
        teacher_positions = []
        for path in paths:
            dump = torch.load(path, weights_only=True)
            positions = dump["model.forward_batch_info.positions"].long()
            ids = dump["model.forward_batch_info.input_ids"]
            torch.testing.assert_close(
                tensors["token_ids"][positions], ids.to(torch.int32), rtol=0, atol=0
            )
            prediction = int(positions[-1]) + 1
            if prediction >= manifest.sequence.prompt_length:
                row = prediction - manifest.sequence.prompt_length
                raw = dump["logits_processor"][0, : manifest.teacher.vocab_size].float()
                ids = tensors["teacher_topk_ids"][row].long()
                actual = tensors["teacher_topk_logits"][row]
                torch.testing.assert_close(actual, raw[ids], rtol=0, atol=0)
                self.assertEqual(actual.min().item(), raw.topk(128).values[-1].item())
                torch.testing.assert_close(
                    tensors["teacher_logsumexp"][row],
                    raw.logsumexp(-1),
                    rtol=0,
                    atol=1e-5,
                )
                teacher_positions.append(prediction)
            for layer in manifest.kv.layers:
                prefix = f"model.layers.{layer.layer_id}.self_attn"
                key = dump[prefix + ".attn.input_k"]
                value = dump[prefix + ".attn.input_v"]
                for component, values, dim in (
                    ("k", key, layer.key_head_dim),
                    ("v", value, layer.value_head_dim),
                ):
                    name = f"target_{component}.{layer.layer_id}"
                    if name not in reference:
                        reference[name] = torch.zeros(
                            (256, layer.num_kv_heads, dim), dtype=values.dtype
                        )
                    reference[name][positions] = values.reshape(
                        -1, layer.num_kv_heads, dim
                    )
        for name, values in reference.items():
            torch.testing.assert_close(
                tensors[name], values[: n - 1], rtol=0, atol=0, msg=name
            )
        self.assertEqual(
            teacher_positions, list(range(manifest.sequence.prompt_length, n))
        )
        print(json.dumps({"runtime_kv_teacher_exact": manifest.sample_id}), flush=True)

    def check_reference(self, samples):
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        try:
            with torch.inference_mode():
                for manifest, tensors in samples:
                    tokens = tensors["token_ids"].long().cuda()[None]
                    output = model(tokens, use_cache=True)
                    p, n = (
                        manifest.sequence.prompt_length,
                        manifest.sequence.total_length,
                    )
                    raw = output.logits[0, p - 1 : n - 1].float().cpu()
                    ids = tensors["teacher_topk_ids"].long()
                    expected = raw.gather(1, ids)
                    errors = {
                        "logits_max_abs": (expected - tensors["teacher_topk_logits"])
                        .abs()
                        .max()
                        .item()
                    }
                    torch.testing.assert_close(
                        tensors["teacher_topk_logits"], expected, rtol=0.015, atol=0.15
                    )
                    torch.testing.assert_close(
                        tensors["teacher_logsumexp"],
                        raw.logsumexp(-1),
                        rtol=0.01,
                        atol=0.1,
                    )
                    for layer in manifest.kv.selected_layer_ids:
                        cache = output.past_key_values.layers[layer]
                        for component, values in (
                            ("k", cache.keys),
                            ("v", cache.values),
                        ):
                            expected_kv = values[0, :, : n - 1].transpose(0, 1).cpu()
                            actual = tensors[f"target_{component}.{layer}"]
                            difference = actual.float() - expected_kv.float()
                            worst = tuple(
                                int(i)
                                for i in torch.unravel_index(
                                    difference.abs().argmax(), difference.shape
                                )
                            )
                            errors[f"{component}.{layer}_max_abs"] = (
                                (actual - expected_kv).abs().max().item()
                            )
                            print(
                                json.dumps(
                                    {
                                        "layer": layer,
                                        "component": component,
                                        "worst_index": worst,
                                        "actual_at_worst": actual[worst].item(),
                                        "reference_at_worst": expected_kv[worst].item(),
                                        "per_position_max": difference.abs()
                                        .flatten(1)
                                        .amax(1)
                                        .topk(5)
                                        .indices.tolist(),
                                        "max_abs": errors[
                                            f"{component}.{layer}_max_abs"
                                        ],
                                        "reference_max_abs": expected_kv.abs()
                                        .max()
                                        .item(),
                                        "relative_rms": (
                                            (actual.float() - expected_kv.float())
                                            .square()
                                            .mean()
                                            .sqrt()
                                            / expected_kv.float().square().mean().sqrt()
                                        ).item(),
                                    }
                                ),
                                flush=True,
                            )
                            # Capture correctness is checked bit-for-bit against
                            # online attention above. Different BF16 engines can
                            # produce different KV, even with identical weights;
                            # retain this optional cross-engine diagnostic gate.
                            if ASSERT_HF_KV:
                                with self.subTest(
                                    sample=manifest.sample_id,
                                    layer=layer,
                                    component=component,
                                ):
                                    torch.testing.assert_close(
                                        actual, expected_kv, rtol=0.03, atol=0.125
                                    )
                    print(
                        json.dumps(
                            {
                                "sample_id": manifest.sample_id,
                                "sequence": [p, n - p],
                                "parity": errors,
                            }
                        ),
                        flush=True,
                    )
        finally:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--assert-hf-kv", action="store_true")
    args, remaining = parser.parse_known_args()
    MODEL_PATH = args.model_path
    ASSERT_HF_KV = args.assert_hf_kv
    unittest.main(argv=[__file__, *remaining])
