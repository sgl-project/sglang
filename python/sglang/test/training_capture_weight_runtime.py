"""Real target replacement must never publish a sample spanning two weights."""

import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.srt.training_capture.identity import local_safetensors_digest
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_pressure_runtime import ARPressureCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot


class CaptureWeightRuntimeBase(ARPressureCaptureRuntimeBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.replacement = cls.root / "replacement"
        cls.replacement.mkdir()
        changed = False
        parameter = "model.layers.0.self_attn.v_proj.weight"
        for source in Path(cls.model).iterdir():
            if not source.is_file():
                continue
            destination = cls.replacement / source.name
            if source.suffix != ".safetensors":
                shutil.copyfile(source, destination)
                continue
            with safe_open(source, framework="pt", device="cpu") as stream:
                names = stream.keys()
                if parameter not in names:
                    destination.symlink_to(source.resolve())
                    continue
                tensors = {name: stream.get_tensor(name) for name in names}
                tensors[parameter] = -tensors[parameter]
                save_file(tensors, destination, metadata=stream.metadata())
                changed = True
        assert changed, "fixture requires the Qwen3 layer-zero V projection"
        cls.revisions = [
            local_safetensors_digest(Path(cls.model)),
            local_safetensors_digest(cls.replacement),
        ]
        assert cls.revisions[0] != cls.revisions[1]

    def exercise_weight_update(self, *, cuda_graph):
        root = self.root / f"weights-{cuda_graph}"
        root.mkdir()
        prompt = [17, 19, 23, 29] * 4
        publications_before = len(self.catalog.publications)
        expected, roots, update_states = {}, [], {}

        def launch(model, version):
            folder = root / version
            folder.mkdir()
            roots.append(folder)
            path = folder / "capture.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_id": "runtime-weight-update",
                        "model_id": self.model_id,
                        "producer_revision": "checkout-under-test",
                        "selected_layer_ids": [0, 14, 27],
                        "expected_weights_revision": self.revisions[int(version)],
                        "catalog_endpoint": self.catalog.endpoint,
                        "journal_directory": str(folder / "journal"),
                        "store": self.store_setup,
                        "sample_ratio": 1.0,
                        "max_sample_tokens": 256,
                        "max_inflight_samples": 2,
                        "max_host_bytes": 64 << 20,
                        "kv_d2h_batch_tokens": 16,
                        "teacher_d2h_batch_tokens": 16,
                        "max_device_bytes": 8 << 20,
                        "storage_chunk_tokens": 64,
                    }
                )
            )
            url = f"http://127.0.0.1:{self.new_bootstrap_port()}"
            launch_process = test_utils._launch_server_process

            def observed(command, *args):
                return launch_process(
                    [
                        sys.executable,
                        "-m",
                        "sglang.test.training_capture_weight_server",
                    ]
                    + command[2:],
                    *args,
                )

            with patch.object(test_utils, "_launch_server_process", observed):
                process = test_utils.popen_launch_server(
                    str(model),
                    url,
                    timeout=300,
                    env={**os.environ, "TRAINING_CAPTURE_TEST_OUTPUT": str(folder)},
                    other_args=[
                        "--training-capture-config",
                        str(path),
                        "--skip-server-warmup",
                        "--skip-tokenizer-init",
                        "--attention-backend",
                        "triton",
                        "--mem-fraction-static",
                        "0.25",
                        "--max-total-tokens",
                        "1024",
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
                        *([] if cuda_graph else ["--disable-overlap-schedule"]),
                    ],
                )
            self.addCleanup(self.stop_process, process)
            return process, url, folder

        def wait_for(folder, url, predicate):
            deadline = time.monotonic() + 30
            while True:
                state = self.all_states(folder, url)["pp0-tp0"]
                if predicate(state):
                    return state
                self.assertLess(time.monotonic(), deadline, state)
                time.sleep(0.03)

        def post(url, path, body):
            response = requests.post(url + path, json=body, timeout=120)
            self.assertEqual(response.status_code, 200, response.text)
            return response

        def payload(rid, length=4):
            return {
                "rid": rid,
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": length,
                    "ignore_eos": True,
                    "logit_bias": {"100": 100.0},
                },
            }

        def generate(url, rid):
            value = post(url, "/generate", payload(rid)).json()
            self.assertEqual(value["output_ids"], [100] * 4)
            return prompt + value["output_ids"]

        process, url, folder = launch(self.model, "0")
        try:
            wait_for(folder, url, lambda s: s["states"].get("available") == 2)
            rid = f"weights-{cuda_graph}-old"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(url, rid)
            self.catalog.wait_publications(publications_before + 1)
            wait_for(folder, url, lambda s: s["states"].get("available") == 2)
            rid = f"weights-{cuda_graph}-interrupted"
            paused = False
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(post, url, "/generate", payload(rid, 128))
                try:
                    deadline = time.monotonic() + 30
                    boundary_path = folder / "pause-boundary.json"
                    while not boundary_path.exists():
                        if future.done():
                            self.fail(
                                f"request ended before pause: {future.result().text}"
                            )
                        self.assertLess(
                            time.monotonic(), deadline, "pause boundary missing"
                        )
                        time.sleep(0.03)
                    paused = True
                    post(url, "/pause_generation", {"mode": "in_place"})
                    before = wait_for(
                        folder, url, lambda s: s["states"].get("active", 0) == 1
                    )
                    admissions = [
                        json.loads(line)
                        for line in (folder / "admissions-pp0-tp0.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    capture_id = next(
                        x["capture_id"] for x in admissions if x["rid"] == rid
                    )
                    boundary = json.loads(boundary_path.read_text())
                    self.assertEqual(boundary["capture_id"], capture_id)
                    self.assertEqual(boundary["output_ids"], [100])
                    result = post(
                        url,
                        "/update_weights_from_disk",
                        {
                            "model_path": str(self.replacement),
                            "flush_cache": False,
                            "weight_version": "fixture-version-b",
                        },
                    ).json()
                    self.assertTrue(result["success"], result)
                    mutation = json.loads((folder / "weight-mutation.json").read_text())
                    self.assertTrue(mutation["success"])
                    self.assertNotEqual(
                        mutation["before_sha256"], mutation["after_sha256"]
                    )
                    self.assertEqual(
                        mutation["expected_after_sha256"], mutation["after_sha256"]
                    )
                    with self.catalog.condition:
                        self.assertTrue(
                            self.catalog.condition.wait_for(
                                lambda: (
                                    self.catalog.captures[capture_id]["state"]
                                    == "FAILED"
                                ),
                                timeout=20,
                            )
                        )
                        self.assertEqual(
                            self.catalog.captures[capture_id]["reason"],
                            "target_weights_update",
                        )
                    after = wait_for(folder, url, lambda s: s["host_pool"]["free"] >= 1)
                    self.assertEqual(after["disabled_reason"], "target_weights_update")
                    self.assertEqual(
                        after["counters"]["failed_target_weights_update"], 1
                    )
                    self.assertEqual(after["admission"]["effective_ratio"], 0)
                    post(url, "/continue_generation", {})
                    paused = False
                    last = future.result(timeout=120).json()
                    self.assertEqual(last["output_ids"], [100] * 128)
                    self.assertEqual(
                        last["meta_info"]["finish_reason"]["type"], "length"
                    )
                finally:
                    if paused:
                        post(url, "/continue_generation", {})

            post(url, "/flush_cache", {})
            generate(url, f"weights-{cuda_graph}-disabled")
            post(url, "/control_training_capture", {"action": "resume"})
            generate(url, f"weights-{cuda_graph}-still-disabled")
            final = self.all_states(folder, url)["pp0-tp0"]
            self.assertEqual(final["disabled_reason"], "target_weights_update")
            self.assertEqual(
                final["counters"]["admitted"], before["counters"]["admitted"]
            )
            self.assertEqual(final["host_pool"]["quarantined"], 0)
            self.assertEqual(len(self.catalog.publications), publications_before + 1)
            update_states = {
                "boundary": boundary,
                "mutation": mutation,
                "before": before,
                "after": after,
                "final": final,
            }
        finally:
            self.stop_process(process)

        process, url, folder = launch(self.replacement, "1")
        try:
            wait_for(folder, url, lambda s: s["states"].get("available") == 2)
            rid = f"weights-{cuda_graph}-new"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(url, rid)
            self.catalog.wait_publications(publications_before + 2)
            new_state = wait_for(
                folder, url, lambda s: s["states"].get("available") == 2
            )
            self.assertIsNone(new_state["disabled_reason"])
            self.assertEqual(
                new_state["counters"].get("cuda_graph_forwards", 0) > 0, cuda_graph
            )
        finally:
            self.stop_process(process)

        references = [
            torch.load(path, weights_only=True)
            for folder in roots
            for path in (folder / "capture-reference").rglob("*.pt")
        ]
        publications = self.catalog.wait_publications(publications_before + 2)[
            publications_before:
        ]
        self.assertEqual(len(publications), 2)
        snapshots = []
        with closing(MooncakeSnapshotStore.connect(self.store_setup)) as reader:
            for version, publication in enumerate(publications):
                manifest, tensors = read_snapshot(reader, publication)
                self.assertEqual(
                    manifest.teacher.weights_revision, self.revisions[version]
                )
                self.assertEqual(
                    tensors["token_ids"].tolist(),
                    expected[manifest.provenance.trace_id],
                )
                check_capture_snapshot(
                    self, manifest, tensors, references, capture_mode="autoregressive"
                )
                snapshots.append((manifest, tensors))
        self.assertNotEqual(
            snapshots[0][0].teacher.fingerprint_sha256,
            snapshots[1][0].teacher.fingerprint_sha256,
        )
        self.assertEqual(
            snapshots[0][0].teacher.tokenizer_revision,
            snapshots[1][0].teacher.tokenizer_revision,
        )
        self.assertFalse(
            torch.equal(snapshots[0][1]["target_v.0"], snapshots[1][1]["target_v.0"])
        )
        self.assertFalse(
            torch.equal(
                snapshots[0][1]["teacher_topk_logits"],
                snapshots[1][1]["teacher_topk_logits"],
            )
        )
        self.assertFalse(self.catalog.errors)
        print(
            json.dumps(
                {
                    "weight_update": {
                        "cuda_graph": cuda_graph,
                        "revisions": self.revisions,
                        "states": update_states,
                        "new_state": new_state,
                        "post_exit_snapshots": len(snapshots),
                        "objects": sum(len(m.objects) for m, _ in snapshots),
                        "payload_bytes": sum(
                            o.nbytes for m, _ in snapshots for o in m.objects
                        ),
                        "publications": publications,
                    }
                }
            ),
            flush=True,
        )
