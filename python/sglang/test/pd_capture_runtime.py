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


def free_port(host="127.0.0.1"):
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@unittest.skipUnless(
    torch.cuda.is_available()
    and shutil.which("mooncake_master")
    and importlib.util.find_spec("mooncake"),
    "CUDA and Mooncake required",
)
class PDCaptureRuntimeBase(CustomTestCase):
    prefill_host = "127.0.0.1"
    decode_host = "127.0.0.1"
    transfer_protocol = "tcp"
    ib_device = None
    teacher_d2h_batch_tokens = 1
    target_attention_backend = None
    observer_module = "sglang.test.pd_capture_server"
    validate_prefill_graph = False

    def new_bootstrap_port(self):
        return free_port(self.prefill_host)

    def reference_paths(self, root):
        return sorted(root.rglob("*.pt"))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.drafts = {}
        cls.model = os.environ.get("TRAINING_CAPTURE_TEST_MODEL", "Qwen/Qwen3-0.6B")
        if not Path(cls.model).is_dir():
            from huggingface_hub import snapshot_download

            cls.model = snapshot_download(cls.model)
        cls.catalog = TestCaptureCatalog()
        cls.addClassCleanup(cls.catalog.close)
        cls.start_store()
        cls.reader = MooncakeSnapshotStore.connect(cls.store_setup)
        cls.addClassCleanup(cls.reader.close)

    @classmethod
    def start_store(cls):
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

    @staticmethod
    def stop_process(process):
        if process.poll() is not None:
            return
        kill_process_tree(process.pid, wait_timeout=20)
        process.wait(timeout=20)

    def launch(
        self,
        role,
        root,
        *,
        replay,
        tp_size,
        pp_size,
        draft=None,
        ragged_mode="static",
        enable_overlap=None,
        extra_args=(),
        prefill_backend="disabled",
    ):
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
            "teacher_d2h_batch_tokens": self.teacher_d2h_batch_tokens,
            "max_device_bytes": 8 << 20,
            "storage_chunk_tokens": 64,
            "http_timeout_seconds": 2.0,
        }
        path = folder / "capture.json"
        path.write_text(json.dumps(config))
        host = self.prefill_host if role == "prefill" else self.decode_host
        url = f"http://{host}:{free_port(host)}"
        launch = test_utils._launch_server_process

        def observed_server(command, *args):
            return launch(
                [sys.executable, "-m", self.observer_module] + command[2:],
                *args,
            )

        graph_config = {
            "prefill": {"backend": prefill_backend},
            "decode": {
                "backend": "full" if replay else "disabled",
                "bs": [1, 2, 4],
                "max_bs": 4,
            },
        }
        if prefill_backend != "disabled":
            graph_config["prefill"].update(bs=[16, 32, 64, 128], max_bs=128)
        if prefill_backend == "full":
            graph_config["prefill"]["full_prefill_max_req"] = 4
        with patch.object(test_utils, "_launch_server_process", observed_server):
            process = popen_launch_server(
                self.model,
                url,
                timeout=240,
                env={
                    **os.environ,
                    "MOONCAKE_PROTOCOL": self.transfer_protocol,
                    "SGLANG_RAGGED_VERIFY_MODE": ragged_mode if draft else "static",
                },
                other_args=[
                    *(
                        ["--disaggregation-ib-device", self.ib_device]
                        if self.ib_device
                        else []
                    ),
                    *(
                        [
                            "--speculative-algorithm",
                            "DSPARK",
                            "--speculative-draft-model-path",
                            str(draft),
                            "--speculative-draft-attention-backend",
                            "triton",
                        ]
                        if draft
                        else []
                    ),
                    *(
                        ["--speculative-dspark-sps-table-path", str(draft / "sps.json")]
                        if draft and (draft / "sps.json").exists()
                        else []
                    ),
                    "--disaggregation-mode",
                    role,
                    "--tp-size",
                    str(tp_size),
                    "--pp-size",
                    str(pp_size),
                    "--disaggregation-transfer-backend",
                    "mooncake",
                    "--disaggregation-bootstrap-port",
                    str(self.bootstrap_port),
                    *(
                        []
                        if draft and role == "decode"
                        else ["--disaggregation-decode-enable-radix-cache"]
                    ),
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--skip-tokenizer-init",
                    "--attention-backend",
                    self.target_attention_backend
                    or ("fa3" if draft and ragged_mode == "compact" else "triton"),
                    *(
                        ["--disable-flashinfer-autotune"]
                        if self.target_attention_backend == "flashinfer"
                        else []
                    ),
                    "--mem-fraction-static",
                    "0.20",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-config",
                    json.dumps(graph_config),
                    *(
                        ["--enable-torch-compile-debug-mode"]
                        if prefill_backend == "tc_piecewise"
                        else []
                    ),
                    *(
                        []
                        if (replay if enable_overlap is None else enable_overlap)
                        and pp_size == 1
                        else ["--disable-overlap-schedule"]
                    ),
                    *extra_args,
                ],
            )
        self.addCleanup(self.stop_process, process)
        return process, url

    def get_draft(self, kind):
        if kind in self.drafts:
            return self.drafts[kind]
        destination = self.root / f"pd-{kind}-draft"
        if kind == "target_hidden":
            from sglang.test.dspark_ragged_capture_runtime import (
                export_confidence_draft,
            )

            export_confidence_draft(self.model, destination)
        else:
            self.assertEqual(kind, "target_kv")
            from sglang.test.dspark_target_kv_runtime import export_synthetic_kv_draft

            root = self.root / "draft-seed"
            root.mkdir()
            self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 4000
            prefill, self.prefill_url = self.launch(
                "prefill", root, replay=False, tp_size=1, pp_size=1
            )
            decode, self.decode_url = self.launch(
                "decode", root, replay=False, tp_size=1, pp_size=1
            )
            first = len(self.catalog.publications)
            try:
                self.generate("draft-seed", [1, 2, 3], 3)
                sample = read_snapshot(
                    self.reader, self.catalog.wait_publications(first + 1)[-1]
                )
            finally:
                self.stop_process(decode)
                self.stop_process(prefill)
            export_synthetic_kv_draft(self.model, destination, *sample)
        self.drafts[kind] = destination
        return destination

    def generate(self, rid, prompt, count, *, biased=False):
        batched = isinstance(rid, list)
        self.wait_capture_capacity(len(rid) if batched else 1)
        rooms = [self.bootstrap_room + i + 1 for i in range(len(rid) if batched else 1)]
        self.bootstrap_room = rooms[-1]
        sampling = {"temperature": 0, "ignore_eos": True}
        if biased:
            sampling["logit_bias"] = {"100": 100.0}
        payload = {
            "rid": rid,
            "input_ids": prompt,
            "bootstrap_host": self.prefill_host,
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

    def wait_capture_capacity(self, count):
        # Publication and the background refill of capture leases are separate.
        # Cohort admission reserves lazily when the request reaches the router.
        deadline = time.monotonic() + 20
        while True:
            state = self.capture_state()
            self.assertIsNone(state["disabled_reason"], state)
            if (
                "request_router" in state
                or state["states"].get("available", 0) >= count
            ):
                return
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.05)

    def abort_capture(self, rid, *, distributed, prompt=None):
        self.wait_capture_capacity(1)
        before = self.capture_state()
        self.bootstrap_room += 1
        payload = {
            "rid": rid,
            "input_ids": prompt if prompt is not None else [1, 2, 3, 4],
            "bootstrap_host": self.prefill_host,
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
            reason = "cohort_failed" if distributed else "request_aborted_or_retracted"
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
        if distributed:
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

    def exercise(
        self,
        *,
        replay,
        prefill_tp=1,
        decode_tp=1,
        prefill_pp=1,
        decode_pp=1,
        draft_kind=None,
        prefill_draft=False,
        ragged_mode="static",
        prefill_backend="disabled",
        enable_overlap=None,
    ):
        baseline_key = (
            "pd_prefill_outputs",
            prefill_tp,
            decode_tp,
            prefill_pp,
            decode_pp,
        )
        if (
            self.validate_prefill_graph
            and prefill_backend != "disabled"
            and baseline_key not in self.drafts
        ):
            self.exercise(
                replay=False,
                prefill_tp=prefill_tp,
                decode_tp=decode_tp,
                prefill_pp=prefill_pp,
                decode_pp=decode_pp,
                enable_overlap=False,
            )
        draft = self.get_draft(draft_kind) if draft_kind else None
        suffix = (
            f"{replay}-{draft_kind}-{prefill_draft}-{ragged_mode}"
            if draft
            else str(replay)
        )
        if self.validate_prefill_graph:
            suffix += f"-prefill-{prefill_backend}-overlap-{enable_overlap}"
        root = self.root / (
            f"p{prefill_tp}x{prefill_pp}-d{decode_tp}x{decode_pp}-replay-{suffix}"
        )
        root.mkdir()
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 5000
        prefill, self.prefill_url = self.launch(
            "prefill",
            root,
            replay=replay,
            tp_size=prefill_tp,
            pp_size=prefill_pp,
            draft=draft if prefill_draft else None,
            ragged_mode=ragged_mode,
            prefill_backend=prefill_backend,
            enable_overlap=enable_overlap,
        )
        decode, self.decode_url = self.launch(
            "decode",
            root,
            replay=replay,
            tp_size=decode_tp,
            pp_size=decode_pp,
            draft=draft,
            ragged_mode=ragged_mode,
            enable_overlap=enable_overlap,
        )
        if draft and ragged_mode != "static":
            setting = requests.post(
                self.decode_url + "/set_internal_state",
                json={"server_args": {"dspark_force_budget_frac": 0.625}},
                timeout=20,
            )
            self.assertEqual(setting.status_code, 200, setting.text)
            updates = setting.json()
            updates = updates if isinstance(updates, list) else [updates]
            self.assertTrue(
                all(
                    item.get("updated") if isinstance(item, dict) else item is True
                    for item in updates
                ),
                updates,
            )
        acceptance_offsets = (
            {
                path: len(path.read_text().splitlines())
                for path in draft.glob("acceptance-tp*.jsonl")
            }
            if draft
            else {}
        )
        first = len(self.catalog.publications)
        responses = {}
        outputs = {}
        observation_offsets = (
            {
                path: len(path.read_text().splitlines())
                for path in draft.glob("observations*.jsonl")
            }
            if draft_kind == "target_kv"
            else {}
        )
        prompt = [1] + [16, 17, 18, 19] * 38
        cases = [
            ("single", [1, 2, 3], 1, False),
            ("chunked", prompt, 7, True),
            ("cached", prompt, 5, True),
        ]
        if draft or self.validate_prefill_graph:
            cases.append(("rejected", [1, 9, 8, 3], 6, False))
        if self.validate_prefill_graph:
            cases.extend(
                [
                    ("cached-extension", prompt + list(range(600, 619)), 7, True),
                    ("unbiased", list(range(8000, 8013)), 6, False),
                ]
            )
        for rid, tokens, count, biased in cases:
            name = f"{rid}-{suffix}"
            result = self.generate(name, tokens, count, biased=biased)
            outputs[rid] = result["output_ids"]
            responses[hashlib.sha256(name.encode()).hexdigest()] = (tokens, result)
            self.catalog.wait_publications(first + len(responses), timeout=45)
        for iteration in range(2 if self.validate_prefill_graph else 1):
            labels = [
                (
                    f"batch-{iteration}-{i}"
                    if self.validate_prefill_graph
                    else f"batch-{i}"
                )
                for i in range(2)
            ]
            names = [f"{label}-{suffix}" for label in labels]
            prompts = (
                [
                    list(
                        range(
                            2000 + iteration * 1000 + i * 100,
                            2000 + iteration * 1000 + i * 100 + length,
                        )
                    )
                    for i, length in enumerate((33, 37))
                ]
                if self.validate_prefill_graph
                else [[1, 3, 9, 2], [1, 7, 8, 3, 2, 4, 5]]
            )
            for label, name, tokens, result in zip(
                labels,
                names,
                prompts,
                self.generate(names, prompts, 3, biased=True),
                strict=True,
            ):
                outputs[label] = result["output_ids"]
                responses[hashlib.sha256(name.encode()).hexdigest()] = (tokens, result)
            self.catalog.wait_publications(first + len(responses), timeout=45)
        if self.validate_prefill_graph and prefill_backend != "disabled":
            self.assertEqual(outputs, self.drafts[baseline_key])
        expected = len(responses)
        publications = self.catalog.wait_publications(first + expected, timeout=45)
        paths = self.reference_paths(root)
        if draft:
            paths.extend(sorted(draft.rglob("*.pt")))
        references = [
            frame
            for path in paths
            if (frame := torch.load(path, weights_only=True))["trace_id"] in responses
        ]
        prefill_replays = []
        if self.validate_prefill_graph:
            prefill_frames = [r for r in references if r.get("pd_role") == "prefill"]
            prefill_replays = [r for r in prefill_frames if r["cuda_graph"]]
            if prefill_backend == "disabled":
                self.assertFalse(prefill_replays)
            else:
                self.assertEqual(
                    {(r["tp_rank"], r["pp_rank"]) for r in prefill_replays},
                    {(tp, pp) for tp in range(prefill_tp) for pp in range(prefill_pp)},
                )
                for frame in prefill_replays:
                    self.assertIsNotNone(frame["prefill_graph"])
                    self.assertEqual(
                        frame["prefill_graph"]["capture_hidden_mode"], "NULL"
                    )
                    self.assertFalse(frame["prefill_graph"]["output_hidden_states"])
                self.assertTrue(
                    any(
                        r["prefill_graph"]["raw_tokens"]
                        < r["prefill_graph"]["padded_tokens"]
                        for r in prefill_replays
                    )
                )
                self.assertTrue(
                    any(
                        r["extend_prefix_length"] >= len(prompt)
                        for r in prefill_replays
                    )
                )
                if prefill_backend == "full":
                    self.assertTrue(
                        any(
                            r["batch_size"] < r["prefill_graph"]["request_slots"]
                            for r in prefill_replays
                        )
                    )
                buffers = {}
                for frame in prefill_replays:
                    key = (
                        frame["tp_rank"],
                        frame["pp_rank"],
                        frame["prefill_graph"]["input_buffer"],
                    )
                    buffers.setdefault(key, set()).add(
                        frame["prefill_graph"]["replay_id"]
                    )
                self.assertTrue(any(len(ids) > 1 for ids in buffers.values()))
            self.assertTrue(any(not r["predictions"] for r in prefill_frames))
            decode_frames = [
                r
                for r in references
                if r.get("forward_mode") and r.get("pd_role") != "prefill"
            ]
            self.assertTrue(decode_frames)
            self.assertTrue(
                all(
                    r["forward_mode"] in ("DECODE", "TARGET_VERIFY")
                    for r in decode_frames
                )
            )
        capture_mode = (
            "pd_speculative_accepted_target_path" if draft else "pd_autoregressive"
        )
        if draft:
            decode_references = [
                frame
                for path in draft.rglob("*.pt")
                if (frame := torch.load(path, weights_only=True))["trace_id"]
                in responses
            ]
            self.assertTrue(decode_references)
            self.assertTrue(
                all(frame["verify_width"] is not None for frame in decode_references)
            )
            self.assertTrue(any(frame["num_commit"] > 1 for frame in decode_references))
            self.assertTrue(
                any(
                    frame["num_commit"] < frame["verify_width"]
                    for frame in decode_references
                )
            )
        if draft_kind == "target_kv":
            observations = [
                json.loads(line)
                for path in draft.glob("observations*.jsonl")
                for line in path.read_text().splitlines()[
                    observation_offsets.get(path, 0) :
                ]
            ]
            contexts = [item for item in observations if item["kind"] == "context"]
            self.assertTrue(
                any(
                    item["previous_end"] is None
                    and item["projected_end"] == len(prompt)
                    for item in contexts
                )
            )
            self.assertTrue(any(item["kind"] == "projection" for item in observations))
            if replay:
                self.assertTrue(
                    any(
                        item["kind"] == "target_verify" and item["cuda_graph"]
                        for item in observations
                    )
                )
        if draft and ragged_mode != "static":
            acceptance = [
                json.loads(line)
                for path in draft.glob("acceptance-tp*.jsonl")
                for line in path.read_text().splitlines()[
                    acceptance_offsets.get(path, 0) :
                ]
            ]
            self.assertTrue(
                any(
                    item["verify_lens"] and len(set(item["verify_lens"])) > 1
                    for item in acceptance
                ),
                acceptance,
            )
            self.assertTrue(
                any(any(item["cap_trim_lens"]) for item in acceptance), acceptance
            )
            if ragged_mode == "compact":
                self.assertTrue(
                    any(
                        frame["verify_count"] < frame["verify_width"]
                        for frame in decode_references
                    )
                )
                if replay:
                    self.assertTrue(
                        any(frame["verify_padding"] > 0 for frame in decode_references)
                    )
                    self.assertTrue(
                        any(item["folded"] for item in acceptance), acceptance
                    )
        if replay:
            self.assertTrue(any(frame["cuda_graph"] for frame in references))
        self.assertTrue(any(frame["batch_size"] > 1 for frame in references))
        for publication in publications[first:]:
            manifest, tensors = read_snapshot(self.reader, publication)
            self.assertEqual(manifest.topology.tp_size, decode_tp)
            self.assertEqual(manifest.topology.pp_size, decode_pp)
            tokens, result = responses.pop(manifest.provenance.trace_id)
            self.assertEqual(
                tensors["token_ids"].tolist(), tokens + result["output_ids"]
            )
            check_capture_snapshot(
                self, manifest, tensors, references, capture_mode=capture_mode
            )
        self.assertFalse(responses)
        for fault_index, fault in enumerate(("missing", "stale")):
            with self.catalog.condition:
                failures_before = sum(
                    v["state"] == "FAILED" for v in self.catalog.captures.values()
                )
            self.generate(
                f"{fault}-{suffix}",
                (
                    list(
                        range(
                            9000 + 20 * fault_index,
                            9013 + 20 * fault_index,
                        )
                    )
                    if self.validate_prefill_graph
                    else [1, 3, 8, 2]
                ),
                3,
                biased=True,
            )
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
        state = self.abort_capture(
            f"abort-{suffix}",
            distributed=decode_tp > 1 or decode_pp > 1,
            prompt=list(range(9100, 9113)) if self.validate_prefill_graph else None,
        )
        self.assertEqual(len(self.catalog.publications), first + expected)
        self.assertGreaterEqual(state["counters"]["pd_handoff_committed"], expected + 1)
        self.assertGreaterEqual(state["counters"]["failed_pd_handoff_failed"], 1)
        self.assertEqual(state["counters"].get("admission_backpressure", 0), 0, state)
        self.assertEqual(state["counters"]["admitted"], expected + 3, state)
        if draft:
            self.assertGreater(state["counters"]["speculative_verify_forwards"], 0)
            self.assertGreater(state["counters"]["speculative_commits_copied"], 0)
        self.assertIsNone(prefill.poll())
        self.assertIsNone(decode.poll())
        self.stop_process(prefill)
        self.stop_process(decode)
        if self.validate_prefill_graph and prefill_backend != "disabled":
            for fault in ("missing", "stale", "abort"):
                trace = hashlib.sha256(f"{fault}-{suffix}".encode()).hexdigest()
                frames = [
                    frame
                    for path in self.reference_paths(root / "prefill")
                    if (frame := torch.load(path, weights_only=True))["trace_id"]
                    == trace
                    and frame.get("pd_role") == "prefill"
                ]
                self.assertTrue(
                    any(r["cuda_graph"] and r["predictions"] for r in frames), fault
                )
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        for publication in publications[first:]:
            # Verify every object digest after both producers have exited.
            read_snapshot(reader, publication)
        if self.validate_prefill_graph and prefill_backend == "disabled":
            self.drafts[baseline_key] = outputs
        print(
            json.dumps(
                {
                    "prefill_tp": prefill_tp,
                    "decode_tp": decode_tp,
                    "prefill_pp": prefill_pp,
                    "decode_pp": decode_pp,
                    "replay": replay,
                    "prefill_backend": prefill_backend,
                    "prefill_replay_frames": len(prefill_replays),
                    "enable_overlap": enable_overlap,
                    "draft_kind": draft_kind,
                    "prefill_draft": prefill_draft,
                    "ragged_mode": ragged_mode,
                    "post_exit_snapshots": expected,
                    "capture": state,
                },
                sort_keys=True,
            )
        )
