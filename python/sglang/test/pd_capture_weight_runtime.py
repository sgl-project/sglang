"""One-sided P/D weight changes and unequal fresh identities must fail capture."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import patch

import requests
import torch

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_utils import read_snapshot
from sglang.test.training_capture_weight_runtime import CaptureWeightRuntimeBase


class PDCaptureWeightRuntimeBase(CaptureWeightRuntimeBase):
    observer_module = "sglang.test.pd_capture_weight_server"
    teacher_d2h_batch_tokens = 16

    def state(self, role):
        response = requests.get(self.urls[role] + "/server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def post(self, role, endpoint, payload):
        response = requests.post(self.urls[role] + endpoint, json=payload, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def wait_capture_capacity(self, count):
        if self.state("decode")["disabled_reason"]:
            return
        super().wait_capture_capacity(count)

    def wait_until(self, predicate):
        deadline = time.monotonic() + 30
        while not predicate():
            self.assertLess(time.monotonic(), deadline, self.state("decode"))
            time.sleep(0.02)

    def failed_capture(self, folder, rid, reason):
        path = folder / "decode" / "control-gates" / "pp0-tp0" / (rid + ".admission")
        self.wait_until(path.exists)
        capture_id = json.loads(path.read_text())["capture_id"]
        self.assertIsNotNone(capture_id)
        with self.catalog.condition:
            self.assertTrue(
                self.catalog.condition.wait_for(
                    lambda: self.catalog.captures[capture_id]["state"] == "FAILED",
                    timeout=20,
                )
            )
            record = self.catalog.captures[capture_id]
            self.assertEqual(record["reason"], reason)
            self.assertNotIn(capture_id, self.catalog.publications)
            self.assertFalse(record["registered"])
            self.assertFalse(record["written"])
        return capture_id

    def exercise_pd_weights(self, *, role, replay):
        root = self.root / f"pd-weights-{role}-{replay}"
        root.mkdir()
        self.urls = {}
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 19000
        first = len(self.catalog.publications)
        expected, failed = {}, {}

        def start(model, stage, side):
            stage.mkdir(exist_ok=True)
            with patch.object(self, "model", str(model)):
                process, url = self.launch(
                    side, stage, replay=replay, tp_size=1, pp_size=1
                )
            self.urls[side] = url
            setattr(self, side + "_url", url)
            return process

        def generate(name, prompt):
            result = self.generate(name, prompt, 4, biased=True)
            self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
            return prompt + result["output_ids"]

        prompt = [17, 19, 23, 29] * 4
        initial = root / "initial"
        prefill = start(self.model, initial, "prefill")
        decode = start(self.model, initial, "decode")
        try:
            rid = f"control-weights-{role}-{replay}-old"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(rid, prompt)
            self.catalog.wait_publications(first + 1)
            self.wait_capture_capacity(1)
            rid = f"control-gated-weights-{role}-{replay}"
            gates = initial / "prefill" / "control-gates"
            gates.mkdir(exist_ok=True)
            hold = gates / (rid + ".hold")
            hold.touch()
            entered = gates / "pp0-tp0" / (rid + ".entered")
            paused = False
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(generate, rid, list(range(1000, 1200)))
                try:
                    self.wait_until(entered.exists)
                    boundary = json.loads(entered.read_text())
                    self.assertTrue(boundary["selected"])
                    self.assertGreater(boundary["chunk_end"], 0)
                    self.assertLess(boundary["chunk_end"], boundary["prompt_length"])
                    self.assertEqual(self.state("decode")["states"].get("active"), 1)
                    self.post(role, "/pause_generation", {"mode": "in_place"})
                    paused = True
                    result = self.post(
                        role,
                        "/update_weights_from_disk",
                        {
                            "model_path": str(self.replacement),
                            "flush_cache": False,
                        },
                    ).json()
                    self.assertTrue(result["success"], result)
                    mutation = json.loads(
                        (initial / role / "weight-mutation.json").read_text()
                    )
                    self.assertNotEqual(
                        mutation["before_sha256"], mutation["after_sha256"]
                    )
                    self.assertEqual(
                        mutation["expected_after_sha256"], mutation["after_sha256"]
                    )
                    self.assertEqual(
                        self.state(role)["disabled_reason"], "target_weights_update"
                    )
                finally:
                    if paused:
                        self.post(role, "/continue_generation", {})
                    self.post(
                        "prefill",
                        "/set_internal_state",
                        {"server_args": {"training_capture_test_release:" + rid: 1}},
                    )
                    hold.unlink(missing_ok=True)
                self.assertEqual(
                    pending.result(timeout=120), list(range(1000, 1200)) + [100] * 4
                )
            failed[rid] = self.failed_capture(
                initial,
                rid,
                "pd_handoff_failed" if role == "prefill" else "target_weights_update",
            )
            handoff = json.loads((gates / "pp0-tp0" / (rid + ".handoff")).read_text())
            self.assertEqual(handoff["payload_present"], role == "decode")
            self.post(role, "/control_training_capture", {"action": "resume"})
            self.post(role, "/flush_cache", {})
            rid = f"control-weights-{role}-{replay}-after-update"
            generate(rid, [31, 37, 41, 43])
            if role == "prefill":
                failed[rid] = self.failed_capture(initial, rid, "pd_handoff_failed")
            else:
                admission = (
                    initial
                    / "decode"
                    / "control-gates"
                    / "pp0-tp0"
                    / (rid + ".admission")
                )
                self.assertIsNone(json.loads(admission.read_text())["capture_id"])
            self.assertEqual(
                self.state(role)["disabled_reason"], "target_weights_update"
            )
            self.assertEqual(len(self.catalog.publications), first + 1)
            updated = {side: self.state(side) for side in ("prefill", "decode")}
            self.assertEqual(updated["decode"]["host_pool"]["quarantined"], 0)
        finally:
            self.stop_process(decode)
            self.stop_process(prefill)

        self.bootstrap_port = self.new_bootstrap_port()
        mismatch = root / "mismatch"
        prefill = start(self.replacement, mismatch, "prefill")
        decode = start(self.model, mismatch, "decode")
        try:
            rid = f"control-weights-{role}-{replay}-mismatch"
            generate(rid, [47, 53, 59, 61])
            failed[rid] = self.failed_capture(mismatch, rid, "pd_handoff_failed")
            mismatch_states = {side: self.state(side) for side in ("prefill", "decode")}
            self.assertEqual(
                mismatch_states["prefill"]["counters"]["pd_context_rejected"], 1
            )
            self.assertTrue(
                all(s["disabled_reason"] is None for s in mismatch_states.values())
            )
            self.assertEqual(len(self.catalog.publications), first + 1)
            self.stop_process(decode)
            matched = root / "matched"
            decode = start(self.replacement, matched, "decode")
            rid = f"control-weights-{role}-{replay}-new"
            expected[hashlib.sha256(rid.encode()).hexdigest()] = generate(rid, prompt)
            self.catalog.wait_publications(first + 2)
            self.wait_capture_capacity(4)
            final = {side: self.state(side) for side in ("prefill", "decode")}
            self.assertTrue(all(s["disabled_reason"] is None for s in final.values()))
            self.assertEqual(final["decode"]["host_pool"]["quarantined"], 0)
            self.assertEqual(
                final["decode"]["counters"].get("cuda_graph_forwards", 0) > 0, replay
            )
        finally:
            self.stop_process(decode)
            self.stop_process(prefill)

        references = [
            torch.load(path, weights_only=True) for path in self.reference_paths(root)
        ]
        publications = self.catalog.wait_publications(first + 2)[first:]
        self.assertEqual(len(publications), 2)
        objects = payload_bytes = 0
        with closing(MooncakeSnapshotStore.connect(self.store_setup)) as reader:
            for version, publication in enumerate(publications):
                manifest, tensors = read_snapshot(reader, publication)
                self.assertEqual(
                    manifest.teacher.weights_revision, self.revisions[version]
                )
                self.assertEqual(
                    tensors["token_ids"].tolist(),
                    expected.pop(manifest.provenance.trace_id),
                )
                check_capture_snapshot(
                    self,
                    manifest,
                    tensors,
                    references,
                    capture_mode="pd_autoregressive",
                )
                objects += len(manifest.objects)
                payload_bytes += sum(obj.nbytes for obj in manifest.objects)
        self.assertFalse(expected)
        self.assertFalse(self.catalog.errors)
        print(
            json.dumps(
                {
                    "pd_weight_update": {
                        "updated_role": role,
                        "replay": replay,
                        "revisions": self.revisions,
                        "boundary": boundary,
                        "mutation": mutation,
                        "handoff": handoff,
                        "updated": updated,
                        "mismatch": mismatch_states,
                        "final": final,
                        "failed_captures": failed,
                        "post_exit_snapshots": len(publications),
                        "objects": objects,
                        "payload_bytes": payload_bytes,
                    }
                }
            ),
            flush=True,
        )
