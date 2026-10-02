"""Control real P/D endpoints without aborting inference or publishing partial data."""

import hashlib
import json
import shutil
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot

register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-small")


class TestPDCaptureControl(PDCaptureRuntimeBase):
    observer_module = "sglang.test.pd_capture_control_server"
    teacher_d2h_batch_tokens = 16
    draft_kind = None

    def state(self, role):
        url = self.prefill_url if role == "prefill" else self.decode_url
        response = requests.get(url + "/server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def control(self, role, action):
        url = self.prefill_url if role == "prefill" else self.decode_url
        response = requests.post(
            url + "/control_training_capture", json={"action": action}, timeout=10
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["success"], response.text)
        state = response.json()["results"][0]["state"]
        self.assertEqual(state["admission_paused"], action != "resume")
        self.assertIsNone(state["disabled_reason"], state)
        return state

    def wait_until(self, predicate):
        deadline = time.monotonic() + 30
        while not predicate():
            self.assertLess(time.monotonic(), deadline, self.state("decode"))
            time.sleep(0.01)

    def wait_capture_capacity(self, count):
        if not self.capture_state()["admission_paused"]:
            super().wait_capture_capacity(count)

    def failures(self):
        with self.catalog.condition:
            return {
                key: record.get("reason")
                for key, record in self.catalog.captures.items()
                if record["state"] == "FAILED"
            }

    def assert_failed(self, previous, reason):
        self.wait_until(lambda: len(self.failures()) > len(previous))
        added = {k: v for k, v in self.failures().items() if k not in previous}
        self.assertEqual(list(added.values()), [reason])

    def remember(self, rid, prompt, result, expected):
        expected[hashlib.sha256(rid.encode()).hexdigest()] = (prompt, result)

    def gated_request(self, root, rid, prompt, action):
        gates = root / "prefill" / "control-gates"
        gates.mkdir(exist_ok=True)
        hold = gates / (rid + ".hold")
        entered = gates / (rid + ".entered")
        hold.touch()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.generate, rid, prompt, 16, biased=True)
            try:
                self.wait_until(entered.exists)
                captured = json.loads(entered.read_text())
                self.assertTrue(captured["attempted"])
                self.assertTrue(captured["selected"])
                self.assertGreater(captured["chunk_end"], 0)
                self.assertLess(captured["chunk_end"], len(prompt))
                self.assertEqual(self.state("decode")["states"].get("active"), 1)
                action()
            finally:
                hold.unlink(missing_ok=True)
            result = pending.result(timeout=120)
        handoff = json.loads((gates / (rid + ".handoff")).read_text())
        return result, handoff

    def exercise_controls(self, replay):
        root = self.root / f"control-replay-{replay}"
        root.mkdir()
        draft = (
            Path(shutil.copytree(self.get_draft(self.draft_kind), root / "draft"))
            if self.draft_kind
            else None
        )
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 12000
        prefill, self.prefill_url = self.launch(
            "prefill", root, replay=replay, tp_size=1, pp_size=1
        )
        decode, self.decode_url = self.launch(
            "decode", root, replay=replay, tp_size=1, pp_size=1, draft=draft
        )
        first = len(self.catalog.publications)
        expected, handoffs = {}, {}
        short = [1, 16, 17, 18]

        def published(rid, prompt=short):
            result = self.generate(rid, prompt, 16, biased=True)
            self.remember(rid, prompt, result, expected)
            self.catalog.wait_publications(first + len(expected), timeout=30)
            return result

        published("control-baseline")
        before_d, before_p = self.state("decode"), self.state("prefill")
        self.control("decode", "pause")
        self.control("prefill", "pause")
        self.generate("control-both-paused", short, 16, biased=True)
        self.control("prefill", "resume")
        self.generate("control-only-d-paused", short, 16, biased=True)
        self.assertEqual(
            self.state("decode")["counters"]["admitted"],
            before_d["counters"]["admitted"],
        )
        self.assertEqual(
            self.state("prefill")["counters"]["pd_selected"],
            before_p["counters"]["pd_selected"],
        )
        self.assertEqual(len(self.catalog.publications), first + len(expected))
        self.control("decode", "resume")
        published("control-first-resume")

        def pause_inflight():
            self.control("decode", "pause")
            self.control("prefill", "pause")

        prompt = list(range(1000, 1200))
        rid = "control-gated-drain"
        result, handoffs[rid] = self.gated_request(root, rid, prompt, pause_inflight)
        self.assertTrue(handoffs[rid]["payload_present"])
        self.remember(rid, prompt, result, expected)
        self.catalog.wait_publications(first + len(expected), timeout=30)
        self.control("prefill", "resume")
        self.control("decode", "resume")

        for index, roles in enumerate(
            (("prefill",), ("decode",), ("prefill", "decode"))
        ):
            previous = self.failures()
            before = self.state("decode")["counters"].get("pd_handoff_committed", 0)
            prompt = list(range(2000 + index * 300, 2200 + index * 300))
            rid = "control-gated-abort-" + "-".join(roles)

            def abort_and_resume(roles=roles):
                for role in roles:
                    self.control(role, "abort")
                for role in roles:
                    self.control(role, "resume")

            _, handoffs[rid] = self.gated_request(root, rid, prompt, abort_and_resume)
            reason = "operator_aborted" if "decode" in roles else "pd_handoff_failed"
            self.assert_failed(previous, reason)
            self.assertEqual(len(self.catalog.publications), first + len(expected))
            self.assertEqual(
                self.state("decode")["counters"].get("pd_handoff_committed", 0), before
            )
            self.assertEqual(handoffs[rid]["payload_present"], "prefill" not in roles)
            if "prefill" in roles:
                self.assertLess(
                    handoffs[rid]["capture_epoch"], handoffs[rid]["current_epoch"]
                )
            published(f"control-recovered-{index}")

        # A paused P with running D must lose the sample, not the inference.
        previous = self.failures()
        self.control("prefill", "pause")
        self.generate("control-only-p-paused", short, 16, biased=True)
        self.assert_failed(previous, "pd_handoff_failed")
        self.assertEqual(len(self.catalog.publications), first + len(expected))
        self.control("prefill", "resume")
        published("control-final-resume")

        # Abort D after handoff while actual decode (including graph replay) runs.
        previous = self.failures()
        committed = self.state("decode")["counters"]["pd_handoff_committed"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                self.generate, "control-running-decode", short, 200, biased=True
            )
            self.wait_until(
                lambda: self.state("decode")["counters"]["pd_handoff_committed"]
                > committed
            )
            self.assertEqual(self.state("decode")["states"].get("active"), 1)
            self.control("decode", "abort")
            pending.result(timeout=120)
        self.assert_failed(previous, "operator_aborted")
        self.control("decode", "resume")
        published("control-after-running-abort")
        self.control("decode", "pause")
        self.wait_until(
            lambda: not any(
                n
                for state, n in self.state("decode")["states"].items()
                if state != "available"
            )
        )
        final_d, final_p = self.state("decode"), self.state("prefill")
        self.assertEqual(final_d["host_pool"]["quarantined"], 0)
        self.assertEqual(final_d["counters"].get("admission_backpressure", 0), 0)
        if replay:
            self.assertGreater(final_d["counters"].get("cuda_graph_forwards", 0), 0)
            self.assertGreater(final_d["counters"].get("overlap_forwards", 0), 0)
        if draft:
            self.assertGreater(
                final_d["counters"].get("speculative_verify_forwards", 0), 0
            )
            self.assertGreater(
                final_d["counters"].get("speculative_commits_copied", 0), 0
            )
        self.assertFalse(self.catalog.errors)
        self.assertIsNone(prefill.poll())
        self.assertIsNone(decode.poll())
        self.stop_process(prefill)
        self.stop_process(decode)

        references = [
            torch.load(path, weights_only=True) for path in self.reference_paths(root)
        ]
        publications = self.catalog.wait_publications(first + len(expected))[first:]
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        for publication in publications:
            manifest, tensors = read_snapshot(reader, publication)
            prompt, result = expected.pop(manifest.provenance.trace_id)
            self.assertEqual(
                tensors["token_ids"].tolist(), prompt + result["output_ids"]
            )
            check_capture_snapshot(
                self,
                manifest,
                tensors,
                references,
                capture_mode=(
                    "pd_speculative_accepted_target_path"
                    if draft
                    else "pd_autoregressive"
                ),
            )
        self.assertFalse(expected)
        print(
            json.dumps(
                {
                    "pd_control": True,
                    "replay": replay,
                    "draft_kind": self.draft_kind,
                    "post_exit_snapshots": len(publications),
                    "handoffs": handoffs,
                    "decode": final_d,
                    "prefill": final_p,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def test_eager_controls(self):
        self.exercise_controls(False)

    def test_graph_overlap_controls(self):
        self.exercise_controls(True)


class TestDSparkPDCaptureControl(TestPDCaptureControl):
    draft_kind = "target_kv"


if __name__ == "__main__":
    unittest.main()
