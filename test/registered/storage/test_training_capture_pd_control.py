"""Control real P/D endpoints without aborting inference or publishing partial data."""

import hashlib
import json
import shutil
import time
import unittest
from collections import Counter
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
    tp_size = 1
    pp_size = 1

    def rank_names(self):
        return [
            f"pp{pp}-tp{tp}" for pp in range(self.pp_size) for tp in range(self.tp_size)
        ]

    def all_states(self, role):
        root = self.control_root / role / "control-gates"
        root.mkdir(exist_ok=True)
        nonce = str(time.monotonic_ns())
        (root / "state-request").write_text(nonce)
        self.state(role)
        deadline = time.monotonic() + 10
        states = {}
        while len(states) != len(self.rank_names()):
            for rank in self.rank_names():
                path = root / rank / "state.json"
                if path.exists():
                    value = json.loads(path.read_text())
                    if value["nonce"] == nonce:
                        states[rank] = value["state"]
            self.assertLess(time.monotonic(), deadline, (role, nonce, states))
            time.sleep(0.01)
        return states

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
        self.control_counts[role]["control_" + action] += 1
        for rank, observed in self.all_states(role).items():
            self.assertEqual(observed["admission_paused"], action != "resume", rank)
            self.assertIsNone(observed["disabled_reason"], (rank, observed))
            for counter, count in self.control_counts[role].items():
                self.assertEqual(observed["counters"].get(counter), count, rank)
        return state

    def wait_until(self, predicate):
        deadline = time.monotonic() + 30
        while not predicate():
            self.assertLess(time.monotonic(), deadline, self.state("decode"))
            time.sleep(0.01)

    def wait_capture_capacity(self, count):
        state = self.capture_state()
        if state["admission_paused"]:
            return
        if "request_router" not in state:
            return super().wait_capture_capacity(count)
        # Resume clears the manual flag before the background all-rank vote
        # restores ticket availability. A correctness probe needs both ready.
        self.wait_until(
            lambda: all(
                item["test_service_admission_ready"]
                and item["states"].get("available", 0) >= count
                for item in self.all_states("decode").values()
            )
        )

    def failures(self):
        with self.catalog.condition:
            return {
                key: record.get("reason")
                for key, record in self.catalog.captures.items()
                if record["state"] == "FAILED"
            }

    def assert_failed(self, previous, reason, rid):
        root = self.control_root / "decode" / "control-gates"
        paths = [root / rank / (rid + ".admission") for rank in self.rank_names()]
        self.wait_until(lambda: all(path.exists() for path in paths))
        capture_ids = {json.loads(path.read_text())["capture_id"] for path in paths}
        self.assertEqual(len(capture_ids), 1)
        capture_id = capture_ids.pop()
        self.assertIsNotNone(capture_id)
        self.assertNotIn(capture_id, previous)
        self.wait_until(lambda: capture_id in self.failures())
        added = {k: v for k, v in self.failures().items() if k not in previous}
        distributed = self.tp_size * self.pp_size > 1
        self.assertEqual(added[capture_id], "cohort_failed" if distributed else reason)
        self.failed_requests[rid] = capture_id
        if not distributed:
            self.assertEqual(set(added), {capture_id})
            return
        self.check_unbound_failures(added)

    def check_unbound_failures(self, failures):
        # Abort also retires spare tickets. They must never have been admitted
        # to another request, written any payload, or become a publication.
        root = self.control_root / "decode" / "control-gates"
        admitted = {
            json.loads(path.read_text())["capture_id"]
            for path in root.glob("*/*.admission")
        }
        for key in set(failures) - set(self.failed_requests.values()):
            self.assertGreater(self.tp_size * self.pp_size, 1)
            self.assertNotIn(key, admitted)
            self.assertEqual(failures[key], "cohort_failed")
            with self.catalog.condition:
                self.assertFalse(self.catalog.captures[key]["registered"])
                self.assertFalse(self.catalog.captures[key]["written"])
                self.assertNotIn(key, self.catalog.publications)
            self.failed_unbound.add(key)

    def remember(self, rid, prompt, result, expected):
        expected[hashlib.sha256(rid.encode()).hexdigest()] = (prompt, result)

    def gated_request(self, root, rid, prompt, action):
        gates = root / "prefill" / "control-gates"
        gates.mkdir(exist_ok=True)
        hold = gates / (rid + ".hold")
        entered = [gates / rank / (rid + ".entered") for rank in self.rank_names()]
        hold.touch()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.generate, rid, prompt, 16, biased=True)
            try:
                self.wait_until(lambda: all(path.exists() for path in entered))
                for path in entered:
                    captured = json.loads(path.read_text())
                    self.assertTrue(captured["attempted"], path)
                    self.assertTrue(captured["selected"], path)
                    self.assertGreater(captured["chunk_end"], 0)
                    self.assertLess(captured["chunk_end"], len(prompt))
                self.assertEqual(self.state("decode")["states"].get("active"), 1)
                action()
            finally:
                response = requests.post(
                    self.prefill_url + "/set_internal_state",
                    json={"server_args": {"training_capture_test_release:" + rid: 1}},
                    timeout=10,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(all(response.json()), response.text)
                for state in self.all_states("prefill").values():
                    self.assertIn(rid, state["test_released_gates"])
                hold.unlink(missing_ok=True)
            result = pending.result(timeout=120)
        handoff = {
            rank: json.loads((gates / rank / (rid + ".handoff")).read_text())
            for rank in self.rank_names()
        }
        return result, handoff

    def exercise_controls(self, replay):
        root = self.root / f"control-replay-{replay}"
        root.mkdir()
        self.control_root = root
        self.control_counts = {role: Counter() for role in ("prefill", "decode")}
        self.failed_requests, self.failed_unbound = {}, set()
        draft = (
            Path(shutil.copytree(self.get_draft(self.draft_kind), root / "draft"))
            if self.draft_kind
            else None
        )
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 12000
        prefill, self.prefill_url = self.launch(
            "prefill", root, replay=replay, tp_size=self.tp_size, pp_size=self.pp_size
        )
        decode, self.decode_url = self.launch(
            "decode",
            root,
            replay=replay,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            draft=draft,
        )
        first = len(self.catalog.publications)
        initial_failures = self.failures()
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
        self.assertTrue(
            all(value["payload_present"] for value in handoffs[rid].values())
        )
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
            self.assert_failed(previous, reason, rid)
            self.assertEqual(len(self.catalog.publications), first + len(expected))
            self.assertEqual(
                self.state("decode")["counters"].get("pd_handoff_committed", 0), before
            )
            for value in handoffs[rid].values():
                self.assertEqual(value["payload_present"], "prefill" not in roles)
                if "prefill" in roles:
                    self.assertLess(value["capture_epoch"], value["current_epoch"])
            published(f"control-recovered-{index}")

        # A paused P with running D must lose the sample, not the inference.
        previous = self.failures()
        self.control("prefill", "pause")
        self.generate("control-only-p-paused", short, 16, biased=True)
        self.assert_failed(previous, "pd_handoff_failed", "control-only-p-paused")
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
        self.assert_failed(previous, "operator_aborted", "control-running-decode")
        self.control("decode", "resume")
        published("control-after-running-abort")
        self.control("decode", "pause")
        self.wait_until(
            lambda: all(
                not observed.get("test_invalid_captures")
                and not any(
                    n for state, n in observed["states"].items() if state != "available"
                )
                for observed in self.all_states("decode").values()
            )
        )
        failures = {
            key: reason
            for key, reason in self.failures().items()
            if key not in initial_failures
        }
        self.check_unbound_failures(failures)
        self.assertEqual(len(self.failed_requests), 5)
        self.assertEqual(
            set(failures), set(self.failed_requests.values()) | self.failed_unbound
        )
        final_d, final_p = self.state("decode"), self.state("prefill")
        rank_states = {role: self.all_states(role) for role in ("prefill", "decode")}
        self.assertEqual(
            sum(s["counters"]["ready"] for s in rank_states["decode"].values()),
            len(expected),
        )
        for state in rank_states["decode"].values():
            self.assertEqual(state["host_pool"]["quarantined"], 0)
            self.assertEqual(state["counters"].get("admission_backpressure", 0), 0)
            self.assertIsNone(state["disabled_reason"])
            writer = state.get("cohort_writer")
            if writer is not None:
                self.assertIsNone(writer["error"], writer)
            if replay:
                self.assertGreater(state["counters"].get("cuda_graph_forwards", 0), 0)
            if replay and self.pp_size == 1:
                self.assertGreater(state["counters"].get("overlap_forwards", 0), 0)
        self.assertEqual(final_d["host_pool"]["quarantined"], 0)
        self.assertEqual(final_d["counters"].get("admission_backpressure", 0), 0)
        if replay:
            self.assertGreater(final_d["counters"].get("cuda_graph_forwards", 0), 0)
            if self.pp_size == 1:
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
            self.assertEqual(manifest.topology.tp_size, self.tp_size)
            self.assertEqual(manifest.topology.pp_size, self.pp_size)
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
                    "tp_size": self.tp_size,
                    "pp_size": self.pp_size,
                    "rank_states": rank_states,
                    "failed_request_captures": self.failed_requests,
                    "failed_unbound_captures": sorted(self.failed_unbound),
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
