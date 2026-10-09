"""Natural P/D KV exhaustion, CPU restore and captured-sample retirement."""

import hashlib
import json
import os
import time
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot


class PDCapturePressureBase(PDCaptureRuntimeBase):
    draft_kind = "target_kv"
    observer_module = "sglang.test.pd_capture_pressure_server"

    def all_capture_states(self):
        root = self.pressure_root / "decode" / "pressure-states"
        root.mkdir(exist_ok=True)
        nonce = str(time.monotonic_ns())
        (root / "request").write_text(nonce)
        response = requests.get(self.decode_url + "/server_info", timeout=10)
        response.raise_for_status()
        deadline = time.monotonic() + 10
        states = {}
        while len(states) != len(self.pressure_ranks):
            for tp, pp in self.pressure_ranks:
                rank = f"pp{pp}-tp{tp}"
                path = root / (rank + ".json")
                if path.exists():
                    value = json.loads(path.read_text())
                    if value["nonce"] == nonce:
                        states[rank] = value["state"]
            self.assertLess(time.monotonic(), deadline, states)
            time.sleep(0.01)
        return states

    def wait_capture_idle(self, *, min_available=0):
        deadline = time.monotonic() + 30
        while True:
            states = self.all_capture_states()
            if all(
                state["states"].get("available", 0) == state["reservations"]
                and state["states"].get("available", 0) >= min_available
                and state.get("queued", 0) == 0
                and state.get("cohort_writer", {}).get("pending", 0) == 0
                for state in states.values()
            ):
                for state in states.values():
                    self.assertEqual(state["host_pool"]["quarantined"], 0)
                    self.assertIsNone(state["disabled_reason"])
                    self.assertFalse(state["admission_paused"])
                return states
            self.assertLess(time.monotonic(), deadline, states)
            time.sleep(0.03)

    def retraction_metrics(self, tp_size, pp_size):
        response = requests.get(self.decode_url + "/metrics", timeout=10)
        response.raise_for_status()
        values = {(tp, pp): 0 for tp in range(tp_size) for pp in range(pp_size)}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name == "sglang:num_retracted_requests_total":
                    rank = (
                        int(sample.labels.get("tp_rank", 0)),
                        int(sample.labels.get("pp_rank", 0)),
                    )
                    values[rank] += sample.value
        return values

    @patch.dict(os.environ, {"SGLANG_TEST_RETRACT": "0"})
    def exercise_pressure(self, *, replay, enable_overlap=False, tp_size=1, pp_size=1):
        draft = self.get_draft(self.draft_kind) if self.draft_kind else None
        suffix = f"{tp_size}-{pp_size}-{replay}-{enable_overlap}"
        ranks = {(tp, pp) for tp in range(tp_size) for pp in range(pp_size)}
        distributed = tp_size * pp_size > 1
        root = self.root / f"pressure-{suffix}"
        root.mkdir()
        self.pressure_root, self.pressure_ranks = root, ranks
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 8000
        extra = [
            "--max-total-tokens",
            "512",
            "--num-reserved-decode-tokens",
            "16",
            "--enable-metrics",
            "--enable-metrics-for-all-schedulers",
        ]
        prefill, self.prefill_url = self.launch(
            "prefill",
            root,
            replay=False,
            tp_size=tp_size,
            pp_size=pp_size,
            draft=draft,
            extra_args=extra,
        )
        decode, self.decode_url = self.launch(
            "decode",
            root,
            replay=replay,
            tp_size=tp_size,
            pp_size=pp_size,
            draft=draft,
            enable_overlap=enable_overlap,
            extra_args=extra,
        )
        # Single-rank leases are prepared asynchronously; cohort admission is lazy.
        before = self.wait_capture_idle(min_available=0 if distributed else 4)
        metric_before = self.retraction_metrics(tp_size, pp_size)
        first = len(self.catalog.publications)
        ids = [f"pressure-batch-{suffix}-{index}" for index in range(4)]
        prompts = [[17 + index, 900 + index] * 8 for index in range(4)]
        results = self.generate(ids, prompts, 192, biased=True)
        self.assertEqual(len(results), 4)
        retired, successful = set(), {}
        retractions = 0
        for rid, prompt, result in zip(ids, prompts, results, strict=True):
            self.assertEqual(result["meta_info"]["id"], rid)
            self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
            count = result["meta_info"]["num_retractions"]
            retractions += count
            if count:
                retired.add(rid)
            else:
                successful[rid] = prompt + result["output_ids"]
        self.assertTrue(retired, "workload did not trigger natural P/D retraction")
        self.assertTrue(successful, "no captured request survived pressure")
        after = self.wait_capture_idle(min_available=1)
        metric_after = self.retraction_metrics(tp_size, pp_size)
        for rank in ranks:
            self.assertEqual(metric_after[rank] - metric_before[rank], retractions)
        for rank, state in after.items():
            old = before[rank]["counters"]
            self.assertEqual(state["counters"]["admitted"] - old.get("admitted", 0), 4)
            failures = {
                name: count - old.get(name, 0)
                for name, count in state["counters"].items()
                if name.startswith("failed_") and count > old.get(name, 0)
            }
            allowed = {"failed_request_aborted_or_retracted"}
            if distributed:
                # A peer's retraction can invalidate this rank before local release.
                allowed.add("failed_peer_capture_failed")
            self.assertLessEqual(failures.keys(), allowed, rank)
            self.assertEqual(sum(failures.values()), len(retired), (rank, failures))
        self.catalog.wait_publications(first + len(successful), timeout=30)
        self.assertEqual(len(self.catalog.publications), first + len(successful))

        fresh_id = f"pressure-fresh-{suffix}"
        fresh = self.generate(fresh_id, prompts[0], 4, biased=True)
        self.assertEqual(fresh["meta_info"]["num_retractions"], 0)
        successful[fresh_id] = prompts[0] + fresh["output_ids"]
        publications = self.catalog.wait_publications(
            first + len(successful), timeout=30
        )[first:]
        final = self.wait_capture_idle()
        for rank, state in final.items():
            self.assertEqual(
                state["counters"]["admitted"]
                - before[rank]["counters"].get("admitted", 0),
                5,
                rank,
            )
        self.assertEqual(
            sum(s["counters"].get("ready", 0) for s in final.values())
            - sum(s["counters"].get("ready", 0) for s in before.values()),
            len(successful),
        )
        self.assertEqual(len(self.catalog.publications), first + len(successful))

        observations = [
            json.loads(line)
            for path in (sorted(draft.glob("observations*.jsonl")) if draft else [])
            for line in path.read_text().splitlines()
        ]
        rebuilt = [
            item
            for item in observations
            if item["kind"] == "context"
            and item["rid"] in retired
            and item["retraction_ct"] > 0
            and item["previous_end"] is None
        ]
        for tp, pp in ranks if draft else []:
            self.assertEqual(
                {
                    item["rid"]
                    for item in rebuilt
                    if item["tp_rank"] == tp and item["pp_rank"] == pp
                },
                retired,
            )
        for item in rebuilt:
            self.assertEqual(item["projected_end"], item["prefix_end"])
            self.assertGreaterEqual(item["projected_end"], 16)

        restored = [
            json.loads(line)
            for path in sorted((root / "decode").glob("pressure-restore-*.jsonl"))
            for line in path.read_text().splitlines()
        ]
        for tp, pp in ranks:
            local = [
                item
                for item in restored
                if item["tp_rank"] == tp and item["pp_rank"] == pp
            ]
            self.assertEqual({item["rid"] for item in local}, retired)
            self.assertEqual(len(local), retractions)
            self.assertTrue(all(item["restored_tokens"] >= 16 for item in local))

        failed_captures = {}
        with self.catalog.condition:
            for rid in retired:
                capture_ids = {
                    item["capture_id"]
                    for item in restored
                    if item["rid"] == rid and item["retraction_ct"] == 1
                }
                self.assertNotIn(None, capture_ids)
                self.assertEqual(len(capture_ids), 1)
                capture_id = capture_ids.pop()
                record = self.catalog.captures[capture_id]
                self.assertEqual(record["state"], "FAILED")
                self.assertEqual(
                    record["reason"],
                    "cohort_failed" if distributed else "request_aborted_or_retracted",
                )
                self.assertNotIn(capture_id, self.catalog.publications)
                failed_captures[rid] = capture_id

        rank_observations, first_admissions = {}, None
        for tp, pp in sorted(ranks):
            rank = f"pp{pp}-tp{tp}"
            admissions = [
                json.loads(line)
                for line in (root / "decode" / f"pressure-admissions-{rank}.jsonl")
                .read_text()
                .splitlines()
            ]
            by_request = {a["rid"]: a["capture_id"] for a in admissions}
            self.assertEqual(len(admissions), 5, rank)
            self.assertEqual(set(by_request), {*ids, fresh_id}, rank)
            self.assertEqual(len(set(by_request.values())), 5, rank)
            self.assertTrue(
                any(set(a["active_capture_rids"]) == set(ids) for a in admissions),
                (rank, "four simultaneous capture contexts required"),
            )
            if first_admissions is None:
                first_admissions = by_request
            self.assertEqual(by_request, first_admissions, rank)
            for rid, capture_id in failed_captures.items():
                self.assertEqual(by_request[rid], capture_id, (rank, rid))
            for rid in successful:
                self.assertIn(by_request[rid], self.catalog.publications, (rank, rid))
            events = [
                json.loads(line)
                for line in (root / "decode" / f"pressure-retractions-{rank}.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual({rid for e in events for rid in e["retracted"]}, retired)
            self.assertEqual(sum(len(e["retracted"]) for e in events), retractions)
            for event in events:
                self.assertFalse(event["debug_retract"])
                self.assertFalse(event["aborted"])
                self.assertLess(
                    event["available_before"], event["required_next_decode"]
                )
                self.assertGreater(event["available_after"], event["available_before"])
            rank_observations[rank] = {
                "admissions": by_request,
                "peak_active_captures": max(
                    len(a["active_capture_rids"]) for a in admissions
                ),
                "metric_retractions": metric_after[(tp, pp)] - metric_before[(tp, pp)],
                "events": events,
            }

        references = [
            torch.load(path, weights_only=True)
            for path in self.reference_paths(root)
            + (sorted(draft.rglob("*.pt")) if draft else [])
        ]
        traces = {
            hashlib.sha256(rid.encode()).hexdigest(): tokens
            for rid, tokens in successful.items()
        }
        references = [frame for frame in references if frame["trace_id"] in traces]
        decode_references = [
            frame
            for frame in references
            if frame.get("forward_mode") and frame.get("pd_role") != "prefill"
        ]
        for tp, pp in ranks:
            local = [
                f
                for f in decode_references
                if f["tp_rank"] == tp and f["pp_rank"] == pp
            ]
            self.assertTrue(any(f["batch_size"] == 4 for f in local))
            self.assertEqual(any(f["cuda_graph"] for f in local), replay)
            if enable_overlap and not draft:
                self.assertTrue(any(f["result_lag"] == 1 for f in local))
            self.assertEqual(any(f["predictions"] for f in local), pp == pp_size - 1)
            rank_observations[f"pp{pp}-tp{tp}"].update(
                source_frames=len(local),
                graph_frames=sum(f["cuda_graph"] for f in local),
                batch_sizes=sorted({f["batch_size"] for f in local}),
            )
        self.stop_process(prefill)
        self.stop_process(decode)
        reader = MooncakeSnapshotStore.connect(self.store_setup)
        self.addCleanup(reader.close)
        objects, tensor_bytes = 0, 0
        for publication in publications:
            manifest, tensors = read_snapshot(reader, publication)
            self.assertEqual(
                (manifest.topology.tp_size, manifest.topology.pp_size),
                (tp_size, pp_size),
            )
            self.assertEqual(
                tensors["token_ids"].tolist(), traces.pop(manifest.provenance.trace_id)
            )
            check_capture_snapshot(
                self,
                manifest,
                tensors,
                references,
                capture_mode="pd_speculative_accepted_target_path"
                if draft
                else "pd_autoregressive",
            )
            objects += len(manifest.objects)
            tensor_bytes += sum(t.nbytes for t in manifest.objects)
        self.assertFalse(traces)
        self.assertFalse(self.catalog.errors)
        print(
            json.dumps(
                {
                    "pd_memory_pressure": {
                        "tp_size": tp_size,
                        "pp_size": pp_size,
                        "draft_kind": self.draft_kind,
                        "replay": replay,
                        "enable_overlap": enable_overlap,
                        "kv_pool_tokens": 512,
                        "requested_path_tokens": 832,
                        "completed_requests": 4,
                        "output_tokens": 768,
                        "num_retractions": retractions,
                        "retired_capture_requests": sorted(retired),
                        "post_exit_snapshots": len(publications),
                        "producers_exited": prefill.poll() is not None
                        and decode.poll() is not None,
                        "tensor_objects": objects,
                        "tensor_bytes": tensor_bytes,
                        "rank_observations": rank_observations,
                        "rebuilt_contexts": rebuilt,
                        "kv_restores": restored,
                        "failed_captures": failed_captures,
                        "capture_before": before,
                        "capture_after": final,
                    }
                }
            ),
            flush=True,
        )
