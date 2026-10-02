"""Natural P/D KV exhaustion, CPU restore and captured-sample retirement."""

import hashlib
import json
import os
import time
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_utils import read_snapshot


class PDCapturePressureBase(PDCaptureRuntimeBase):
    def wait_capture_idle(self, *, min_available=0):
        deadline = time.monotonic() + 30
        while True:
            state = self.capture_state()
            if (
                state["states"].get("available", 0) == state["reservations"]
                and state["states"].get("available", 0) >= min_available
                and state.get("queued", 0) == 0
                and state.get("cohort_writer", {}).get("pending", 0) == 0
            ):
                self.assertEqual(state["host_pool"]["quarantined"], 0)
                self.assertIsNone(state["disabled_reason"])
                return state
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.03)

    def retraction_metrics(self, pp_size):
        response = requests.get(self.decode_url + "/metrics", timeout=10)
        response.raise_for_status()
        values = {rank: 0 for rank in range(pp_size)}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name == "sglang:num_retracted_requests_total":
                    values[int(sample.labels.get("pp_rank", 0))] += sample.value
        return values

    @patch.dict(os.environ, {"SGLANG_TEST_RETRACT": "0"})
    def exercise_pressure(self, *, replay, enable_overlap=False, pp_size=1):
        draft = self.get_draft("target_kv")
        suffix = f"{pp_size}-{replay}-{enable_overlap}"
        root = self.root / f"pressure-{suffix}"
        root.mkdir()
        self.bootstrap_port, self.bootstrap_room = self.new_bootstrap_port(), 8000
        extra = [
            "--max-total-tokens",
            "512",
            "--num-reserved-decode-tokens",
            "16",
            "--enable-metrics",
        ]
        prefill, self.prefill_url = self.launch(
            "prefill",
            root,
            replay=False,
            tp_size=1,
            pp_size=pp_size,
            draft=draft,
            extra_args=extra,
        )
        decode, self.decode_url = self.launch(
            "decode",
            root,
            replay=replay,
            tp_size=1,
            pp_size=pp_size,
            draft=draft,
            enable_overlap=enable_overlap,
            extra_args=extra,
        )
        # Single-rank leases are prepared asynchronously; cohort admission is lazy.
        before = self.wait_capture_idle(min_available=4 if pp_size == 1 else 0)
        metric_before = self.retraction_metrics(pp_size)
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
        metric_after = self.retraction_metrics(pp_size)
        for rank in range(pp_size):
            self.assertEqual(metric_after[rank] - metric_before[rank], retractions)
        self.assertEqual(
            after["counters"]["admitted"] - before["counters"].get("admitted", 0),
            4,
            after,
        )
        failures = {
            name: count - before["counters"].get(name, 0)
            for name, count in after["counters"].items()
            if name.startswith("failed_")
        }
        allowed = {"failed_request_aborted_or_retracted"}
        if pp_size > 1:
            # A peer's retraction can invalidate this rank before local release.
            allowed.add("failed_peer_capture_failed")
        self.assertLessEqual(failures.keys(), allowed)
        self.assertEqual(sum(failures.values()), len(retired), failures)
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
        self.assertEqual(
            final["counters"]["admitted"] - before["counters"].get("admitted", 0), 5
        )
        self.assertEqual(len(self.catalog.publications), first + len(successful))

        observations = [
            json.loads(line)
            for path in sorted(draft.glob("observations*.jsonl"))
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
        for rank in range(pp_size):
            self.assertEqual(
                {item["rid"] for item in rebuilt if item.get("pp_rank", 0) == rank},
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
        for rank in range(pp_size):
            local = [item for item in restored if item["pp_rank"] == rank]
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
                    "cohort_failed" if pp_size > 1 else "request_aborted_or_retracted",
                )
                self.assertNotIn(capture_id, self.catalog.publications)
                failed_captures[rid] = capture_id

        references = [
            torch.load(path, weights_only=True)
            for path in self.reference_paths(root) + sorted(draft.rglob("*.pt"))
        ]
        traces = {
            hashlib.sha256(rid.encode()).hexdigest(): tokens
            for rid, tokens in successful.items()
        }
        references = [frame for frame in references if frame["trace_id"] in traces]
        self.assertTrue(any(frame["batch_size"] == 4 for frame in references))
        self.assertEqual(any(frame["cuda_graph"] for frame in references), replay)
        for publication in publications:
            manifest, tensors = read_snapshot(self.reader, publication)
            self.assertEqual(
                tensors["token_ids"].tolist(), traces.pop(manifest.provenance.trace_id)
            )
            check_capture_snapshot(
                self,
                manifest,
                tensors,
                references,
                capture_mode="pd_speculative_accepted_target_path",
            )
        self.assertFalse(traces)
        self.assertFalse(self.catalog.errors)
        self.stop_process(prefill)
        self.stop_process(decode)
        for publication in publications:
            read_snapshot(self.reader, publication)
        print(
            json.dumps(
                {
                    "pd_memory_pressure": {
                        "pp_size": pp_size,
                        "replay": replay,
                        "enable_overlap": enable_overlap,
                        "kv_pool_tokens": 512,
                        "requested_path_tokens": 832,
                        "completed_requests": 4,
                        "output_tokens": 768,
                        "num_retractions": retractions,
                        "retired_capture_requests": sorted(retired),
                        "post_exit_snapshots": len(publications),
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
