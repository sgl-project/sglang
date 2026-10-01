"""Real cache eviction and retract/resume through the serving HTTP interface."""

import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot


def exercise_capture_lifecycle(test, *, model_path, directory, cuda_graph):
    root = Path(directory) / f"lifecycle-{cuda_graph}"
    root.mkdir()
    config = json.loads(Path(test.capture_path).read_text())
    config.update(journal_directory=str(root / "journal"))
    path = root / "capture.json"
    path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def observed_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.training_capture_overlap_server"]
            + command[2:],
            *args,
        )

    def state():
        response = requests.get(url + "/server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def wait_available():
        deadline = time.monotonic() + 20
        while True:
            current = state()
            if (
                current["states"].get("available", 0) == current["reservations"]
                and current["reservations"] > 0
            ):
                return current
            test.assertLess(time.monotonic(), deadline, current)
            time.sleep(0.03)

    def metrics():
        response = requests.get(url + "/metrics", timeout=10)
        response.raise_for_status()
        result = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                result[sample.name] = result.get(sample.name, 0.0) + sample.value
        return result

    first_publication = len(test.catalog.publications)
    results = []

    def generate(prompt, name):
        wait_available()
        response = requests.post(
            url + "/generate",
            json={
                "rid": f"lifecycle-{cuda_graph}-{name}",
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        test.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        results.append(value)
        test.catalog.wait_publications(first_publication + len(results), timeout=20)
        return value

    process = None
    publications = []
    try:
        with patch.object(test_utils, "_launch_server_process", observed_server):
            process = test_utils.popen_launch_server(
                model_path,
                url,
                timeout=240,
                other_args=[
                    "--training-capture-config",
                    str(path),
                    "--skip-server-warmup",
                    "--enable-metrics",
                    "--enable-cache-report",
                    "--attention-backend",
                    "triton",
                    "--mem-fraction-static",
                    "0.25",
                    "--max-total-tokens",
                    "256",
                    "--max-running-requests",
                    "4",
                    "--chunked-prefill-size",
                    "128",
                    "--cuda-graph-backend-decode",
                    "full" if cuda_graph else "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    *([] if cuda_graph else ["--disable-overlap-schedule"]),
                ],
            )
        prompt = [100, 200, 300, 400] * 40
        original = generate(prompt, "initial")
        hit = generate(prompt, "hit")
        test.assertEqual(original["meta_info"]["cached_tokens"], 0)
        test.assertGreater(hit["meta_info"]["cached_tokens"], 0)
        test.assertEqual(hit["output_ids"], original["output_ids"])

        before_eviction = metrics().get("sglang:evicted_tokens_total", 0.0)
        replacement = generate([500, 600, 700, 800] * 40, "replacement")
        after_eviction = metrics()["sglang:evicted_tokens_total"]
        test.assertGreater(after_eviction, before_eviction)
        miss = generate(prompt, "after-eviction")
        test.assertEqual(miss["meta_info"]["cached_tokens"], 0)
        test.assertEqual(miss["output_ids"], original["output_ids"])

        before_retract = wait_available()
        rid = f"lifecycle-{cuda_graph}-retracted"
        with test.catalog.condition:
            failed_before = {
                capture_id
                for capture_id, record in test.catalog.captures.items()
                if record["state"] == "FAILED"
            }
        paused = False
        last = None
        try:
            with requests.post(
                url + "/generate",
                json={
                    "rid": rid,
                    "input_ids": [23, 41, 99, 100],
                    "stream": True,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 128,
                        "ignore_eos": True,
                        "logit_bias": {"100": 100.0},
                    },
                },
                stream=True,
                timeout=60,
            ) as response:
                test.assertEqual(response.status_code, 200)
                retracted = False
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    last = json.loads(line[6:])
                    if retracted:
                        continue
                    test.assertIsNone(last["meta_info"]["finish_reason"])
                    paused = True
                    status = requests.post(
                        url + "/pause_generation", json={"mode": "retract"}, timeout=20
                    )
                    test.assertEqual(status.status_code, 200, status.text)
                    # Catalog failure is the completion fence: the request has
                    # released its capture lease while generation is paused.
                    with test.catalog.condition:
                        test.assertTrue(
                            test.catalog.condition.wait_for(
                                lambda: any(
                                    capture_id not in failed_before
                                    and record["state"] == "FAILED"
                                    and record.get("reason")
                                    == "request_aborted_or_retracted"
                                    for capture_id, record in test.catalog.captures.items()
                                ),
                                timeout=20,
                            ),
                            "retracted request did not fail its capture",
                        )
                        test.assertEqual(
                            len(test.catalog.publications),
                            first_publication + len(results),
                        )
                    status = requests.post(
                        url + "/continue_generation", json={}, timeout=20
                    )
                    test.assertEqual(status.status_code, 200, status.text)
                    paused = False
                    retracted = True
                test.assertTrue(retracted, "stream ended before retract")
        finally:
            if paused:
                requests.post(url + "/continue_generation", json={}, timeout=20)
        test.assertIsNotNone(last)
        test.assertGreaterEqual(last["meta_info"]["num_retractions"], 1)
        test.assertEqual(last["meta_info"]["finish_reason"]["type"], "length")
        test.assertEqual(last["output_ids"], [100] * 128)
        after_retract = wait_available()
        test.assertEqual(
            after_retract["counters"]["admitted"],
            before_retract["counters"]["admitted"] + 1,
        )
        test.assertEqual(
            after_retract["counters"]["ready"], before_retract["counters"]["ready"]
        )
        test.assertEqual(after_retract["host_pool"]["quarantined"], 0)

        recovered = generate(prompt, "after-retract")
        test.assertEqual(recovered["output_ids"], original["output_ids"])
        final = wait_available()
        test.assertEqual(final["counters"]["ready"], len(results))
        test.assertEqual(final["host_pool"]["quarantined"], 0)
        test.assertEqual(final["enable_overlap"], cuda_graph)
        test.assertEqual(
            final["counters"].get("cuda_graph_forwards", 0) > 0, cuda_graph
        )
        actual_metrics = metrics()
        for metric, key in (
            ("kv_staging_allocated_bytes", "device_allocated_bytes"),
            ("kv_staging_limit_bytes", "device_limit_bytes"),
        ):
            test.assertEqual(
                actual_metrics[f"sglang:training_capture_{metric}"],
                final["host_pool"][key],
            )

        publications = test.catalog.wait_publications(first_publication + len(results))[
            first_publication:
        ]
        references = [
            torch.load(file, weights_only=True)
            for file in sorted((root / "capture-reference").glob("*.pt"))
        ]
        slots = {}
        for item in references:
            slots.setdefault(item["trace_id"], set()).update(item["kv_slots"].tolist())
        initial_slots = slots[
            hashlib.sha256(original["meta_info"]["id"].encode()).hexdigest()
        ]
        replacement_slots = slots[
            hashlib.sha256(replacement["meta_info"]["id"].encode()).hexdigest()
        ]
        reused_slots = initial_slots & replacement_slots
        test.assertTrue(reused_slots, "eviction did not reuse the original KV slots")
        expected = {
            hashlib.sha256(result["meta_info"]["id"].encode()).hexdigest(): result[
                "output_ids"
            ]
            for result in results
        }
        for publication in publications:
            manifest, tensors = test.read_sample(publication)
            trace_id = manifest.provenance.trace_id
            test.assertNotEqual(trace_id, hashlib.sha256(rid.encode()).hexdigest())
            test.assertEqual(
                tensors["token_ids"][manifest.sequence.prompt_length :].tolist(),
                expected.pop(trace_id),
            )
            check_capture_snapshot(
                test, manifest, tensors, references, capture_mode="autoregressive"
            )
        test.assertFalse(expected)
        test.assertFalse(test.catalog.errors)
        print(
            json.dumps(
                {
                    "lifecycle_capture": final,
                    "cuda_graph_enabled": cuda_graph,
                    "samples": len(publications),
                    "evicted_tokens": after_eviction - before_eviction,
                    "reused_kv_slots": len(reused_slots),
                    "cached_tokens": [
                        item["meta_info"]["cached_tokens"] for item in results
                    ],
                    "resumed_output_tokens": len(last["output_ids"]),
                    "num_retractions": last["meta_info"]["num_retractions"],
                    "staging_http_metrics_verified": True,
                }
            ),
            flush=True,
        )
    finally:
        if process is not None:
            kill_process_tree(process.pid)
            process.wait(timeout=20)
    for publication in publications:
        test.read_sample(publication)
