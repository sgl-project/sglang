"""Verify real KV pool exhaustion, draft context rebuild and capture retirement."""

import hashlib
import json
import time

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.test.dspark_capture_observer import check_capture_snapshot


def exercise_dspark_capture_pressure(
    test, *, url, directory, cuda_graph, enable_overlap, tp_size=1, pp_size=1
):
    ranks = {(tp, pp) for tp in range(tp_size) for pp in range(pp_size)}
    distributed = tp_size * pp_size > 1

    def state():
        response = requests.get(url + "/server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def wait_available(count):
        deadline = time.monotonic() + 30
        while True:
            current = state()
            if current["states"].get("available", 0) >= count:
                return current
            test.assertLess(time.monotonic(), deadline, current)
            time.sleep(0.03)

    def retraction_metric():
        response = requests.get(url + "/metrics", timeout=10)
        response.raise_for_status()
        values = {rank: 0 for rank in ranks}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name == "sglang:num_retracted_requests_total":
                    rank = (
                        int(sample.labels.get("tp_rank", 0)),
                        int(sample.labels.get("pp_rank", 0)),
                    )
                    values[rank] += sample.value
        return values

    def completed_count(current):
        if distributed:
            return current["cohort_writer"]["counters"].get("stored", 0)
        return current["counters"].get("ready", 0)

    before = wait_available(4)
    metric_before = retraction_metric()
    first_publication = len(test.catalog.publications)
    prefix = f"pressure-{cuda_graph}-{enable_overlap}"
    request_ids = [f"{prefix}-{index}" for index in range(4)]
    prompts = [[17 + index, 900 + index] * 8 for index in range(4)]
    params = {
        "temperature": 0,
        "max_new_tokens": 192,
        "ignore_eos": True,
        "logit_bias": {"100": 100.0},
    }
    # Four disjoint 208-token paths exceed the 512-token pool. The server's
    # ordinary admission estimate is low, and its debug retract flag is off.
    response = requests.post(
        url + "/generate",
        json={"rid": request_ids, "input_ids": prompts, "sampling_params": params},
        timeout=120,
    )
    test.assertEqual(response.status_code, 200, response.text)
    results = response.json()
    test.assertEqual(len(results), 4)
    retired, successful = set(), {}
    num_retractions = 0
    for result, rid, prompt in zip(results, request_ids, prompts, strict=True):
        test.assertEqual(result["meta_info"]["id"], rid)
        test.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
        test.assertEqual(result["output_ids"], [100] * 192)
        count = result["meta_info"]["num_retractions"]
        num_retractions += count
        if count:
            retired.add(rid)
        else:
            successful[rid] = prompt + result["output_ids"]
    test.assertTrue(retired, "workload did not trigger automatic retraction")
    test.assertTrue(successful, "no captured request survived memory pressure")
    after = wait_available(4)
    metric_after = retraction_metric()
    for rank in ranks:
        test.assertEqual(metric_after[rank] - metric_before[rank], num_retractions)
    test.assertEqual(
        after["counters"]["admitted"] - before["counters"].get("admitted", 0), 4, after
    )
    failures = {
        name: count - before["counters"].get(name, 0)
        for name, count in after["counters"].items()
        if name.startswith("failed_") and count > before["counters"].get(name, 0)
    }
    allowed = {"failed_request_aborted_or_retracted"}
    if distributed:
        allowed.add("failed_peer_capture_failed")
    test.assertLessEqual(failures.keys(), allowed)
    test.assertEqual(sum(failures.values()), len(retired), failures)
    test.assertEqual(completed_count(after) - completed_count(before), len(successful))
    test.assertEqual(after["host_pool"]["quarantined"], 0)
    test.catalog.wait_publications(first_publication + len(successful), timeout=30)
    test.assertEqual(
        len(test.catalog.publications), first_publication + len(successful)
    )

    # Reusing capacity after the failed capture must admit a new complete sample.
    fresh_id = prefix + "-fresh"
    response = requests.post(
        url + "/generate",
        json={
            "rid": fresh_id,
            "input_ids": prompts[0],
            "sampling_params": params | {"max_new_tokens": 4},
        },
        timeout=60,
    )
    test.assertEqual(response.status_code, 200, response.text)
    fresh = response.json()
    test.assertEqual(fresh["output_ids"], [100] * 4)
    test.assertEqual(fresh["meta_info"]["num_retractions"], 0)
    successful[fresh_id] = prompts[0] + fresh["output_ids"]
    publications = test.catalog.wait_publications(
        first_publication + len(successful), timeout=30
    )[first_publication:]
    final = wait_available(4)
    test.assertEqual(final["host_pool"]["quarantined"], 0)
    test.assertEqual(final["counters"]["admitted"] - before["counters"]["admitted"], 5)
    test.assertEqual(completed_count(final) - completed_count(before), len(successful))

    observations = [
        json.loads(line)
        for path in sorted(directory.glob("observations*.jsonl"))
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
    test.assertEqual({item["rid"] for item in rebuilt}, retired)
    for tp, pp in ranks:
        test.assertEqual(
            {
                item["rid"]
                for item in rebuilt
                if item.get("tp_rank", 0) == tp and item.get("pp_rank", 0) == pp
            },
            retired,
        )
    for item in rebuilt:
        test.assertEqual(item["projected_end"], item["prefix_end"])
        test.assertGreaterEqual(item["projected_end"], 16)

    references = [
        torch.load(path, weights_only=True)
        for path in sorted((directory / "capture-reference").rglob("*.pt"))
    ]
    by_trace = {
        hashlib.sha256(rid.encode()).hexdigest(): tokens
        for rid, tokens in successful.items()
    }
    for publication in publications:
        manifest, tensors = test.read_sample(publication)
        test.assertEqual(
            tensors["token_ids"].tolist(), by_trace.pop(manifest.provenance.trace_id)
        )
        check_capture_snapshot(test, manifest, tensors, references)
    test.assertFalse(by_trace)
    test.assertFalse(test.catalog.errors)
    print(
        json.dumps(
            {
                "dspark_memory_pressure": {
                    "tp_size": tp_size,
                    "pp_size": pp_size,
                    "cuda_graph": cuda_graph,
                    "enable_overlap": enable_overlap,
                    "kv_pool_tokens": 512,
                    "requested_path_tokens": 4 * 208,
                    "completed_requests": len(results),
                    "output_tokens": 4 * 192,
                    "num_retractions": num_retractions,
                    "retired_capture_requests": sorted(retired),
                    "ready_samples": len(publications),
                    "rebuilt_contexts": rebuilt,
                    "capture_after": final,
                }
            }
        ),
        flush=True,
    )
    return publications
