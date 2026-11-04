#!/usr/bin/env python3
"""Verify the preserved final-runtime acceptance data without contacting services."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def archive(name):
    directory = ROOT / name
    for entry in read_json(directory / "archive.json"):
        path = directory / entry["file"]
        data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        assert len(data) == entry["original_bytes"], path
        assert hashlib.sha256(data).hexdigest() == entry["original_sha256"], path
    return directory


def trials(name, router=False):
    directory = archive(name)
    prefix = "router-" if router else ""
    kind = "router_trial" if router else "trial"
    raw = gzip.decompress((directory / f"{prefix}raw.jsonl.gz").read_bytes())
    rows = [row for line in raw.splitlines() if (row := json.loads(line))["kind"] == kind]
    summary = read_json(directory / f"{prefix}summary.json")
    schedule = read_json(directory / f"{prefix}schedule.json")
    assert len(rows) == len(schedule) == summary["attempts"], name
    assert len({row["id"] for row in rows}) == len(rows), name
    assert {row["id"] for row in rows} == {row["id"] for row in schedule}, name
    assert all(row["valid"] for row in rows), name
    for group in {row["group"] for row in rows}:
        members = [row for row in rows if row["group"] == group]
        if router:
            expected = summary["groups"][group]["all_completed_ttft_ms"]["mean"]
            assert abs(statistics.mean(row["resume"]["ttft_ms"] for row in members) - expected) < 1e-7
        for row in members:
            hits = row["resume_cache"]
            assert hits["device"] == 0
            assert hits["cpu"] + hits["storage"] >= row["prefix_tokens"]
    return rows, summary


def main():
    pressure, summary = trials("final-host-pressure")
    assert len(pressure) == 6
    prepared = [row for row in pressure if row["group"] == "prefetch"]
    assert len(prepared) == 3
    assert all(row["prefetch_ready_before_resume"] and row["resume_cache"]["cpu"] == 8192 for row in prepared)
    pair = next(iter(summary["paired"].values()))
    assert pair["paired_mean_ttft_bootstrap"]["ci"][1] < 0
    assert pair["paired_mean_cycle_bootstrap"]["ci"][1] < 0
    count = len(pressure)
    for name, expected in [("router-p6-b3", 7), ("final-router-legacy", 3), ("final-router-reorg", 2)]:
        rows, _ = trials(name, router=True)
        assert len(rows) == expected
        routed = [row for row in rows if row["group"] == "b3_router"]
        assert all(row["selected_worker_url"] == "http://127.0.0.1:51803" for row in routed)
        assert all(row["prefetch_ready_before_resume"] for row in routed)
        count += len(rows)
    short, _ = trials("final-short-window")
    assert len(short) == 18
    count += len(short)
    for name in ["router-p6-l3-protection", "final-protection"]:
        proof = read_json(archive(name) / "proof.json")
        assert all(view["claims"][0]["claim"]["phase"] == "active" for view in proof["task_views"])
        assert proof["released_first"]["claims"][0]["claim"]["phase"] == "released"
        assert proof["remaining_second"]["claims"][0]["claim"]["phase"] == "active"
        assert proof["released_second"]["claims"][0]["claim"]["phase"] == "released"
        assert proof["released_second"]["claims"][0]["claim"]["backend_quiescent"]
        if name == "final-protection":
            claim = proof["host_active"]["operations"][0]["claim"]
            assert claim["state"] == "active" and claim["effective_protection"] == "eviction_exclusion"
            assert proof["host_released"]["operations"][0]["claim"]["state"] in ("released", "expired")
    for filename in ["final-router-feedback.json", "final-router-reorg-feedback.json"]:
        feedback = read_json(ROOT / filename)["data"]
        assert not feedback["global_snapshot_complete"] and len(feedback["owners"]) == 2
        assert all(owner["mode"] == "feedback" and owner["error"] is None for owner in feedback["owners"])
    for filename, expected in [("control.txt", "154 passed"), ("gpu.txt", "2 passed"),
                               ("router-unit-serial.txt", "1114 passed"), ("router-proxy.txt", "241 passed")]:
        assert expected in (ROOT / "checks" / filename).read_text()
    print(f"Verified {count} additive trials, real owner feedback, shared protection and final test receipts.")


if __name__ == "__main__":
    main()
