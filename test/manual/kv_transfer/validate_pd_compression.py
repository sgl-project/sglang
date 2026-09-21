"""Prepare, run and audit bounded r2 correctness workloads (not a benchmark).

`prepare` needs the model tokenizer; `run`/`audit` use only the standard library.
Log commands are explicit, read-only collector commands, executed without a shell.
No raw KV is exported. Each run requires an unused output directory.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

LENGTHS = [1, 32, 1023, 1024, 1025, 4097, 8192]
CACHE_PHASES = {"native", "passthrough", "lz4", "force-l2"}
PHASES = ["off", "native", "passthrough", "lz4", "force", "force-l2"]
BASE = (
    "A field research team records the temperature, rainfall, and river level each morning. "
    "They compare observations across seasons and explain which measurements support their conclusions. "
    "Reliable records include the date, location, instruments, and units. "
    "Summarize the main lessons from these observations in clear English. "
)
L2_DOCUMENTS = (
    "TARGET ALPHA",
    "DOCUMENT BRAVO",
    "DOCUMENT CHARLIE",
    "DOCUMENT DELTA",
    "DOCUMENT ECHO",
    "DOCUMENT FOXTROT",
)


def host_cached_tokens(meta):
    """A missing provenance record is a host miss, never a successful restore."""
    return (meta.get("cached_tokens_details") or {}).get("host", 0)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    names = iter(
        "Hello Zebra Copper Delta Ocean River Garden Alpha Bravo Charlie Echo Foxtrot "
        "Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra "
        "Tango Uniform Victor Whiskey Xray Yankee Azure Bronze Crimson Emerald Indigo "
        "Jade Lavender Magenta Navy Olive Purple Ruby Silver Teal Violet White Yellow".split()
    )
    used = set()

    def document():
        for name in names:
            ids = tokenizer.encode(name + ". " + BASE * 500, add_special_tokens=False)
            if ids[0] not in used:
                used.add(ids[0])
                require(len(ids) >= 8192, "Insufficient real-text tokens")
                return ids
        raise ValueError("Tokenizer requires additional distinct document prefixes")

    shared = tokenizer.encode(BASE * 400, add_special_tokens=False)
    # Reproduce the original r1 one-token -> 32-token pair, not just another
    # pair of the same lengths. These hashes are from its saved request logs.
    require(
        sha(shared[:1])
        == "bb92d74f1bc9cdecef174885f4461ee391d54dcd2ad4a229c79c89543c2a6b92"
        and sha(shared[:32])
        == "341b4ee36d9271eb013acd8cc839d8bcc13de5c4bef7b2c8ce1e134ce1e1226d",
        "Tokenizer/text differs from the original r1 shared-prefix case",
    )
    used.add(shared[0])
    workload = {
        "version": 1,
        "tokenizer": args.tokenizer,
        "warmup": document()[:16],
        "lengths": [document()[:n] for n in LENGTHS],
        "shared": [shared[:1], shared[:32]],
        "concurrency": [document()[:4097] for _ in range(4)],
        # Preserve r1's documents, tokenizer options, lengths and ordering.
        # DOCUMENT prefixes intentionally share a short path, exercising splits.
        "l2": [
            [
                tokenizer.encode(
                    f"{name} cycle {cycle}. " + BASE * 400, add_special_tokens=False
                )[:8192]
                for name in L2_DOCUMENTS
            ]
            for cycle in (1, 2, 3)
        ],
    }
    with args.output.open("x") as f:
        json.dump(workload, f)


def records(text, marker):
    for line in text.splitlines():
        if marker + " " in line:
            yield line, json.loads(line.split(marker + " ", 1)[1])


def latest_state(text):
    found = list(records(text, "KV_COMPRESSION_STATS"))
    if found:
        line, s = found[-1]
        r, h = s["runtime"], s["l2"]
        idle = (
            s["pending_backups"] == 0
            and s["active_backup"] is None
            and s["active_restore"] is None
            and s["quarantined"] == 0
            and all(
                r[k] == 0
                for k in (
                    "queued_tasks",
                    "running_tasks",
                    "inflight_objects",
                    "quarantined",
                    "resident_bytes",
                )
            )
            and h["reserved_bytes"] == h["retired_bytes"] == 0
            and all(
                h.get(k, -1) == 0
                for k in (
                    "active_readers",
                    "active_writers",
                    "stage_waiters",
                    "stage_users",
                    "quarantined_objects",
                )
            )
        )
        return line, idle, s
    found = list(records(text, "KV_COMPRESSION_CACHE_STATE"))
    if found:
        line, s = found[-1]
        return line, s["writes"] == s["loads"] == 0, s
    return None, False, None


def collect(command, timeout=30):
    return subprocess.run(
        shlex.split(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    ).stdout


def drain(command, output, label, timeout=180):
    start = time.monotonic()
    previous = latest_state(collect(command))[0]
    stable = 0
    with (output / "drain.jsonl").open("a") as f:
        while time.monotonic() - start < timeout:
            time.sleep(min(5, timeout - (time.monotonic() - start)))
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                break
            line, idle, state = latest_state(
                collect(command, timeout=min(30, remaining))
            )
            if line is None or line == previous:
                continue
            previous = line
            stable = stable + 1 if idle else 0
            f.write(
                json.dumps(
                    {
                        "label": label,
                        "elapsed": time.monotonic() - start,
                        "idle": idle,
                        "consecutive": stable,
                        "state": state,
                    }
                )
                + "\n"
            )
            f.flush()
            if stable == 3:
                return
    raise TimeoutError(f"{label}: no three fresh idle observations within {timeout}s")


def request(url, ids, rid):
    body = {
        "input_ids": ids,
        "rid": rid,
        "return_logprob": True,
        "logprob_start_len": -1,
        "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def run(args):
    work = json.loads(args.workload.read_text())
    require(work["version"] == 1, "Unsupported workload")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "phase": args.phase,
        "scenario": args.scenario,
        "workload_sha256": sha(work),
        "started_ns": time.time_ns(),
        "router": args.router,
        "drain_timeout": 180,
    }
    chosen = (
        {args.scenario}
        if args.scenario != "all"
        else {"lengths", "shared", "concurrency", "l2"}
    )
    if args.phase not in CACHE_PHASES:
        require(args.scenario != "l2", "This phase has no L2")
        chosen.discard("l2")
    config["expected_cases"] = [["warmup", 0, "generation"]]
    if "lengths" in chosen:
        require(
            [len(ids) for ids in work["lengths"]] == LENGTHS,
            "Incomplete length workload",
        )
        config["expected_cases"] += [["lengths", 0, str(n)] for n in LENGTHS]
    if "shared" in chosen:
        require(
            [len(ids) for ids in work["shared"]] == [1, 32],
            "Incomplete shared-prefix workload",
        )
        require(work["shared"][0] == work["shared"][1][:1], "Shared prefix differs")
        config["expected_cases"] += [["shared", 0, c] for c in ("prime-1", "shared-32")]
    if "concurrency" in chosen:
        require(len(work["concurrency"]) == 4, "Four concurrent requests required")
        config["expected_cases"] += [["concurrency", 0, str(i)] for i in range(4)]
    if "l2" in chosen:
        require(
            len(work["l2"]) == 3
            and all(
                len(docs) == 6 and all(len(d) == 8192 for d in docs)
                for docs in work["l2"]
            ),
            "Three complete L2 workloads required",
        )
        config["expected_cases"] += [
            ["l2", cycle, case]
            for cycle in (1, 2, 3)
            for case in (
                "cold",
                "repeat",
                "pressure-0",
                "pressure-1",
                "pressure-2",
                "pressure-3",
                "pressure-4",
                "restore",
            )
        ]
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    # Save both effective server information and the pre-run logs for diagnosis.
    with urllib.request.urlopen(
        args.router.rstrip("/") + "/get_server_info", timeout=30
    ) as r:
        (args.output / "server-info.json").write_bytes(r.read())
    for role in ("prefill", "decode"):
        (args.output / (role + ".initial.log")).write_text(
            collect(getattr(args, role + "_log_command"))
        )

    def issue(group, case, ids, cycle=0):
        rid = f"r2-{config['started_ns']}-{group}-{cycle}-{case}"
        start = time.monotonic()
        response = request(args.router, ids, rid)
        row = {
            "group": group,
            "case": case,
            "cycle": cycle,
            "rid": rid,
            "input_tokens": len(ids),
            "input_sha256": sha(ids),
            "seconds": time.monotonic() - start,
            "result": response,
        }
        # Responses are persisted before validation, including failed requests.
        return row

    def save(row):
        with (args.output / "requests.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        validate_response(row)
        return row

    try:
        save(issue("warmup", "generation", work["warmup"]))
        if args.phase in CACHE_PHASES:
            drain(args.prefill_log_command, args.output, "warmup")
        if "lengths" in chosen:
            for ids in work["lengths"]:
                row = save(issue("lengths", str(len(ids)), ids))
                require(
                    row["result"]["meta_info"].get("cached_tokens", 0) == 0,
                    "Independent prefix unexpectedly hit",
                )
        if "shared" in chosen:
            for label, ids in zip(("prime-1", "shared-32"), work["shared"]):
                row = save(issue("shared", label, ids))
                if label == "shared-32" and args.phase in CACHE_PHASES:
                    require(
                        row["result"]["meta_info"].get("cached_tokens") == 1,
                        "Did not exercise one-token prefix hit",
                    )
        if "concurrency" in chosen:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [
                    executor.submit(issue, "concurrency", str(i), ids)
                    for i, ids in enumerate(work["concurrency"])
                ]
                for job in jobs:
                    save(job.result())
        if "l2" in chosen:
            for cycle, docs in enumerate(work["l2"], 1):
                save(issue("l2", "cold", docs[0], cycle))
                drain(args.prefill_log_command, args.output, f"cycle-{cycle}-target")
                save(issue("l2", "repeat", docs[0], cycle))
                for i, ids in enumerate(docs[1:]):
                    save(issue("l2", f"pressure-{i}", ids, cycle))
                    time.sleep(8)  # Preserve the r1 eight-request sequence.
                restored = save(issue("l2", "restore", docs[0], cycle))
                require(
                    host_cached_tokens(restored["result"]["meta_info"]) > 0,
                    "Host miss: cached_tokens_details is missing/null or reports host=0; original response retained",
                )
                drain(args.prefill_log_command, args.output, f"cycle-{cycle}-complete")
        if args.phase in CACHE_PHASES:
            drain(args.prefill_log_command, args.output, "final")
    finally:
        for role in ("prefill", "decode"):
            (args.output / (role + ".log")).write_text(
                collect(getattr(args, role + "_log_command"))
            )
    audit(args.output)


def result_rows(directory):
    return [
        json.loads(line)
        for line in (directory / "requests.jsonl").read_text().splitlines()
    ]


def validate_response(row):
    m = row["result"]["meta_info"]
    require(
        m.get("finish_reason", {}).get("type") == "length",
        "Unexpected request termination",
    )
    require(
        m.get("prompt_tokens") == row["input_tokens"]
        and m.get("completion_tokens") == 64,
        "Incomplete generation",
    )
    require(len(m.get("output_token_logprobs", [])) == 64, "Incomplete output tokens")


def complete_rows(directory, config):
    rows = result_rows(directory)
    keys = [(r["group"], r["cycle"], r["case"]) for r in rows]
    require(
        len(set(keys)) == len(keys)
        and set(keys) == {tuple(k) for k in config["expected_cases"]},
        "Missing/duplicate cases: incomplete runs cannot pass audit",
    )
    for row in rows:
        validate_response(row)
    return rows


def audit(directory):
    config = json.loads((directory / "config.json").read_text())
    rows = complete_rows(directory, config)
    phase = config["phase"]
    pre = (directory / "prefill.log").read_text()
    dec = (directory / "decode.log").read_text()
    if phase not in ("off", "native"):
        for role, log in (("prefill", pre), ("decode", dec)):
            settings = list(records(log, "PD_KV_COMPRESSION_CONFIG"))
            require(settings, f"Missing effective compression config: {role}")
            cfg = settings[-1][1]
            require(
                cfg["mode"] == ("passthrough" if phase == "passthrough" else "lz4")
                and cfg["force"] == phase.startswith("force")
                and cfg["verify"]
                and cfg["chunk_tokens"] == 1024
                and cfg["workspace_bytes"] == 512 * 1024**2
                and cfg["shared_l2"] == (role == "prefill" and phase in CACHE_PHASES),
                f"Effective config does not match phase {phase}: {role} {cfg}",
            )
    traces = {}
    for _, t in records(pre + "\n" + dec, "KV_COMPRESSION_HANDOFF"):
        traces.setdefault(t["rid"], {}).setdefault(t["stage"], []).append(t)
    rooms = {}
    for row in rows:
        stages = traces.get(row["rid"], {})
        require(
            set(stages) >= {"prefill_sampled", "metadata_written", "decode_received"},
            f"Missing handoff trace for {row['rid']}",
        )
        first = row["result"]["meta_info"]["output_token_logprobs"][0][1]
        require(
            all(t["token_id"] == first for ts in stages.values() for t in ts),
            f"Handoff token mismatch: {row['rid']} {stages}",
        )
        written = stages["metadata_written"][-1]
        received = stages["decode_received"][-1]
        require(
            written["room"] == received["room"] == received["received_room"],
            "Metadata room mismatch",
        )
        rooms[row["rid"]] = written["room"]

    sends = [
        s
        for _, s in records(pre, "PD_KV_COMPRESSION_SEND")
        if s["room"] in rooms.values()
    ]
    receives = [
        s
        for _, s in records(dec, "PD_KV_COMPRESSION_RECV")
        if s["room"] in rooms.values()
    ]
    if phase in ("off", "native"):
        require(
            not sends and not receives,
            "Uncompressed control used the compressed transport",
        )
    if phase not in ("off", "native"):
        require(sends and receives, "Missing P/D payload evidence")
        require(
            {s["room"] for s in sends} == set(rooms.values()),
            "Missing request payload evidence",
        )
        for row in rows:
            chunks = [s for s in sends if s["room"] == rooms[row["rid"]]]
            require(
                sum(s["objects"] for s in chunks) == row["input_tokens"]
                and {s["chunk"] for s in chunks}
                == set(range((row["input_tokens"] + 1023) // 1024)),
                f"Incomplete page/chunk coverage: {row['rid']}",
            )
        received_keys = {
            (r["room"], r["page_start"] // 1024, r["raw_bytes"]): r for r in receives
        }
        require(
            len(received_keys) == len(receives) == len(sends),
            "Duplicate or unmatched chunks",
        )
        for s in sends:
            r = received_keys.get((s["room"], s["chunk"], s["raw_bytes"]))
            require(
                r and r["verified"] and r["wire_bytes"] == s["wire_bytes"],
                "Missing verified writeback",
            )
            if phase.startswith("force"):
                require(
                    s["lz4_objects"] == s["objects"] and s["raw_objects"] == 0,
                    "Forced path fell back",
                )
                require(
                    r["lz4_objects"] == s["objects"],
                    "Decode did not decompress every object",
                )

    restores = {r["rid"]: r for _, r in records(pre, "KV_COMPRESSION_L2_RESTORE")}
    l2_rows = [r for r in rows if r["group"] == "l2"]
    for row in l2_rows:
        if row["case"] == "restore" and phase != "native":
            r = restores.get(row["rid"])
            require(
                r and r["native_missing_pages"] > 0, "Missing GPU eviction evidence"
            )
            require(
                r["verified_pages"] == r["adopted_pages"] > 0,
                "L2 writeback was not verified",
            )
            if phase == "force-l2":
                require(r["lz4_pages"] == r["adopted_pages"], "L2 restored raw objects")
    if l2_rows and phase != "native":
        for cycle in (1, 2, 3):
            cycle_rows = {r["case"]: r for r in l2_rows if r["cycle"] == cycle}
            require(len(cycle_rows) == 8, f"Incomplete L2 cycle {cycle}")
            cold_room = rooms[cycle_rows["cold"]["rid"]]
            cold_refs = {
                ref
                for s in sends
                if s["room"] == cold_room
                for ref in (s.get("refs") or [])
            }
            require(cold_refs, "TRACE_REUSE is required")
            for case in ("repeat", "restore"):
                room = rooms[cycle_rows[case]["rid"]]
                found = {
                    ref: source
                    for s in sends
                    if s["room"] == room
                    for ref, source in zip(
                        s.get("refs") or [], s.get("ref_sources") or []
                    )
                    if ref in cold_refs
                }
                require(
                    len(found) >= 8191 and all(v == "host" for v in found.values()),
                    f"{case}: retained prefix was re-encoded or identity changed",
                )
    if phase in CACHE_PHASES:
        observations = [
            json.loads(line)
            for line in (directory / "drain.jsonl").read_text().splitlines()
        ]
        completed = {
            s["label"]
            for s in observations
            if s["consecutive"] == 3 and s["elapsed"] <= 180
        }
        needed = {"warmup", "final"}
        if l2_rows:
            needed |= {
                f"cycle-{i}-{step}"
                for i in (1, 2, 3)
                for step in ("target", "complete")
            }
        require(needed <= completed, "Missing bounded natural-drain evidence")
        _, idle, state = latest_state(pre)
        require(idle, "Final background work has not drained")
        if phase == "native":
            require(state.get("native_control"), "Native control used compressed L2")
        if phase != "native":
            require(
                state["operations"]["backup_failures"]
                == state["operations"]["restore_failures"]
                == 0,
                "Background failure occurred",
            )
            if phase == "force-l2":
                require(
                    state["operations"]["published_lz4_pages"] > 0,
                    "No LZ4 L2 publication",
                )
    summary = {
        "phase": phase,
        "requests": len(rows),
        "handoff_verified": True,
        "verified_pd_chunks": len(receives),
        "l2_restores": sum(r["case"] == "restore" for r in l2_rows),
        "same_path_output_comparison": "pending: run compare",
        "performance_claim": False,
    }
    (directory / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def compare(baseline, tested):
    configs = [json.loads((p / "config.json").read_text()) for p in (baseline, tested)]
    require(
        configs[0]["workload_sha256"] == configs[1]["workload_sha256"],
        "Different workloads",
    )

    def indexed(p, config):
        rows = complete_rows(p, config)
        result = {(r["group"], r["cycle"], r["case"]): r for r in rows}
        require(len(result) == len(rows), "Duplicate cases")
        return result

    before, after = indexed(baseline, configs[0]), indexed(tested, configs[1])
    require(before.keys() == after.keys(), "Missing or different cases")
    differences = []
    for key, a in before.items():
        b = after[key]
        ma, mb = a["result"]["meta_info"], b["result"]["meta_info"]
        require(a["input_sha256"] == b["input_sha256"], "Input mismatch")

        def cache_path(m):
            detail = m.get("cached_tokens_details") or {}
            return (
                m.get("cached_tokens", 0),
                *(detail.get(k, 0) for k in ("device", "host", "storage")),
            )

        path_a, path_b = cache_path(ma), cache_path(mb)
        require(
            path_a == path_b,
            f"Different cache execution paths at {key}: {path_a} != {path_b}",
        )
        ai, bi = ([v[1] for v in m["output_token_logprobs"]] for m in (ma, mb))
        require(len(ai) == len(bi) == 64, "Incomplete token outputs")
        if ai != bi:
            differences.append(
                {
                    "case": key,
                    "first_difference": next(
                        i for i, (x, y) in enumerate(zip(ai, bi)) if x != y
                    ),
                    "baseline_tokens": ai,
                    "tested_tokens": bi,
                }
            )
    report = {
        "baseline": str(baseline),
        "cases": len(after),
        "differences": differences,
        "passed": not differences,
    }
    (tested / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    require(
        not differences,
        "Same-path token mismatch; see comparison.json and handoff traces",
    )
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("run")
    p.add_argument("--workload", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--router", required=True)
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument(
        "--scenario",
        choices=["all", "lengths", "shared", "concurrency", "l2"],
        default="all",
    )
    p.add_argument("--prefill-log-command", required=True)
    p.add_argument("--decode-log-command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("tested", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "audit":
        audit(args.directory)
    else:
        compare(args.baseline, args.tested)


if __name__ == "__main__":
    main()
