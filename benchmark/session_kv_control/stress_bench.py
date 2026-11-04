#!/usr/bin/env python3
"""Sustained paired L2 retention and L3 working-set HTTP benchmarks."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import traceback
import uuid

from bench_report import distribution
from session_kv_bench import (
    BenchError, Control, HTTP, Journal, build_tokens, cache_details, digest,
    fingerprint, get_metadata, intervals_size, ready_tokens, require_completed,
    require_pin_released, require_pin_success, sleep_until, status,
)


def paired_schedule(seed, count, candidate):
    rng, schedule = random.Random(seed), []
    for block in range(count):
        arms = ["on_demand", candidate]
        rng.shuffle(arms)
        # The order inside a paired block is identical for both arms.
        for arm in arms:
            schedule.append({"block": block, "arm": arm, "id": f"b{block}-{arm}"})
    return schedule


def l2_salt(args, spec, kind, index):
    return "stress-"+kind+"-"+digest([args.run_id, spec["block"], spec["arm"], index])[:32]


def bounded_jobs(items, concurrency, work):
    """Return every planned outcome, including worker exceptions, in input order."""
    def invoke(item):
        start = time.monotonic()
        try:
            row = work(item)
        except Exception as exc:
            row = {"valid": False, "errors": [str(exc)], "traceback": traceback.format_exc()}
        return {"item": item, "started_monotonic": start,
                "completed_monotonic": time.monotonic(), **row}
    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(invoke, items))


def output_equal(left, right):
    """Compare text and IDs when available; differing availability is not proof."""
    fields = [k for k in ("output_ids", "text") if left.get(k) is not None or right.get(k) is not None]
    return bool(fields) and all(left.get(k) == right.get(k) for k in fields)


def parity(a, b, metadata, expected_mem):
    if a.cap["engine_id"] == b.cap["engine_id"]:
        raise BenchError("source and destination Engine IDs must differ")
    for field in ("namespace", "cache_salt_encoding", "weight_version"):
        if a.cap.get(field) != b.cap.get(field):
            raise BenchError(f"incompatible Engine {field}")
    for field in ("layout_id", "components"):
        if a.executor.get(field) != b.executor.get(field):
            raise BenchError(f"incompatible Engine {field}")
    if a.backend_namespace != b.backend_namespace:
        raise BenchError("backend namespace mismatch")
    if a.cap.get("policies") or b.cap.get("policies"):
        raise BenchError("disable implicit policies for explicit-control comparison")
    left, right = (fingerprint(metadata[k]) for k in ("a", "b"))
    differences = {k: [left[k], right[k]] for k in left if left[k] != right[k]}
    # Pool sizes are boot observations; automatic allocation can differ slightly
    # between GPU models. The explicit cap remains subject to exact parity.
    observed = differences.pop("max_total_num_tokens", None)
    if observed and (left.get("max_total_tokens") is not None
                     or any(type(x) is not int or x <= 0 for x in observed)
                     or abs(observed[0]-observed[1])/max(observed) > .01):
        differences["max_total_num_tokens"] = observed
    if differences:
        raise BenchError("Engine configuration mismatch: " + json.dumps(differences))
    if left.get("mem_fraction_static") != expected_mem:
        raise BenchError("serving mem_fraction_static differs from requested stress configuration")
    if left.get("max_total_tokens") is not None:
        raise BenchError("stress profile requires automatic KV capacity without max_total_tokens cap")
    if not left.get("enable_hierarchical_cache") or left.get("hicache_storage_backend") != "mooncake":
        raise BenchError("stress profile requires HiCache and Mooncake")
    return {"fingerprint_a": left, "fingerprint_b": right,
            "automatic_pool_difference": observed, "automatic_pool_tolerance_fraction": .01}


def host_tokens(control, target, trial):
    reply = control.residency(target, "host", trial)
    return {"tokens": intervals_size([x.get("descriptor", {}) for x in reply.get("items", [])]),
            "response": reply}


def demote(control, target, length, trial):
    reply = control.transfer("demote", target, {"tier": "gpu"}, {"tier": "host"}, trial)
    require_completed(reply, length, "GPU to host DEMOTE")
    return reply


def publish(control, target, length, trial):
    reply = control.transfer("publish", target, {"tier": "gpu"},
                             {"tier": "storage", "backend": "mooncake"}, trial)
    require_completed(reply, length, "storage publication")
    return reply


def flush(control, timeout, trial):
    reply = control.http.request("/flush_cache?timeout="+str(timeout), {}, trial=trial)
    if not isinstance(reply, str) or not reply.startswith("Cache flushed."):
        raise BenchError("flush did not certify local cache reset")
    control.refresh()  # No workers exist during refresh.


class Telemetry:
    """Bounded, non-overlapping requests; unavailable metrics stay explicit."""
    def __init__(self, http, journal, interval):
        self.http, self.journal, self.interval = http, journal, interval
        self.stop_event, self.label = threading.Event(), "startup"
        self.thread = threading.Thread(target=self.run, name="stress-telemetry", daemon=True)

    def sample(self):
        for path in ("/metrics", "/v1/loads"):
            try:
                value = self.http.request(path, optional=True, trial="telemetry-"+self.label)
                self.journal.emit("telemetry", phase=self.label, path=path, value=value)
            except Exception as exc:
                self.journal.emit("telemetry", phase=self.label, path=path, error=str(exc))

    def run(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop_event.set()
        self.thread.join()


def generation_row(control, tokens, salt, count, seed, trial, phase):
    row = {"valid": False, "errors": [], "phase": phase, "id": trial,
           "cache_salt": salt, "input_tokens": len(tokens), "input_sha256": digest(tokens)}
    try:
        row["generation"] = control.http.generate(tokens, salt, count, trial=trial, seed=seed, origin=phase)
        row["cache"] = cache_details(row["generation"]["meta_info"])
        row["valid"] = True
    except Exception as exc:
        row["errors"].append(str(exc))
        row["traceback"] = traceback.format_exc()
    return row


def materialize(row, control, target, length, trial, storage=False):
    if row["valid"]:
        try:
            if storage:
                row["publication"] = publish(control, target, length, trial)
            row["demote"] = demote(control, target, length, trial)
        except Exception as exc:
            row["valid"] = False
            row["errors"].append(str(exc))
    return row


def prefetch_resume(args, b, case, trial, candidate, controls):
    """A control future can drain late; it cannot move the generation deadline."""
    start = time.monotonic()
    deadline = start+args.tool_wait_s
    observed, lock = {}, threading.Lock()
    target = b.target(case["prefix"], case["salt"])

    def observe(op):
        with lock:
            observed.update(operation=copy.deepcopy(op), observed_monotonic=time.monotonic())

    def prepare():
        sleep_until(deadline-min(args.lead_s, args.tool_wait_s))
        return b.transfer("prefetch", target, {"tier": "storage", "backend": "mooncake"},
                          {"tier": "host"}, trial, observe)

    pending = controls.submit(prepare) if candidate else None
    row = {"id": trial, "phase": "resume", "valid": False, "errors": [],
           "tool_wait_s": args.tool_wait_s, "requested_lead_s": args.lead_s,
           "client_cycle_started_monotonic": start, "request_deadline_monotonic": deadline}
    try:
        sleep_until(deadline)
        with lock:
            row["prefetch_at_deadline"] = copy.deepcopy(observed)
        row.update(generation_row(b, case["prefix"]+case["suffix"], case["salt"],
                                  args.output_tokens, args.seed, trial, "resume"))
        if row.get("generation"):
            row["deadline_dispatch_delay_ms"] = (row["generation"]["started_monotonic"]-deadline)*1000
        if row.get("cache"):
            row["repeated_prefill_tokens"] = max(0, len(case["prefix"])-sum(row["cache"][k] for k in ("device", "cpu", "storage")))
    except Exception as exc:
        row["errors"].append(str(exc))
        row["traceback"] = traceback.format_exc()
    finally:
        if pending:
            try:
                # transfer has bounded timeout plus cancellation drain ownership.
                row["prefetch"] = pending.result()
                operation = row["prefetch"]["operation"]
                row["prefetch_status"] = status(operation)
                row["prepared_tokens"] = ready_tokens(operation)
                completed = row["prefetch"].get("timing", {}).get("completion_observed_monotonic")
                sent = row.get("generation", {}).get("started_monotonic")
                row["prefetch_ready_before_resume"] = bool(sent is not None and completed is not None
                    and completed <= sent and row["prepared_tokens"] >= len(case["prefix"])
                    and row["prefetch_status"] == "completed")
                row["prefetch_late"] = not row["prefetch_ready_before_resume"]
                # Late storage hits may share the same read. Keep uncertainty.
                row["unused_prepared_tokens_lower_bound"] = max(0, row["prepared_tokens"]-
                    sum(row.get("cache", {}).get(k, 0) for k in ("cpu", "storage")))
                row["prefetch_usage_ambiguous"] = bool(row["prefetch_late"] and row.get("cache", {}).get("storage"))
                require_completed(row["prefetch"], len(case["prefix"]), "prefetch")
            except Exception as exc:
                row["errors"].append("prefetch: "+str(exc))
        if row.get("generation"):
            try:
                row["demote"] = demote(b, target, len(case["prefix"]), trial)
            except Exception as exc:
                row["errors"].append("demote: "+str(exc))
        row["client_cycle_ms"] = (time.monotonic()-start)*1000
        row["valid"] = bool(row.get("generation")) and not row["errors"]
    return row


def l2_arm(args, b, cases, material, spec, journal):
    start, claims = time.monotonic(), []
    arm = {**spec, "rows": [], "errors": [], "cleanup": [], "valid": False,
           "hot_salts": [c["salt"] for c in cases]}
    candidate = spec["arm"] == "pin"
    try:
        flush(b, args.timeout_s, spec["id"])
        def prime(index):
            case = cases[index]
            trial = f"{spec['id']}-prime-{index}"
            row = generation_row(b, case["prefix"], case["salt"], 1, args.seed, trial, "prime")
            target = b.target(case["prefix"], case["salt"])
            return materialize(row, b, target, args.length, trial, storage=True)
        arm["rows"] += bounded_jobs(list(range(len(cases))), args.concurrency, prime)
        if any(not row["valid"] for row in arm["rows"]):
            raise BenchError("hot prefix priming failed")
        if candidate:
            for index, case in enumerate(cases):
                claim = f"{spec['id']}-claim-{index}"
                claims.append((case, claim))  # Release even an ambiguous acquire.
                op = b.pin(b.target(case["prefix"], case["salt"]), claim, "acquire", 1,
                           args.pin_ttl_s, spec["id"])
                require_pin_success(op, args.length, claim, b.context["engine_epoch"])
                journal.emit("pin_acquire", trial=spec["id"], operation=op)
        pressure_start = time.monotonic()
        def pressure(index):
            salt = l2_salt(args, spec, "pressure", index)
            trial = f"{spec['id']}-pressure-{index}"
            row = generation_row(b, material[0], salt, 1, args.seed, trial, "pressure")
            return materialize(row, b, b.target(material[0], salt), args.length, trial)
        arm["rows"] += bounded_jobs(list(range(args.pressure_prefixes)), args.concurrency, pressure)
        arm["pressure_wall_ms"] = (time.monotonic()-pressure_start)*1000
        arm["hot_residency_before_resume"] = [host_tokens(b, b.target(c["prefix"], c["salt"]), spec["id"]) for c in cases]
        if candidate:
            for case, claim in claims:
                actual = b.claim(claim, spec["id"])
                require_pin_success({"update": {"status": "completed", "quiescent": True},
                                     "public_result": {"claim": actual}}, args.length, claim, b.context["engine_epoch"])
        resume_start = time.monotonic()
        def resume(index):
            case = cases[index]
            row = generation_row(b, case["prefix"]+case["suffix"], case["salt"], args.output_tokens,
                                 args.seed, f"{spec['id']}-resume-{index}", "resume")
            row["case"] = index
            if row.get("cache"):
                row["repeated_prefill_tokens"] = max(0, args.length-sum(row["cache"][k] for k in ("device", "cpu", "storage")))
            return row
        arm["rows"] += bounded_jobs(list(range(len(cases))), args.concurrency, resume)
        arm["foreground_wall_ms"] = (time.monotonic()-resume_start)*1000
    except Exception as exc:
        arm["errors"].append(str(exc))
        arm["traceback"] = traceback.format_exc()
    finally:
        for case, claim in claims:
            receipt = {"claim_id": claim, "valid": False}
            try:
                receipt["release"] = b.pin(b.target(case["prefix"], case["salt"]), claim, "release", 2, 0, spec["id"])
                receipt["query"] = b.claim(claim, spec["id"])
                require_pin_released(receipt["query"], claim, b.context["engine_epoch"])
                receipt["valid"] = True
            except Exception as exc:
                receipt["error"] = str(exc)
                arm["errors"].append("PIN cleanup: "+str(exc))
            arm["cleanup"].append(receipt)
        arm["planned_requests"] = 2*len(cases)+args.pressure_prefixes
        arm["wall_ms"] = (time.monotonic()-start)*1000
        arm["valid"] = not arm["errors"] and len(arm["rows"]) == arm["planned_requests"] and all(r["valid"] for r in arm["rows"])
    return arm


def l3_arm(args, b, cases, spec, journal):
    start = time.monotonic()
    arm = {**spec, "rows": [], "errors": [], "planned_requests": len(cases)*args.passes, "valid": False}
    try:
        flush(b, args.timeout_s, spec["id"])
        jobs = [(p, index) for p in range(args.passes) for index in spec["dataset_order"]]
        # Fixed client lanes give each arm the exact same per-client order.
        lanes = [jobs[i::args.concurrency] for i in range(args.concurrency)]
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as controls:
            def client(lane):
                rows = []
                for passage, index in lanes[lane]:
                    trial = f"{spec['id']}-pass{passage}-case{index}"
                    row = prefetch_resume(args, b, cases[index], trial, spec["arm"] == "prefetch", controls)
                    row.update(case=index, passage=passage, client=lane)
                    journal.emit("request_result", trial=trial, block=spec["block"], arm=spec["arm"], **row)
                    rows.append(row)
                return {"rows": rows, "valid": True}
            clients = bounded_jobs(list(range(args.concurrency)), args.concurrency, client)
            for result in clients:
                arm["rows"] += result.get("rows", [])
                arm["errors"] += result.get("errors", [])
    except Exception as exc:
        arm["errors"].append(str(exc))
        arm["traceback"] = traceback.format_exc()
    arm["wall_ms"] = (time.monotonic()-start)*1000
    arm["valid"] = not arm["errors"] and len(arm["rows"]) == arm["planned_requests"] and all(r["valid"] for r in arm["rows"])
    return arm


def summarize(arms):
    summary = {"arms": [], "paired": [], "physical_io_bytes": None, "physical_host_peak_bytes": None}
    for arm in arms:
        rows = arm["rows"]
        completed = [r for r in rows if r.get("generation")]
        resumes = [r for r in completed if r.get("phase") == "resume"]
        wall = arm["wall_ms"]/1000
        counts = {"block": arm["block"], "arm": arm["arm"], "valid": arm["valid"],
                  "planned_requests": arm["planned_requests"], "recorded_requests": len(rows),
                  "completed_requests": len(completed),
                  "failed_requests": arm["planned_requests"]-len(completed),
                  "invalid_requests": sum(not r.get("valid") for r in rows),
                  "unattempted_requests": arm["planned_requests"]-len(rows),
                  "errors": arm["errors"], "wall_ms": arm["wall_ms"],
                  "achieved_completed_requests_per_s": len(completed)/wall if wall else None,
                  "achieved_output_tokens_per_s": sum(r["generation"]["meta_info"].get("completion_tokens", 0) for r in completed)/wall if wall else None,
                  "resume_ttft_ms": distribution([r["generation"]["ttft_ms"] for r in resumes]),
                  "resume_latency_ms": distribution([r["generation"]["latency_ms"] for r in resumes]),
                  "client_cycle_ms": distribution([r["client_cycle_ms"] for r in rows if "client_cycle_ms" in r]),
                  "pressure_ttft_ms": distribution([r["generation"]["ttft_ms"] for r in completed if r.get("phase") == "pressure"]),
                  "repeated_prefill_tokens": sum(r.get("repeated_prefill_tokens", 0) for r in resumes),
                  "resume_tier_tokens": {k: sum(r.get("cache", {}).get(k, 0) for r in resumes) for k in ("device", "cpu", "storage")},
                  "prefetch_ready_before_resume": sum(r.get("prefetch_ready_before_resume", False) for r in rows),
                  "prefetch_late": sum(r.get("prefetch_late", False) for r in rows),
                  "prefetch_statuses": {}, "unused_prepared_tokens_lower_bound": sum(r.get("unused_prepared_tokens_lower_bound", 0) for r in rows),
                  "prefetch_usage_ambiguous": sum(r.get("prefetch_usage_ambiguous", False) for r in rows),
                  "cleanup_failures": sum(not c["valid"] for c in arm.get("cleanup", []))}
        for row in rows:
            if "prefetch_status" in row:
                key = row["prefetch_status"]
                counts["prefetch_statuses"][key] = counts["prefetch_statuses"].get(key, 0)+1
        summary["arms"].append(counts)
    for block in sorted({a["block"] for a in arms}):
        pair = [a for a in summary["arms"] if a["block"] == block]
        if len(pair) != 2:
            continue
        base = next(a for a in pair if a["arm"] == "on_demand")
        candidate = next(a for a in pair if a["arm"] != "on_demand")
        delta = lambda key: (candidate[key]-base[key]) if candidate[key] is not None and base[key] is not None else None
        bttft, cttft = base["resume_ttft_ms"]["mean"], candidate["resume_ttft_ms"]["mean"]
        summary["paired"].append({"block": block, "candidate": candidate["arm"], "valid": base["valid"] and candidate["valid"],
                                  "mean_resume_ttft_delta_ms": cttft-bttft if bttft is not None and cttft is not None else None,
                                  "wall_delta_ms": delta("wall_ms"),
                                  "completed_requests_per_s_delta": delta("achieved_completed_requests_per_s")})
    summary["notes"] = ["All completed requests contribute; failed and unattempted jobs remain in denominators.",
        "Paired blocks, rather than correlated requests, are the experimental units; no confidence claim from three blocks.",
        "Wall throughput includes flush, tool waits, control drain, DEMOTE and cleanup; L2 additionally includes priming and pressure.",
        "L3 source publication is reported separately. Per-request DEMOTE applies to both arms. No per-request flush.",
        "Tier tokens are logical cache attribution, not physical I/O or peak memory. Late storage reuse may share reads."]
    return summary


def validate_outputs(arm, expected):
    for row in arm["rows"]:
        if row.get("phase") != "resume" or not row.get("generation"):
            continue
        key = row["case"]
        prior = expected.setdefault(key, row["generation"])
        row["output_equal"] = output_equal(prior, row["generation"])
        if not row["output_equal"]:
            row["valid"] = arm["valid"] = False
            row["errors"].append("greedy output differs across conditions")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    args.run_id = args.run_id or uuid.uuid4().hex
    journal = Journal(output/"raw.jsonl")
    arms, setup = [], []
    try:
        key = os.environ.get(args.api_key_env) if args.api_key_env else None
        a, b = [Control(HTTP(url, journal, args.timeout_s, key), "stress-"+args.run_id,
                        args.poll_ms, args.timeout_s) for url in (args.engine_a, args.engine_b)]
        if a.http.url == b.http.url:
            raise BenchError("distinct source and destination URLs required")
        metadata = {"a": get_metadata(a.http), "b": get_metadata(b.http)}
        comparison = parity(a, b, metadata, args.expected_mem_fraction)
        page = b.public.get("prefix", {}).get("page_size") or b.executor.get("page_size", b.executor.get("directions", {}).get("page_alignment", 1))
        if args.length % page or args.length > 65536:
            raise BenchError("prefix must align to serving page size and fit control target budget")
        for name in ("a", "b"):
            for field in ("model_path", "tokenizer_path"):
                declared = (metadata[name].get("model") or {}).get(field)
                if declared and Path(declared).resolve() != Path(args.model).resolve():
                    raise BenchError(f"local tokenizer differs from serving {field}")
        material = build_tokens(args.model, args.length, args.seed)
        context = comparison["fingerprint_b"].get("context_length")
        if context and args.length+len(material[1])+args.output_tokens > context:
            raise BenchError("prefix, suffix and output exceed context capacity")
        count = args.hot_sessions if args.command == "l2" else args.sessions
        working = (args.pressure_prefixes if args.command == "l2" else args.sessions)*args.length
        if working <= args.host_capacity_tokens and not args.allow_small_working_set:
            raise BenchError("working set must exceed measured host capacity; smoke requires --allow-small-working-set")
        if args.command == "l2" and args.hot_sessions*args.length >= args.host_capacity_tokens:
            raise BenchError("PIN hot set must fit host capacity with room for churn")
        protection_budget = None
        if args.command == "l2":
            serving = metadata["b"]["server"] or {}
            config = serving.get("server_args", serving).get("kv_control_config", {}) or {}
            if isinstance(config, str):
                config = json.loads(config)
            protection_budget = config.get("protection_limits", {}).get("max_protected_tokens", 65536)
            if args.hot_sessions*args.length > protection_budget:
                raise BenchError(f"hot set exceeds native L2 protection budget {protection_budget}; configure explicit Engine protection_limits")
        schedule = paired_schedule(args.seed, args.rounds if args.command == "l2" else args.blocks,
                                   "pin" if args.command == "l2" else "prefetch")
        for spec in schedule:
            order = list(range(count))
            random.Random(args.seed+spec["block"]).shuffle(order)
            spec["dataset_order"] = order
        cases = [{"prefix": material[0], "suffix": material[1], "salt": "stress-data-"+digest([args.run_id, i])[:32]} for i in range(count)]
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except Exception:
            revision = None
        manifest = {"args": vars(args), "metadata": metadata, "capabilities": {"a": a.cap, "b": b.cap},
                    "comparison": comparison, "source_revision": revision, "python": sys.version,
                    "functional_only": args.allow_small_working_set, "working_set_tokens": working,
                    "declared_measured_host_capacity_tokens": args.host_capacity_tokens,
                    "native_l2_protection_budget_tokens": protection_budget,
                    "module_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                    "input": {"prefix_sha256": digest(material[0]), "suffix_sha256": digest(material[1]),
                              "mode": "synthetic_ledger", "material": {"prefix": material[0], "suffix": material[1]}, "cases": cases},
                    "clock": "client monotonic; TTFT starts at actual generate send; periodic telemetry identical in both arms"}
        write_json(output/"manifest.json", manifest)
        write_json(output/"schedule.json", schedule)
        journal.emit("manifest", **manifest)
        # Shorter independent telemetry timeout prevents optional sampling from
        # extending cleanup by a full inference timeout.
        telemetry_http = HTTP(args.engine_b, journal, min(args.timeout_s, 5), key)
        with Telemetry(telemetry_http, journal, args.telemetry_interval_s) as telemetry:
            if args.command == "l3":
                telemetry.label = "source-publication"
                def source(index):
                    case = cases[index]
                    trial = "publication-"+str(index)
                    row = generation_row(a, case["prefix"], case["salt"], 1, args.seed, trial, "publication")
                    if row["valid"]:
                        try:
                            row["publication"] = publish(a, a.target(case["prefix"], case["salt"]), args.length, trial)
                        except Exception as exc:
                            row["valid"] = False
                            row["errors"].append(str(exc))
                    return row
                started = time.monotonic()
                setup = bounded_jobs(list(range(count)), args.concurrency, source)
                write_json(output/"source-publication.json", {"rows": setup, "wall_ms": (time.monotonic()-started)*1000})
                if any(not row["valid"] for row in setup):
                    raise BenchError("source publication failed; comparison cannot begin")
            outputs = {}
            for spec in schedule:
                telemetry.label = spec["id"]
                b.caller = "stress-"+args.run_id+"-"+spec["id"]
                # Refresh occurs inside arm after flush and before all workers.
                if args.command == "l2":
                    cases = [{**c, "salt": l2_salt(args, spec, "hot", i)} for i, c in enumerate(cases)]
                    arm = l2_arm(args, b, cases, material, spec, journal)
                else:
                    arm = l3_arm(args, b, cases, spec, journal)
                validate_outputs(arm, outputs)
                arms.append(arm)
                journal.emit("arm_result", **arm)
                write_json(output/(spec["id"]+".json"), arm)
                write_json(output/"summary.json", summarize(arms))
                # Phase-boundary metadata is kept outside measured arm time.
                journal.emit("phase_metadata", phase=spec["id"], server=b.http.request("/server_info", optional=True))
                print(json.dumps({"arm": spec["id"], "valid": arm["valid"], "wall_ms": arm["wall_ms"],
                                  "completed": sum(bool(r.get("generation")) for r in arm["rows"]), "errors": arm["errors"]}), flush=True)
        return 0 if arms and all(a["valid"] for a in arms) else 2
    except Exception as exc:
        journal.emit("run_failure", error=str(exc), traceback=traceback.format_exc())
        write_json(output/"failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        journal.file.close()
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        write_json(output/"checksums.json", checksums)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="command", required=True)
    for command in ("l2", "l3"):
        sub = subs.add_parser(command)
        for name in ("engine-a", "engine-b", "model", "output"):
            sub.add_argument("--"+name, required=True)
        sub.add_argument("--host-capacity-tokens", required=True, type=int)
        sub.add_argument("--allow-small-working-set", action="store_true")
        sub.add_argument("--expected-mem-fraction", type=float, default=.8)
        sub.add_argument("--run-id")
        sub.add_argument("--seed", type=int, default=1729)
        sub.add_argument("--length", type=int, default=16384)
        sub.add_argument("--concurrency", type=int, default=8)
        sub.add_argument("--output-tokens", type=int, default=32)
        sub.add_argument("--timeout-s", type=float, default=120)
        sub.add_argument("--poll-ms", type=float, default=50)
        sub.add_argument("--telemetry-interval-s", type=float, default=5)
        sub.add_argument("--api-key-env")
        if command == "l2":
            sub.add_argument("--rounds", type=int, default=3)
            sub.add_argument("--hot-sessions", type=int, default=4)
            sub.add_argument("--pressure-prefixes", type=int, default=53)
            sub.add_argument("--pin-ttl-s", type=float, default=900)
        else:
            sub.add_argument("--blocks", type=int, default=3)
            sub.add_argument("--sessions", type=int, default=64)
            sub.add_argument("--passes", type=int, default=1)
            sub.add_argument("--tool-wait-s", type=float, default=3)
            sub.add_argument("--lead-s", type=float, default=2.5)
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    for name in ("host_capacity_tokens", "length", "concurrency", "output_tokens", "rounds", "hot_sessions", "pressure_prefixes", "blocks", "sessions", "passes"):
        if hasattr(args, name) and getattr(args, name) < 1:
            cli.error(name+" must be positive")
    for name in ("timeout_s", "poll_ms", "telemetry_interval_s", "expected_mem_fraction", "pin_ttl_s"):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            cli.error(name+" must be finite and positive")
    for name in ("tool_wait_s", "lead_s"):
        if hasattr(args, name) and (not math.isfinite(getattr(args, name)) or getattr(args, name) < 0):
            cli.error(name+" must be finite and nonnegative")
    try:
        return run(args)
    except Exception as exc:
        print("STRESS BENCHMARK FAILED: "+str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
