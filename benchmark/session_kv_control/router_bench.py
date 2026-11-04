#!/usr/bin/env python3
"""B2/B3 paired trials: same Router prepares B; direct B versus free Router pick."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
import traceback
import urllib.error
import uuid

from bench_inputs import load_trace, tokenizer_identity
from bench_report import distribution
from bench_statistics import bootstrap_mean_ci, stable_seed
from session_kv_bench import (
    BenchError, Control, HTTP, Journal, build_tokens, cache_details, csv_numbers,
    digest, fingerprint, get_metadata, intervals_size, require_completed, sleep_until, unwrap,
)


class Router:
    def __init__(self, control_http, formal_http, poll_ms=50):
        self.http, self.formal, self.poll_ms = control_http, formal_http, poll_ms
        self.capabilities = unwrap(self.http.request("/v1/kv/capabilities"))
        if not self.capabilities.get("enabled") or self.capabilities.get("lifecycle") != "explicit":
            raise BenchError("Router explicit lifecycle capability is required")

    def command(self, body, trial):
        # Retry only a lost transport response, with exactly the same identity.
        for attempt in range(2):
            try:
                return unwrap(self.http.request("/v1/kv/control", body, trial=trial))
            except BenchError as exc:
                cause = exc.__cause__
                if attempt or not isinstance(cause, (OSError, urllib.error.URLError)) or isinstance(cause, urllib.error.HTTPError):
                    raise
                self.http.journal.emit("router_transport_retry", trial=trial, command_kind=body.get("kind"))

    def event(self, task, event, revision, event_id, trial, **fields):
        return self.command({"kind": "session_event", "task": task, "event": event,
            "revision": revision, "event_id": event_id, **fields}, trial)

    def query(self, task, message_id, trial):
        return self.command({"kind": "query", "task": task, "message_id": message_id}, trial)

    def topology(self, engines, trial):
        metrics = self.formal.request("/metrics", trial=trial)
        if not isinstance(metrics, str):
            raise BenchError("Router metrics must expose worker health attribution")
        matches = re.findall(r'^sgl_router_worker_health\{worker_url="([^"\\]+)"\}\s+([01])(?:\.0)?\s*$', metrics, re.M)
        healthy = {url.rstrip("/") for url, state in matches if state == "1"}
        if healthy != set(engines):
            raise BenchError("Router must expose exactly the two expected healthy worker URLs: " + json.dumps(sorted(healthy)))
        self.http.journal.emit("router_topology", trial=trial, healthy_workers=sorted(healthy),
                               evidence="Router circuit-breaker health gauge plus Engine capability RPCs")
        return sorted(healthy)


def inspect_preparation(view, b, length):
    """Validate a frozen logical scope and attributed current Engine observation."""
    if view.get("state") == "on_demand_fallback":
        return False
    if view.get("state") == "unknown":
        raise BenchError("Router preparation intent became unknown")
    if view.get("missing_scope") or view.get("materialization_required") or view.get("admission_gaps"):
        raise BenchError("Router cannot materialize/prepare the frozen full input scope")
    known = view.get("known_scope", [])
    if len(known) != 1 or known[0].get("branch") != "main" or known[0].get("end_token") != length:
        raise BenchError("Router froze a different prefix or branch")
    operations = view.get("operations", [])
    if len(operations) != 1:
        raise BenchError("Router must prepare one exact B-owner operation")
    op = operations[0]
    if (op.get("worker_url", "").rstrip("/") != b.http.url or op.get("engine_id") != b.cap["engine_id"]
            or op.get("owner_epoch") != b.cap["engine_epoch"] or op.get("action") != "sglang.prefetch"):
        raise BenchError("Router preparation has wrong worker/epoch/action attribution")
    if op.get("error"):
        raise BenchError("Router preparation observation failed: " + str(op["error"]))
    result = op.get("result") or {}
    status = result.get("update", {}).get("status")
    if status in ("failed", "rejected", "unsupported", "missing", "deferred", "cancelled", "partial"):
        raise BenchError("Router child did not complete full preparation: " + str(status))
    scope = op.get("ready_scope")
    if scope is None:
        return False
    if (scope.get("tier") != "host" or scope.get("provenance") != "engine_scoped_query"
            or scope.get("start_token") != 0 or type(scope.get("end_token")) is not int
            or not 0 <= scope["end_token"] <= length):
        raise BenchError("Router ready scope is not a compatible current host query")
    return scope["end_token"] == length


class PreparationObserver:
    """Independent lifecycle/RPC thread; the formal-request timer never joins it."""
    def __init__(self, router, a, b, task, length, deadline, trial):
        self.router, self.a, self.b, self.task = router, a, b, task
        self.length, self.deadline, self.trial = length, deadline, trial
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.view, self.observed_at, self.decision = None, None, None
        self.error, self.intent_admitted, self.submission_attempted = None, False, False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def run(self):
        try:
            remaining = lambda: max(0, round((self.deadline-time.monotonic())*1000))
            self.router.event(self.task, "tool_wait", 2, "tool-wait", self.trial,
                target={"kind": "branch", "branch_id": "main", "revision": 1},
                worker_url=self.a.http.url, pressure=False, expected_wait_ms=remaining())
            if self.stop_event.is_set():
                return
            self.submission_attempted = True
            view = self.router.event(self.task, "resume_expected", 3, "prepare-b", self.trial,
                target={"kind": "branch", "branch_id": "main", "revision": 1},
                worker_url=self.b.http.url, expected_wait_ms=remaining())
            self.decision = copy.deepcopy(view)
            if view.get("state") != "on_demand_fallback":
                self.intent_admitted = True
            while True:
                inspect_preparation(view, self.b, self.length)
                with self.lock:
                    self.view, self.observed_at = copy.deepcopy(view), time.monotonic()
                if self.stop_event.wait(self.router.poll_ms/1000) or view.get("state") == "on_demand_fallback":
                    return
                view = self.router.query(self.task, "prepare-b", self.trial)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.router.http.journal.emit("router_observer_error", trial=self.trial, error=self.error)

    def snapshot(self):
        with self.lock:
            return {"view": copy.deepcopy(self.view), "observed_at": self.observed_at,
                    "error": self.error}

    def stop(self, timeout):
        self.stop_event.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise BenchError("preparation observer RPC still in flight; cleanup ownership unknown")


def flush_local(control, trial, timeout):
    result = control.http.request("/flush_cache?timeout="+str(timeout), {}, trial=trial)
    if not isinstance(result, str) or not result.startswith("Cache flushed."):
        raise BenchError("local flush did not certify success")
    control.refresh()


def assert_cold(control, tokens, salt, trial):
    target = control.target(tokens, salt)
    observed = {}
    for tier in ("gpu", "host"):
        observed[tier] = control.residency(target, tier, trial)
        if observed[tier].get("items"):
            raise BenchError(f"{control.http.url} target is already resident on {tier}")
    return observed


def observed_prepared_tokens(view, length):
    """Observed lower bound, including historical transfer effects after cancel."""
    maximum = 0
    for op in (view or {}).get("operations", []):
        scope = op.get("ready_scope") or {}
        if scope.get("tier") == "host" and scope.get("start_token") == 0:
            maximum = max(maximum, scope.get("end_token", 0))
        result = op.get("result") or {}
        public = result.get("public_result") or {}
        spans = public.get("ready_scope")
        if spans is None:
            spans = [x for x in result.get("update", {}).get("target_outcomes", [])
                     if "destination_ready" in x.get("completed_steps", [])]
        maximum = max(maximum, intervals_size(spans))
    if maximum > length:
        raise BenchError("preparation evidence extends beyond frozen prefix")
    return maximum


def cleanup(router, task, observer, trial, timeout):
    result = {"state": "unknown", "quiescent": False}
    try:
        observer_stopped = True
        if observer:
            try:
                observer.stop(timeout+1)
            except BenchError as exc:
                observer_stopped = False
                result["observer_stop_error"] = str(exc)
        result["finish_receipt"] = router.event(task, "task_finished", 5, "finish", trial, cascade=False)
        if observer and observer.submission_attempted and observer.decision is None:
            # A lost response may still own a job. Query its stable message ID;
            # unknown remains explicit instead of guessing there was no job.
            observer.intent_admitted = True
        if observer and observer.intent_admitted:
            result["cancel_receipt"] = router.command({"kind": "cancel", "task": task, "message_id": "prepare-b"}, trial)
            deadline = time.monotonic()+timeout
            while True:
                view = router.query(task, "prepare-b", trial)
                result["last_view"] = view
                if view.get("quiescent") is True and all(op.get("quiescent") is True for op in view.get("operations", [])):
                    break
                if time.monotonic() >= deadline:
                    raise BenchError("Router cancellation has not drained within cleanup deadline")
                time.sleep(router.poll_ms/1000)
            result["ack"] = router.command({"kind": "ack", "task": task, "message_id": "prepare-b"}, trial)
            if result["ack"].get("acknowledged") is not True:
                raise BenchError("Router did not acknowledge drained preparation")
        if not observer_stopped:
            raise BenchError("observer transport remains in flight after task cleanup; outcome unknown")
        result.update(state="drained", quiescent=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def trial(args, a, b, router, case, spec, journal):
    row = {"kind": "router_trial", **spec, "valid": False, "invalid_reasons": [], "input_provenance": case["provenance"]}
    tokens, suffix = case["prefix"], case["suffix"]
    salt = "router-bench-"+digest([args.run_id, spec["id"]])[:40]
    task = {"session_id": "bench-"+uuid.uuid4().hex, "incarnation": uuid.uuid4().hex}
    row.update(task=task, cache_salt=salt, input_case=case["id"], prefix_sha256=digest(tokens), suffix_sha256=digest(suffix))
    observer, task_started = None, False
    started = time.monotonic()
    try:
        # Flush both owners before source work and A again after confirmed L3
        # publication. Formal Router selection can then never inherit warm A KV.
        for engine in (a, b):
            flush_local(engine, spec["id"], args.timeout_s)
        row["source_cold"] = assert_cold(a, tokens, salt, spec["id"])
        started = time.monotonic()
        row["source"] = a.http.generate(tokens, salt, 1, trial=spec["id"], seed=args.seed, origin="source")
        source_hits = cache_details(row["source"]["meta_info"])
        if sum(source_hits[t] for t in ("device", "cpu", "storage")):
            raise BenchError("source initial computation unexpectedly reused cache")
        row["publication"] = a.transfer("publish", a.target(tokens, salt), {"tier": "gpu"},
            {"tier": "storage", "backend": args.storage_backend}, spec["id"])
        require_completed(row["publication"], len(tokens), "source L3 publication")
        flush_local(a, spec["id"], args.timeout_s)
        row["cold_before_preparation"] = {control.http.url: assert_cold(control, tokens, salt, spec["id"]) for control in (a, b)}
        row["healthy_workers"] = router.topology([a.http.url, b.http.url], spec["id"])
        row["engine_capabilities_after_flush"] = {"a": copy.deepcopy(a.cap), "b": copy.deepcopy(b.cap)}
        task_started = True
        row["task_started"] = router.event(task, "task_started", 0, "start", spec["id"])
        row["request_observed"] = router.event(task, "request_observed", 1, "observed-source", spec["id"],
            branch_id="main", branch_revision=1, request_id="source", request_incarnation="source-binding",
            input_ids=tokens, cache_salt=salt, worker_url=a.http.url)
        if row["request_observed"].get("state") != "associated":
            raise BenchError("Router did not associate the exact already-generated source input")
        wait_started = time.monotonic()
        deadline = wait_started+spec["tool_wait_s"]
        row["initial_and_publication_ms"] = (wait_started-started)*1000
        observer = PreparationObserver(router, a, b, task, len(tokens), deadline, spec["id"]).start()
        sleep_until(deadline)
        row["actual_tool_wait_ms"] = (time.monotonic()-wait_started)*1000
        row["preparation_at_resume"] = observer.snapshot()
        row["formal_scheduled_monotonic"] = deadline
        if spec["group"] == "b3_router":
            context = {**task, "branch_id": "main", "revision": 4, "branch_revision": 2,
                       "request_id": "resume", "request_incarnation": "resume-binding"}
            response = router.formal.generate(tokens+suffix, salt, args.output_tokens,
                trial=spec["id"], seed=args.seed, kv_context=context)
            row["resume"] = response
            worker = response["response_headers"].get("x-sgl-worker-url")
            if not worker:
                raise BenchError("Router response omitted x-sgl-worker-url; owner attribution unknown")
            worker = worker.rstrip("/")
            if worker not in (a.http.url, b.http.url):
                raise BenchError("Router selected an unverified worker: " + worker)
        else:
            destination = a if spec["group"] == "route_change" else b
            response = destination.http.generate(tokens+suffix, salt, args.output_tokens, trial=spec["id"], seed=args.seed)
            row["resume"] = response
            worker = destination.http.url
        row["selected_worker_url"] = worker
        row["route_changed_from_prepared_b"] = worker != b.http.url
        row["formal_dispatch_lateness_ms"] = (response["started_monotonic"]-deadline)*1000
        row["complete_cycle_ms"] = (time.monotonic()-started)*1000
        row["resume_cache"] = cache_details(response["meta_info"], require=True)
        hits = row["resume_cache"]
        if hits["device"] or sum(hits[t] for t in ("cpu", "storage")) < len(tokens):
            raise BenchError("formal request bypassed verified cold L3/L2 path or recomputed prefix")
        view = row["preparation_at_resume"]["view"]
        row["prefetch_ready_before_resume"] = inspect_preparation(view, b, len(tokens)) if view else False
        if worker == b.http.url and row["prefetch_ready_before_resume"] and hits["cpu"] < len(tokens):
            raise BenchError("queried B preparation was not reused from host by the actual worker")
        if worker == a.http.url and (hits["cpu"] or hits["storage"] < len(tokens)):
            raise BenchError("unprepared A did not prove full ordinary storage reuse")
        row["repeated_prefill_tokens"] = max(0, len(tokens)-sum(hits[t] for t in ("cpu", "storage")))
        row["preparation_usage"] = ("unused_on_b_route_changed" if worker != b.http.url else
            "confirmed_host_reuse" if row["prefetch_ready_before_resume"] else "ambiguous_or_policy_fallback")
        row["valid"] = True
    except Exception as exc:
        row["invalid_reasons"].append(f"{type(exc).__name__}: {exc}")
        row["exception"] = traceback.format_exc()
    finally:
        if task_started:
            row["cleanup"] = cleanup(router, task, observer, spec["id"], args.cleanup_timeout_s)
            if not row["cleanup"]["quiescent"]:
                row["valid"] = False
                row["invalid_reasons"].append("Router cleanup remains unknown/non-quiescent")
        if observer:
            row["preparation_decision"] = observer.decision
            row["preparation_final"] = observer.snapshot()
            if observer.error:
                row["valid"] = False
                row["invalid_reasons"].append(observer.error)
        if row.get("resume"):
            row.setdefault("complete_cycle_ms", (row["resume"]["completed_monotonic"]-started)*1000)
            row.setdefault("formal_dispatch_lateness_ms", (row["resume"]["started_monotonic"]-row["formal_scheduled_monotonic"])*1000)
        try:
            prepared = max(observed_prepared_tokens(row.get("preparation_at_resume", {}).get("view"), len(tokens)),
                observed_prepared_tokens(row.get("preparation_final", {}).get("view"), len(tokens)),
                observed_prepared_tokens(row.get("cleanup", {}).get("last_view"), len(tokens)))
            row["prepared_tokens_observed"] = prepared
            if row.get("selected_worker_url") and row["selected_worker_url"] != b.http.url:
                row["unused_prepared_tokens_observed"] = prepared
                row["used_prepared_tokens"] = 0
                row["preparation_usage"] = "unused_on_b_route_changed" if prepared else "route_changed_no_preparation_confirmed"
            elif row.get("selected_worker_url") == b.http.url and row.get("prefetch_ready_before_resume"):
                used = min(prepared, row.get("resume_cache", {}).get("cpu", 0))
                row["used_prepared_tokens"] = used
                row["unused_prepared_tokens_observed"] = prepared-used
            else:
                row["used_prepared_tokens"] = None
                row["unused_prepared_tokens_observed"] = None
            row["preparation_accounting"] = "observed scope lower bound; no physical I/O bytes inferred"
        except BenchError as exc:
            row["valid"] = False
            row["invalid_reasons"].append(str(exc))
        row["quiescent_cycle_ms"] = (time.monotonic()-started)*1000
    return row


def report(directory, rows):
    groups = {}
    for group in ("b2_direct_b", "b3_router", "route_change"):
        attempts = [r for r in rows if r["group"] == group]
        completed = [r for r in attempts if r.get("resume")]
        groups[group] = {"attempts": len(attempts), "valid": sum(r["valid"] for r in attempts),
            "completed": len(completed), "all_completed_ttft_ms": distribution([r["resume"]["ttft_ms"] for r in completed]),
            "all_completed_cycle_ms": distribution([r["complete_cycle_ms"] for r in completed if "complete_cycle_ms" in r]),
            "ready_before_resume": sum(r.get("prefetch_ready_before_resume", False) for r in attempts),
            "route_changed": sum(r.get("route_changed_from_prepared_b", False) for r in attempts),
            "unused_prepared_tokens_observed": sum(r.get("unused_prepared_tokens_observed") or 0 for r in attempts),
            "usage_unknown": sum(r.get("unused_prepared_tokens_observed") is None for r in attempts),
            "policy_fallback": sum((r.get("preparation_decision") or {}).get("state") == "on_demand_fallback" for r in attempts)}
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair"], {})[row["group"]] = row
    pairs = []
    for members in by_pair.values():
        if all(members.get(g, {}).get("resume") for g in ("b2_direct_b", "b3_router")):
            a, b = members["b2_direct_b"], members["b3_router"]
            pairs.append({"pair": a["pair"], "valid": a["valid"] and b["valid"],
                "ttft_delta_ms": b["resume"]["ttft_ms"]-a["resume"]["ttft_ms"],
                "selected_worker_url": b.get("selected_worker_url")})
    conditions = {}
    for row in rows:
        label = json.dumps([row["prefix_tokens"], row["tool_wait_s"], row["group"]])
        block = conditions.setdefault(label, {"attempts": 0, "valid": 0, "ttft_ms": [], "route_changed": 0})
        block["attempts"] += 1
        block["valid"] += int(row["valid"])
        block["route_changed"] += int(row.get("route_changed_from_prepared_b", False))
        if row.get("resume"):
            block["ttft_ms"].append(row["resume"]["ttft_ms"])
    for block in conditions.values():
        block["ttft_ms"] = distribution(block["ttft_ms"])
    result = {"attempts": len(rows), "groups": groups, "conditions": conditions, "paired": pairs,
        "paired_all_completed_mean_delta_ci": bootstrap_mean_ci([p["ttft_delta_ms"] for p in pairs], seed=stable_seed("b3-b2")),
        "invalid": [{"id": r["id"], "reasons": r["invalid_reasons"]} for r in rows if not r["valid"]],
        "comparison": "B3 free normal Router selection minus B2 direct B; same Router preparation targeting B in both arms; neither arm is B0",
        "limitations": ["Router transfer-cost EMA is stateful; direct B2 traffic does not trigger its formal-dispatch learning fence",
                        "all completed responses remain in primary statistics, including invalid attribution/control trials",
                        "no GPU benefit established by local HTTP tests; current ready scope is queried evidence, never a residency lease"]}
    Path(directory, "router-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Router B2/B3 配对", "", result["comparison"], "",
        "| group | attempts | completed | valid | mean TTFT ms | ready before resume | changed route | fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in groups.items():
        mean = group["all_completed_ttft_ms"]["mean"]
        lines.append(f"| {name} | {group['attempts']} | {group['completed']} | {group['valid']} | {'—' if mean is None else f'{mean:.3f}'} | {group['ready_before_resume']} | {group['route_changed']} | {group['policy_fallback']} |")
    lines += ["", "完整配对差值、N 与 bootstrap CI 见 router-summary.json。所有完成响应进入主统计；错误归属、控制失败和未完成 cleanup 的条目保留为 invalid。A 在发布后清本地缓存，Router 自由选择 A 时需证明普通 L3 命中。两臂使用相同准备信号，EMA 的历史状态可能不同；直接 B2 请求不会触发 Router 的正式调度学习屏障。", ""]
    Path(directory, "router-summary.md").write_text("\n".join(lines))
    return result


def run(args):
    if args.router_mode != "b2-b3":
        raise BenchError("explicit --router-mode b2-b3 is required")
    control_key = os.environ.get(args.router_control_key_env, "")
    if not control_key:
        raise BenchError("Router control bearer environment variable is missing/empty")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    journal = Journal(root/"router-raw.jsonl")
    args.run_id = args.run_id or uuid.uuid4().hex
    try:
        key = os.environ.get(args.engine_key_env) if args.engine_key_env else None
        formal_key = os.environ.get(args.router_api_key_env) if args.router_api_key_env else None
        a, b = [Control(HTTP(url, journal, args.timeout_s, key), "bench-"+args.run_id,
                        args.poll_ms, args.timeout_s) for url in (args.engine_a, args.engine_b)]
        if a.http.url == b.http.url or a.cap["engine_id"] == b.cap["engine_id"]:
            raise BenchError("B3 requires two distinct Engine owners")
        metadata = {"a": get_metadata(a.http), "b": get_metadata(b.http)}
        for data in metadata.values():
            for field in ("model_path", "tokenizer_path"):
                actual = (data.get("model") or {}).get(field)
                if actual and Path(actual).resolve() != Path(args.model).resolve():
                    raise BenchError("local input tokenizer/model differs from serving Engine")
        if fingerprint(metadata["a"]) != fingerprint(metadata["b"]):
            raise BenchError("Engine model/load/cache capacity differs")
        for field in ("namespace", "weight_version", "cache_salt_encoding"):
            if a.cap.get(field) != b.cap.get(field):
                raise BenchError("incompatible Engine "+field)
        if a.backend_namespace != b.backend_namespace or a.executor.get("layout_id") != b.executor.get("layout_id"):
            raise BenchError("Engine layout/backend_namespace differs")
        if a.cap.get("policies") or b.cap.get("policies"):
            raise BenchError("implicit Engine policies confound B2/B3 comparison")
        router = Router(HTTP(args.router, journal, args.timeout_s, control_key), HTTP(args.router, journal, args.timeout_s, formal_key), args.poll_ms)
        router.topology([a.http.url, b.http.url], "preflight")
        if args.trace_jsonl:
            cases = load_trace(args.trace_jsonl, args.model)
        else:
            cases = []
            for n in args.prefix_lengths:
                prefix, suffix = build_tokens(args.model, n, args.seed+n)
                cases.append({"id": f"synthetic-{n}", "prefix": prefix, "suffix": suffix, "tool_wait_s": args.tool_wait_s,
                              "provenance": {"input_mode": "synthetic_ledger", "model_tokenizer": tokenizer_identity(args.model), "seed": args.seed+n}})
        page = b.public.get("prefix", {}).get("page_size") or b.executor.get("directions", {}).get("page_alignment")
        for case in cases:
            if not page or not 0 < len(case["prefix"]) <= 65536 or len(case["prefix"]) % page:
                raise BenchError("input prefix must align to advertised page size")
            context = fingerprint(metadata["b"]).get("context_length")
            if context and len(case["prefix"])+len(case["suffix"])+args.output_tokens > context:
                raise BenchError("input and output exceed serving context budget")
        manifest = {"args": vars(args), "metadata": metadata, "capabilities": {"a": a.cap, "b": b.cap, "router": router.capabilities},
            "inputs": [{"id": c["id"], "provenance": c["provenance"]} for c in cases],
            "source_flush": "ordinary local flush after confirmed L3 publication; shared backend is retained",
            "preparation": "both arms: Router resume_expected with fixed worker_url B; final B3 worker selected by normal Router policy",
            "credentials": "environment names only; Authorization headers omitted from journal",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        Path(root, "router-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        journal.emit("router_manifest", **manifest)
        rng, schedule = random.Random(args.seed), []
        for case in cases:
            for repetition in range(args.repetitions):
                pair = f"{digest(case['id'])[:8]}-r{repetition}"
                groups = ["b2_direct_b", "b3_router"] + (["route_change"] if args.negative == "route_change" and repetition == 0 else [])
                rng.shuffle(groups)
                for group in groups:
                    schedule.append({"id": pair+"-"+group, "pair": pair, "group": group, "input_case": case["id"],
                        "prefix_tokens": len(case["prefix"]), "tool_wait_s": case["tool_wait_s"] if case.get("tool_wait_s") is not None else args.tool_wait_s})
        Path(root, "router-schedule.json").write_text(json.dumps(schedule, indent=2)+"\n")
        by_id = {c["id"]: c for c in cases}
        for case in cases:
            for repetition in range(args.warmup_repetitions):
                for control in (a, b):
                    warmup = "warmup-"+uuid.uuid4().hex
                    control.http.generate(case["prefix"]+case["suffix"], warmup, args.output_tokens, trial=warmup, seed=args.seed, origin="warmup")
        rows, signatures = [], {}
        for spec in schedule:
            row = trial(args, a, b, router, by_id[spec["input_case"]], spec, journal)
            if row.get("resume"):
                signature = {k: row["resume"].get(k) for k in ("output_ids", "text")}
                if signature != signatures.setdefault(spec["input_case"], signature):
                    row["valid"] = False
                    row["invalid_reasons"].append("deterministic output differs between B2/B3/route-change conditions")
            rows.append(row)
            journal.emit("router_trial", **{k: v for k, v in row.items() if k != "kind"})
            report(root, rows)
            print(json.dumps({"trial": row["id"], "valid": row["valid"], "worker": row.get("selected_worker_url"),
                              "ttft_ms": row.get("resume", {}).get("ttft_ms"), "reasons": row["invalid_reasons"]}), flush=True)
        return 0 if all(r["valid"] for r in rows) else 2
    finally:
        journal.file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router-mode", choices=["b2-b3"], required=True)
    p.add_argument("--router", required=True)
    p.add_argument("--engine-a", required=True)
    p.add_argument("--engine-b", required=True)
    p.add_argument("--router-control-key-env", required=True)
    p.add_argument("--router-api-key-env")
    p.add_argument("--engine-key-env")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trace-jsonl")
    p.add_argument("--prefix-lengths", type=lambda v: csv_numbers(v, int), default=[2048,8192,16384])
    p.add_argument("--tool-wait-s", type=float, default=3)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--warmup-repetitions", type=int, default=1)
    p.add_argument("--negative", choices=["route_change", "none"], default="route_change")
    p.add_argument("--output-tokens", type=int, default=8)
    p.add_argument("--poll-ms", type=float, default=50)
    p.add_argument("--timeout-s", type=float, default=60)
    p.add_argument("--cleanup-timeout-s", type=float, default=60)
    p.add_argument("--storage-backend", default="mooncake")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--run-id")
    args = p.parse_args()
    if (args.repetitions < 1 or args.warmup_repetitions < 0 or args.output_tokens < 1
            or not all(math.isfinite(x) and x > 0 for x in (args.poll_ms, args.timeout_s, args.cleanup_timeout_s))
            or not math.isfinite(args.tool_wait_s) or args.tool_wait_s < 0):
        p.error("invalid timing/repetition/output limits")
    try:
        return run(args)
    except Exception as exc:
        print(f"ROUTER BENCHMARK FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
