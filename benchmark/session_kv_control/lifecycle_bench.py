#!/usr/bin/env python3
"""Fixed-work lifecycle replay with bounded asynchronous control cleanup.

The arms share a serving binary, input plan and fixed generation deadlines.
This measures simulated rollout tasks, not an actual trainer's waiting time.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from bench_report import distribution
from bench_statistics import bootstrap_mean_ci, stable_seed
from router_bench import Router, flush_local
from session_kv_bench import (
    HTTP,
    BenchError,
    Control,
    HTTPBenchError,
    Journal,
    build_tokens,
    cache_details,
    digest,
    get_metadata,
    ready_tokens,
    sleep_until,
    status,
)
from stress_bench import output_equal, parity, publish

ARMS = ("on_demand", "fixed_prefetch", "lifecycle")
ARM_EVIDENCE_FIELDS = {
    "rows",
    "server_before",
    "server_after",
    "costs_before",
    "costs_after",
}


def arm_metrics(arm):
    return {key: value for key, value in arm.items() if key not in ARM_EVIDENCE_FIELDS}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cost_snapshot(engine, trial):
    # Older/mock Engines may not expose these observations. Preserve unknown
    # instead of estimating physical traffic from logical request hit counts.
    try:
        return engine.http.request(
            "/v1/kv/control",
            {"kind": "capacity", "context": engine.context},
            optional=True,
            trial=trial,
        )
    except BenchError as error:
        return {"success": False, "observation_error": str(error)}


class AsyncControls:
    """Bound running plus queued jobs; callers see backpressure and final drain."""

    def __init__(self, workers, capacity):
        if not 0 < workers <= capacity:
            raise ValueError("control capacity must be at least worker count")
        self.pool = futures.ThreadPoolExecutor(max_workers=workers)
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.active = set()
        self.peak = 0
        self.errors = []
        self.outcomes = {
            "jobs": 0,
            "strict_preparation_complete": 0,
            "policy_fallback": 0,
            "prepared_tokens_observed": 0,
            "child_statuses": {},
        }

    def submit(self, work):
        started = time.monotonic()
        self.slots.acquire()
        waited_ms = (time.monotonic() - started) * 1000
        try:
            pending = self.pool.submit(work)
        except BaseException:
            self.slots.release()
            raise
        with self.lock:
            self.active.add(pending)
            self.peak = max(self.peak, len(self.active))

        def finished(job):
            # add_done_callback can run inline if a short job already finished.
            with self.lock:
                self.active.discard(job)
                self.outcomes["jobs"] += 1
                if job.exception() is not None:
                    self.errors.append(repr(job.exception()))
                elif isinstance(job.result(), dict) and (
                    job.result().get("errors") or not job.result().get("quiescent")
                ):
                    self.errors.append(
                        {
                            "id": job.result().get("id"),
                            "errors": job.result().get("errors", []),
                            "quiescent": job.result().get("quiescent"),
                        }
                    )
                if job.exception() is None and isinstance(job.result(), dict):
                    result = job.result()
                    for key in (
                        "strict_preparation_complete",
                        "policy_fallback",
                        "prepared_tokens_observed",
                    ):
                        self.outcomes[key] += result.get(key, 0)
                    for state in result.get("child_statuses", []):
                        counts = self.outcomes["child_statuses"]
                        counts[state] = counts.get(state, 0) + 1
            self.slots.release()

        pending.add_done_callback(finished)
        return waited_ms

    def drain(self):
        started = time.monotonic()
        self.pool.shutdown(wait=True)
        return {
            "wall_ms": (time.monotonic() - started) * 1000,
            "peak_pending": self.peak,
            "errors": self.errors,
            "outcomes": self.outcomes,
        }


def replay_plan(args):
    """Same per-lane order and exogenous waits in every policy arm."""
    rng = random.Random(args.seed)
    lanes = [[] for _ in range(args.concurrency)]
    for turn in range(args.passes):
        order = list(range(args.cases))
        rng.shuffle(order)
        for position, case in enumerate(order):
            actual = max(
                0,
                args.tool_wait_s + rng.uniform(-args.wait_jitter_s, args.wait_jitter_s),
            )
            predicted = max(
                0,
                actual + rng.uniform(-args.prediction_error_s, args.prediction_error_s),
            )
            lanes[position % args.concurrency].append(
                {
                    "id": f"p{turn}-c{case}",
                    "case": case,
                    "turn": turn,
                    "tool_wait_s": actual,
                    "predicted_wait_s": predicted,
                    "cancelled": rng.random() < args.cancel_fraction,
                    "route_change": rng.random() < args.route_change_fraction,
                }
            )
    return lanes


def execute_lane(lane, work, deadline=None, clock=None):
    """Keep clients active until the deadline; the finite plan is a safety cap."""
    clock = clock or time.monotonic
    rows = []
    for item in lane:
        admitted = clock()
        if deadline is not None and admitted >= deadline:
            break
        rows.append(work(item, admitted))
    stopped = clock()
    return rows, deadline is not None and stopped < deadline, stopped


def cases_for_arm(args, material, spec):
    common = int(args.prefix_tokens * args.shared_prefix_fraction)
    common -= common % args.page_size
    result = []
    for index, tokens in enumerate(material):
        # A family shares physical salt and ancestor bytes; independent task
        # incarnations keep concurrent branch revisions unambiguous.
        family = index // args.branches_per_family
        leader = family * args.branches_per_family
        prefix = material[leader][:common] + tokens[common:]
        salt_index = 0 if args.salt_mode == "shared" else family
        salt_scope = (
            f"block-{spec['block']}"
            if getattr(args, "reuse_published_sources", False)
            else spec["id"]
        )
        salt = "lifecycle-" + digest([args.run_id, salt_scope, salt_index])[:40]
        result.append(
            {
                "id": index,
                "family": family,
                "branch": index % args.branches_per_family,
                "prefix": prefix,
                "salt": salt,
            }
        )
    return result


def logical_working_set(cases, page_size):
    """Deduplicate same-salt prefix pages; this is logical KV, not I/O bytes."""
    pages = set()
    for case in cases:
        parent = hashlib.sha256(case["salt"].encode()).digest()
        for start in range(0, len(case["prefix"]), page_size):
            parent = hashlib.sha256(
                parent
                + json.dumps(
                    case["prefix"][start : start + page_size], separators=(",", ":")
                ).encode()
            ).digest()
            pages.add(parent)
    return len(pages) * page_size


def resume_prompt(case):
    # A fixed synthetic tool-result token makes the complete published prefix
    # reusable. Repeating the prefix alone forces its final page to be computed
    # for logits and would count that mandatory work as a recovery miss.
    return case["prefix"] + [case["prefix"][-1]]


class PressureSampler:
    """One optional low-rate sampler; unknown or stale data cannot assert pressure."""

    def __init__(self, http, journal, threshold):
        self.http, self.journal, self.threshold = http, journal, threshold
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            sample = {"sample_monotonic": time.monotonic()}
            try:
                text = self.http.request(
                    "/metrics", optional=True, trial="lifecycle-pressure"
                )
                values = {}
                for kind in ("used", "total"):
                    matches = re.findall(
                        r"^sglang:hicache_host_"
                        + kind
                        + r"_tokens(?:\{[^\n]*\})?\s+([0-9.eE+-]+)\s*$",
                        text or "",
                        re.M,
                    )
                    # TP1/DP1 only: ambiguous multiple pool gauges stay unknown.
                    if len(matches) != 1:
                        raise BenchError(
                            "one authoritative host pool gauge is required"
                        )
                    values[kind] = float(matches[0])
                if (
                    not all(math.isfinite(value) for value in values.values())
                    or not 0 <= values["used"] <= values["total"]
                    or values["total"] <= 0
                ):
                    raise BenchError("invalid host pool capacity gauges")
                sample.update(
                    used_tokens=values["used"],
                    total_tokens=values["total"],
                    usage=values["used"] / values["total"],
                )
            except Exception as exc:
                sample["error"] = str(exc)
            with self.lock:
                self.latest = sample
            self.journal.emit("lifecycle_pressure", **sample)
            self.stop.wait(2)

    def snapshot(self):
        with self.lock:
            sample = copy.deepcopy(self.latest)
        age = time.monotonic() - sample.get("sample_monotonic", float("-inf"))
        return {
            **sample,
            "sample_age_s": age if math.isfinite(age) else None,
            "pressure": age <= 6 and sample.get("usage", -1) >= self.threshold,
        }


def metadata_budget(args):
    total = args.blocks * len(args.arms) * args.cases * args.passes
    events = (
        args.blocks
        * args.cases
        * args.passes
        * sum(5 if arm == "lifecycle" else 3 for arm in args.arms)
    )
    required = {
        "max_operations": events,
        "limits.max_tombstones": total,
        "limits.max_sessions": args.concurrency
        + (args.control_backlog or args.concurrency * 2),
    }
    if args.router_budget_file:
        config = json.loads(Path(args.router_budget_file).read_text())
        actual = {
            "max_operations": config["max_operations"],
            "limits.max_tombstones": config["limits"]["max_tombstones"],
            "limits.max_sessions": config["limits"]["max_sessions"],
        }
        if any(actual[key] < value for key, value in required.items()):
            raise BenchError(
                "Router metadata configuration is smaller than replay requirements: "
                + json.dumps(required)
            )
        return {
            "required": required,
            "configured": actual,
            "scope": "dedicated Router initially without earlier lifecycle receipts/tombstones; active index/material/job budgets also apply",
        }
    if total > 5000:
        raise BenchError(
            "long replay requires --router-budget-file for metadata preflight"
        )
    return {"required": required, "configured": None}


def inspect_ready(view, engine, length):
    """Readiness is a scoped observation, never inferred from job acceptance."""
    if not view or view.get("state") in ("on_demand_fallback", "unknown"):
        return False
    scopes = view.get("known_scope", [])
    if (
        len(scopes) != 1
        or scopes[0].get("branch") != "main"
        or scopes[0].get("end_token") != length
    ):
        raise BenchError("Router froze a different input scope")
    for op in view.get("operations", []):
        if op.get("action") != "sglang.prefetch":
            continue
        if (
            op.get("worker_url", "").rstrip("/") != engine.http.url
            or op.get("engine_id") != engine.cap["engine_id"]
            or op.get("owner_epoch") != engine.cap["engine_epoch"]
        ):
            raise BenchError("Router preparation owner/epoch differs from target")
        scope = op.get("ready_scope") or {}
        if (
            scope.get("tier") == "host"
            and scope.get("provenance") == "engine_scoped_query"
            and scope.get("start_token") == 0
            and scope.get("end_token") == length
        ):
            return True
    return False


def drain_router(router, task, message_id, trial, timeout):
    """A lost submit response still owns its stable message ID until drained."""
    router.command({"kind": "cancel", "task": task, "message_id": message_id}, trial)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = router.query(task, message_id, trial)
        if time.monotonic() >= deadline:
            break
        if view.get("quiescent") is True and all(
            op.get("quiescent") is True for op in view.get("operations", [])
        ):
            # ACK revalidates shared jobs; a query does not reserve quiescence.
            try:
                ack = router.command(
                    {"kind": "ack", "task": task, "message_id": message_id}, trial
                )
            except HTTPBenchError as error:
                if (
                    error.status_code != 400
                    or error.error.get("code") != "rejected"
                    or error.error.get("reason") != "operation/claim not quiescent"
                ):
                    raise
            else:
                if not isinstance(ack, dict) or ack.get("acknowledged") is not True:
                    raise BenchError("Router drain acknowledgement failed")
                return view
        time.sleep(min(router.poll_ms / 1000, max(0, deadline - time.monotonic())))
    raise BenchError("Router intent did not become quiescent")


class Preparation:
    """One bounded control owner; inference reads snapshots without joining it."""

    def __init__(
        self,
        args,
        router,
        destination,
        task,
        case,
        spec,
        deadline,
        journal,
        pressure=None,
    ):
        self.args, self.router, self.b = args, router, destination
        self.task, self.case, self.spec = task, case, spec
        self.deadline, self.journal = deadline, journal
        self.pressure = pressure
        self.done = threading.Event()
        self.request_started = threading.Event()
        self.lock = threading.Lock()
        self.observed = {}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.observed)

    def observe(self, operation):
        with self.lock:
            self.observed = {
                "operation": copy.deepcopy(operation),
                "observed_monotonic": time.monotonic(),
            }

    def run(self):
        args, spec = self.args, self.spec
        trial = spec["trial"]
        result = {"id": trial, "errors": [], "quiescent": False}
        attempted, admitted = False, False
        wait_attempted, native_quiescent = False, True
        try:
            predicted_deadline = (
                self.deadline - spec["tool_wait_s"] + spec["predicted_wait_s"]
            )
            if spec["arm"] == "fixed_prefetch":
                # Cancellation before submission skips work; cancellation after
                # submission drains the native operation normally via Control.
                due = max(
                    self.deadline - spec["tool_wait_s"],
                    predicted_deadline - args.fixed_lead_s,
                )
                if not self.done.wait(max(0, due - time.monotonic())):
                    native_quiescent = False
                    result["prefetch"] = self.b.transfer(
                        "prefetch",
                        self.b.target(self.case["prefix"], self.case["salt"]),
                        {"tier": "storage", "backend": "mooncake"},
                        {"tier": "host"},
                        trial,
                        self.observe,
                    )
                    native_quiescent = True
                    operation = result["prefetch"]["operation"]
                    result["prefetch_status"] = status(operation)
                    result["child_statuses"] = [status(operation)]
                    result["prepared_tokens_observed"] = ready_tokens(operation)
                    result["strict_preparation_complete"] = status(
                        operation
                    ) == "completed" and ready_tokens(operation) >= len(
                        self.case["prefix"]
                    )
            elif spec["arm"] == "lifecycle" and not self.done.is_set():
                remaining = max(
                    0, round((predicted_deadline - time.monotonic()) * 1000)
                )
                pressure = (
                    self.pressure.snapshot() if self.pressure else {"pressure": False}
                )
                result["pressure_evidence"] = pressure
                wait_attempted = True
                result["tool_wait"] = self.router.event(
                    self.task,
                    "tool_wait",
                    2,
                    "wait",
                    trial,
                    target={"kind": "branch", "branch_id": "main", "revision": 1},
                    worker_url=self.b.http.url,
                    source={"tier": "host"},
                    pressure=pressure["pressure"],
                    expected_wait_ms=remaining,
                )
                attempted = True
                view = self.router.event(
                    self.task,
                    "resume_expected",
                    3,
                    "prepare",
                    trial,
                    target={"kind": "branch", "branch_id": "main", "revision": 1},
                    worker_url=self.b.http.url,
                    expected_wait_ms=max(
                        0, round((predicted_deadline - time.monotonic()) * 1000)
                    ),
                )
                admitted = view.get("state") != "on_demand_fallback"
                result["policy_fallback"] = not admitted
                result["decision"] = copy.deepcopy(view)
                while True:
                    result["policy_fallback"] = result["policy_fallback"] or any(
                        op.get("policy_fallback") for op in view.get("operations", [])
                    )
                    prepared = inspect_ready(view, self.b, len(self.case["prefix"]))
                    with self.lock:
                        self.observed = {
                            "view": copy.deepcopy(view),
                            "observed_monotonic": time.monotonic(),
                        }
                    if (
                        prepared
                        or self.request_started.is_set()
                        or self.done.wait(self.router.poll_ms / 1000)
                        or not admitted
                    ):
                        break
                    view = self.router.query(self.task, "prepare", trial)
                result["child_statuses"] = [
                    (op.get("result") or {}).get("update", {}).get("status", "unknown")
                    for op in view.get("operations", [])
                    if op.get("action") == "sglang.prefetch"
                ]
                result["strict_preparation_complete"] = inspect_ready(
                    view, self.b, len(self.case["prefix"])
                )
                result["prepared_tokens_observed"] = (
                    max(
                        (op.get("ready_scope") or {}).get("end_token", 0)
                        for op in view.get("operations", [])
                    )
                    if view.get("operations")
                    else 0
                )
        except Exception as exc:
            result["errors"].append(str(exc))
            result["traceback"] = traceback.format_exc()
        finally:
            # Retiring the task early could invalidate the formal request's
            # revision. Even failed preparation keeps ownership until consumption.
            self.done.wait()
            try:
                result["finish"] = self.router.event(
                    self.task, "task_finished", 5, "finish", trial, cascade=False
                )
                if result.get("tool_wait", {}).get("operations") or (
                    wait_attempted and "tool_wait" not in result
                ):
                    result["demote_last_view"] = drain_router(
                        self.router, self.task, "wait", trial, args.cleanup_timeout_s
                    )
                # If submission lost its response, query/cancel conservatively.
                if admitted or (attempted and "decision" not in result):
                    result["last_view"] = drain_router(
                        self.router, self.task, "prepare", trial, args.cleanup_timeout_s
                    )
                    result["policy_fallback"] = result.get(
                        "policy_fallback", False
                    ) or any(
                        op.get("policy_fallback")
                        for op in result["last_view"].get("operations", [])
                    )
                result["quiescent"] = native_quiescent
            except Exception as exc:
                result["errors"].append("cleanup: " + str(exc))
            result["final_observation"] = self.snapshot()
            result["completed_monotonic"] = time.monotonic()
            self.journal.emit("lifecycle_control", **result)
        return result


def resume_task(args, a, b, router, case, spec, controls, journal, pressure=None):
    row = {
        **spec,
        "kind": "lifecycle_task",
        "errors": [],
        "valid": False,
        "case": case["id"],
        "family": case["family"],
        "branch": case["branch"],
    }
    started = time.monotonic()
    task = {
        "session_id": "lifecycle-" + digest([args.run_id, spec["trial"]])[:40],
        "incarnation": "1",
    }
    row.update(
        task=task, cache_salt=case["salt"], input_sha256=digest(resume_prompt(case))
    )
    preparation = None
    try:
        router.event(task, "task_started", 0, "start", spec["trial"])
        associated = router.event(
            task,
            "request_observed",
            1,
            "source",
            spec["trial"],
            branch_id="main",
            branch_revision=1,
            request_id="source",
            request_incarnation="source",
            input_ids=resume_prompt(case),
            cache_salt=case["salt"],
            worker_url=a.http.url,
        )
        if associated.get("state") != "associated":
            raise BenchError("source input was not associated")
        wait_started = time.monotonic()
        deadline = wait_started + spec["tool_wait_s"]
        row.update(wait_started_monotonic=wait_started, scheduled_monotonic=deadline)
        preparation = Preparation(
            args, router, b, task, case, spec, deadline, journal, pressure
        )
        row["control_backpressure_ms"] = controls.submit(preparation.run)
        # A saturated control backlog can delay injection; retain that lateness.
        sleep_until(deadline)
        preparation.request_started.set()
        row["preparation_at_resume"] = preparation.snapshot()
        observed = row["preparation_at_resume"].get("observed_monotonic")
        row["preparation_observation_age_ms"] = (
            max(0, (time.monotonic() - observed) * 1000)
            if observed is not None
            else None
        )
        if not spec["cancelled"]:
            destination = a if spec["route_change"] else b
            context = {
                **task,
                "branch_id": "main",
                "revision": 4,
                "branch_revision": 2,
                "request_id": "resume",
                "request_incarnation": "resume",
            }
            http = router.formal if args.via_router else destination.http
            row["generation"] = http.generate(
                resume_prompt(case),
                case["salt"],
                args.output_tokens,
                trial=spec["trial"],
                seed=args.seed,
                kv_context=context if args.via_router else None,
            )
            generation = row["generation"]
            worker = (
                generation["response_headers"].get("x-sgl-worker-url")
                if args.via_router
                else destination.http.url
            )
            if worker is None or worker.rstrip("/") not in (a.http.url, b.http.url):
                raise BenchError(
                    "formal response worker attribution is missing/unknown"
                )
            row["selected_worker_url"] = worker.rstrip("/")
            row["cache"] = cache_details(generation["meta_info"], require=True)
            row["dispatch_lateness_ms"] = max(
                0, (generation["started_monotonic"] - deadline) * 1000
            )
            row["repeated_prefill_tokens"] = max(
                0,
                len(case["prefix"])
                - sum(row["cache"][k] for k in ("device", "cpu", "storage")),
            )
            snapshot = row["preparation_at_resume"]
            if "view" in snapshot:
                row["ready_before_resume"] = inspect_ready(
                    snapshot["view"], b, len(case["prefix"])
                )
            elif "operation" in snapshot:
                row["ready_before_resume"] = status(
                    snapshot["operation"]
                ) == "completed" and ready_tokens(snapshot["operation"]) >= len(
                    case["prefix"]
                )
            else:
                row["ready_before_resume"] = False
        row["valid"] = True
    except Exception as exc:
        row["errors"].append(str(exc))
        row["traceback"] = traceback.format_exc()
        if preparation is None:
            # No asynchronous owner exists yet; retire the partially admitted
            # task here and keep any failure explicit.
            try:
                router.event(
                    task, "task_finished", 5, "finish", spec["trial"], cascade=False
                )
            except Exception as cleanup_error:
                row["errors"].append("cleanup: " + str(cleanup_error))
    finally:
        if preparation:
            preparation.done.set()
        row["completed_monotonic"] = time.monotonic()
        row["task_latency_ms"] = (row["completed_monotonic"] - started) * 1000
        journal.emit("lifecycle_task", **{k: v for k, v in row.items() if k != "kind"})
    return row


def arm_summary(spec, rows, wall_s, drain, args, source_s, signatures):
    complete = [row for row in rows if row.get("generation")]
    for row in complete:
        key = (spec["block"], row["case"])
        signature = {k: row["generation"].get(k) for k in ("output_ids", "text")}
        row["output_equal"] = output_equal(
            signature, signatures.setdefault(key, signature)
        )
        if not row["output_equal"]:
            row["valid"] = False
            row["errors"].append("paired deterministic output differs")
    expected = sum(not row["cancelled"] for row in rows)
    accepted = sum(row["output_equal"] for row in complete)
    return {
        **spec,
        "planned_tasks": len(rows),
        "expected_effective_tasks": expected,
        "completed": len(complete),
        "effective_completed": accepted,
        "valid_tasks": sum(row["valid"] for row in rows),
        "cancelled_tasks": sum(row["cancelled"] for row in rows),
        "wall_s": wall_s,
        "source_publication_s": source_s,
        "complete_job_wall_s": wall_s + source_s,
        "all_completed_tasks_per_s": len(complete) / wall_s,
        "effective_tasks_per_s": accepted / wall_s,
        "complete_job_effective_tasks_per_s": accepted / (wall_s + source_s),
        "meets_minimum_duration": wall_s >= args.min_arm_seconds,
        "meets_minimum_resumes": len(complete) >= args.min_arm_resumes,
        "all_completed_ttft_ms": distribution(
            [row["generation"]["ttft_ms"] for row in complete]
        ),
        "all_completed_latency_ms": distribution(
            [row["generation"]["latency_ms"] for row in complete]
        ),
        "all_completed_task_latency_ms": distribution(
            [row["task_latency_ms"] for row in complete]
        ),
        "dispatch_lateness_ms": distribution(
            [row.get("dispatch_lateness_ms", 0) for row in complete]
        ),
        "control_backpressure_ms": distribution(
            [row.get("control_backpressure_ms", 0) for row in rows]
        ),
        "ready_before_resume": sum(
            row.get("ready_before_resume", False) for row in complete
        ),
        "repeated_prefill_tokens": sum(
            row.get("repeated_prefill_tokens", 0) for row in complete
        ),
        "drain": drain,
        "rows": rows,
    }


def summarize(arms):
    pairs = []
    for block in sorted({arm["block"] for arm in arms}):
        members = {arm["arm"]: arm for arm in arms if arm["block"] == block}
        base = members.get("on_demand")
        if not base:
            continue
        for name, candidate in members.items():
            if name == "on_demand":
                continue
            duration = bool(base.get("duration_s") or candidate.get("duration_s"))
            comparable = not duration and (
                base["expected_effective_tasks"]
                == candidate["expected_effective_tasks"]
                and base["completed"] == base["expected_effective_tasks"]
                and candidate["completed"] == candidate["expected_effective_tasks"]
            )
            pairs.append(
                {
                    "block": block,
                    "candidate": name,
                    "comparison_mode": "duration" if duration else "fixed_work",
                    "fixed_work_complete": None if duration else comparable,
                    "effective_throughput_change_percent": 100
                    * (
                        candidate["effective_tasks_per_s"]
                        / base["effective_tasks_per_s"]
                        - 1
                    )
                    if base["effective_tasks_per_s"]
                    else None,
                    "wall_change_percent": None
                    if duration
                    else 100 * (candidate["wall_s"] / base["wall_s"] - 1),
                }
            )
    return {
        "arms": [arm_metrics(arm) for arm in arms],
        "paired": pairs,
        "paired_block_throughput_ci_percent": {
            name: bootstrap_mean_ci(
                [
                    pair["effective_throughput_change_percent"]
                    for pair in pairs
                    if pair["candidate"] == name
                    and pair["effective_throughput_change_percent"] is not None
                ],
                seed=stable_seed(name),
            )
            for name in ("fixed_prefetch", "lifecycle")
        },
        "measurement": "completed non-cancelled simulated rollout tasks matching source output / work wall including registration, tool waits, control backpressure and final drain",
        "trainer_waiting_time": None,
        "limitations": [
            "closed-loop clients; fixed work or bounded duration admission; dispatch lateness is recorded, not an open-loop arrival test",
            "natural cache residency permits GPU hits; all hit tiers and repeated prefill are retained",
            "family branches share input ancestry/salt but have independent logical task incarnations",
            "source publication precedes work; complete-job throughput includes it; lab reset is reported separately",
            "physical I/O bytes are not inferred from cached-token counts",
        ],
    }


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    journal = Journal(root / "lifecycle-raw.jsonl")
    args.run_id = args.run_id or uuid.uuid4().hex
    try:
        key = os.environ.get(args.engine_key_env) if args.engine_key_env else None
        bearer = os.environ.get(args.router_control_key_env, "")
        if not bearer:
            raise BenchError(
                "Router control bearer environment variable is missing/empty"
            )
        a, b = [
            Control(
                HTTP(url, journal, args.timeout_s, key),
                "lifecycle-" + args.run_id,
                args.poll_ms,
                args.timeout_s,
            )
            for url in (args.engine_a, args.engine_b)
        ]
        metadata = {"a": get_metadata(a.http), "b": get_metadata(b.http)}
        configuration = parity(a, b, metadata, args.expected_mem_fraction)
        context = configuration["fingerprint_b"].get("context_length")
        if context and args.prefix_tokens + 1 + args.output_tokens > context:
            raise BenchError("input and output exceed serving context budget")
        router = Router(
            HTTP(args.router, journal, args.timeout_s, bearer),
            HTTP(
                args.router,
                journal,
                args.timeout_s,
                os.environ.get(args.router_api_key_env)
                if args.router_api_key_env
                else None,
            ),
            args.poll_ms,
        )
        router.topology([a.http.url, b.http.url], "preflight")
        args.page_size = b.public.get("prefix", {}).get("page_size") or b.executor.get(
            "directions", {}
        ).get("page_alignment")
        if not args.page_size or args.prefix_tokens % args.page_size:
            raise BenchError("prefix must align to Engine page size")
        for value in metadata.values():
            for field in ("model_path", "tokenizer_path"):
                actual = (value.get("model") or {}).get(field)
                if actual and Path(actual).resolve() != Path(args.model).resolve():
                    raise BenchError("local tokenizer/model differs from Engine")
        material = [
            build_tokens(args.model, args.prefix_tokens, args.seed + index)[0]
            for index in range(args.cases)
        ]
        lanes = replay_plan(args)
        rng, schedule = random.Random(args.seed), []
        for block in range(args.blocks):
            names = list(args.arms)
            rng.shuffle(names)
            schedule.extend(
                {"block": block, "arm": name, "id": f"b{block}-{name}"}
                for name in names
            )
        manifest = {
            "args": vars(args),
            "metadata": metadata,
            "configuration": configuration,
            "metadata_budget": metadata_budget(args),
            "capabilities": {"a": a.cap, "b": b.cap, "router": router.capabilities},
            "module_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            "input_material": material,
            "input_mode": "synthetic_ledger",
            "output_reference": "source warm-cache generation; initial generation also retained to expose full-prefill versus cached-prefix numerical differences",
            "tool_result": "one fixed synthetic token after the published prefix, equal to its final token",
            "fixed_plan": lanes,
            "schedule": schedule,
            "execution_mode": "duration" if args.duration_s else "fixed_work",
            "acceptance": "duration mode executes a per-lane plan prefix until the admission deadline; plan exhaustion fails; fixed-work mode executes the complete plan; never pad by idle sleep",
            "pressure": "optional sampled host usage; only a recent unambiguous pool gauge above threshold asserts pressure",
        }
        write_json(root / "lifecycle-manifest.json", manifest)
        journal.emit("lifecycle_manifest", **manifest)
        arms, signatures, published = [], {}, {}
        for spec in schedule:
            cases = cases_for_arm(args, material, spec)
            working_set = logical_working_set(cases, args.page_size)
            if (
                args.host_capacity_tokens
                and working_set < args.host_capacity_tokens
                and not args.allow_small_working_set
            ):
                raise BenchError(
                    "deduplicated logical working set is smaller than declared host capacity"
                )
            reset_started = time.monotonic()
            previous = (
                published.get(spec["block"]) if args.reuse_published_sources else None
            )
            for engine in (b,) if previous else (a, b):
                flush_local(engine, spec["id"], args.timeout_s)
            reset_s = time.monotonic() - reset_started
            # Publication is sequential to keep source setup independent of the
            # bounded speculative-read pipeline under evaluation.
            source_started = time.monotonic()
            source_records = {}
            for case in cases:
                trial = spec["id"] + "-source-" + str(case["id"])
                identity = {
                    "id": trial,
                    "case": case["id"],
                    "prefix_sha256": digest(case["prefix"]),
                    "cache_salt": case["salt"],
                }
                if previous:
                    original = previous["records"][case["id"]]
                    journal.emit(
                        "lifecycle_publication",
                        **identity,
                        generation=original["generation"],
                        reused_from=original["id"],
                        reference_only=True,
                    )
                    continue
                initial = a.http.generate(
                    resume_prompt(case),
                    case["salt"],
                    args.output_tokens,
                    trial=trial,
                    seed=args.seed,
                    origin="publication",
                )
                # Match the cached-prefix execution shape used by restoration.
                # Full-prefill and warm-cache BF16 generation may differ at a
                # near tie even with unmodified GPU residency. Retain both.
                response = a.http.generate(
                    resume_prompt(case),
                    case["salt"],
                    args.output_tokens,
                    trial=trial,
                    seed=args.seed,
                    origin="publication_reference",
                )
                reference = signatures.setdefault((spec["block"], case["id"]), response)
                if not output_equal(reference, response):
                    raise BenchError(
                        "source reference output changed across policy arms"
                    )
                publication = publish(
                    a,
                    a.target(case["prefix"], case["salt"]),
                    len(case["prefix"]),
                    trial,
                )
                record = {
                    **identity,
                    "generation": response,
                    "initial_generation": initial,
                    "initial_reference_equal": output_equal(initial, response),
                    "publication": publication,
                    "reused_from": None,
                    "reference_only": False,
                }
                source_records[case["id"]] = record
                journal.emit("lifecycle_publication", **record)
            source_phase_s = time.monotonic() - source_started
            actual_source_s = 0 if previous else source_phase_s
            source_calls = 0 if previous else len(cases) * 2
            if args.reuse_published_sources:
                if not previous:
                    published[spec["block"]] = {
                        "arm": spec["id"],
                        "wall_s": actual_source_s,
                        "records": source_records,
                    }
                source_s = published[spec["block"]]["wall_s"] / len(args.arms)
            else:
                source_s = actual_source_s
            source_phase = {
                "source_generation_calls": source_calls,
                "source_execution_wall_s": actual_source_s,
                "source_phase_wall_s": source_phase_s,
                "source_reused_from": previous["arm"] if previous else None,
                "source_allocation_parts": len(args.arms)
                if args.reuse_published_sources
                else 1,
                "source_allocation_definition": "equal share of measured common block setup"
                if args.reuse_published_sources
                else "measured per-arm setup",
            }
            journal.emit(
                "lifecycle_source_phase",
                id=spec["id"],
                allocated_wall_s=source_s,
                **source_phase,
            )
            # Source A retains its natural GPU/host state. Destination B starts
            # cold and subsequently evicts naturally for the whole arm.
            controls = AsyncControls(
                args.control_workers or args.concurrency,
                args.control_backlog or args.concurrency * 2,
            )
            before = {
                "a": a.http.request("/server_info", optional=True),
                "b": b.http.request("/server_info", optional=True),
            }
            costs_before = {
                engine.http.url: cost_snapshot(engine, spec["id"]) for engine in (a, b)
            }
            work_started = time.monotonic()
            admission_deadline = (
                work_started + args.duration_s if args.duration_s else None
            )
            pressure = (
                PressureSampler(
                    HTTP(args.engine_b, journal, min(5, args.timeout_s), key),
                    journal,
                    args.pressure_threshold,
                )
                if args.demote_under_pressure
                else None
            )
            if pressure:
                pressure.thread.start()

            def lane_work(lane):
                def work(item, admitted):
                    return resume_task(
                        args,
                        a,
                        b,
                        router,
                        cases[item["case"]],
                        {
                            **spec,
                            **item,
                            "arm_id": spec["id"],
                            "trial": spec["id"] + "-" + item["id"],
                            "admitted_monotonic": admitted,
                        },
                        controls,
                        journal,
                        pressure,
                    )

                return execute_lane(lane, work, admission_deadline)

            try:
                with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    completed_lanes = list(pool.map(lane_work, lanes))
                    rows = [
                        row for lane_rows, _, _ in completed_lanes for row in lane_rows
                    ]
                inference_finished = time.monotonic()
            finally:
                drain = controls.drain()
                if pressure:
                    pressure.stop.set()
                    pressure.thread.join()
            wall_s = time.monotonic() - work_started
            arm = arm_summary(spec, rows, wall_s, drain, args, source_s, signatures)
            arm.update(
                lab_reset_s=reset_s,
                request_work_s=inference_finished - work_started,
                started_monotonic=work_started,
                completed_monotonic=work_started + wall_s,
                logical_working_set_tokens=working_set,
                host_capacity_tokens=args.host_capacity_tokens,
                duration_s=args.duration_s,
                admission_deadline_monotonic=admission_deadline,
                duration_budget_exhausted_lanes=[
                    i
                    for i, (_, exhausted, _) in enumerate(completed_lanes)
                    if exhausted
                ],
                lane_stopped_monotonic=[end for _, _, end in completed_lanes],
                plan_capacity_tasks=sum(map(len, lanes)),
                **source_phase,
            )
            arm["server_before"] = before
            arm["costs_before"] = costs_before
            arm["costs_after"] = {
                engine.http.url: cost_snapshot(engine, spec["id"]) for engine in (a, b)
            }
            arm["server_after"] = {
                "a": a.http.request("/server_info", optional=True),
                "b": b.http.request("/server_info", optional=True),
            }
            write_json(root / (spec["id"] + ".json"), arm)
            arms.append(arm)
            journal.emit("lifecycle_arm", **arm_metrics(arm))
            write_json(root / "lifecycle-summary.json", summarize(arms))
            print(json.dumps(arm_metrics(arm)), flush=True)
        return (
            0
            if all(
                arm["completed"] == arm["expected_effective_tasks"]
                and arm["valid_tasks"] == arm["planned_tasks"]
                and arm["meets_minimum_duration"]
                and arm["meets_minimum_resumes"]
                and not arm["drain"]["errors"]
                and not arm["duration_budget_exhausted_lanes"]
                for arm in arms
            )
            else 2
        )
    except Exception as exc:
        journal.emit(
            "lifecycle_failure", error=str(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        journal.file.close()
        write_json(
            root / "lifecycle-checksums.json",
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.iterdir()
                if p.is_file() and p.name != "lifecycle-checksums.json"
            },
        )


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (
        "router",
        "router-control-key-env",
        "engine-a",
        "engine-b",
        "model",
        "output",
    ):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--engine-key-env")
    cli.add_argument("--router-api-key-env")
    cli.add_argument("--via-router", action="store_true")
    cli.add_argument("--arms", type=lambda value: value.split(","), default=list(ARMS))
    cli.add_argument(
        "--reuse-published-sources",
        action="store_true",
        help="publish immutable source contexts once per block; allocate setup equally across arms",
    )
    cli.add_argument("--concurrency", type=int, default=8)
    cli.add_argument("--cases", type=int, default=64)
    cli.add_argument("--passes", type=int, default=2)
    cli.add_argument(
        "--duration-s",
        type=float,
        default=0,
        help="continuously admit work for this many seconds per arm; passes bounds the plan and must not exhaust early",
    )
    cli.add_argument("--blocks", type=int, default=5)
    cli.add_argument("--prefix-tokens", type=int, default=16384)
    cli.add_argument("--output-tokens", type=int, default=32)
    cli.add_argument("--tool-wait-s", type=float, default=3)
    cli.add_argument("--wait-jitter-s", type=float, default=0)
    cli.add_argument("--prediction-error-s", type=float, default=0)
    cli.add_argument("--fixed-lead-s", type=float, default=2.5)
    cli.add_argument("--shared-prefix-fraction", type=float, default=0)
    cli.add_argument("--branches-per-family", type=int, default=1)
    cli.add_argument("--salt-mode", choices=("family", "shared"), default="family")
    cli.add_argument("--cancel-fraction", type=float, default=0)
    cli.add_argument("--route-change-fraction", type=float, default=0)
    cli.add_argument("--control-workers", type=int, default=0)
    cli.add_argument("--control-backlog", type=int, default=0)
    cli.add_argument("--min-arm-seconds", type=float, default=0)
    cli.add_argument("--min-arm-resumes", type=int, default=0)
    cli.add_argument("--expected-mem-fraction", type=float, default=0.8)
    cli.add_argument(
        "--host-capacity-tokens",
        type=int,
        default=0,
        help="observed host capacity for deduplicated working-set pressure gate",
    )
    cli.add_argument("--allow-small-working-set", action="store_true")
    cli.add_argument(
        "--router-budget-file",
        help="protected Router JSON; only numeric metadata limits are retained",
    )
    cli.add_argument("--demote-under-pressure", action="store_true")
    cli.add_argument("--pressure-threshold", type=float, default=0.9)
    cli.add_argument("--timeout-s", type=float, default=120)
    cli.add_argument("--cleanup-timeout-s", type=float, default=120)
    cli.add_argument("--poll-ms", type=float, default=100)
    cli.add_argument("--seed", type=int, default=1729)
    cli.add_argument("--run-id")
    return cli


def validate(args):
    if (
        not args.arms
        or len(set(args.arms)) != len(args.arms)
        or set(args.arms) - set(ARMS)
    ):
        raise ValueError(
            "arms must be distinct on_demand,fixed_prefetch,lifecycle choices"
        )
    for name in (
        "concurrency",
        "cases",
        "passes",
        "blocks",
        "prefix_tokens",
        "output_tokens",
        "branches_per_family",
    ):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    for name in ("timeout_s", "cleanup_timeout_s", "poll_ms", "expected_mem_fraction"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name + " must be finite and positive")
    for name in (
        "tool_wait_s",
        "wait_jitter_s",
        "prediction_error_s",
        "fixed_lead_s",
        "min_arm_seconds",
        "duration_s",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(name + " must be finite and nonnegative")
    for name in (
        "shared_prefix_fraction",
        "cancel_fraction",
        "route_change_fraction",
        "pressure_threshold",
    ):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            raise ValueError(name + " must be in [0,1]")
    if (
        args.control_workers < 0
        or args.control_backlog < 0
        or args.min_arm_resumes < 0
        or args.host_capacity_tokens < 0
    ):
        raise ValueError("control bounds and minimum resumes must be nonnegative")
    if (args.control_backlog or args.concurrency * 2) < (
        args.control_workers or args.concurrency
    ):
        raise ValueError("control backlog must be at least worker count")
    if args.via_router and args.route_change_fraction:
        raise ValueError("route-change replay requires direct controlled destinations")
    if args.cancel_fraction == 1:
        raise ValueError("at least some effective tasks must be planned")
    if args.duration_s and args.cases < args.concurrency:
        raise ValueError("duration mode requires at least one case per client")


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        validate(args)
        return run(args)
    except Exception as exc:
        print("LIFECYCLE BENCHMARK FAILED: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
