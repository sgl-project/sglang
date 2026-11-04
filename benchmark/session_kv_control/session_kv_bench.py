#!/usr/bin/env python3
"""HTTP-only paired session-KV benchmark. No server/process/package management."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

from bench_inputs import estimate_kv_bytes_per_token, load_trace, tokenizer_identity
from bench_load import BackgroundLoad
from bench_statistics import PrefetchCostEMA
from bench_report import distribution, percentile, summarize, write_summary

ACTIVE = {"received", "queued", "accepted", "running", "cancelling"}
GOOD = {"completed"}
GROUPS = {"on_demand", "prefetch", "demote", "pin_demote", "pin_demote_prefetch"}
NEGATIVES = {"unused", "missing", "cancel", "route_change"}
CAPACITY_KEYS = (
    "model_path", "dtype", "kv_cache_dtype", "tp_size", "pp_size", "dp_size",
    "mem_fraction_static", "max_total_tokens", "max_total_num_tokens", "page_size", "context_length",
    "max_running_requests", "max_prefill_tokens", "chunked_prefill_size", "disable_overlap_schedule",
    "disable_cuda_graph", "disable_piecewise_cuda_graph", "disable_radix_cache", "disable_chunked_prefix_cache",
    "enable_hierarchical_cache", "hicache_ratio", "hicache_size",
    "hicache_mem_layout", "hicache_host_memory_mode", "hicache_io_backend",
    "hicache_write_policy", "hicache_storage_backend", "hicache_storage_prefetch_policy",
    "radix_tree_core", "radix_cache_backend", "attention_backend", "sampling_backend", "random_seed",
    "hicache_storage_key_scheme", "hicache_storage_head_partition", "hicache_storage_layer_partition",
    "hicache_storage_backend_extra_config",
)


class BenchError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def unwrap(reply):
    if isinstance(reply, dict) and "success" in reply:
        if not reply["success"]:
            raise BenchError("control rejected: " + json.dumps(reply.get("error", reply)))
        return reply["data"]
    return reply


def results(reply):
    data = unwrap(reply)
    return data.get("results", [data] if "update" in data else [])


def status(op):
    return op.get("update", {}).get("status", "unknown")


def intervals_size(spans, component="full"):
    """Union ranges; repeated component/branch results never inflate token use."""
    ranges = []
    for span in spans:
        c = span.get("coverage", span)
        if c.get("component") not in (None, component):
            continue
        if span.get("coverage_kind", "tokens") != "tokens":
            continue
        lo, hi = c.get("start_token", 0), c.get("end_token", 0)
        if isinstance(lo, int) and isinstance(hi, int) and 0 <= lo < hi:
            ranges.append((lo, hi))
    end, total = 0, 0
    for lo, hi in sorted(ranges):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total


def ready_tokens(op):
    pub = (op or {}).get("public_result") or {}
    if "ready_scope" in pub:
        return intervals_size(pub["ready_scope"])
    return intervals_size([
        x for x in (op or {}).get("update", {}).get("target_outcomes", [])
        if "destination_ready" in x.get("completed_steps", [])
    ])


def cache_details(meta, require=False):
    raw = meta.get("cached_tokens_details")
    if raw is None:
        if require and type(meta.get("cached_tokens")) is int and meta["cached_tokens"] == 0:
            return {"device": 0, "cpu": 0, "storage": 0, "details_present": False,
                    "provenance": "explicit_zero_total"}
        if meta.get("cached_tokens", 0) == 0 and not require:
            return {"device": 0, "cpu": 0, "storage": 0, "details_present": False}
        raise BenchError("cached_tokens_details missing; cannot identify hit tier")
    out = {"device": raw.get("device", 0), "cpu": raw.get("cpu", raw.get("host", 0)),
           "storage": raw.get("storage", 0), "details_present": True, "raw": raw}
    if any(type(out[k]) is not int or out[k] < 0 for k in ("device", "cpu", "storage")):
        raise BenchError("invalid tier token counts")
    total = sum(out[k] for k in ("device", "cpu", "storage"))
    if "cached_tokens" in meta and total != meta["cached_tokens"]:
        raise BenchError(f"tier accounting mismatch: {total} != {meta['cached_tokens']}")
    return out


def sse_objects(lines):
    """Parse complete SSE events, including comments and multiline data."""
    data = []
    for line in lines:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        line = line.rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        raise BenchError("truncated SSE event")


class Journal:
    def __init__(self, path):
        self.file = open(path, "x", buffering=1)
        self.lock = threading.Lock()

    def emit(self, kind, **fields):
        with self.lock:
            self.file.write(json.dumps({"kind": kind, "wall_time": time.time(),
                                       "monotonic": time.monotonic(), **fields}, ensure_ascii=False) + "\n")


class HTTP:
    def __init__(self, url, journal, timeout=60, key=None):
        self.url, self.journal, self.timeout = url.rstrip("/"), journal, timeout
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = "Bearer " + key

    def request(self, path, body=None, *, optional=False, trial=None):
        start, cpu = time.perf_counter(), time.thread_time()
        started_monotonic = time.monotonic()
        payload, code = None, None
        try:
            req = urllib.request.Request(self.url + path,
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                code = response.status
                raw = response.read().decode()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
            return payload
        except urllib.error.HTTPError as exc:
            code = exc.code
            payload = exc.read().decode(errors="replace")
            if optional and code in (404, 405, 503):
                return None
            raise BenchError(f"HTTP {self.url}{path}: {code}: {payload[:2048]}") from exc
        except (OSError, TimeoutError) as exc:
            if optional:
                return None
            raise BenchError(f"HTTP {self.url}{path}: {exc}") from exc
        finally:
            self.journal.emit("http", trial=trial, url=self.url, path=path,
                              request=body, response=payload, status_code=code,
                              elapsed_ms=(time.perf_counter()-start)*1000,
                              client_cpu_ms=(time.thread_time()-cpu)*1000,
                              started_monotonic=started_monotonic)

    def generate(self, tokens, salt, count, *, trial, seed, origin="resume", kv_context=None):
        body = {"input_ids": tokens, "cache_salt": salt, "stream": True,
                "sampling_params": {"temperature": 0, "max_new_tokens": count,
                                    "ignore_eos": True, "sampling_seed": seed}}
        if kv_context is not None:
            body["kv_context"] = kv_context
        start, cpu = time.perf_counter(), time.thread_time()
        started_monotonic = time.monotonic()
        first = None
        events, last, response_headers = [], None, {}
        try:
            req = urllib.request.Request(self.url + "/generate", data=json.dumps(body).encode(), headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_headers = {"x-sgl-worker-url": response.headers.get("x-sgl-worker-url")}
                # HTTPResponse iteration uses readline(): yield at an SSE line
                # boundary without requests.iter_lines' 512-byte chunk gate.
                for event in sse_objects(response):
                    now = time.perf_counter()
                    if now-start > self.timeout:
                        raise BenchError("generation exceeded total stream timeout")
                    if "error" in event:
                        raise BenchError("stream error: " + json.dumps(event))
                    if isinstance(event, list):
                        raise BenchError("unexpected batched generation response")
                    # Ignore empty heartbeat/prefill metadata. An empty text token still
                    # counts when output_ids or completion_tokens prove token emission.
                    emitted = event.get("output_ids") or event.get("meta_info", {}).get("completion_tokens", 0) > 0
                    if first is None and emitted:
                        first = now
                    events.append({"elapsed_ms": (now-start)*1000, "event": event})
                    last = event
            if last is None or first is None:
                raise BenchError("stream ended without an emitted token")
            if last.get("meta_info", {}).get("finish_reason") is None:
                raise BenchError("stream ended without finish_reason")
            if last.get("meta_info", {}).get("completion_tokens") != count:
                raise BenchError("generation did not produce requested fixed output length")
            if last.get("output_ids") is not None and len(last["output_ids"]) != count:
                raise BenchError("output_ids length disagrees with requested output length")
            out = {"ttft_ms": (first-start)*1000, "latency_ms": (time.perf_counter()-start)*1000,
                   "client_cpu_ms": (time.thread_time()-cpu)*1000, "output_ids": last.get("output_ids"),
                   "text": last.get("text"), "meta_info": last.get("meta_info", {}), "url": self.url,
                   "started_monotonic": started_monotonic, "completed_monotonic": time.monotonic()}
            out["response_headers"] = response_headers
            return out
        finally:
            self.journal.emit("generate", trial=trial, origin=origin, url=self.url,
                              request=body, events=events, started_monotonic=started_monotonic,
                              response_headers=response_headers,
                              elapsed_ms=(time.perf_counter()-start)*1000)


class Control:
    def __init__(self, http, caller, poll_ms=20, timeout=60):
        self.http, self.caller, self.poll_ms, self.timeout = http, caller, poll_ms, timeout
        self.refresh()

    def refresh(self):
        self.cap = unwrap(self.http.request("/v1/kv/capabilities"))
        if not self.cap.get("enabled"):
            raise BenchError(f"KV control disabled on {self.http.url}")
        self.executor = self.cap.get("executor", {})
        self.public = self.cap.get("public_protocol", {})
        self.backend_namespace = self.public.get("backend_namespace", self.executor.get("directions", {}).get("backend_namespace"))
        self.context = {k: self.cap[k] for k in ("engine_id", "engine_epoch", "namespace")}
        self.context["caller_id"] = self.caller

    def supports(self, action, src=None, dst=None):
        if action in self.public.get("actions", {}):
            if "1.0" not in self.public["actions"][action]:
                return False
            paths = self.public.get("paths", {}).get(action)
            return src is None or paths is None or [src, dst] in [list(x) for x in paths]
        return action in self.executor.get("actions", [])

    def prefetch_action(self):
        if self.supports("sglang.prefetch", "storage", "host"):
            return "sglang.prefetch"
        if self.supports("sglang.copy_to"):
            paths = self.executor.get("directions", {}).get("copy_to")
            if paths is None or ["storage", "host"] in [list(x) for x in paths]:
                return "sglang.copy_to"
        raise BenchError("capabilities do not advertise executable storage→host prefetch")

    def command(self, command, trial=None):
        return unwrap(self.http.request("/v1/kv/control", command, trial=trial))

    def target(self, tokens, salt):
        # TokensTarget is accepted by both old and new actions. Keep the raw
        # logical salt: Engine applies namespace encoding exactly once.
        return {"kind": "tokens", "namespace": self.cap["namespace"], "token_ids": tokens,
                "cache_salt": salt, "layout_id": self.executor["layout_id"]}

    def submit(self, action, payload, trial, action_id=None):
        message = "bench-" + uuid.uuid4().hex
        reply = self.command({"kind": "submit", "context": self.context,
                              "envelope": {"protocol_version": "0.1", "message_id": message,
                                           "actions": [{"action_id": action_id or "one", "action_type": action,
                                                        "action_version": "1.0", "payload": payload}]}}, trial)
        ops = reply.get("results", [])
        if len(ops) != 1:
            raise BenchError("expected one operation: " + json.dumps(reply))
        return ops[0]

    def wait(self, op, trial, observe=None, timing=None):
        start = time.perf_counter()
        cancelled_at = None
        while True:
            if observe:
                observe(op)
            if status(op) not in ACTIVE and op.get("update", {}).get("quiescent"):
                if timing is not None:
                    timing["completion_observed_monotonic"] = time.monotonic()
                    timing["completion_elapsed_ms"] = (timing["completion_observed_monotonic"]-timing["started_monotonic"])*1000
                    timing["completion_interval_ms"] = [
                        (timing["last_nonterminal_rpc_started_monotonic"]-timing["started_monotonic"])*1000,
                        timing["completion_elapsed_ms"]]
                self.command({"kind": "ack", "context": self.context,
                              "message_id": op["ref"]["message_id"]}, trial)
                if cancelled_at is not None:
                    raise BenchError("operation timeout; cancellation drained: " + json.dumps(op))
                return op
            if time.perf_counter()-start > self.timeout:
                if cancelled_at is None:
                    # Keep cancellation ownership until a quiescent receipt.
                    op = self.command({"kind": "cancel", "ref": op["ref"]}, trial)
                    cancelled_at = time.perf_counter()
                    self.http.journal.emit("control_timeout", trial=trial, cancel=op)
                    continue
                if time.perf_counter()-cancelled_at > self.timeout:
                    self.http.journal.emit("control_drain_unknown", trial=trial, operation=op,
                                           reason="second bounded cancellation window expired")
                    raise BenchError("operation timeout; cancellation remains non-quiescent: " + json.dumps(op))
            time.sleep(self.poll_ms/1000)
            rpc_started = time.monotonic()
            reply = self.command({"kind": "query", "context": self.context,
                                  "message_id": op["ref"]["message_id"]}, trial)
            matches = [x for x in reply.get("results", []) if x["ref"]["operation_id"] == op["ref"]["operation_id"]]
            if len(matches) != 1:
                raise BenchError("query lost operation")
            op = matches[0]
            if timing is not None and status(op) in ACTIVE:
                timing["last_nonterminal_rpc_started_monotonic"] = rpc_started

    def transfer(self, what, target, src, dst, trial, observe=None):
        action = {"prefetch": "sglang.prefetch", "demote": "sglang.demote", "publish": "sglang.copy_to"}[what]
        if what == "prefetch":
            action = self.prefetch_action()
        elif not self.supports(action, src["tier"], dst["tier"]):
            raise BenchError(f"capabilities do not advertise {action} on requested path")
        if what == "publish":
            paths = self.executor.get("directions", {}).get("copy_to")
            if paths is not None and [src["tier"], dst["tier"]] not in [list(x) for x in paths]:
                raise BenchError("capabilities do not advertise executable storage publication")
        if action == "sglang.copy_to":
            payload = {"target": target, "src": src, "dst": dst}
        else:
            src, dst = copy.deepcopy(src), copy.deepcopy(dst)
            for location in (src, dst):
                if location["tier"] == "storage" and self.public.get("backend_namespace_selection"):
                    location["backend_namespace"] = self.backend_namespace
            payload = {"target": target, "source": src, "destination": dst,
                       "allow_partial": False, "timeout_ms": round(self.timeout*1000)}
        start = time.monotonic()
        timing = {"started_monotonic": start, "last_nonterminal_rpc_started_monotonic": start,
                  "definition": "client-observed completion including queue, I/O and host registration; excludes ack; not pure I/O"}
        op = self.submit(action, payload, trial)
        timing["submission_returned_monotonic"] = time.monotonic()
        if observe:
            observe(op)
        op = self.wait(op, trial, observe, timing)
        return {"action": action, "elapsed_ms": (time.monotonic()-start)*1000,
                "operation": op, "timing": timing, "io_time_ms": None, "physical_io_bytes": None}

    def residency(self, target, tier, trial):
        return self.command({"kind": "residency", "context": self.context,
                             "target": target, "location": {"tier": tier}, "limit": 256}, trial)

    def pin(self, target, claim, mode, revision, ttl, trial):
        payload = {"target": target, "location": {"tier": "host"}, "claim_id": claim,
                   "mode": mode, "revision": revision, "required_protection": "eviction_exclusion"}
        if mode != "release":
            payload["ttl_seconds"] = ttl
        return self.wait(self.submit("sglang.pin", payload, trial), trial)

    def claim(self, claim_id, trial):
        return self.command({"kind": "claim", "context": self.context, "claim_id": claim_id}, trial)


def sleep_until(deadline):
    while time.monotonic() < deadline:
        time.sleep(max(0, min(.05, deadline-time.monotonic())))


def require_completed(transfer, expected_tokens, label):
    op = transfer.get("operation", transfer)
    if status(op) not in GOOD or not op.get("update", {}).get("quiescent"):
        raise BenchError(f"{label} did not complete: {json.dumps(op)}")
    if ready_tokens(op) < expected_tokens:
        raise BenchError(f"{label} lacks full ready_scope ({ready_tokens(op)}/{expected_tokens})")


def require_pin_success(op, expected_tokens, claim_id, owner_epoch):
    """Protection is independent of transfer readiness; P4b may have no ready_scope."""
    claim = (op.get("public_result") or {}).get("claim") or {}
    if (status(op) != "completed" or not op.get("update", {}).get("quiescent")
            or claim.get("state") != "active" or claim.get("tombstone")
            or claim.get("claim_id") != claim_id or claim.get("owner_epoch") != owner_epoch
            or claim.get("effective_protection") != "eviction_exclusion"
            or not isinstance(claim.get("remaining_ttl"), (int, float))
            or not math.isfinite(claim["remaining_ttl"]) or claim["remaining_ttl"] <= 0
            or intervals_size(claim.get("actual_scope", [])) < expected_tokens
            or (claim.get("location") or {}).get("tier") != "host"):
        raise BenchError("PIN lacks active, bounded, owner-confirmed protection for full scope: " + json.dumps(op))
    return claim


def require_pin_released(claim, claim_id, owner_epoch):
    if (claim.get("claim_id") != claim_id or claim.get("owner_epoch") != owner_epoch
            or claim.get("state") not in ("released", "expired", "invalidated")
            or not claim.get("tombstone") or claim.get("effective_protection") is not None
            or (claim.get("remaining_ttl") is not None and claim["remaining_ttl"] > 0)):
        raise BenchError("PIN release query lacks terminal unprotected claim: " + json.dumps(claim))


def apply_host_pressure(args, b, prefix, trial):
    rows = []
    count, length = getattr(args, "host_pressure_prefixes", 0), getattr(args, "host_pressure_tokens", 0)
    if not count or not length:
        return rows
    tokens = (prefix*((length+len(prefix)-1)//len(prefix)))[:length]
    for index in range(count):
        salt = "pressure-" + digest([args.run_id, trial, index])[:40]
        row = {"index": index, "tokens": length, "salt": salt}
        row["generation"] = b.http.generate(tokens, salt, 1, trial=trial, seed=args.seed, origin="host_pressure")
        target = b.target(tokens, salt)
        if b.supports("sglang.demote", "gpu", "host"):
            row["transfer"] = b.transfer("demote", target, {"tier": "gpu"}, {"tier": "host"}, trial)
            row["mode"] = "demote_to_host"
        else:
            row["transfer"] = b.transfer("publish", target, {"tier": "gpu"}, {"tier": "host"}, trial)
            row["mode"] = "host_copy_with_gpu_residency_possible"
        require_completed(row["transfer"], length, "host pressure materialization")
        row["host_residency"] = b.residency(target, "host", trial)
        row["host_tokens_observed"] = intervals_size([x.get("descriptor", {}) for x in row["host_residency"].get("items", [])])
        if row["host_tokens_observed"] < length:
            raise BenchError("host pressure copy was not present at residency observation")
        rows.append(row)
    return rows


def get_metadata(http):
    data = {}
    for name, path in (("model", "/model_info"), ("server", "/server_info"),
                       ("l3_layout", "/get_hicache_l3_cache_layout")):
        data[name] = http.request(path, optional=True)
        if data[name] is None and name in ("model", "server"):
            data[name] = http.request("/get_" + name + "_info")
    return data


def fingerprint(metadata):
    info = metadata["server"] or {}
    args = info.get("server_args", info)
    output = {k: args.get(k) for k in CAPACITY_KEYS}
    extra = output.get("hicache_storage_backend_extra_config")
    if isinstance(extra, str):
        try:
            output["hicache_storage_backend_extra_config"] = json.loads(extra)
        except json.JSONDecodeError:
            pass
    return output


def snapshot(http, trial, phase):
    try:
        data = http.request("/metrics", optional=True, trial=trial)
    except Exception as exc:
        data = {"unavailable": str(exc)}
    if isinstance(data, str):
        data = "\n".join(x for x in data.splitlines()
                         if not x.startswith("#") and any(k in x.lower() for k in ("hicache", "kv_control", "storage", "prefetch", "cache_host")))
    http.journal.emit("metrics", trial=trial, phase=phase, url=http.url, metrics=data)


def build_tokens(model, length, seed):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rng = random.Random(seed)
    words = "amber river copper ledger orchard spiral basalt paper window maple vector session cache tool model network number value memory result".split()
    text = "Read the reference ledger below and answer the question after the tool result.\n"
    text += "\n".join(f"entry {i}: " + " ".join(rng.choices(words, k=24)) + f" {rng.randrange(1_000_000)}" for i in range(length//10+40))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < length:
        raise BenchError("generated token material too short")
    suffix = tokenizer.encode("\nTool result: all ledger entries are valid. Answer with a short description of the ledger.\nAnswer:", add_special_tokens=False)
    return tokens[:length], suffix


def one_trial(args, a, b, material, spec, journal):
    trial = {"kind": "trial", **spec, "valid": False, "invalid_reasons": []}
    prefix, suffix = material
    salt = "bench-" + digest([args.run_id, args.seed, spec["id"]])[:40]
    trial.update(cache_salt=salt, prefix_sha256=digest(prefix), resumed_input_tokens=len(prefix)+len(suffix))
    bytes_per_token = getattr(args, "estimated_kv_bytes_per_token", None)
    trial["estimated_full_kv_bytes"] = len(prefix)*bytes_per_token if bytes_per_token else None
    source_target, dest_target = a.target(prefix, salt), b.target(prefix, salt)
    group = spec["group"]
    shared, shared_lock = {}, threading.Lock()
    pin_acquired, future, load = False, None, None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def observe(op):
        with shared_lock:
            shared["operation"] = copy.deepcopy(op)
            shared["observed_at"] = time.monotonic()
            shared.setdefault("first_observed_at", shared["observed_at"])
            if ready_tokens(op) >= len(prefix) and status(op) == "completed":
                shared.setdefault("first_ready_observed_at", shared["observed_at"])

    def work(wait_start):
        data = {}
        nonlocal pin_acquired
        if "demote" in group:
            data["demote"] = a.transfer("demote", source_target, {"tier": "gpu"}, {"tier": "host"}, spec["id"])
            require_completed(data["demote"], len(prefix), "source demote")
        if group.startswith("pin_"):
            # Even a malformed/unknown receipt may have created a bounded claim;
            # attempt explicit release during cleanup and query its actual state.
            pin_acquired = True
            data["pin"] = a.pin(source_target, spec["id"], "acquire", 1,
                                max(args.pin_ttl_s, spec["tool_wait_s"]+args.timeout_s+5), spec["id"])
            require_pin_success(data["pin"], len(prefix), spec["id"], a.context["engine_epoch"])
        if "prefetch" in group or group in NEGATIVES:
            sleep_until(wait_start + max(0, spec["tool_wait_s"]-spec["lead_s"]))
            target = copy.deepcopy(dest_target)
            if group == "missing":
                target["cache_salt"] += "-missing"
            if group == "cancel":
                action = b.prefetch_action()
                payload = {"target": target}
                if action == "sglang.prefetch":
                    location = {"tier": "storage", "backend": args.storage_backend}
                    if b.public.get("backend_namespace_selection"):
                        location["backend_namespace"] = b.backend_namespace
                    payload.update(source=location, destination={"tier": "host"}, allow_partial=False, timeout_ms=round(args.timeout_s*1000))
                else:
                    payload.update(src={"tier": "storage", "backend": args.storage_backend}, dst={"tier": "host"})
                op = b.submit(action, payload, spec["id"])
                observe(op)
                cancelled = b.command({"kind": "cancel", "ref": op["ref"]}, spec["id"])
                data["cancel_receipt"] = cancelled
                data["prefetch"] = {"action": action, "operation": b.wait(cancelled, spec["id"], observe)}
            else:
                data["prefetch"] = b.transfer("prefetch", target,
                    {"tier": "storage", "backend": args.storage_backend}, {"tier": "host"}, spec["id"], observe)
        return data

    start = time.monotonic()
    try:
        if args.cold_mode == "flush":
            body = b.http.request("/flush_cache?timeout="+str(args.timeout_s), {}, trial=spec["id"])
            if not isinstance(body, str) or not body.startswith("Cache flushed."):
                raise BenchError("flush response did not certify local cache reset")
            b.refresh()
            dest_target = b.target(prefix, salt)
        trial["host_pressure"] = apply_host_pressure(args, b, prefix, spec["id"])
        # Unique salt also excludes B session/radix content from all previous trials.
        for tier in ("gpu", "host"):
            residency = b.residency(dest_target, tier, spec["id"])
            if residency.get("items"):
                raise BenchError(f"destination {tier} already contains target")
        snapshot(b.http, spec["id"], "before")
        start = time.monotonic()
        trial["source"] = a.http.generate(prefix, salt, 1, trial=spec["id"], seed=args.seed, origin="source")
        trial["source_cache"] = cache_details(trial["source"]["meta_info"])
        if sum(trial["source_cache"][k] for k in ("device", "cpu", "storage")):
            raise BenchError("source initial computation unexpectedly hit cache")
        trial["publication"] = a.transfer("publish", source_target,
            {"tier": "gpu"}, {"tier": "storage", "backend": args.storage_backend}, spec["id"])
        require_completed(trial["publication"], len(prefix), "storage publication")
        wait_start = time.monotonic()
        trial["initial_and_publication_ms"] = (wait_start-start)*1000
        if getattr(args, "background_concurrency", 0):
            bg_length = getattr(args, "background_prefix_tokens", 256)
            bg_tokens = (prefix*((bg_length+len(prefix)-1)//len(prefix)))[:bg_length]
            load = BackgroundLoad(b.http, bg_tokens, getattr(args, "background_output_tokens", 16),
                                  args.background_concurrency, spec["id"], args.seed, args.run_id).start()
        future = executor.submit(work, wait_start)
        sleep_until(wait_start + spec["tool_wait_s"])
        trial["actual_tool_wait_ms"] = (time.monotonic()-wait_start)*1000
        trial["effective_requested_lead_s"] = min(spec["tool_wait_s"], spec["lead_s"])
        with shared_lock:
            trial["prefetch_at_resume"] = copy.deepcopy(shared)
        at_resume = trial["prefetch_at_resume"].get("operation", {})
        first_observed = trial["prefetch_at_resume"].get("first_observed_at")
        trial["control_acceptance_lead_ms"] = ((time.monotonic()-first_observed)*1000 if first_observed is not None else None)
        trial["prefetch_ready_by_resume"] = ready_tokens(at_resume) >= len(prefix) and status(at_resume) == "completed"
        trial["prefetch_ready_before_resume"] = trial["prefetch_ready_by_resume"]
        if group not in ("unused", "missing", "cancel"):
            resume_http = a.http if group == "route_change" else b.http
            trial["resume"] = resume_http.generate(prefix+suffix, salt, args.output_tokens,
                                                   trial=spec["id"], seed=args.seed)
            trial["resume_cache"] = cache_details(trial["resume"]["meta_info"], require=True)
            hits = trial["resume_cache"]
            trial["repeated_prefill_tokens"] = max(0, len(prefix)-min(len(prefix), sum(hits[k] for k in ("device", "cpu", "storage"))))
            if group in ("on_demand", "demote", "pin_demote"):
                if hits["device"] or hits["cpu"] or hits["storage"] < len(prefix):
                    trial["invalid_reasons"].append("on-demand baseline did not prove full storage reuse with zero device/CPU hits")
            elif group != "route_change":
                if hits["device"]:
                    trial["invalid_reasons"].append("destination device hit unexpectedly bypassed L3→L2 preparation")
                if trial["prefetch_ready_by_resume"] and hits["cpu"] < len(prefix):
                    trial["invalid_reasons"].append("completed prefetch was not reused from CPU/host")
                if hits["cpu"]+hits["storage"] < len(prefix):
                    trial["invalid_reasons"].append("resume did not reuse full published prefix")
        trial["complete_cycle_ms"] = (time.monotonic()-start)*1000
        if load:
            trial["background_load"] = load.stop()
            load = None
            if trial["background_load"]["total_failures"]:
                trial["invalid_reasons"].append("background inference requests failed; configured load was not maintained")
        trial["background"] = future.result(timeout=args.timeout_s+spec["tool_wait_s"]+2)
        operation = trial["background"].get("prefetch", {}).get("operation", {})
        if operation:
            prepared = ready_tokens(operation)
            completed_at = trial["background"]["prefetch"].get("timing", {}).get("completion_observed_monotonic")
            if completed_at is not None and trial.get("resume"):
                trial["prefetch_ready_before_resume"] = (prepared >= len(prefix) and status(operation) == "completed"
                    and completed_at <= trial["resume"]["started_monotonic"])
            if (trial.get("resume") and group not in NEGATIVES and trial["prefetch_ready_before_resume"]
                    and trial.get("resume_cache", {}).get("cpu", 0) < len(prefix)):
                trial["invalid_reasons"].append("completed preparation before actual request send was not fully reused from host")
            cpu_used = trial.get("resume_cache", {}).get("cpu", 0) if group != "route_change" else 0
            trial["prepared_tokens"] = prepared
            # In-flight at arrival is not unused work. Demand/control may share
            # or race a physical transfer and host attribution alone cannot tell.
            ambiguous = bool(trial.get("resume") and group != "route_change" and
                             not trial["prefetch_ready_before_resume"])
            trial["prefetch_usage_ambiguous"] = ambiguous
            trial["used_prepared_tokens"] = None if ambiguous else min(prepared, cpu_used)
            trial["unused_prepared_tokens"] = None if ambiguous else prepared-trial["used_prepared_tokens"]
            trial["used_prefetch_ratio"] = trial["used_prepared_tokens"]/prepared if prepared and not ambiguous else None
            trial["prefetch_command_completed"] = status(operation) == "completed"
            trial["prefetch_outcome_status"] = status(operation)
            trial["prefetch_attribution"] = ("ambiguous_at_arrival" if ambiguous else
                "no_resume_or_routed_elsewhere" if group in ("unused", "route_change") else "ready_before_resume_host_reuse")
            if group not in NEGATIVES and status(operation) != "completed":
                trial["invalid_reasons"].append("prefetch control did not complete: " + status(operation))
            if group == "missing" and (status(operation) not in ("missing", "partial") or prepared):
                trial["invalid_reasons"].append("missing-storage negative did not report missing content with zero ready scope")
            if group == "cancel" and not operation.get("update", {}).get("quiescent"):
                trial["invalid_reasons"].append("cancel did not drain")
            if group == "cancel":
                trial["cancel_raced_completion"] = status(operation) == "completed"
        trial["valid"] = not trial["invalid_reasons"]
    except Exception as exc:
        trial["invalid_reasons"].append(f"{type(exc).__name__}: {exc}")
        trial["exception"] = traceback.format_exc()
        if future:
            try:
                trial["background"] = future.result(timeout=args.timeout_s+spec["tool_wait_s"]+2)
            except Exception as background_error:
                trial["background_error"] = str(background_error)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if load:
            try:
                trial["background_load"] = load.stop()
                if trial["background_load"]["total_failures"]:
                    trial["invalid_reasons"].append("background inference request failures during drain")
                    trial["valid"] = False
            except Exception as exc:
                trial["invalid_reasons"].append("background drain: " + str(exc))
                trial["valid"] = False
        if pin_acquired:
            try:
                trial["pin_release"] = a.pin(source_target, spec["id"], "release", 2, None, spec["id"])
                if status(trial["pin_release"]) != "completed":
                    trial["invalid_reasons"].append("PIN release failed")
                    trial["valid"] = False
                trial["pin_release_claim"] = a.claim(spec["id"], spec["id"])
                require_pin_released(trial["pin_release_claim"], spec["id"], a.context["engine_epoch"])
            except Exception as exc:
                trial["invalid_reasons"].append("PIN release: " + str(exc))
                trial["valid"] = False
        snapshot(b.http, spec["id"], "after")
        trial["quiescent_cycle_ms"] = (time.monotonic()-start)*1000
    return trial


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    journal = Journal(output / "raw.jsonl")
    try:
        return run_with_journal(args, output, journal)
    finally:
        journal.file.close()


def observe_cost_row(costs, cost_key, row, bytes_per_token):
    if row.get("kind") != "trial":
        return
    transfer = row.get("background", {}).get("prefetch", {})
    operation = transfer.get("operation", {})
    if status(operation) != "completed" or not operation.get("update", {}).get("quiescent"):
        return
    if ready_tokens(operation) < row["prefix_tokens"]:
        return
    elapsed = transfer.get("timing", {}).get("completion_elapsed_ms", transfer.get("elapsed_ms"))
    if elapsed is None:
        return
    no_resume = row.get("group") == "unused"
    ready = row.get("prefetch_ready_before_resume", row.get("prefetch_ready_by_resume", False))
    costs.observe(cost_key(row["prefix_tokens"]), elapsed,
        row["prefix_tokens"]*bytes_per_token if bytes_per_token else None,
        contaminated=bool(row.get("resume") and not ready and not no_resume))


def run_with_journal(args, output, journal):
    args.run_id = args.run_id or uuid.uuid4().hex
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    a_http, b_http = (HTTP(url, journal, args.timeout_s, key) for url in (args.engine_a, args.engine_b))
    if a_http.url == b_http.url:
        raise BenchError("source and destination must be distinct engines")
    meta = {"a": get_metadata(a_http), "b": get_metadata(b_http)}
    for metadata in meta.values():
        for field in ("model_path", "tokenizer_path"):
            observed = (metadata.get("model") or {}).get(field)
            if observed and Path(observed).resolve() != Path(args.model).resolve():
                raise BenchError(f"local input {field} differs from serving Engine: {observed}")
    a, b = (Control(http, "bench-"+args.run_id, args.poll_ms, args.timeout_s) for http in (a_http, b_http))
    if a.cap["engine_id"] == b.cap["engine_id"]:
        raise BenchError("engine IDs must differ")
    for field in ("namespace", "cache_salt_encoding", "weight_version"):
        if a.cap.get(field) != b.cap.get(field):
            raise BenchError(f"incompatible source/destination {field}")
    for field in ("layout_id", "components"):
        if a.executor.get(field) != b.executor.get(field):
            raise BenchError(f"incompatible source/destination {field}")
    if a.backend_namespace != b.backend_namespace:
        raise BenchError("source/destination backend_namespace differs")
    differences = {k: [fingerprint(meta["a"])[k], fingerprint(meta["b"])[k]] for k in CAPACITY_KEYS
                   if fingerprint(meta["a"])[k] != fingerprint(meta["b"])[k]}
    if differences:
        raise BenchError("engine model/load/cache configuration differs: " + json.dumps(differences))
    if a.cap.get("policies") or b.cap.get("policies"):
        raise BenchError("disable implicit KV policies for isolated explicit-control comparison")
    if args.router:
        router = HTTP(args.router, journal, args.timeout_s, key)
        meta["router"] = router.request("/v1/kv/capabilities", optional=True)
        # Engine-targeted traffic stays fixed. B3 selection requires a Router's
        # verified attribution contract; never infer worker identity from HTTP 200.
    if args.reference_url:
        ref = HTTP(args.reference_url, journal, args.timeout_s, key)
        meta["reference"] = get_metadata(ref)
        meta["reference"]["capabilities"] = ref.request("/v1/kv/capabilities", optional=True)
    selected = args.groups.split(",")
    if selected == ["auto"]:
        selected = ["on_demand", "prefetch"]
        if a.supports("sglang.demote", "gpu", "host"):
            selected.append("demote")
            if a.supports("sglang.pin"):
                selected += ["pin_demote", "pin_demote_prefetch"]
    unknown = set(selected)-GROUPS
    if unknown:
        raise BenchError("unknown groups: " + str(unknown))
    if "on_demand" not in selected:
        raise BenchError("on_demand is required for a paired comparison")
    negative = args.negative.split(",") if args.negative else []
    if set(negative)-NEGATIVES:
        raise BenchError("unknown negative scenarios")
    page = a.public.get("prefix", {}).get("page_size") or a.executor.get("page_size", a.executor.get("directions", {}).get("page_alignment", 1))
    trace_path = getattr(args, "trace_jsonl", None)
    if trace_path:
        cases = load_trace(trace_path, args.model)
    else:
        identity = tokenizer_identity(args.model)
        cases = []
        for length in args.prefix_lengths:
            prefix, suffix = build_tokens(args.model, length, args.seed+length)
            cases.append({"id": f"synthetic-{length}", "prefix": prefix, "suffix": suffix,
                          "tool_wait_s": None, "lead_s": None,
                          "provenance": {"input_mode": "synthetic_ledger", "declared_origin": "generated_by_harness",
                              "model_tokenizer": identity, "seed": args.seed+length,
                              "prefix_tokens_sha256": digest(prefix), "suffix_tokens_sha256": digest(suffix)}})
    for length in [len(case["prefix"]) for case in cases]:
        if length > 65536:
            raise BenchError("input prefix exceeds control target token budget")
        if length % page:
            raise BenchError(f"prefix length {length} must align to page size {page}")
    pressure_length = getattr(args, "host_pressure_tokens", 0)
    if pressure_length % page:
        raise BenchError("host pressure token length must align to page size")
    context_limit = fingerprint(meta["b"]).get("context_length")
    if context_limit:
        if any(len(case["prefix"])+len(case["suffix"])+args.output_tokens > context_limit for case in cases):
            raise BenchError("input trace plus output exceeds serving context capacity")
        if pressure_length+1 > context_limit or (getattr(args, "background_concurrency", 0) and
            args.background_prefix_tokens+args.background_output_tokens > context_limit):
            raise BenchError("pressure/background request exceeds context capacity")
    estimate = estimate_kv_bytes_per_token(args.model, getattr(args, "kv_bytes_per_token", None))
    args.estimated_kv_bytes_per_token = estimate["bytes_per_token"]
    costs = PrefetchCostEMA(getattr(args, "ema_alpha", .25), getattr(args, "ema_initial_ms", 500), getattr(args, "ema_safety_factor", 1.5))
    def cost_key(length):
        return json.dumps([a.cap["namespace"], a.backend_namespace, a.executor.get("layout_id"), length])
    history_meta = None
    history_path = getattr(args, "cost_history_jsonl", None)
    if history_path:
        history_raw = Path(history_path).read_bytes()
        history = [json.loads(x) for x in history_raw.splitlines()]
        historical_manifest = next((x for x in history if x.get("kind") == "manifest"), {})
        historical_cap = historical_manifest.get("capabilities", {}).get("a", {})
        historical_backend = historical_cap.get("public_protocol", {}).get("backend_namespace")
        if historical_cap.get("namespace") != a.cap["namespace"] or historical_backend != a.backend_namespace:
            raise BenchError("cost history namespace/backend_namespace incompatible")
        if historical_cap.get("executor", {}).get("layout_id") != a.executor.get("layout_id"):
            raise BenchError("cost history layout incompatible")
        for row in history:
            observe_cost_row(costs, cost_key, row, estimate["bytes_per_token"])
        history_meta = {"file": str(Path(history_path).resolve()), "sha256": hashlib.sha256(history_raw).hexdigest(),
                        "mode": "historical observations only; contaminated costs stored separately"}
    bandwidth = None
    if getattr(args, "bandwidth_profile_json", None):
        path = Path(args.bandwidth_profile_json).resolve()
        raw = path.read_bytes()
        bandwidth = {"source_file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "declared_profile": json.loads(raw),
                     "applied_by_harness": False, "independently_measured": False}
    manifest = {"args": vars(args), "metadata": meta, "capabilities": {"a": a.cap, "b": b.cap},
                "groups": selected, "negative_scenarios": negative, "comparison_fingerprint": fingerprint(meta["b"]),
                "python": sys.version, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "load": {"foreground": "one source then one resumed request", "background_concurrency": getattr(args, "background_concurrency", 0),
                         "host_pressure_prefixes": getattr(args, "host_pressure_prefixes", 0), "host_pressure_tokens": pressure_length,
                         "bandwidth_profile": bandwidth},
                "inputs": [{"id": c["id"], "prefix_tokens": len(c["prefix"]), "suffix_tokens": len(c["suffix"]),
                            "provenance": c["provenance"]} for c in cases],
                "kv_byte_estimate": estimate, "cost_history": history_meta,
                "module_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("bench_*.py")},
                "clock": "client monotonic; no cross-process timestamp subtraction"}
    Path(output, "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    journal.emit("manifest", **manifest)
    planned, rng = [], random.Random(args.seed)
    for case in cases:
        length = len(case["prefix"])
        waits = [case["tool_wait_s"]] if case["tool_wait_s"] is not None else args.tool_waits
        leads = [case["lead_s"]] if case["lead_s"] is not None else args.lead_times
        if getattr(args, "lead_policy", "fixed") == "ema":
            leads = [None]
        for wait, lead, repetition in itertools.product(waits, leads, range(args.repetitions)):
            case_key = digest(case["id"])[:8]
            pair = f"p{length}-{case_key}-w{wait}-l{lead}-r{repetition}"
            groups = selected + (negative if repetition == 0 else [])
            groups = groups.copy()
            rng.shuffle(groups)
            for group in groups:
                planned.append({"id": pair+"-"+group, "pair": pair, "input_case": case["id"], "prefix_tokens": length,
                                "tool_wait_s": wait, "lead_s": lead, "repetition": repetition, "group": group,
                                "input_provenance": case["provenance"]})
    Path(output, "schedule.json").write_text(json.dumps(planned, indent=2)+"\n")
    records, outputs, sources = [], {}, {}
    materials = {c["id"]: (c["prefix"], c["suffix"]) for c in cases}
    for case_id, (prefix, suffix) in materials.items():
        for repetition in range(args.warmup_repetitions):
            for name, http in (("a", a.http), ("b", b.http)):
                warmup_id = f"warmup-{name}-{digest(case_id)[:8]}-{repetition}"
                salt = "warmup-" + digest([args.run_id, warmup_id])[:40]
                response = http.generate(prefix+suffix, salt, args.output_tokens,
                                         trial=warmup_id, seed=args.seed, origin="warmup")
                journal.emit("warmup", id=warmup_id, engine=name, response=response)
                if sum(cache_details(response["meta_info"])[k] for k in ("device", "cpu", "storage")):
                    raise BenchError("warmup unexpectedly hit cache")
    pair_leads = {}
    for spec in planned:
        if getattr(args, "lead_policy", "fixed") == "ema":
            selection = pair_leads.setdefault(spec["pair"], costs.choose(cost_key(spec["prefix_tokens"]), spec["tool_wait_s"],
                spec["prefix_tokens"]*estimate["bytes_per_token"] if estimate["bytes_per_token"] else None))
            spec = {**spec, "lead_s": selection["lead_s"], "lead_selection": dict(selection)}
            journal.emit("lead_selection", trial=spec["id"], pair=spec["pair"], selection=selection)
        trial = one_trial(args, a, b, materials[spec["input_case"]], spec, journal)
        if trial.get("resume"):
            generation = trial["resume"]
            signature = {"output_ids": generation["output_ids"], "text": generation["text"]}
            expected = outputs.setdefault(spec["input_case"], signature)
            trial["deterministic_output_equal"] = signature == expected
            if signature != expected:
                trial["invalid_reasons"].append("deterministic resumed output differs across conditions")
                trial["valid"] = False
        if trial.get("source"):
            signature = {k: trial["source"][k] for k in ("output_ids", "text")}
            trial["deterministic_source_equal"] = signature == sources.setdefault(spec["input_case"], signature)
            if not trial["deterministic_source_equal"]:
                trial["invalid_reasons"].append("deterministic source output differs across conditions")
                trial["valid"] = False
        observe_cost_row(costs, cost_key, trial, estimate["bytes_per_token"])
        trial["cost_ema_after_trial"] = copy.deepcopy(costs.states)
        journal.emit("trial", **{k: v for k, v in trial.items() if k != "kind"})
        records.append(trial)
        write_summary(output, records)
        print(json.dumps({"trial": spec["id"], "valid": trial["valid"], "reasons": trial["invalid_reasons"],
                          "ttft_ms": trial.get("resume", {}).get("ttft_ms"),
                          "cache": trial.get("resume_cache")}, ensure_ascii=False), flush=True)
    summary = write_summary(output, records)
    Path(output, "prefetch-cost-ema.json").write_text(json.dumps({"states": costs.states,
        "byte_estimate": estimate, "definition": "client-observed completion; actual I/O unknown; uncontended samples only inform lead choice"}, indent=2)+"\n")
    return 0 if summary["valid_attempts"] == summary["attempts"] else 2


def csv_numbers(text, cast):
    values = [cast(x) for x in text.split(",")]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated finite nonnegative numbers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--engine-a", required=True)
    bench.add_argument("--engine-b", required=True)
    bench.add_argument("--router", help="record Router capabilities; routing remains fixed to verified B")
    bench.add_argument("--reference-url", help="record stock server configuration separately; not pooled with B")
    bench.add_argument("--model", required=True, help="local tokenizer directory")
    bench.add_argument("--trace-jsonl", help="local token_prefix/prefix_text + tool_suffix + tool_wait_s records")
    bench.add_argument("--output", required=True, help="new run directory (raw.jsonl must not already exist)")
    bench.add_argument("--seed", type=int, default=1729)
    bench.add_argument("--run-id")
    bench.add_argument("--prefix-lengths", type=lambda x: csv_numbers(x, int), default=[2048, 8192, 16384])
    bench.add_argument("--tool-waits", type=lambda x: csv_numbers(x, float), default=[0.05, 1.0])
    bench.add_argument("--lead-times", type=lambda x: csv_numbers(x, float), default=[0.0, 0.05, 0.5])
    bench.add_argument("--lead-policy", choices=["fixed", "ema"], default="fixed")
    bench.add_argument("--ema-alpha", type=float, default=.25)
    bench.add_argument("--ema-initial-ms", type=float, default=500)
    bench.add_argument("--ema-safety-factor", type=float, default=1.5)
    bench.add_argument("--cost-history-jsonl", help="seed EMA from compatible prior raw.jsonl; late costs kept separate")
    bench.add_argument("--kv-bytes-per-token", type=int, help="explicit logical FULL KV byte estimate; never physical I/O")
    bench.add_argument("--repetitions", type=int, default=5)
    bench.add_argument("--warmup-repetitions", type=int, default=1,
                       help="separate full-length warmups per engine and prefix size")
    bench.add_argument("--groups", default="auto")
    bench.add_argument("--negative", default="unused,missing,cancel,route_change")
    bench.add_argument("--output-tokens", type=int, default=32)
    bench.add_argument("--storage-backend", default="mooncake")
    bench.add_argument("--cold-mode", choices=["salt", "flush"], default="salt")
    bench.add_argument("--poll-ms", type=float, default=20)
    bench.add_argument("--timeout-s", type=float, default=60)
    bench.add_argument("--pin-ttl-s", type=float, default=120)
    bench.add_argument("--background-concurrency", type=int, default=0, help="opt-in closed-loop inference workers on B during wait+resume")
    bench.add_argument("--background-prefix-tokens", type=int, default=256)
    bench.add_argument("--background-output-tokens", type=int, default=16)
    bench.add_argument("--host-pressure-prefixes", type=int, default=0, help="opt-in unrelated prefixes materialized on B host before each trial")
    bench.add_argument("--host-pressure-tokens", type=int, default=0)
    bench.add_argument("--bandwidth-profile-json", help="record external bandwidth configuration/provenance only; no shaping")
    bench.add_argument("--api-key-env", help="environment variable holding API key; value is never journaled")
    report = sub.add_parser("summarize")
    report.add_argument("directory")
    args = parser.parse_args()
    if args.command == "summarize":
        records = [json.loads(x) for x in Path(args.directory, "raw.jsonl").read_text().splitlines()]
        print(json.dumps(write_summary(args.directory, records), indent=2))
        return 0
    if args.repetitions < 1 or args.output_tokens < 1 or args.warmup_repetitions < 0 or any(x < 1 for x in args.prefix_lengths):
        parser.error("repetitions, output tokens and prefix lengths must be positive")
    if not all(math.isfinite(x) and x > 0 for x in (args.poll_ms, args.timeout_s, args.pin_ttl_s)):
        parser.error("poll/timeout must be positive")
    if (args.background_concurrency < 0 or args.background_prefix_tokens < 1 or args.background_output_tokens < 2
            or args.host_pressure_prefixes < 0 or args.host_pressure_tokens < 0
            or bool(args.host_pressure_prefixes) != bool(args.host_pressure_tokens)):
        parser.error("invalid background or host-pressure configuration; pressure requires both count and tokens")
    try:
        return run(args)
    except Exception as exc:
        print(f"BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
