"""Export a paused hybrid request, then restore it on a different native server.

This is a functional HTTP reproducer, not a training or throughput benchmark.
Run only against dedicated test servers. It never launches or kills a server.
The export command intentionally leaves its source paused after releasing the
request's private slots. Stop that source through its original process owner
before resuming on a newly started server. Snapshot files must be accessible at
one identical trusted filesystem path on all participating ranks and servers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def require(value, message):
    if not value:
        raise ValueError(message)


def file_ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


class HTTP:
    def __init__(self, url, out, timeout):
        self.url, self.out, self.timeout = url.rstrip("/"), Path(out), timeout
        self.out.mkdir(parents=True, exist_ok=True)
        self.refs = []
        self.counter = 0

    def call(self, route, payload=None):
        request = Request(
            self.url + route,
            data=json.dumps(payload, allow_nan=False).encode()
            if payload is not None
            else None,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
        except HTTPError as error:
            raw, status = error.read(), error.code
        self.counter += 1
        path = (
            self.out
            / f"http-{self.counter:04d}-{route.strip('/').replace('/', '-')}.json"
        )
        record = {
            "route": route,
            "payload": payload,
            "status": status,
            "elapsed_s": time.perf_counter() - started,
            "body": raw.decode("utf-8", errors="replace"),
        }
        write_json(path, record)
        self.refs.append(file_ref(path))
        if status < 200 or status >= 300:
            raise RuntimeError(f"{route}: HTTP {status}; immutable response: {path}")
        value = json.loads(raw)
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(f"{route}: engine error; immutable response: {path}")
        return value, record["elapsed_s"]


def token_output(row):
    meta = row.get("meta_info", {})
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list):
        raise ValueError("Actual SGLang token IDs and behavior logprobs are missing")
    ids, probs = [], []
    for entry in pairs:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError("Invalid native output_token_logprobs entry")
        prob, token = entry[:2]
        if (
            type(token) is not int
            or token < 0
            or type(prob) not in (int, float)
            or not math.isfinite(prob)
            or prob > 1e-5
        ):
            raise ValueError("Native token/logprob entry is not finite and valid")
        ids.append(token)
        probs.append(prob)
    reason = meta.get("finish_reason")
    if isinstance(reason, dict):
        reason = reason.get("type")
    spans = meta.get("weight_versions", meta.get("weight_version_spans"))
    version = meta.get("weight_version")
    if spans is not None:
        if not isinstance(spans, list) or not spans:
            raise ValueError("Invalid native weight version spans")
        cursor = 0
        versions = set()
        for span in spans:
            if (
                not isinstance(span, dict)
                or type(span.get("start")) is not int
                or type(span.get("end")) is not int
                or span["start"] != cursor
                or span["end"] < cursor
                or span["end"] > len(ids)
                or span.get("version") is None
            ):
                raise ValueError(
                    "Native version spans do not cover actual generated tokens"
                )
            cursor = span["end"]
            versions.add(span["version"])
        if cursor != len(ids) or len(versions) != 1:
            raise ValueError(
                "Attempt contains mixed or incompletely observed model versions"
            )
        observed = next(iter(versions))
        if version is not None and observed != version:
            raise ValueError(
                "Scalar version disagrees with original native version spans"
            )
        version = observed
    return {
        "token_ids": ids,
        "logprobs": probs,
        "stop_reason": reason,
        "weight_version": version,
        "raw_meta": meta,
    }


class SSE:
    def __init__(self, http, payload, stem):
        self.http, self.payload = http, payload
        self.path = http.out / (stem + ".jsonl")
        self.condition = threading.Condition()
        self.events, self.error, self.done = [], None, False
        self.start_s = time.perf_counter()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        request = Request(
            self.http.url + "/generate",
            data=json.dumps(self.payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with (
                urlopen(request, timeout=self.http.timeout) as response,
                self.path.open("w") as out,
            ):
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    row = json.loads(data)
                    if row.get("error"):
                        raise RuntimeError("SGLang stream returned an engine error")
                    event = {"monotonic_s": time.perf_counter(), "raw": row}
                    out.write(json.dumps(event, allow_nan=False) + "\n")
                    out.flush()
                    with self.condition:
                        self.events.append(event)
                        self.condition.notify_all()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            write_json(
                self.path.with_suffix(".error.json"),
                {"status": error.code, "body": body},
            )
            self.error = RuntimeError(f"generate HTTP {error.code}: {body}")
        except BaseException as error:
            self.error = error
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def wait_tokens(self, count, after=None):
        deadline = time.monotonic() + self.http.timeout
        with self.condition:
            while True:
                for event in self.events:
                    if after is not None and event["monotonic_s"] <= after:
                        continue
                    actual = event["raw"].get("meta_info", {}).get("completion_tokens")
                    if type(actual) is int and actual >= count:
                        return event
                if self.done:
                    raise RuntimeError(
                        f"Actual request ended before token {count}"
                    ) from self.error
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"No actual SSE token {count}")
                self.condition.wait(min(left, 1))

    def finish(self):
        self.thread.join(self.http.timeout)
        if self.thread.is_alive():
            raise TimeoutError("Actual SGLang generation did not finish")
        if self.error:
            raise RuntimeError("Actual SGLang generation failed") from self.error
        if not self.events:
            raise ValueError("No SGLang generation output")
        self.http.refs.append(file_ref(self.path))
        return self.events[-1]


def identity_from_inspect(value):
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Snapshot inspect did not return engine-generated identity")
    if not identity.get("model_path") or not identity.get(
        "actual_model_config_file_sha256"
    ):
        raise ValueError("Snapshot identity has no actual model_path/config SHA")
    if identity.get("weight_version") is None:
        raise ValueError("Actual engine weight_version is missing")
    return identity


def source_prefix(ack):
    ids = ack.get("output_ids")
    probs = ack.get("output_logprobs")
    if (
        not isinstance(ids, list)
        or not isinstance(probs, list)
        or len(ids) != len(probs)
    ):
        raise ValueError(
            "Snapshot ACK has no actual generated token/logprob checkpoint"
        )
    return {"token_ids": ids, "logprobs": probs}


def rank_rows(value, tp_size):
    rows = value.get("ranks", [])
    require(len(rows) == tp_size, "Missing TP rank responses")
    by_rank = {row.get("tp_rank"): row for row in rows}
    require(set(by_rank) == set(range(tp_size)), "Duplicate or foreign TP rank")
    require(
        all(row.get("status") == "PASS" for row in rows),
        "A native rank rejected the operation",
    )
    return by_rank


def inspect_engine(http):
    value, _ = http.call("/token_kv_snapshot", {"operation": "inspect"})
    require(value.get("status") == "PASS", "Unsupported engine")
    identity = identity_from_inspect(value)
    require(
        type(identity.get("tp_size")) is int and identity["tp_size"] > 0,
        "Missing native TP size",
    )
    rank_rows(value, identity["tp_size"])
    return value, identity


def checked_manifest(reference):
    require(isinstance(reference, dict), "Missing committed manifest reference")
    observed = file_ref(reference["path"])
    require(observed == reference, "Committed manifest bytes/SHA differ")
    return json.loads(Path(reference["path"]).read_text())


def export_source(args):
    http = HTTP(args.engine_url, args.out, args.timeout)
    initial, identity = inspect_engine(http)
    rid, snapshot_id = "source_" + uuid4().hex, "snapshot_" + uuid4().hex
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    stream = SSE(
        http,
        {
            "rid": rid,
            "text": prompt,
            "stream": True,
            "return_logprob": True,
            "logprob_start_len": -1,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        "source-generation",
    )
    stream.wait_tokens(args.interrupt_at)
    require(not stream.done, "Source finished before interruption")
    http.call("/pause_generation", {"mode": "in_place"})
    guard = {
        "identity": identity,
        "expected_weight_version": identity["weight_version"],
        "snapshot_id": snapshot_id,
        "storage_dir": str(args.storage_dir),
        "target_rid": rid,
    }
    exported, _ = http.call("/token_kv_snapshot", {**guard, "operation": "export"})
    require(
        exported.get("snapshot_committed") is True,
        "All TP ranks must be committed before release",
    )
    checked_manifest(exported.get("manifest_ref"))
    for row in rank_rows(exported, identity["tp_size"]).values():
        require(
            set(row.get("state_kinds", []))
            == {"attention_kv", "mamba_conv", "mamba_temporal"},
            "Incomplete hybrid state",
        )
    prefix = source_prefix(exported)
    require(
        len(prefix["token_ids"]) >= args.interrupt_at,
        "Export has no interrupted token history",
    )
    released, _ = http.call(
        "/token_kv_snapshot",
        {**guard, "operation": "release", "release_full_gpu_pool": False},
    )
    for row in rank_rows(released, identity["tp_size"]).values():
        require(
            row.get("old_slots_released") is True and row.get("detached") is True,
            "Source request still owns native slots",
        )
    result = {
        "status": "CHECKPOINT_READY_SOURCE_LEFT_PAUSED",
        "identity": identity,
        "source_rid": rid,
        "snapshot_id": snapshot_id,
        "storage_dir": str(args.storage_dir),
        "manifest_ref": exported["manifest_ref"],
        "source_binding": exported.get("source_binding"),
        "source_inspect": initial,
        "export": exported,
        "release": released,
        "prefix": prefix,
        "max_new_tokens": args.max_new_tokens,
        "whole_gpu_returned": False,
        "performance": "NOT_RUN",
    }
    write_json(args.out / "source-checkpoint.json", result)
    return result


def restore_target(args):
    source = json.loads(args.source_receipt.read_text())
    require(
        source.get("status") == "CHECKPOINT_READY_SOURCE_LEFT_PAUSED",
        "Source checkpoint did not finish",
    )
    checked_manifest(source.get("manifest_ref"))
    http = HTTP(args.engine_url, args.out, args.timeout)
    initial, identity = inspect_engine(http)
    require(identity == source["identity"], "Source/target identity differs")
    require(
        initial.get("engine_instance_id")
        and initial["engine_instance_id"]
        != source["source_inspect"]["engine_instance_id"],
        "Target must be a different engine instance",
    )
    rid = "target_" + uuid4().hex
    descriptor = {
        key: source[key]
        for key in ("snapshot_id", "storage_dir", "manifest_ref", "source_binding")
    }
    descriptor.update(
        identity=identity, expected_weight_version=identity["weight_version"]
    )
    stream = SSE(
        http,
        {"rid": rid, "stream": True, "token_kv_restore": descriptor},
        "target-generation",
    )
    last = stream.finish()
    output = token_output(last["raw"])
    prefix = source["prefix"]
    count = len(prefix["token_ids"])
    require(
        output["token_ids"][:count] == prefix["token_ids"]
        and output["logprobs"][:count] == prefix["logprobs"],
        "Previously generated tokens or logprobs changed",
    )
    require(
        count < len(output["token_ids"]) <= source["max_new_tokens"]
        and output["stop_reason"] not in (None, "abort", "aborted"),
        "Native continuation did not complete within its saved budget",
    )
    require(
        output["weight_version"] == identity["weight_version"],
        "Native output version is absent or differs",
    )
    inspected, after_identity = inspect_engine(http)
    require(after_identity == identity, "Target version changed during restore")
    source_rows = rank_rows(source["export"], identity["tp_size"])
    observed = []
    for rank, row in rank_rows(inspected, identity["tp_size"]).items():
        restored = row.get("portable_restore_by_rid", {}).get(rid)
        require(isinstance(restored, dict), "Missing actual native import receipt")
        require(
            restored.get("request_reconstructed") is True
            and restored.get("from_retained_cpu_record") is False
            and restored.get("loaded_from_snapshot") is True
            and restored.get("allocated_new_slots") is True
            and restored.get("digestmatch") is True,
            "Target did not reconstruct and read back the actual state",
        )
        for key in (
            "state_sha256",
            "state_inventory_sha256",
            "state_bytes",
            "computed_tokens",
        ):
            require(
                restored.get(key) is not None
                and restored[key] == source_rows[rank].get(key),
                "Restored state differs from its source TP partition",
            )
        metrics = row.get("resume_metrics_by_rid", {}).get(rid)
        require(
            isinstance(metrics, dict)
            and metrics.get("actual_input_counter_status")
            == "OBSERVED_COMPLETED_NATIVE_FORWARD"
            and metrics.get("actual_completed_forward_iters")
            and type(metrics.get("actual_processed_input_tokens")) is int
            and metrics["actual_processed_input_tokens"] > 0,
            "No completed native forward input observation",
        )
        require(
            metrics.get("actual_recomputed_tokens") == 0,
            "Native forward recomputed part of the committed prefix",
        )
        observed.append({"tp_rank": rank, "restore": restored, "metrics": metrics})
    require(
        len(
            {
                (
                    x["metrics"]["committed_prefix_tokens"],
                    x["metrics"]["actual_recomputed_tokens"],
                    x["metrics"]["actual_processed_input_tokens"],
                )
                for x in observed
            }
        )
        == 1,
        "TP ranks disagree on processed history",
    )
    result = {
        "status": "PASS",
        "scope": "FRESH_INSTANCE_NATIVE_REQUEST_RESTORE",
        "source_receipt": file_ref(args.source_receipt),
        "initial_inspect": initial,
        "target_inspect": inspected,
        "target_rid": rid,
        "output": output,
        "ranks": observed,
        "actual_recomputed_history_tokens_per_rank": observed[0]["metrics"][
            "actual_recomputed_tokens"
        ],
        "source_process_exit_verified": False,
        "whole_gpu_returned": False,
        "performance": "NOT_RUN",
    }
    write_json(args.out / "restore-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    export = sub.add_parser("export")
    restore = sub.add_parser("resume")
    for command in (export, restore):
        command.add_argument("--engine-url", required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--timeout", type=float, default=180)
    export.add_argument(
        "--prompt", default="Describe the integers from one to one hundred in order."
    )
    export.add_argument("--prompt-file", type=Path)
    export.add_argument("--storage-dir", type=Path, required=True)
    export.add_argument("--interrupt-at", type=int, default=8)
    export.add_argument("--max-new-tokens", type=int, default=128)
    export.add_argument("--temperature", type=float, default=0.7)
    restore.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(math.isfinite(args.timeout) and args.timeout > 0, "Invalid timeout")
    if args.phase == "export":
        require(
            0 < args.interrupt_at < args.max_new_tokens,
            "Interruption must leave some original generation budget",
        )
        require(
            args.storage_dir.is_absolute(),
            "Storage path must be absolute on all servers",
        )
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = export_source(args) if args.phase == "export" else restore_target(args)
    except Exception as error:
        write_json(
            args.out / "failure.json",
            {
                "status": "FAILED",
                "type": type(error).__name__,
                "error": str(error),
                "performance": "NOT_RUN",
            },
        )
        raise
    print(json.dumps({"status": result["status"], "evidence": str(args.out)}))


if __name__ == "__main__":
    main()
