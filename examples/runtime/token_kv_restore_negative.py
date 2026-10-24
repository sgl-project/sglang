from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from token_kv_restore import (
    HTTP,
    checked_manifest,
    inspect_engine,
    rank_rows,
    require,
    write_json,
)

EXPECTED = [
    (
        "invalid_weight_version",
        "Descriptor weight version differs from snapshot identity",
    ),
    ("invalid_cache_layout", "model/version/cache layout identity mismatch"),
    ("invalid_source_binding", "Snapshot trajectory/source binding differs"),
    (
        "invalid_manifest_ref",
        "Descriptor manifest reference differs from committed snapshot",
    ),
]


def _ranks(identity: dict) -> int:
    for key in ("tensor_parallel_size", "tp_size", "world_size"):
        value = identity.get(key)
        if isinstance(value, int):
            return value
    raise ValueError("inspect identity does not expose tensor-parallel rank count")


def _descriptor(receipt: dict) -> dict:
    identity = copy.deepcopy(receipt["identity"])
    descriptor = {
        "snapshot_id": receipt["snapshot_id"],
        "storage_dir": receipt["storage_dir"],
        "manifest_ref": copy.deepcopy(receipt["manifest_ref"]),
        "source_binding": copy.deepcopy(receipt["source_binding"]),
        "identity": identity,
        "expected_weight_version": identity["weight_version"],
    }
    return descriptor


def _different_sha256(value: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64,
        "manifest_ref.sha256 must be 64 hex characters",
    )
    int(value, 16)
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


def _read_body(response, deadline: float) -> bytes:
    chunks, size = [], 0
    while size < 65537:
        if response.isclosed():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("negative response deadline reached")
        owner = response
        for _ in range(4):
            sock = getattr(getattr(owner, "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(remaining)
                break
            owner = getattr(owner, "fp", None)
        else:
            raise RuntimeError("cannot bind the response socket to its deadline")
        chunk = response.read1(min(4096, 65537 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def _bounded_post(
    engine_url: str, payload: dict, timeout: float
) -> tuple[int, str, str, str, float]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        engine_url.rstrip("/") + "/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = _read_body(response, started + timeout)
    except urllib.error.HTTPError as error:
        status = error.code
        content_type = error.headers.get("Content-Type", "") if error.headers else ""
        with error:
            raw = _read_body(error, started + timeout)
    body = raw[:65536].decode("utf-8", errors="replace")
    return (
        status,
        content_type,
        body,
        "TRUNCATED" if len(raw) > 65536 else "COMPLETE",
        time.monotonic() - started,
    )


def _records(body: str, content_type: str) -> list[dict]:
    records: list[dict] = []
    if "application/json" in content_type.lower() or (body.lstrip()[:1] in "{["):
        records.append(json.loads(body))
        return records
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            item = line[5:].strip()
            if item and item != "[DONE]":
                records.append(json.loads(item))
    require(records, "response contained neither JSON nor SSE data records")
    return records


def _error_text(records: list[dict]) -> str:
    texts: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("error", "message") and isinstance(child, str):
                    texts.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(records)
    return "\n".join(texts)


def _has_generation(records: list[dict]) -> bool:
    for record in records:
        for key in ("token_ids", "output_ids", "text", "output_token_logprobs"):
            if record.get(key):
                return True
        nested = [v for v in record.values() if isinstance(v, dict)]
        if nested and _has_generation(nested):
            return True
    return False


def _rank_evidence(raw_inspect: object) -> tuple[str, list[str]]:
    rows = rank_rows(raw_inspect, 2)
    if any(
        not isinstance(row.get(field), dict)
        for row in rows.values()
        for field in ("portable_restore_by_rid", "resume_metrics_by_rid")
    ):
        return "UNKNOWN", []
    return "PRESENT", [
        str(row[field])
        for row in rows.values()
        for field in ("portable_restore_by_rid", "resume_metrics_by_rid")
    ]


def _validate_engine_role(
    receipt: dict, raw_inspect: dict, identity: dict, role: str
) -> None:
    source_instance = receipt["source_inspect"]["engine_instance_id"]
    current_instance = raw_inspect.get("engine_instance_id")
    require(
        isinstance(source_instance, str) and isinstance(current_instance, str),
        "engine instance identity missing",
    )
    if role == "target":
        require(
            source_instance != current_instance,
            "source and target engine instances must be distinct",
        )
        return
    require(role == "released-source", "unknown explicit engine role")
    require(
        source_instance == current_instance,
        "released-source must be the original engine instance",
    )
    release = receipt["release"]
    require(
        release.get("operation") == "release"
        and release.get("snapshot_id") == receipt["snapshot_id"],
        "release differs from the exported snapshot",
    )
    require(
        release.get("identity") == identity
        and receipt["export"].get("snapshot_committed") is True,
        "released source identity or export commit differs",
    )
    exported = rank_rows(receipt["export"], 2)
    for rank, row in rank_rows(release, 2).items():
        require(
            row.get("rid") == receipt["source_rid"]
            and row.get("old_slots_released") is True
            and row.get("detached") is True,
            "all original TP request slots must be released and detached",
        )
        require(row.get("identity") == identity, "release rank identity differs")
        for key in (
            "state_sha256",
            "state_inventory_sha256",
            "computed_tokens",
            "state_bytes",
        ):
            require(
                row.get(key) is not None and row[key] == exported[rank].get(key),
                "release rank differs from exported state",
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HTTP-negative correctness entry for SGLang portable snapshot restore"
    )
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument(
        "--engine-role", choices=("target", "released-source"), default="target"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--total-timeout", type=float, default=150)
    args = parser.parse_args()
    require(0 < args.timeout <= 30, "--timeout must be in (0, 30]")
    require(0 < args.total_timeout <= 180, "--total-timeout must be in (0, 180]")
    args.out.mkdir(parents=True, exist_ok=False)

    try:
        receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
        require(
            receipt.get("status") == "CHECKPOINT_READY_SOURCE_LEFT_PAUSED",
            "source receipt status mismatch",
        )
        checked_manifest(receipt["manifest_ref"])

        deadline = time.monotonic() + args.total_timeout
        http = HTTP(args.engine_url, args.out, args.timeout)
        raw_inspect, identity = inspect_engine(http)
        require(
            identity == receipt["identity"],
            "target identity differs from source identity",
        )
        require(_ranks(identity) == 2, "actual inspect identity is not TP2")
        source_instance = receipt["source_inspect"]["engine_instance_id"]
        target_instance = raw_inspect.get("engine_instance_id")
        require(
            isinstance(source_instance, str) and isinstance(target_instance, str),
            "engine instance identity missing",
        )
        _validate_engine_role(receipt, raw_inspect, identity, args.engine_role)

        initial_observation, initial_native = _rank_evidence(raw_inspect)
        results = []
        passed = 0
        for index, (name, expected_cause) in enumerate(EXPECTED):
            descriptor = _descriptor(receipt)
            if index == 0:
                descriptor["expected_weight_version"] = (
                    str(descriptor["expected_weight_version"]) + "_NEGATIVE"
                )
            elif index == 1:
                require(
                    isinstance(descriptor["identity"].get("cache_layout"), dict),
                    "cache_layout must be a dict",
                )
                descriptor["identity"]["cache_layout"]["_negative_probe"] = True
            elif index == 2:
                require(
                    descriptor["source_binding"] is None
                    or isinstance(descriptor["source_binding"], dict),
                    "source_binding must be null or a dict",
                )
                descriptor["source_binding"] = {
                    **(descriptor["source_binding"] or {}),
                    "_negative_probe": True,
                }
            else:
                descriptor["manifest_ref"]["sha256"] = _different_sha256(
                    descriptor["manifest_ref"]["sha256"]
                )
            payload = {
                "rid": str(uuid.uuid4()),
                "stream": True,
                "token_kv_restore": descriptor,
            }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {
                    "arm": name,
                    "outcome": "INCOMPLETE",
                    "reason": "total timeout exhausted before POST",
                }
                results.append(result)
                break
            try:
                status, content_type, body, read_state, elapsed = _bounded_post(
                    args.engine_url, payload, min(args.timeout, remaining)
                )
            except Exception as error:
                results.append(
                    {
                        "arm": name,
                        "outcome": "INCOMPLETE",
                        "reason": f"{type(error).__name__}: {error}",
                        "payload": payload,
                    }
                )
                break
            write_json(
                args.out / f"{index + 1:02d}-{name}.json",
                dict(
                    payload=payload,
                    status=status,
                    content_type=content_type,
                    body=body,
                    read_state=read_state,
                    elapsed_s=elapsed,
                    expected_cause=expected_cause,
                ),
            )
            try:
                records = _records(body, content_type)
                error_text = _error_text(records)
                rejected = expected_cause in error_text
                generated = _has_generation(records)
                require(read_state == "COMPLETE", f"truncated response in {name}")
                require(
                    any(
                        isinstance(r, dict)
                        and (r.get("error") or r.get("object") == "error")
                        for r in records
                    ),
                    f"response has no structured error in {name}",
                )
                require(not generated, f"unexpected generated token content in {name}")
                require(rejected, f"expected rejection cause absent in {name}")
                outcome = "PASS_HTTP_REJECTION"
                passed += 1
            except Exception as error:
                results.append(
                    {
                        "arm": name,
                        "outcome": "FAIL",
                        "reason": f"{type(error).__name__}: {error}",
                        "status": status,
                        "content_type": content_type,
                        "body": body,
                        "read_state": read_state,
                        "elapsed": elapsed,
                        "expected_cause": expected_cause,
                        "payload": payload,
                    }
                )
                break
            results.append(
                {
                    "arm": name,
                    "outcome": outcome,
                    "status": status,
                    "content_type": content_type,
                    "body": body,
                    "read_state": read_state,
                    "elapsed": elapsed,
                    "expected_cause": expected_cause,
                    "payload": payload,
                }
            )

        if passed == len(EXPECTED):
            try:
                require(
                    time.monotonic() < deadline,
                    "total deadline reached before final inspect",
                )
                http.timeout = min(args.timeout, deadline - time.monotonic())
                final_raw, final_identity = inspect_engine(http)
                require(
                    final_identity == identity,
                    "final inspect identity differs from initial identity",
                )
                evidence_state, evidence = _rank_evidence(final_raw)
                invalid_rids = (
                    [
                        record["payload"]["rid"]
                        for record in results
                        if str(evidence).find(record["payload"]["rid"]) >= 0
                    ]
                    if evidence_state == "PRESENT"
                    else []
                )
                require(
                    not invalid_rids,
                    f"invalid RID appeared in rank evidence: {invalid_rids}",
                )
                require(
                    initial_observation == "PRESENT" and evidence_state == "PRESENT",
                    "initial/final native evidence is missing",
                )
                require(
                    evidence == initial_native,
                    "native restore/input-counter observations changed during incompatible requests",
                )
                final = {
                    "status": "PASS_HTTP_REJECTIONS_ONLY"
                    if passed == len(EXPECTED)
                    else "INCOMPLETE",
                    "rejection_proofs": f"{passed}/{len(EXPECTED)}",
                    "native_import_observation": evidence_state,
                    "corrupt_disk_HTTP": "NOT_RUN",
                    "valid_restore": "NOT_RUN_BY_THIS_ENTRY",
                    "final_identity_fresh": final_identity == identity,
                    "engine_role": args.engine_role,
                    "native_observations_unchanged": evidence == initial_native,
                    "engine_instance_id": target_instance,
                    "arms": results,
                }
            except Exception as error:
                final = {
                    "status": "INCOMPLETE",
                    "reason": f"final inspect failed: {type(error).__name__}: {error}",
                    "arms": results,
                }
        else:
            final = {
                "status": "INCOMPLETE"
                if all(r["outcome"] != "FAIL" for r in results)
                else "FAIL",
                "rejection_proofs": f"{passed}/{len(EXPECTED)}",
                "corrupt_disk_HTTP": "NOT_RUN",
                "valid_restore": "NOT_RUN_BY_THIS_ENTRY",
                "arms": results,
            }
        write_json(args.out / "result.json", final)
        print(
            json.dumps(
                {
                    "status": final["status"],
                    "rejection_proofs": final.get("rejection_proofs"),
                    "native_import_observation": final.get("native_import_observation"),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
        return 0 if final["status"] == "PASS_HTTP_REJECTIONS_ONLY" else 1
    except Exception as error:
        final = {
            "status": "INCOMPLETE",
            "reason": f"{type(error).__name__}: {error}",
            "corrupt_disk_HTTP": "NOT_RUN",
            "valid_restore": "NOT_RUN_BY_THIS_ENTRY",
        }
        write_json(args.out / "result.json", final)
        print(json.dumps(final))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
