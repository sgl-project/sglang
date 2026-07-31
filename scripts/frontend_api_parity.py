#!/usr/bin/env python3
"""Capture and compare Python/Rust frontend API behavior.

The harness deliberately treats the Python HTTP response as the reference rather
than re-implementing protocol expectations in the comparator. A case file defines
requests plus the small set of unstable response paths (IDs/timestamps) that may be
normalized. Everything else is compared recursively and in order, including SSE
event ordering.

Typical workflow:

  # 1. Run the Python frontend, then capture its behavior.
  python3 scripts/frontend_api_parity.py capture \
      --base-url http://127.0.0.1:30000 \
      --cases test/fixtures/frontend_api_parity/stage1_generation.json \
      --output /tmp/python-stage1.json \
      --label python \
      --revision "$(git rev-parse HEAD)"

  # 2. Run the Rust frontend, then compare it with the saved Python behavior.
  python3 scripts/frontend_api_parity.py compare \
      --base-url http://127.0.0.1:30000 \
      --cases test/fixtures/frontend_api_parity/stage1_generation.json \
      --reference /tmp/python-stage1.json \
      --write-actual /tmp/rust-stage1.json \
      --label rust

  # 3. Recompare two saved snapshots without live servers.
  python3 scripts/frontend_api_parity.py diff \
      --reference /tmp/python-stage1.json \
      --actual /tmp/rust-stage1.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from typing import Any, Iterable

import requests

SCHEMA_VERSION = 1
NORMALIZED_VALUE = "<frontend-api-parity:normalized>"
MAX_REPORTED_DIFFERENCES = 100


class HarnessError(RuntimeError):
    """A bad case, fixture, response, or invocation."""


def load_json(path: str | os.PathLike[str]) -> Any:
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"failed to load JSON from {path}: {exc}") from exc


def _require_type(value: Any, expected: type, context: str) -> None:
    if not isinstance(value, expected):
        raise HarnessError(
            f"{context} must be {expected.__name__}, got {type(value).__name__}"
        )


def validate_case_spec(spec: Any) -> dict[str, Any]:
    _require_type(spec, dict, "case spec")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError(
            f"case spec schema_version must be {SCHEMA_VERSION}, "
            f"got {spec.get('schema_version')!r}"
        )

    cases = spec.get("cases")
    _require_type(cases, list, "case spec.cases")
    if not cases:
        raise HarnessError("case spec.cases must not be empty")

    names: set[str] = set()
    for index, case in enumerate(cases):
        context = f"case[{index}]"
        _require_type(case, dict, context)
        name = case.get("name")
        if not isinstance(name, str) or not name:
            raise HarnessError(f"{context}.name must be a non-empty string")
        if name in names:
            raise HarnessError(f"duplicate case name: {name!r}")
        names.add(name)

        request = case.get("request")
        _require_type(request, dict, f"{context}.request")
        method = request.get("method", "POST")
        if not isinstance(method, str) or not method:
            raise HarnessError(f"{context}.request.method must be a string")
        path = request.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise HarnessError(f"{context}.request.path must start with '/'")
        if "headers" in request:
            _require_type(request["headers"], dict, f"{context}.request.headers")
            if not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in request["headers"].items()
            ):
                raise HarnessError(
                    f"{context}.request.headers keys and values must be strings"
                )

        response_mode = case.get("response_mode", "auto")
        if response_mode not in {"auto", "json", "sse", "text"}:
            raise HarnessError(
                f"{context}.response_mode must be auto, json, sse, or text"
            )

        paths = case.get("normalize_paths", [])
        _require_type(paths, list, f"{context}.normalize_paths")
        if not all(isinstance(path, str) and path.startswith("/") for path in paths):
            raise HarnessError(
                f"{context}.normalize_paths entries must be JSON pointers"
            )

        tolerance = case.get("float_tolerance", 0.0)
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or tolerance < 0
            or not math.isfinite(float(tolerance))
        ):
            raise HarnessError(
                f"{context}.float_tolerance must be a finite non-negative number"
            )

    return spec


def load_case_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_case_spec(load_json(path))


def case_spec_digest(spec: dict[str, Any]) -> str:
    encoded = json.dumps(
        spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_cases(
    spec: dict[str, Any], selected_names: Iterable[str] | None
) -> list[dict[str, Any]]:
    selected = list(selected_names or [])
    if not selected:
        return list(spec["cases"])

    by_name = {case["name"]: case for case in spec["cases"]}
    unknown = [name for name in selected if name not in by_name]
    if unknown:
        raise HarnessError(f"unknown case name(s): {', '.join(unknown)}")
    return [by_name[name] for name in selected]


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_sse_lines(lines: Iterable[str | bytes]) -> list[dict[str, Any]]:
    """Parse an SSE byte/line stream while preserving event order and wire fields."""

    events: list[dict[str, Any]] = []
    event_name: str | None = None
    event_id: str | None = None
    retry: int | str | None = None
    data_lines: list[str] = []
    saw_field = False

    def dispatch() -> None:
        nonlocal event_name, event_id, retry, data_lines, saw_field
        if not saw_field:
            return
        event: dict[str, Any] = {"data": _json_or_text("\n".join(data_lines))}
        if event_name is not None:
            event["event"] = event_name
        if event_id is not None:
            event["id"] = event_id
        if retry is not None:
            event["retry"] = retry
        events.append(event)
        event_name = None
        event_id = None
        retry = None
        data_lines = []
        saw_field = False

    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = raw_line
        line = line.rstrip("\r\n")
        if not line:
            dispatch()
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if not separator:
            value = ""

        if field == "data":
            data_lines.append(value)
            saw_field = True
        elif field == "event":
            event_name = value
            saw_field = True
        elif field == "id":
            event_id = value
            saw_field = True
        elif field == "retry":
            retry = int(value) if value.isdigit() else value
            saw_field = True

    dispatch()
    return events


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def capture_response(response: Any, response_mode: str) -> dict[str, Any]:
    content_type = _media_type(response.headers.get("content-type"))
    captured: dict[str, Any] = {
        "status": response.status_code,
        "content_type": content_type,
    }

    is_sse = content_type == "text/event-stream"
    if is_sse:
        captured["events"] = parse_sse_lines(response.iter_lines(decode_unicode=True))
        return captured

    text = response.text
    should_try_json = response_mode in {"auto", "json", "sse"} or (
        content_type == "application/json" or content_type.endswith("+json")
    )
    if should_try_json:
        try:
            captured["body"] = response.json()
            return captured
        except (ValueError, json.JSONDecodeError):
            pass

    captured["text"] = text
    return captured


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise HarnessError(f"normalization path must be a JSON pointer: {pointer!r}")
    return [
        token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")
    ]


def _replace_pointer(node: Any, tokens: list[str], depth: int = 0) -> int:
    token = tokens[depth]
    last = depth == len(tokens) - 1
    replacements = 0

    if isinstance(node, dict):
        keys = list(node) if token == "*" else ([token] if token in node else [])
        for key in keys:
            if last:
                node[key] = NORMALIZED_VALUE
                replacements += 1
            else:
                replacements += _replace_pointer(node[key], tokens, depth + 1)
        return replacements

    if isinstance(node, list):
        if token == "*":
            indexes = range(len(node))
        else:
            try:
                index = int(token)
            except ValueError:
                return 0
            indexes = [index] if 0 <= index < len(node) else []
        for index in indexes:
            if last:
                node[index] = NORMALIZED_VALUE
                replacements += 1
            else:
                replacements += _replace_pointer(node[index], tokens, depth + 1)
        return replacements

    return 0


def normalize_response(
    response: dict[str, Any],
    paths: Iterable[str],
    *,
    require_match: bool = True,
) -> dict[str, Any]:
    normalized = copy.deepcopy(response)
    for pointer in paths:
        tokens = _pointer_tokens(pointer)
        count = _replace_pointer(normalized, tokens)
        if count == 0 and require_match:
            raise HarnessError(
                f"normalization path {pointer!r} matched no response values"
            )
    return normalized


def parse_cli_headers(values: Iterable[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition("=")
        if not separator or not name:
            raise HarnessError(f"header must use NAME=VALUE syntax, got {value!r}")
        headers[name] = header_value
    return headers


def execute_case(
    session: Any,
    base_url: str,
    case: dict[str, Any],
    cli_headers: dict[str, str],
    timeout: float,
    *,
    require_normalization_match: bool = True,
) -> dict[str, Any]:
    request = case["request"]
    headers = dict(cli_headers)
    headers.update(request.get("headers", {}))
    url = base_url.rstrip("/") + request["path"]
    response = session.request(
        request.get("method", "POST").upper(),
        url,
        headers=headers or None,
        json=request.get("json") if "json" in request else None,
        params=request.get("params"),
        timeout=timeout,
        stream=case.get("response_mode") == "sse",
    )
    captured = capture_response(response, case.get("response_mode", "auto"))
    return normalize_response(
        captured,
        case.get("normalize_paths", []),
        require_match=require_normalization_match,
    )


def capture_snapshot(
    spec: dict[str, Any],
    *,
    base_url: str,
    label: str,
    revision: str | None,
    cli_headers: dict[str, str],
    timeout: float,
    selected_names: Iterable[str] | None = None,
    require_normalization_match: bool = True,
    session: Any = requests,
) -> dict[str, Any]:
    cases = select_cases(spec, selected_names)
    results = []
    for case in cases:
        results.append(
            {
                "name": case["name"],
                "request": copy.deepcopy(case["request"]),
                "float_tolerance": float(case.get("float_tolerance", 0.0)),
                "response": execute_case(
                    session,
                    base_url,
                    case,
                    cli_headers,
                    timeout,
                    require_normalization_match=require_normalization_match,
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "case_spec_sha256": case_spec_digest(spec),
        "source": {
            "label": label,
            "revision": revision,
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "cases": results,
    }


def validate_snapshot(snapshot: Any, context: str) -> dict[str, Any]:
    _require_type(snapshot, dict, context)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError(
            f"{context}.schema_version must be {SCHEMA_VERSION}, "
            f"got {snapshot.get('schema_version')!r}"
        )
    _require_type(snapshot.get("cases"), list, f"{context}.cases")
    names = [case.get("name") for case in snapshot["cases"]]
    if any(not isinstance(name, str) or not name for name in names):
        raise HarnessError(f"{context} contains a case without a valid name")
    if len(names) != len(set(names)):
        raise HarnessError(f"{context} contains duplicate case names")
    return snapshot


def select_snapshot_cases(
    snapshot: dict[str, Any], selected_names: Iterable[str] | None
) -> dict[str, Any]:
    selected = list(selected_names or [])
    if not selected:
        return snapshot

    by_name = {case["name"]: case for case in snapshot["cases"]}
    unknown = [name for name in selected if name not in by_name]
    if unknown:
        raise HarnessError(
            f"reference snapshot is missing selected case(s): {', '.join(unknown)}"
        )

    filtered = copy.deepcopy(snapshot)
    filtered["cases"] = [by_name[name] for name in selected]
    return filtered


def _short(value: Any, limit: int = 240) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _child_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def compare_values(
    expected: Any,
    actual: Any,
    *,
    tolerance: float,
    path: str = "$",
    differences: list[str] | None = None,
) -> list[str]:
    if differences is None:
        differences = []
    if len(differences) >= MAX_REPORTED_DIFFERENCES:
        return differences

    expected_is_number = isinstance(expected, (int, float)) and not isinstance(
        expected, bool
    )
    actual_is_number = isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if expected_is_number and actual_is_number:
        if not math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance
        ):
            differences.append(
                f"{path}: expected {_short(expected)}, got {_short(actual)} "
                f"(absolute tolerance {tolerance})"
            )
        return differences

    if type(expected) is not type(actual):
        differences.append(
            f"{path}: expected {type(expected).__name__} {_short(expected)}, "
            f"got {type(actual).__name__} {_short(actual)}"
        )
        return differences

    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(f"{_child_path(path, key)}: missing from actual")
        for key in sorted(actual_keys - expected_keys):
            differences.append(f"{_child_path(path, key)}: unexpected in actual")
        for key in sorted(expected_keys & actual_keys):
            compare_values(
                expected[key],
                actual[key],
                tolerance=tolerance,
                path=_child_path(path, key),
                differences=differences,
            )
        return differences

    if isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append(
                f"{path}: expected list length {len(expected)}, got {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_values(
                expected_item,
                actual_item,
                tolerance=tolerance,
                path=f"{path}[{index}]",
                differences=differences,
            )
        return differences

    if expected != actual:
        differences.append(f"{path}: expected {_short(expected)}, got {_short(actual)}")
    return differences


def compare_snapshots(reference: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    validate_snapshot(reference, "reference")
    validate_snapshot(actual, "actual")
    differences: list[str] = []

    if reference.get("case_spec_sha256") != actual.get("case_spec_sha256"):
        differences.append(
            "case spec digest differs: reference and actual were not captured "
            "from the same complete case file"
        )

    reference_cases = {case["name"]: case for case in reference["cases"]}
    actual_cases = {case["name"]: case for case in actual["cases"]}
    for name in sorted(reference_cases.keys() - actual_cases.keys()):
        differences.append(f"case {name!r}: missing from actual snapshot")
    for name in sorted(actual_cases.keys() - reference_cases.keys()):
        differences.append(f"case {name!r}: unexpected in actual snapshot")

    for name in sorted(reference_cases.keys() & actual_cases.keys()):
        expected_case = reference_cases[name]
        actual_case = actual_cases[name]
        case_differences: list[str] = []
        compare_values(
            expected_case.get("request"),
            actual_case.get("request"),
            tolerance=0.0,
            path="$.request",
            differences=case_differences,
        )
        compare_values(
            expected_case.get("response"),
            actual_case.get("response"),
            tolerance=float(expected_case.get("float_tolerance", 0.0)),
            path="$.response",
            differences=case_differences,
        )
        differences.extend(f"case {name!r}: {item}" for item in case_differences)

    if len(differences) > MAX_REPORTED_DIFFERENCES:
        return differences[:MAX_REPORTED_DIFFERENCES] + [
            f"... {len(differences) - MAX_REPORTED_DIFFERENCES} more differences"
        ]
    return differences


def write_snapshot(path: str | os.PathLike[str], snapshot: dict[str, Any]) -> None:
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(payload)
    os.replace(temporary, destination)


def _add_live_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="HTTP header added to every request; may be repeated",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="selected_cases",
        help="run one named case; may be repeated (default: all)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare Python/Rust frontend API behavior."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="capture normalized responses from a live server"
    )
    _add_live_options(capture)
    capture.add_argument("--output", required=True)

    compare = subparsers.add_parser(
        "compare", help="compare a live server with a saved reference snapshot"
    )
    _add_live_options(compare)
    compare.add_argument("--reference", required=True)
    compare.add_argument(
        "--write-actual", help="optionally save the live server snapshot"
    )

    offline = subparsers.add_parser("diff", help="compare two saved snapshots")
    offline.add_argument("--reference", required=True)
    offline.add_argument("--actual", required=True)
    return parser


def _print_differences(differences: list[str]) -> None:
    if not differences:
        print("frontend API parity: PASS")
        return
    print(f"frontend API parity: FAIL ({len(differences)} difference(s))")
    for difference in differences:
        print(f"- {difference}")


def run(args: argparse.Namespace) -> int:
    if args.command == "diff":
        reference = validate_snapshot(load_json(args.reference), "reference")
        actual = validate_snapshot(load_json(args.actual), "actual")
        differences = compare_snapshots(reference, actual)
        _print_differences(differences)
        return 1 if differences else 0

    if args.timeout <= 0:
        raise HarnessError("--timeout must be positive")
    spec = load_case_spec(args.cases)
    headers = parse_cli_headers(args.header)
    snapshot = capture_snapshot(
        spec,
        base_url=args.base_url,
        label=args.label,
        revision=args.revision,
        cli_headers=headers,
        timeout=args.timeout,
        selected_names=args.selected_cases,
        require_normalization_match=args.command == "capture",
    )

    if args.command == "capture":
        write_snapshot(args.output, snapshot)
        print(f"captured {len(snapshot['cases'])} case(s) to {args.output}")
        return 0

    if args.write_actual:
        write_snapshot(args.write_actual, snapshot)
        print(f"captured actual responses to {args.write_actual}")
    reference = validate_snapshot(load_json(args.reference), "reference")
    reference = select_snapshot_cases(reference, args.selected_cases)
    differences = compare_snapshots(reference, snapshot)
    _print_differences(differences)
    return 1 if differences else 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (HarnessError, requests.RequestException) as exc:
        print(f"frontend API parity harness error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
