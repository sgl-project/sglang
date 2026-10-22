"""Opt-in, fail-closed pytest execution. Stdlib-only for cheap contract tests."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize(text):
    # Never publish credentials in URLs, headers, environment dumps, or logs.
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    text = re.sub(
        r"(?im)((?:authorization|password|token|secret|api[_-]?key)\s*[:=]\s*)[^\s,]+",
        r"\1[REDACTED]",
        text,
    )
    for key, value in os.environ.items():
        if re.search(r"PASSWORD|TOKEN|SECRET|API_KEY", key) and len(value) >= 4:
            text = text.replace(value, "[REDACTED]")
    return text


class InvalidResults(ValueError):
    def __init__(self, counts, bad_skips):
        self.counts = counts
        super().__init__(
            f"Invalid test results: {counts}; unexpected skips: {bad_skips}"
        )


def inspect_junit(path, allowed_skips=()):
    # Reports are untrusted input. Bound memory and prohibit DTD/entity expansion;
    # decode UTF-8 first so alternate encodings cannot bypass the declaration guard.
    with Path(path).open("rb") as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("JUnit report exceeds 16 MiB")
    text = data.decode("utf-8")
    if "\x00" in text or re.search(r"<!\s*(DOCTYPE|ENTITY)", text, re.I):
        raise ValueError("JUnit DTD/entities are forbidden")
    root = ET.fromstring(text)
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError("Invalid JUnit root")
    cases = list(root.iter("testcase"))
    skipped = [c for c in cases if c.find("skipped") is not None]
    bad_skips = [
        f"{c.get('classname')}.{c.get('name')}"
        for c in skipped
        if f"{c.get('classname')}.{c.get('name')}" not in allowed_skips
    ]
    counts = {
        "collected": len(cases),
        "executed": len(cases) - len(skipped),
        "skipped": len(skipped),
        "failures": len(list(root.iter("failure"))) + len(list(root.iter("error"))),
    }
    if counts["executed"] == 0 or counts["failures"] or bad_skips:
        raise InvalidResults(counts, bad_skips)
    return counts


def stop_process(process):
    # pytest and all of its server/distributed descendants share this session.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_pytest_files(
    files,
    report_dir,
    timeout_per_file,
    suite_timeout=None,
    continue_on_error=False,
    allowed_skips=(),
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    selected = [str(Path(f).resolve()) for f in files]
    manifest = {
        "selected": selected,
        "files": [],
        "success": False,
        "allowed_skips": list(allowed_skips),
    }
    summary = directory / "summary.json"

    def save():
        summary.write_text(json.dumps(manifest, indent=2) + "\n")

    save()  # An interrupted run leaves an explicitly incomplete report.
    if not selected or len(set(selected)) != len(selected):
        return 1
    started = time.monotonic()
    deadline = started + suite_timeout if suite_timeout is not None else float("inf")
    for filename in selected:
        name = (
            Path(filename).stem
            + "-"
            + hashlib.sha256(filename.encode()).hexdigest()[:12]
        )
        junit = directory / f"{name}.xml"
        log = directory / f"{name}.log"
        if junit.exists():
            # Reject stale evidence rather than accepting a previous run's XML.
            raise ValueError(f"Report already exists: {junit}")
        record = {
            "file": filename,
            "junit": junit.name,
            "log": log.name,
            "passed": False,
            "executed": 0,
            "skipped": 0,
        }
        manifest["files"].append(record)
        tic = time.monotonic()
        process = None
        handlers = {}

        def interrupted(signum, frame):
            raise InterruptedError(f"runner received signal {signum}")

        try:
            handlers = {
                sig: signal.signal(sig, interrupted)
                for sig in (signal.SIGTERM, signal.SIGINT)
            }
            remaining = min(timeout_per_file, deadline - tic)
            if remaining <= 0:
                raise TimeoutError("suite budget exhausted")
            env = dict(
                os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""
            )
            # -o addopts= prevents repository configuration from changing selection.
            command = [
                sys.executable,
                "-m",
                "pytest",
                filename,
                "-v",
                "-o",
                "addopts=",
                "-o",
                "junit_family=xunit2",
                f"--junitxml={junit}",
            ]
            with log.open("w+", errors="replace") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )

                try:
                    record["returncode"] = process.wait(timeout=remaining)
                finally:
                    stop_process(process)
                    process = None
                    output.seek(0)
                    cleaned = sanitize(output.read())
                    output.seek(0)
                    output.write(cleaned)
                    output.truncate()
            counts = inspect_junit(junit, allowed_skips)
            record.update(counts)
            record["passed"] = record["returncode"] == 0
        except (
            OSError,
            ValueError,
            ET.ParseError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as exc:
            if isinstance(exc, InvalidResults):
                record.update(exc.counts)
            if isinstance(exc, InterruptedError):
                continue_on_error = False
            record["error"] = sanitize(str(exc))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if process is not None:
                stop_process(process)
            if junit.exists():
                junit.write_text(sanitize(junit.read_text(errors="replace")))
            record["elapsed"] = round(time.monotonic() - tic, 3)
            save()
        print(json.dumps(record), flush=True)
        if not record["passed"] and not continue_on_error:
            break
    manifest["elapsed"] = round(time.monotonic() - started, 3)
    manifest["success"] = len(manifest["files"]) == len(selected) and all(
        r["passed"] for r in manifest["files"]
    )
    save()
    return 0 if manifest["success"] else 1
