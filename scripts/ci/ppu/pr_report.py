"""Validate mandatory PPU PR evidence and selected-job outcomes (stdlib only)."""

import argparse
import html
import importlib.util
import json
import os
import re
import stat
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "pr_runner", ROOT / "python/sglang/test/ci/pr_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

SUITES = {
    "cpu": ["stage-a-test-cpu-ppu"],
    "smoke": ["stage-a-test-1-gpu-ppu"],
    "body": ["stage-b-test-1-gpu-ppu"],
    "multi": ["stage-c-test-2-gpu-ppu"],
}
JOBS = {
    "stage-a-test-cpu-ppu": "cpu",
    "stage-a-test-1-gpu-ppu": "ppu_single",
    "stage-b-test-1-gpu-ppu": "ppu_single",
    "stage-c-test-2-gpu-ppu": "ppu_multi",
}

# Suite -> registry backend, for deriving the expected file inventory.
BACKENDS = {
    "stage-a-test-cpu-ppu": "CPU",
    "stage-a-test-1-gpu-ppu": "PPU",
    "stage-b-test-1-gpu-ppu": "PPU",
    "stage-b-kernel-test-1-gpu-ppu": "PPU",
    "stage-c-test-2-gpu-ppu": "PPU",
}


def _register():
    spec = importlib.util.spec_from_file_location(
        "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_inventory(suite):
    """Enabled per-commit registrations for *suite*, repository-relative.

    The expected inventory comes from this checkout's own registry, never from
    worker-produced evidence: a selection pipeline that silently drops a
    registered file must fail verification, not redefine the inventory.
    """
    register = _register()
    backend = register.HWBackend[BACKENDS[suite]]
    files = [
        str(path)
        for path in (ROOT / "test/registered").rglob("*.py")
        if path.name not in ("conftest.py", "__init__.py")
    ]
    return {
        str(Path(r.filename).resolve().relative_to(ROOT))
        for r in register.collect_tests(files, sanity_check=True)
        if r.backend == backend
        and r.effective_suite == suite
        and r.disabled is None
        and not r.nightly
    }


def _repo_relative(path):
    """Normalize a worker-reported path to repository-relative form.

    Worker and verifier checkouts differ (pod vs runner), so anchor at the
    registered-test root instead of trusting the absolute prefix.
    """
    parts = Path(path).parts
    for index in range(len(parts) - 1):
        if parts[index] == "test" and parts[index + 1] == "registered":
            return str(Path(*parts[index:]))
    return None


def export_evidence(directory, destination):
    """Read NAS evidence without rewriting it; publish only a complete local export."""
    source = Path(directory)
    destination = Path(destination).absolute()
    source_resolved = source.resolve(strict=True)
    target = destination.resolve()
    if target == source_resolved or source_resolved in target.parents:
        raise ValueError("Evidence export must be outside the source directory")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Evidence export destination already exists")

    def copy_directory(source_fd, output):
        count = 0
        for name in sorted(os.listdir(source_fd)):
            # Names become artifact paths. Never follow links or publish secret names.
            if (
                not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or runner.sanitize(name) != name
            ):
                raise ValueError("Unsafe evidence entry name")
            mode = os.stat(name, dir_fd=source_fd, follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                child_fd = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=source_fd
                )
                try:
                    child = output / name
                    child.mkdir(mode=0o700)
                    count += copy_directory(child_fd, child)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                if Path(name).suffix not in (
                    ".json",
                    ".xml",
                    ".log",
                    ".tsv",
                    ".status",
                ):
                    continue
                fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd
                )
                with os.fdopen(fd, encoding="utf-8", errors="replace") as handle:
                    metadata = os.fstat(handle.fileno())
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise ValueError(
                            "Evidence must contain only single-link regular files"
                        )
                    cleaned = runner.sanitize(handle.read())
                with (output / name).open("x", encoding="utf-8") as handle:
                    handle.write(cleaned)
                count += 1
            else:
                raise ValueError("Evidence links and special files are not allowed")
        return count

    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Fresh files inherit runner ownership, not NAS mode bits or ACLs.
        # Staging stays outside the upload path until every retained file is sanitized.
        with tempfile.TemporaryDirectory(
            prefix=".ppu-export-", dir=destination.parent
        ) as temp:
            staging = Path(temp) / "evidence"
            staging.mkdir(mode=0o700)
            if not copy_directory(source_fd, staging):
                raise ValueError("No supported evidence files to export")
            staging.rename(destination)
    finally:
        os.close(source_fd)
    return destination


def verify(directory, lane, kernel=False):
    directory = Path(directory)
    if kernel and lane != "body":
        raise ValueError("Kernel evidence only belongs to the body lane")
    worker = json.loads((directory / "worker.json").read_text())
    setup = json.loads((directory / "setup.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    if worker.get("exit_code") != 0 or worker.get("elapsed", -1) < 0:
        raise ValueError("Missing/failed worker completion")
    if setup.get("lane") != lane or setup.get("setup_seconds", -1) < 0:
        raise ValueError("Missing setup timing")
    for key in ("commit", "python", "torch", "checkout_import", "backend"):
        if not environment.get(key):
            raise ValueError(f"Missing environment identity: {key}")
    if lane != "cpu":
        model = json.loads((directory / "model.json").read_text())
        if not all(model.get(key) for key in ("path", "metadata_sha256", "weights")):
            raise ValueError("Missing model identity")
        if not environment.get("digest", "").startswith(
            "sha256:"
        ) or "@sha256:" not in environment.get("image", ""):
            raise ValueError("Missing immutable image identity")
        if len(environment.get("devices", [])) < (2 if lane == "multi" else 1):
            raise ValueError("Missing device identity")
    names = SUITES[lane] + (["stage-b-kernel-test-1-gpu-ppu"] if kernel else [])
    total = {"executed": 0, "skipped": 0}
    for suite in names:
        report = json.loads((directory / suite / "summary.json").read_text())
        if report.get("success") is not True or not report.get("selected"):
            raise ValueError(f"Incomplete suite: {suite}")
        selected = {_repo_relative(path) for path in report["selected"]}
        if selected != expected_inventory(suite):
            raise ValueError(f"Selection/registration mismatch: {suite}")
        if [r["file"] for r in report["files"]] != report["selected"]:
            raise ValueError(f"Missing selected file: {suite}")
        if len(set(report["selected"])) != len(report["selected"]) or report.get(
            "allowed_skips"
        ):
            # No platform skips are currently approved for these PPU suites.
            raise ValueError(f"Duplicate selection or unapproved skip policy: {suite}")
        xmls = {p.name for p in (directory / suite).glob("*.xml")}
        if xmls != {r.get("junit") for r in report["files"]} or len(xmls) != len(
            report["selected"]
        ):
            raise ValueError(f"Missing per-file JUnit: {suite}")
        for record in report["files"]:
            if (
                record.get("passed") is not True
                or record.get("executed", 0) <= 0
                or record.get("returncode") != 0
            ):
                raise ValueError(f"Empty/failed file in {suite}")
            counts = runner.inspect_junit(directory / suite / record["junit"])
            if any(record.get(key) != value for key, value in counts.items()):
                raise ValueError(f"JUnit/summary disagreement: {suite}")
            log = record.get("log", "")
            if (
                Path(log).name != log
                or not (directory / suite / log).is_file()
                or record.get("elapsed", -1) < 0
            ):
                raise ValueError(f"Missing log/timing: {suite}")
            for key in total:
                total[key] += counts[key]
    print(json.dumps({"lane": lane, "suites": names, **total}))
    return total


def summarize(directory, lane):
    """Best-effort failure/success presentation; never substitutes for verify()."""
    directory = Path(directory)

    def read_json(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    setup = read_json(directory / "setup.json")
    worker = read_json(directory / "worker.json")
    aot = read_json(directory / "aot.json")
    environment = read_json(directory / "environment.json")
    timing = {"scheduling_seconds": None, "pod_startup_seconds": None}
    try:
        timestamps = (
            (directory / "environment-pods.timing.tsv").read_text().strip().split("\t")
        )
        created, scheduled, started = [
            datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps
        ]
        if created <= scheduled <= started:
            timing = {
                "scheduling_seconds": (scheduled - created).total_seconds(),
                "pod_startup_seconds": (started - scheduled).total_seconds(),
            }
    except (OSError, ValueError, TypeError):
        pass
    try:
        setup_log = (directory / "setup.log").read_text(errors="replace")
    except OSError:
        setup_log = ""
    cache = "unknown"
    if "PPU sgl-kernel build SUCCESS" in setup_log:
        cache = "source build"
    elif "PPU sgl-kernel installed from cache (no compile)" in setup_log:
        cache = "cache hit"
    key = re.search(r"Wheel cache key: ([a-f0-9]+)", setup_log)
    rows = []
    for path in sorted(directory.glob("*/summary.json")):
        report = read_json(path)
        records = {r.get("file"): r for r in report.get("files", [])}
        for file in report.get("selected", []):
            record = records.get(file, {})
            rows.append(
                {
                    "suite": path.parent.name,
                    "file": file,
                    "executed": record.get("executed", 0),
                    "skipped": record.get("skipped", 0),
                    "seconds": record.get("elapsed"),
                    "passed": record.get("passed", False),
                }
            )
    result = {
        "lane": lane,
        "worker_seconds": worker.get("elapsed"),
        "setup_seconds": setup.get("setup_seconds"),
        "aot_seconds": aot.get("aot_seconds"),
        "aot_cache": cache,
        "aot_cache_key": key[1] if key else None,
        "github_queue_seconds": None,
        **timing,
        "files": rows,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "timing-summary.json").write_text(
        runner.sanitize(json.dumps(result, indent=2))
    )
    lines = [
        f"### PPU PR {lane}",
        "",
        "```json",
        json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2),
        "```",
        "Null timings are unobserved, not zero. Pod startup includes image pull and container initialization. GitHub runner queue time is not measured.",
        "",
        "| Selected file | Executed | Skipped | Seconds | Passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        file = html.escape(row["file"]).replace("|", "&#124;").replace("\n", " ")
        lines.append(
            f"| {file} | {row['executed']} | {row['skipped']} | {row['seconds']} | {row['passed']} |"
        )
    lines += [
        "",
        "Environment: <code>" + html.escape(json.dumps(environment)) + "</code>",
        "",
        "See the run/attempt/job artifact for sanitized per-file logs, model identity, and JUnit evidence.",
        "",
    ]
    text = runner.sanitize("\n".join(lines))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(text)
    print(text)
    return result


def aggregate(needs, cpu_only=False, e2e_requested=False):
    e2e = needs.get("e2e", {})
    if e2e_requested and not cpu_only:
        if (
            e2e.get("result") != "success"
            or e2e.get("outputs", {}).get("evidence_ok") != "true"
        ):
            return False
    elif e2e and e2e.get("result") != "skipped":
        return False
    check = needs.get("check-changes", {})
    if check.get("result") != "success":
        return False
    selection = check.get("outputs", {})
    if any(
        selection.get(lane) not in ("true", "false")
        for lane in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi")
    ):
        return False
    selected = any(
        selection[lane] == "true"
        for lane in ("cpu", "ppu_single", "ppu_kernel", "ppu_multi")
    )
    gate_result = "success" if selected and not cpu_only else "skipped"
    if needs.get("pr-gate", {}).get("result") != gate_result:
        return False
    if selection["ppu_single"] == "true" and selection["cpu"] != "true":
        return False
    if selection["ppu_kernel"] == "true" and selection["ppu_single"] != "true":
        return False
    if selection["ppu_multi"] == "true" and selection["ppu_kernel"] != "true":
        return False
    for job, lane in JOBS.items():
        record = needs.get(job, {})
        if selection[lane] == "true" and (not cpu_only or lane == "cpu"):
            if (
                record.get("result") != "success"
                or record.get("outputs", {}).get("evidence_ok") != "true"
            ):
                return False
        elif record.get("result") != "skipped":
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["verify", "aggregate", "sanitize", "summarize", "export"]
    )
    parser.add_argument("--directory")
    parser.add_argument("--destination")
    parser.add_argument("--lane", choices=SUITES)
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--e2e-requested", action="store_true")
    args = parser.parse_args()
    if args.mode == "aggregate":
        raise SystemExit(
            0
            if aggregate(
                json.loads(os.environ["NEEDS_JSON"]), args.cpu_only, args.e2e_requested
            )
            else 1
        )
    elif args.mode == "verify":
        verify(args.directory, args.lane, args.kernel)
    elif args.mode == "summarize":
        summarize(args.directory, args.lane)
    elif args.mode == "export":
        if not args.directory or not args.destination:
            parser.error("export requires --directory and --destination")
        export_evidence(args.directory, args.destination)
    else:
        for path in Path(args.directory).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".xml", ".log"):
                path.write_text(runner.sanitize(path.read_text(errors="replace")))
