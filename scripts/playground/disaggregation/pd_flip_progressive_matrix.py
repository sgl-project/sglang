#!/usr/bin/env python3
"""Run six real-cluster trials; local smoke tests do not unblock four-node E2E."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from pathlib import PurePosixPath


MODES = ("full", "partial", "zero")
PATHS = ("recovery", "commit")


def resolve_case_paths(output_root, decision_path, mode, repo_root=None):
    repo = Path(repo_root or os.environ["SGLANG_REPO"]).resolve()
    output = Path(output_root).resolve()
    try:
        relative = output.relative_to(repo)
    except ValueError as exc:
        raise ValueError("--output-root must be inside SGLANG_REPO") from exc
    host_case = output / decision_path / mode
    container_case = (
        PurePosixPath("/sgl-workspace/sglang")
        / PurePosixPath(relative.as_posix())
        / decision_path
        / mode
    )
    return host_case, container_case


def _router_request(url, api_key, timeout, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": "Bearer " + api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("router request failed with HTTP %s" % response.status)
        return json.loads(response.read().decode("utf-8"))


def router_get_workers(router_url, api_key, timeout):
    payload = _router_request(
        router_url.rstrip("/") + "/pd_flip/router/workers", api_key, timeout
    )
    workers = payload.get("workers") if isinstance(payload, dict) else None
    if not isinstance(workers, list):
        raise RuntimeError("router workers response is malformed")
    return workers


def router_set_drain(router_url, api_key, timeout, worker_id, draining):
    payload = _router_request(
        router_url.rstrip("/") + "/pd_flip/router/worker/drain",
        api_key,
        timeout,
        {"worker_id": worker_id, "draining": draining},
    )
    worker = payload.get("worker") if isinstance(payload, dict) else None
    if not payload.get("success") or not worker or worker.get("draining") is not draining:
        raise RuntimeError("router did not confirm drain mutation for %s" % worker_id)


def prepare_router_placement(
    router_url, api_key, other_decode_url, source_url, target_url, timeout
):
    workers = router_get_workers(router_url, api_key, timeout)
    by_url = {str(worker.get("url", "")).rstrip("/"): worker for worker in workers}
    urls = [other_decode_url, source_url, target_url]
    selected = []
    for url in urls:
        worker = by_url.get(url.rstrip("/"))
        if not worker or worker.get("role") != "decode":
            raise RuntimeError("required decode worker missing from router: %s" % url)
        selected.append(worker)
    placement = {
        "router_url": router_url,
        "api_key": api_key,
        "timeout": timeout,
        "workers": selected,
    }
    mutations = ((selected[0], True), (selected[2], True), (selected[1], False))
    changed = []
    try:
        for worker, draining in mutations:
            router_set_drain(
                router_url, api_key, timeout, worker["worker_id"], draining
            )
            changed.append(worker)
    except Exception:
        for worker in reversed(changed):
            router_set_drain(
                router_url,
                api_key,
                timeout,
                worker["worker_id"],
                bool(worker.get("draining")),
            )
        raise
    return placement


def release_targets(placement):
    other, _, target = placement["workers"]
    for worker, draining in ((other, bool(other.get("draining"))), (target, False)):
        router_set_drain(
            placement["router_url"],
            placement["api_key"],
            placement["timeout"],
            worker["worker_id"],
            draining,
        )


def restore_router_placement(placement):
    errors = []
    for worker in placement["workers"]:
        try:
            router_set_drain(
                placement["router_url"],
                placement["api_key"],
                placement["timeout"],
                worker["worker_id"],
                bool(worker.get("draining")),
            )
        except Exception as exc:
            errors.append("%s: %s" % (worker["worker_id"], exc))
    if errors:
        raise RuntimeError("router drain restore failures: " + "; ".join(errors))


def restore_router_placement_preserving(placement, original_error=None):
    try:
        restore_router_placement(placement)
    except Exception as restore_error:
        if original_error is None:
            raise
        original_error.add_note(str(restore_error))


def validate_pressure_timeline(timeline, decision_path):
    pressure_start = timeline["pressure_start"]
    controller_start = timeline["controller_start"]
    pressure_end = timeline["pressure_end"]
    controller_end = timeline["controller_end"]
    if not pressure_start < controller_start < pressure_end:
        raise RuntimeError("pressure/controller overlap contract failed")
    if decision_path == "commit" and pressure_end < controller_end:
        raise RuntimeError("commit pressure ended before controller")
    if decision_path == "recovery":
        first_active = timeline.get("first_batch_target_active")
        if first_active is None or not first_active <= pressure_end < controller_end:
            raise RuntimeError("recovery pressure did not stop after first activation")


def run_shell(command, *, env, log_path=None):
    output = None
    if log_path:
        output = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.run(
            command,
            shell=True,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT if output else None,
            check=False,
        ).returncode
    finally:
        if output:
            output.close()


def log_tail(path, limit=2000):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def assert_processes_running(processes):
    for name, process, log_path in processes:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "%s exited early with rc=%s: %s"
                % (name, return_code, log_tail(log_path))
            )


def wait_for_path(path, timeout, interval, processes=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert_processes_running(processes)
        time.sleep(interval)
    raise TimeoutError("timed out waiting for %s" % path)


def wait_for_process(process, timeout, interval, watched=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert_processes_running(watched)
        return_code = process.poll()
        if return_code is not None:
            return return_code
        time.sleep(interval)
    raise TimeoutError("process did not exit before deadline")


def wait_for_store(url, timeout, interval):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(timeout, 3.0)) as response:
                if 200 <= response.status < 300:
                    payload = json.loads(response.read().decode("utf-8"))
                    if isinstance(payload, dict):
                        return payload
        except Exception as exc:
            last_error = exc
        time.sleep(interval)
    raise TimeoutError("dedicated store did not become ready: %r" % last_error)


def validate_store_generation(proof, ready, seen_tokens, seen_generations):
    required = ("token", "pid", "starttime", "generation")
    if any(not proof.get(field) for field in required):
        raise RuntimeError("store generation proof is incomplete")
    if any(str(proof[field]) != str(ready.get(field)) for field in required[1:]):
        raise RuntimeError("store ready identity does not match reset proof")
    if proof["token"] in seen_tokens or proof["generation"] in seen_generations:
        raise RuntimeError("store generation/token was reused across cases")
    seen_tokens.add(proof["token"])
    seen_generations.add(proof["generation"])


def terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return process.returncode


def validate_measurement_exit(return_code, matrix_sent_sigterm):
    if return_code == 0:
        return
    expected_signal = -int(signal.SIGTERM)
    if matrix_sent_sigterm and return_code == expected_signal:
        return
    if return_code == expected_signal:
        raise RuntimeError("measurement exited from unexpected signal rc=%s" % return_code)
    raise RuntimeError("measurement sidecar termination rc=%s" % return_code)


def wait_for_target_active(journal_path, timeout, interval, processes):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        assert_processes_running(processes)
        if journal_path.exists():
            try:
                last = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                last = None
            if isinstance(last, dict) and last.get("phase") == "target_active":
                return last
        time.sleep(interval)
    raise TimeoutError("target_active was not observed in journal: %r" % last)


def run_case(args, mode, decision_path):
    case_dir, container_case_dir = resolve_case_paths(
        args.output_root, decision_path, mode
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    ready = case_dir / "workload.ready.json"
    done = case_dir / "controller.done"
    pressure_stop = case_dir / "pressure.stop"
    pressure_started = case_dir / "pressure.started.json"
    pressure_ended = case_dir / "pressure.ended.json"
    events = case_dir / "migration_events.jsonl"
    journal = case_dir / "pd_flip_session.json"
    generation_file = case_dir / "store.generation.json"
    raw_paths = (
        ready,
        done,
        pressure_stop,
        pressure_started,
        pressure_ended,
        events,
        journal,
        generation_file,
        case_dir / "request_metrics.jsonl",
        case_dir / "errors.jsonl",
        case_dir / "controller.log",
        case_dir / "measurement.log",
    )
    for path in raw_paths:
        path.unlink(missing_ok=True)

    generation_token = uuid.uuid4().hex
    env = os.environ.copy()
    env.update(
        {
            "MODE": mode,
            "DECISION_PATH": decision_path,
            "OUTPUT_DIR": str(case_dir.resolve()),
            "HOST_CASE_DIR": str(case_dir.resolve()),
            "CONTAINER_CASE_DIR": str(container_case_dir),
            "READY_MARKER": str(ready.resolve()),
            "CONTROLLER_DONE_MARKER": str(done.resolve()),
            "PRESSURE_STOP_MARKER": str(pressure_stop.resolve()),
            "PRESSURE_STARTED_MARKER": str(pressure_started.resolve()),
            "PRESSURE_ENDED_MARKER": str(pressure_ended.resolve()),
            "MIGRATION_EVENTS": str(events.resolve()),
            "PD_FLIP_ARTIFACT_DIR": str(container_case_dir),
            "PD_FLIP_SESSION_JOURNAL_PATH": str(
                container_case_dir / "pd_flip_session.json"
            ),
            "STORE_GENERATION_TOKEN": generation_token,
            "STORE_GENERATION_FILE": str(generation_file.resolve()),
        }
    )
    case_started = time.time()
    if run_shell(args.reset_store_cmd, env=env) != 0:
        raise RuntimeError("store reset command failed")
    ready_generation = wait_for_store(
        args.store_ready_url, args.timeout_seconds, args.poll_interval_seconds
    )
    if not generation_file.exists():
        raise RuntimeError("store reset did not write fresh generation proof")
    generation = json.loads(generation_file.read_text(encoding="utf-8"))
    if generation.get("token") != generation_token:
        raise RuntimeError("store generation proof is invalid")
    validate_store_generation(
        generation,
        ready_generation,
        args._seen_store_tokens,
        args._seen_store_generations,
    )

    measure_log_path = case_dir / "measurement.log"
    workload_log_path = case_dir / "workload.log"
    controller_log_path = case_dir / "controller.log"
    measure_log = measure_log_path.open("w", encoding="utf-8")
    measurement = subprocess.Popen(
        args.measure_command,
        shell=True,
        env=env,
        stdout=measure_log,
        stderr=subprocess.STDOUT,
    )
    workload_command = [
        sys.executable,
        str(Path(__file__).with_name("pd_flip_progressive_workload.py")),
        "--base-url",
        args.base_url,
        "--source-url",
        args.source_url,
        "--target-url",
        args.target_url,
        "--admin-api-key-env",
        args.admin_api_key_env,
        "--ready-marker",
        str(ready),
        "--controller-done-marker",
        str(done),
        "--pressure-stop-marker",
        str(pressure_stop),
        "--pressure-started-marker",
        str(pressure_started),
        "--pressure-ended-marker",
        str(pressure_ended),
        "--model",
        args.model,
        "--mode",
        mode,
        "--decision-path",
        decision_path,
        "--output-dir",
        str(case_dir),
        "--coordination-timeout-seconds",
        str(args.timeout_seconds),
    ]
    placement = None
    workload_log = workload_log_path.open("w", encoding="utf-8")
    workload = None
    controller = None
    controller_log = None
    workload_rc = 1
    measurement_rc = None
    measurement_sent_sigterm = False
    timeline = {}
    original_error: BaseException | None = None
    try:
        placement = prepare_router_placement(
            args.router_url,
            os.environ.get(args.router_admin_api_key_env)
            or os.environ[args.admin_api_key_env],
            args.other_decode_url,
            args.source_url,
            args.target_url,
            args.timeout_seconds,
        )
        workload = subprocess.Popen(
            workload_command,
            env=env,
            stdout=workload_log,
            stderr=subprocess.STDOUT,
        )
        watched = [
            ("measurement", measurement, measure_log_path),
            ("workload", workload, workload_log_path),
        ]
        wait_for_path(
            ready, args.timeout_seconds, args.poll_interval_seconds, watched
        )
        release_targets(placement)
        wait_for_path(
            pressure_started,
            args.timeout_seconds,
            args.poll_interval_seconds,
            watched,
        )
        timeline.update(json.loads(pressure_started.read_text(encoding="utf-8")))
        controller_log = controller_log_path.open("w", encoding="utf-8")
        timeline["controller_start"] = time.monotonic()
        controller = subprocess.Popen(
            args.controller_command,
            shell=True,
            env=env,
            stdout=controller_log,
            stderr=subprocess.STDOUT,
        )
        if decision_path == "recovery":
            wait_for_target_active(
                journal,
                args.timeout_seconds,
                args.poll_interval_seconds,
                watched
                + [("controller", controller, controller_log_path)],
            )
            timeline["first_batch_target_active"] = time.monotonic()
            pressure_stop.touch()
            wait_for_path(
                pressure_ended,
                args.timeout_seconds,
                args.poll_interval_seconds,
                watched + [("controller", controller, controller_log_path)],
            )
            timeline.update(
                json.loads(pressure_ended.read_text(encoding="utf-8"))
            )
        controller_rc = wait_for_process(
            controller,
            args.timeout_seconds,
            args.poll_interval_seconds,
            watched,
        )
        timeline["controller_end"] = time.monotonic()
        if controller_rc:
            raise RuntimeError(
                "controller failed rc=%s: %s"
                % (controller_rc, log_tail(controller_log_path))
            )
        if decision_path == "commit":
            pressure_stop.touch()
        done.touch()
        if not pressure_ended.exists():
            wait_for_path(
                pressure_ended,
                args.timeout_seconds,
                args.poll_interval_seconds,
                [("measurement", measurement, measure_log_path), ("workload", workload, workload_log_path)],
            )
            timeline.update(json.loads(pressure_ended.read_text(encoding="utf-8")))
        validate_pressure_timeline(timeline, decision_path)
        workload_rc = workload.wait(timeout=args.timeout_seconds)
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        done.touch()
        pressure_stop.touch()
        if controller is not None:
            terminate(controller)
        if workload is not None:
            terminate(workload)
        measurement_sent_sigterm = measurement.poll() is None
        measurement_rc = terminate(measurement)
        if placement is not None:
            restore_router_placement_preserving(placement, original_error)
        if controller_log is not None:
            controller_log.close()
        workload_log.close()
        measure_log.close()
    if workload_rc:
        raise RuntimeError(
            "case %s/%s workload failed: %s"
            % (decision_path, mode, log_tail(workload_log_path))
        )
    if not events.exists() or not events.stat().st_size or events.stat().st_mtime < case_started:
        raise RuntimeError("measurement sidecar did not produce fresh non-empty events")
    validate_measurement_exit(measurement_rc, measurement_sent_sigterm)
    (case_dir / "orchestration_timeline.json").write_text(
        json.dumps(timeline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if run_shell(
        args.summarize_command,
        env=env,
        log_path=case_dir / "summarize.log",
    ):
        raise RuntimeError("summarizer failed for %s/%s" % (decision_path, mode))


def run(args):
    if not args.reset_store_cmd.strip():
        raise ValueError("--reset-store-cmd must be non-empty")
    if not os.environ.get(args.admin_api_key_env):
        raise ValueError("admin key environment variable is empty")
    args._seen_store_tokens = set()
    args._seen_store_generations = set()
    for decision_path in PATHS:
        for mode in MODES:
            run_case(args, mode, decision_path)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--other-decode-url", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--admin-api-key-env", default="ADMIN_API_KEY")
    parser.add_argument(
        "--router-admin-api-key-env", default="PD_FLIP_ROUTER_ADMIN_API_KEY"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reset-store-cmd", required=True)
    parser.add_argument("--store-ready-url", required=True)
    parser.add_argument("--measure-command", required=True)
    parser.add_argument("--controller-command", required=True)
    parser.add_argument("--summarize-command", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
