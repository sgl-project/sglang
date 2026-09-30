#!/usr/bin/env python3
"""Single-GPU experiment queue; stop idle load before starting a submitted task."""

import argparse
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            if os.geteuid() == 0:
                owner = path.parent.stat()
                os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_group(process, grace=5):
    """Reap the task and terminate any descendants in its dedicated group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may have exited while a child GPU process is still alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Worker:
    def __init__(self, root, idle_command, interval=0.25):
        self.root = Path(root).resolve()
        self.idle_command = idle_command
        self.interval = interval
        self.stop = threading.Event()
        self.idle = None
        self.idle_log = None
        self.next_idle_start = 0
        self.active = None
        self.active_job = None
        self.active_log = None
        self.active_started = 0
        self.last_status = 0
        self.worker_id = uuid.uuid4().hex
        owner = self.root.stat()
        for name in ("queue", "running", "results", "logs"):
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True)
            if os.geteuid() == 0:
                os.chown(directory, owner.st_uid, owner.st_gid)

    def status(self, mode, force=False):
        if not force and time.monotonic() - self.last_status < 1:
            return
        write_json(
            self.root / "status.json",
            {
                "worker_id": self.worker_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "mode": mode,
                "heartbeat_unix": time.time(),
                "idle_pid": self.idle.pid if self.idle else None,
                "active_job": self.active_job,
                "active_pid": self.active.pid if self.active else None,
            },
        )
        self.last_status = time.monotonic()

    def stop_idle(self):
        if self.idle is not None:
            stop_group(self.idle)
            self.idle = None
        if self.idle_log is not None:
            self.idle_log.close()
            self.idle_log = None

    def start_idle(self):
        if not self.idle_command or time.monotonic() < self.next_idle_start:
            return
        self.idle_log = (self.root / "logs" / "idle.log").open("a")
        try:
            self.idle = subprocess.Popen(
                self.idle_command,
                stdout=self.idle_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            self.idle_log.close()
            self.idle_log = None
            self.next_idle_start = time.monotonic() + 10
            raise

    def finish_active(self, outcome, returncode):
        job_id = self.active_job["job_id"]
        if self.active is not None:
            stop_group(self.active)
        if self.active_log is not None:
            self.active_log.close()
        write_json(
            self.root / "results" / f"{job_id}.json",
            {
                "job_id": job_id,
                "outcome": outcome,
                "returncode": returncode,
                "finished_unix": time.time(),
                "log_path": str(self.root / "logs" / f"{job_id}.log"),
            },
        )
        (self.root / "running" / f"{job_id}.json").unlink(missing_ok=True)
        self.active = self.active_job = self.active_log = None

    def launch(self, path):
        self.stop_idle()
        destination = self.root / "running" / path.name
        path.replace(destination)
        with destination.open() as stream:
            self.active_job = json.load(stream)
        job_id = destination.stem
        self.active_job["job_id"] = job_id
        self.active_log = (self.root / "logs" / f"{job_id}.log").open("a")
        try:
            command = self.active_job["command"]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(part, str) for part in command)
            ):
                raise ValueError("command must be a nonempty argv list")
            environment = os.environ.copy()
            environment.update(self.active_job.get("env", {}))
            self.active = subprocess.Popen(
                command,
                cwd=self.active_job["cwd"],
                env=environment,
                stdout=self.active_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active_started = time.monotonic()
            self.status("experiment", force=True)
        except (OSError, ValueError, KeyError) as error:
            print(f"Launch failed: {error}", file=self.active_log, flush=True)
            self.finish_active("launch_failed", 127)

    def run(self):
        with (self.root / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.geteuid() == 0:
                owner = self.root.stat()
                os.fchown(lock.fileno(), owner.st_uid, owner.st_gid)
            # Do not silently replay arbitrary experiments after a node restart.
            for path in (self.root / "running").glob("*.json"):
                result = self.root / "results" / path.name
                if not result.exists():
                    write_json(
                        result,
                        {
                            "job_id": path.stem,
                            "outcome": "interrupted",
                            "returncode": None,
                        },
                    )
                path.unlink()
            try:
                while not self.stop.is_set():
                    if self.active is not None:
                        rc = self.active.poll()
                        timeout = self.active_job["timeout_seconds"]
                        if rc is not None:
                            self.finish_active("completed" if rc == 0 else "failed", rc)
                        elif time.monotonic() - self.active_started >= timeout:
                            self.finish_active("timed_out", 124)
                        else:
                            self.status("experiment")
                            self.stop.wait(self.interval)
                            continue
                    pending = sorted((self.root / "queue").glob("*.json"))
                    if pending:
                        self.launch(pending[0])
                        continue
                    if (self.root / "PAUSED").exists():
                        self.stop_idle()
                        self.status("paused")
                    else:
                        if self.idle is not None and self.idle.poll() is not None:
                            self.stop_idle()
                            self.next_idle_start = time.monotonic() + 10
                        if self.idle is None:
                            self.start_idle()
                        self.status("idle_load" if self.idle else "idle")
                    self.stop.wait(self.interval)
            finally:
                self.stop_idle()
                if self.active_job is not None:
                    self.finish_active("interrupted", None)
                self.status("stopped", force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--no-idle-load", action="store_true")
    serve.add_argument("--duty-cycle", type=float, default=0.60)
    submit = commands.add_parser("submit")
    submit.add_argument("--cwd", default=os.getcwd())
    submit.add_argument("--timeout-seconds", type=float, default=3600)
    submit.add_argument("--env", action="append", default=[])
    submit.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("status")
    commands.add_parser("pause")
    commands.add_parser("resume")
    args = parser.parse_args()
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "serve":
        idle = (
            []
            if args.no_idle_load
            else [
                sys.executable,
                str(Path(__file__).with_name("cuda_load.py")),
                "--duty-cycle",
                str(args.duty_cycle),
            ]
        )
        worker = Worker(root, idle)
        signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
        signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
        worker.run()
    elif args.action == "submit":
        argv = args.command[1:] if args.command[:1] == ["--"] else args.command
        if (
            not argv
            or not math.isfinite(args.timeout_seconds)
            or args.timeout_seconds <= 0
        ):
            parser.error("provide a command and positive --timeout-seconds")
        env = {}
        for value in args.env:
            key, separator, setting = value.partition("=")
            if not separator or not key:
                parser.error("--env must be KEY=VALUE")
            env[key] = setting
        job_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:12]}"
        (root / "queue").mkdir(exist_ok=True)
        write_json(
            root / "queue" / f"{job_id}.json",
            {
                "job_id": job_id,
                "command": argv,
                "cwd": str(Path(args.cwd).resolve()),
                "env": env,
                "timeout_seconds": args.timeout_seconds,
            },
        )
        print(job_id)
    elif args.action == "status":
        status = root / "status.json"
        data = (
            json.loads(status.read_text())
            if status.exists()
            else {"mode": "not_started"}
        )
        data["queued"] = len(list((root / "queue").glob("*.json")))
        data["paused"] = (root / "PAUSED").exists()
        if "heartbeat_unix" in data:
            data["heartbeat_age_seconds"] = time.time() - data["heartbeat_unix"]
        print(json.dumps(data, indent=2))
    elif args.action == "pause":
        (root / "PAUSED").touch()
    elif args.action == "resume":
        (root / "PAUSED").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
