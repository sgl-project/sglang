"""Detached, guarded command runner. Owns only the process tree it starts.

start JOB.json creates a fresh output directory; status/stop take that directory.
No RPC listener, credentials, process-name killing, or automatic retry.
"""

import argparse
import json
import os
import resource
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from curve_config import inside_repo

GIB = 1024**3


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def values(path):
    return {
        k.rstrip(":"): int(v.split()[0])
        for k, v in (
            line.split(None, 1) for line in Path(path).read_text().splitlines()
        )
    }


def snapshot(root):
    cg = Path("/sys/fs/cgroup")
    stats = values(cg / "memory.stat")
    current = int((cg / "memory.current").read_text())
    limit_text = (cg / "memory.max").read_text().strip()
    host = values("/proc/meminfo")
    limit = int(limit_text) if limit_text != "max" else host["MemTotal"] * 1024
    disk = os.statvfs(root)
    return dict(
        time=time.time(),
        current_bytes=current,
        limit_bytes=limit,
        working_bytes=max(0, current - stats["active_file"] - stats["inactive_file"]),
        host_available_bytes=host["MemAvailable"] * 1024,
        disk_available_bytes=disk.f_bavail * disk.f_frsize,
        oom_events=values(cg / "memory.events"),
        host_oom_kill=values("/proc/vmstat")["oom_kill"],
    )


def violations(now, initial, footprint):
    errors = []
    if now["working_bytes"] > 0.78 * now["limit_bytes"]:
        errors.append("nonreclaimable memory >78%")
    if now["host_available_bytes"] < 200 * GIB:
        errors.append("host reserve <200GiB")
    if now["disk_available_bytes"] < 1024 * GIB:
        errors.append("disk reserve <1TiB")
    if footprint > 80 * GIB:
        errors.append("experiment footprint >80GiB")
    for k in ("oom", "oom_kill", "oom_group_kill"):
        if now["oom_events"].get(k, 0) > initial["oom_events"].get(k, 0):
            errors.append(f"container {k} increased")
    if now["host_oom_kill"] > initial["host_oom_kill"]:
        errors.append("host OOM counter increased; attribution unresolved")
    return errors


def stop_child(proc):
    import psutil

    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except psutil.NoSuchProcess:
            pass
    proc.wait(timeout=10)


def disk_footprint(root):
    """Apparent bytes without following symlinks; deleted compiler temps are gone.

    Permission and I/O errors still propagate and stop the workload. Only ENOENT
    is tolerated for children that disappear during this live filesystem scan.
    """
    root = Path(root)
    root.lstat()  # A missing budget root is an error.
    stack = [root]
    seen = set()
    total = 0
    while stack:
        path = stack.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            if path == root:
                raise
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += info.st_size
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as entries:
                    stack.extend(Path(entry.path) for entry in entries)
            except FileNotFoundError:
                if path == root:
                    raise
    return total


def supervise(job):
    out = inside_repo(job["output"])
    root = inside_repo(job["budget_root"])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lock = None
    if job.get("gpu"):
        lock_path = inside_repo(job.get("gpu_lock_file", root / ".gpu-owner.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(lock_path, "a")
    if lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = snapshot(root)
    footprint = disk_footprint(root)
    reasons = violations(initial, initial, footprint)
    if job.get("gpu"):
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if gpu:
            reasons.append("GPUs already occupied")
    for port in job.get("ports", []):
        try:
            with socket.socket() as sock:
                # Match TCP server restart semantics while still refusing a listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
        except OSError:
            reasons.append(f"port {port} already occupied")
    if reasons:
        save(
            out / "status.json",
            dict(state="safety_stop", reasons=reasons, resources=initial),
        )
        save(out / "baseline.json", initial)
        save(out / "final-resources.json", initial)
        (out / "DONE").write_text("preflight stopped; no child launched\n")
        return
    save(out / "baseline.json", initial)
    interrupted = False

    def interrupt(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    disk_state = {"bytes": footprint, "updated": time.monotonic(), "error": None}
    disk_stop = threading.Event()

    def scan_disk():
        while not disk_stop.wait(30):
            try:
                disk_state["bytes"] = disk_footprint(root)
                disk_state["updated"] = time.monotonic()
            except BaseException as exc:
                disk_state["error"] = repr(exc)
                return

    # Directory scans cannot block the 0.5-second RAM/OOM monitor.
    threading.Thread(target=scan_disk, daemon=True).start()
    proc = None
    tick = 0
    deadline = time.monotonic() + job.get("timeout_s", 14400)
    try:
        with (out / "command.log").open("x") as log:
            proc = subprocess.Popen(
                job["argv"],
                env={**os.environ, **job.get("env", {})},
                cwd=job["cwd"],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        while True:
            now = snapshot(root)
            if disk_state["error"]:
                raise RuntimeError("disk monitor failed: " + disk_state["error"])
            if time.monotonic() - disk_state["updated"] > 90:
                raise RuntimeError(
                    "disk monitor has not completed a scan within 90 seconds"
                )
            footprint = disk_state["bytes"]
            reasons = violations(now, initial, footprint)
            state = dict(
                state="running", pid=proc.pid, resources=now, footprint_bytes=footprint
            )
            if reasons:
                state.update(state="safety_stop", reasons=reasons)
            elif interrupted or (out / "STOP").exists():
                state["state"] = "stopped"
            elif time.monotonic() > deadline:
                state["state"] = "timeout"
            elif proc.poll() is not None:
                state.update(state="exited", returncode=proc.returncode)
            if tick % 4 == 0 or state["state"] != "running":
                save(out / "status.json", state)
                with (out / "resources.jsonl").open("a") as f:
                    f.write(json.dumps(now) + "\n")
            if state["state"] != "running":
                break
            tick += 1
            time.sleep(0.5)
    except BaseException as exc:
        save(out / "status.json", dict(state="monitor_error", error=repr(exc)))
        raise
    finally:
        disk_stop.set()
        if proc:
            stop_child(proc)
        save(out / "final-resources.json", snapshot(root))
        (out / "DONE").write_text("all owned command processes stopped\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "supervise", "status", "stop"])
    p.add_argument("path")
    a = p.parse_args()
    path = inside_repo(a.path)
    if a.action in ("status", "stop"):
        if a.action == "stop":
            (path / "STOP").write_text("explicit stop\n")
        status = path / "status.json"
        print(status.read_text() if status.exists() else '{"state":"starting"}')
        return
    job = json.loads(path.read_text())
    out = inside_repo(job["output"])
    inside_repo(job["budget_root"])
    inside_repo(job["cwd"])
    if a.action == "start":
        out.mkdir(parents=True, exist_ok=False)
        with (out / "supervisor.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, "-B", __file__, "supervise", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"supervisor_pid": child.pid, "output": str(out)}))
    else:
        try:
            supervise(job)
        except BaseException as exc:
            if not (out / "status.json").exists():
                save(out / "status.json", dict(state="monitor_error", error=repr(exc)))
            raise


if __name__ == "__main__":
    main()
