"""Run a subprocess and kill it if stdout goes silent for too long.

Motivating failure mode on Intel BMG runners: the SGLang server prints
`Multi-thread loading shards: 0% Completed`, then produces no further
output for 9 minutes, then the outer 600s `popen_launch_server` timeout
fires — with no stack trace and no diagnostic hint. The Level-Zero
runtime has silently deadlocked mid weight-load.

This wrapper watches the child's combined stdout+stderr. If nothing is
written for `--stall-secs` seconds, we assume the process is wedged, dump
XPU diagnostics, and SIGKILL the process tree. Exit code is 124 on stall
(matches GNU `timeout`), otherwise the child's own exit code.

We stream the child's output through so live logs still work; we're only
observing timing, not capturing.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CMD... under a stdout-progress watchdog."
    )
    p.add_argument(
        "--stall-secs",
        type=int,
        default=int(os.environ.get("XPU_STALL_SECS", "120")),
        help="Kill the child if stdout is silent for this many seconds "
        "(default: 120, override with $XPU_STALL_SECS).",
    )
    p.add_argument(
        "--diagnostic-script",
        type=Path,
        default=None,
        help="Path to a diagnostic script to run when a stall is detected, "
        "before killing the child. Defaults to sibling dump_xpu_diagnostics.sh.",
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run. Everything after -- is passed verbatim.",
    )
    args = p.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        p.error("no command supplied")
    if args.diagnostic_script is None:
        args.diagnostic_script = Path(__file__).resolve().parent / "dump_xpu_diagnostics.sh"
    return args


def _kill_tree(pid: int) -> None:
    # Send SIGTERM to the whole process group first; if the child spawned
    # its own subprocesses (SGLang launches scheduler + tokenizer + detokenizer
    # workers), plain SIGKILL on `pid` leaks them. Falling back to SIGKILL
    # after a grace period matches what shells do on Ctrl-C.
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(50):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    args = parse_args()

    # Use a new process group so we can nuke the whole tree at once.
    proc = subprocess.Popen(
        args.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        start_new_session=True,
    )

    last_output_at = time.monotonic()
    lock = threading.Lock()

    def _pump():
        assert proc.stdout is not None
        nonlocal last_output_at
        for line in proc.stdout:
            with lock:
                last_output_at = time.monotonic()
            sys.stdout.write(line)
            sys.stdout.flush()

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    stalled = False
    while True:
        rc = proc.poll()
        if rc is not None:
            # Child exited on its own.
            pump.join(timeout=2)
            return rc

        with lock:
            idle = time.monotonic() - last_output_at

        if idle >= args.stall_secs:
            stalled = True
            sys.stdout.write(
                f"\n[watchdog] no output for {int(idle)}s "
                f"(threshold {args.stall_secs}s). Assuming XPU stall.\n"
            )
            sys.stdout.flush()
            break

        # Sleep the smaller of 5s or the remaining budget, so we react
        # promptly near the deadline without busy-spinning early.
        time.sleep(min(5.0, max(1.0, args.stall_secs - idle)))

    # Dump diagnostics BEFORE killing so the still-live process shows up in
    # `fuser` / `lsof` output.
    diag = args.diagnostic_script
    if diag.exists():
        sys.stdout.write(f"[watchdog] running diagnostic: {diag}\n")
        sys.stdout.flush()
        try:
            subprocess.run(["bash", str(diag)], timeout=90)
        except subprocess.TimeoutExpired:
            sys.stdout.write("[watchdog] diagnostic script timed out.\n")
    else:
        sys.stdout.write(f"[watchdog] diagnostic script missing: {diag}\n")

    _kill_tree(proc.pid)
    pump.join(timeout=2)
    return 124 if stalled else (proc.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())
