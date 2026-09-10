"""Per-cell execution: server lifecycle, load generation, manifest."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

from asc_bench.cleanup import wait_hbm_freed
from asc_bench.config import BenchConfig, argv_from_args
from asc_bench.expand import Cell
from asc_bench.sla import parse_last_metrics

HEALTH_POLL_S = 2.0
KILL_GRACE_S = 5.0
DEFAULT_PORT = 30000
GSM8K_SCRIPT = Path(__file__).resolve().parents[2] / "gsm8k" / "bench_sglang.py"

# status enum: launching|healthy|benching|done|failed_launch|
#              failed_health_timeout|failed_bench|failed_hbm|failed_unknown
FAILURE_STATUSES = (
    "failed_launch",
    "failed_health_timeout",
    "failed_bench",
    "failed_hbm",
    "failed_unknown",
)


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    @property
    def pid(self) -> int: ...


class ProcessCtl(Protocol):
    """Injection seam so Runner is testable without real subprocesses."""

    def popen(
        self, cmd: list[str], env: dict[str, str], log_path: Path
    ) -> ProcessHandle: ...

    def killpg(self, pgid: int, sig: int) -> None: ...

    def http_ok(self, url: str, timeout: float) -> bool: ...


class RealProcs:
    """Production ProcessCtl: setsid children, logs to files, urllib health."""

    def popen(
        self, cmd: list[str], env: dict[str, str], log_path: Path
    ) -> ProcessHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log_fh:
            return subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
                cwd=str(log_path.parent),
            )

    def killpg(self, pgid: int, sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(pgid, sig)
            else:
                os.kill(pgid, sig)
        except ProcessLookupError:
            pass

    def http_ok(self, url: str, timeout: float) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return getattr(resp, "status", 200) == 200
        except OSError:
            return False


class Manifest:
    """Append-only JSONL: header line, then one status line per transition."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def header(self, payload: dict[str, Any]) -> None:
        self._write({"type": "run_header", **payload})

    def append(
        self,
        cell_id: str,
        cell_hash: str,
        status: str,
        *,
        detail: str | None = None,
        metrics: dict[str, Any] | None = None,
        accuracy: float | None = None,
        durations: dict[str, float] | None = None,
    ) -> None:
        self._write(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cell_id": cell_id,
                "cell_hash": cell_hash,
                "status": status,
                "detail": detail,
                "metrics": metrics,
                "gsm8k_accuracy": accuracy,
                "durations": durations,
            }
        )

    def _write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @classmethod
    def last_status_by_cell(cls, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("cell_id") and record.get("status"):
                    out[record["cell_id"]] = record["status"]
        return out

    @classmethod
    def result_rows(cls, path: Path) -> list[dict[str, Any]]:
        """All terminal per-cell records (done or failed_*), in file order."""
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = record.get("status")
                if status == "done" or status in FAILURE_STATUSES:
                    rows.append(record)
        return rows


def stderr_tail(path: Path, lines: int = 40) -> str | None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    tail = "\n".join(content.splitlines()[-lines:])
    return tail or None


class Runner:
    def __init__(
        self,
        cfg: BenchConfig,
        run_dir: Path,
        procs: ProcessCtl | None = None,
        hbm_probe: Callable[[], dict[int, int] | None] | None = None,
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.procs = procs or RealProcs()
        self.hbm_probe = hbm_probe
        self.log = log
        self.cells_dir = self.run_dir / "cells"
        self.manifest = Manifest(self.run_dir / "manifest.jsonl")

    # -- command builders --------------------------------------------------
    def server_argv(self, cell: Cell) -> list[str]:
        python = self.cfg.run.python or sys.executable
        return (
            [python, "-m", "sglang.launch_server"]
            + self.cfg.model.to_argv()
            + argv_from_args(cell.server_args)
        )

    def bench_argv(self, cell: Cell, bench_jsonl: Path) -> list[str]:
        python = self.cfg.run.python or sys.executable
        port = cell.server_args.get("--port", DEFAULT_PORT)
        argv = [
            python,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang-oai",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dataset-name",
            cell.dataset_name,
        ]
        argv += argv_from_args(cell.workload_args)
        if cell.num_prompts:
            argv += ["--num-prompts", str(cell.num_prompts)]
        if self.cfg.run.warmup_requests > 0:
            argv += ["--warmup-requests", str(self.cfg.run.warmup_requests)]
        argv += [
            "--seed",
            str(self.cfg.run.seed + cell.repeat),
            "--output-file",
            str(bench_jsonl),
            "--tag",
            cell.cell_id,
        ]
        return argv

    def gsm8k_argv(self, cell: Cell) -> list[str]:
        spec = self.cfg.run.gsm8k
        assert spec is not None
        python = self.cfg.run.python or sys.executable
        port = cell.server_args.get("--port", DEFAULT_PORT)
        argv = [
            python,
            str(GSM8K_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--num-questions",
            str(spec.num_questions),
            "--num-shots",
            str(spec.num_shots),
        ]
        if spec.data_path:
            argv += ["--data-path", spec.data_path]
        return argv

    # -- lifecycle ---------------------------------------------------------
    def _kill_tree(self, proc: ProcessHandle) -> None:
        self.procs.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + KILL_GRACE_S
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)
        if proc.poll() is None:
            self.procs.killpg(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def _wait_healthy(self, cell: Cell, proc: ProcessHandle) -> tuple[bool, float]:
        port = cell.server_args.get("--port", DEFAULT_PORT)
        url = f"http://127.0.0.1:{port}/health"
        started = time.monotonic()
        deadline = started + self.cfg.run.health_timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False, time.monotonic() - started
            if self.procs.http_ok(url, timeout=2.0):
                return True, time.monotonic() - started
            time.sleep(HEALTH_POLL_S)
        return False, time.monotonic() - started

    def _hbm_check(
        self,
        cell: Cell,
        baseline: dict[int, int] | None,
        durations: dict[str, float],
    ) -> bool:
        started = time.monotonic()
        freed = wait_hbm_freed(
            self.hbm_probe,
            baseline,
            budget_mb=self.cfg.run.hbm_budget_mb,
            timeout_s=self.cfg.run.hbm_timeout_s,
            poll_s=self.cfg.run.hbm_poll_s,
            log=lambda msg: self.log(f"[{cell.cell_id}] {msg}"),
        )
        durations["hbm_gate_s"] = round(time.monotonic() - started, 1)
        return freed

    def _run_gsm8k(self, cell: Cell, cdir: Path, env: dict[str, str]) -> float | None:
        out_path = cdir / "gsm8k.out"
        proc = None
        try:
            proc = self.procs.popen(self.gsm8k_argv(cell), env, out_path)
            rc = proc.wait(timeout=self.cfg.run.bench_timeout_s)
        except subprocess.TimeoutExpired:
            if proc is not None:
                self._kill_tree(proc)
            return None
        except Exception:  # noqa: BLE001
            return None
        if rc != 0:
            return None
        text = out_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Accuracy:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else None

    def _finish(
        self,
        cell: Cell,
        server: ProcessHandle,
        baseline: dict[int, int] | None,
        durations: dict[str, float],
        status: str,
        **extra: Any,
    ) -> str:
        """Kill the server, run the HBM gate, and record the terminal state."""
        self._kill_tree(server)
        freed = self._hbm_check(cell, baseline, durations)
        if status == "done" and not freed:
            status = "failed_hbm"
            extra["detail"] = "HBM not freed after kill; node may be dirty"
        self._record(cell, status, durations=durations, **extra)
        return status

    def run_cell(self, cell: Cell) -> str:
        """Execute one cell to a terminal status and return that status."""
        cdir = self.cells_dir / cell.cell_id
        cdir.mkdir(parents=True, exist_ok=True)
        bench_jsonl = cdir / "bench.jsonl"
        server_log = cdir / "server.log"
        durations: dict[str, float] = {}

        self._record(cell, "launching", detail=None)
        baseline = self.hbm_probe() if self.hbm_probe else None
        env = {**os.environ, **cell.env}
        try:
            server = self.procs.popen(self.server_argv(cell), env, server_log)
        except Exception as exc:  # noqa: BLE001
            self._record(cell, "failed_launch", detail=str(exc))
            return "failed_launch"

        try:
            healthy, health_s = self._wait_healthy(cell, server)
            durations["health_s"] = round(health_s, 1)
            if not healthy:
                if server.poll() is not None:
                    detail = f"server exited early rc={server.poll()}"
                else:
                    detail = f"health timeout after {self.cfg.run.health_timeout_s}s"
                return self._finish(
                    cell,
                    server,
                    baseline,
                    durations,
                    "failed_health_timeout",
                    detail=detail,
                )
            self._record(cell, "healthy", durations=durations)

            accuracy: float | None = None
            if self.cfg.run.gsm8k is not None:
                accuracy = self._run_gsm8k(cell, cdir, env)
                if accuracy is None:
                    return self._finish(
                        cell,
                        server,
                        baseline,
                        durations,
                        "failed_bench",
                        detail="gsm8k run failed or accuracy not parseable",
                    )

            self._record(cell, "benching", durations=durations)
            bench = None
            try:
                bench = self.procs.popen(
                    self.bench_argv(cell, bench_jsonl), env, cdir / "bench.log"
                )
                bench_rc = bench.wait(timeout=self.cfg.run.bench_timeout_s)
            except subprocess.TimeoutExpired:
                if bench is not None:
                    self._kill_tree(bench)
                bench_rc = -9
            if bench_rc != 0:
                return self._finish(
                    cell,
                    server,
                    baseline,
                    durations,
                    "failed_bench",
                    detail=f"bench_serving rc={bench_rc}",
                )

            metrics = parse_last_metrics(bench_jsonl)
            if metrics is None:
                return self._finish(
                    cell,
                    server,
                    baseline,
                    durations,
                    "failed_bench",
                    detail="no metrics record found in bench output",
                )
            return self._finish(
                cell,
                server,
                baseline,
                durations,
                "done",
                metrics=metrics,
                accuracy=accuracy,
            )
        except Exception as exc:  # noqa: BLE001 - never leave the node dirty
            self._kill_tree(server)
            self._record(
                cell,
                "failed_unknown",
                detail=f"{type(exc).__name__}: {exc}",
                durations=durations,
            )
            return "failed_unknown"

    # -- misc --------------------------------------------------------------
    def _record(self, cell: Cell, status: str, **kwargs: Any) -> None:
        detail = kwargs.get("detail")
        suffix = f" ({detail})" if detail else ""
        self.log(f"[{cell.cell_id}] {status}{suffix}")
        self.manifest.append(cell.cell_id, cell.cell_hash, status, **kwargs)
