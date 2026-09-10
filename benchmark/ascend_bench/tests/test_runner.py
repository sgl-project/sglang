import json
from pathlib import Path

from asc_bench.config import load_config
from asc_bench.expand import expand_cells
from asc_bench.runner import Manifest, Runner

CFG = """
name: unit
model:
  path: Qwen/Qwen3-8B
server:
  args:
    --port: 30000
workload:
  dataset_name: random
sla:
  thresholds:
    p99_ttft_ms: 2000
run:
  repeats: 1
  health_timeout_s: 5
  bench_timeout_s: 60
  hbm_timeout_s: 1
  hbm_poll_s: 0.1
"""


class FakeProc:
    def __init__(self, rc=None):
        self.pid = 4321
        self._rc = rc

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc if self._rc is not None else 0


class FakeProcs:
    def __init__(self, script=None):
        self.script = script or {}
        self.kill_signals = []
        self.procs = []

    def popen(self, cmd, env, log_path):
        if self.script.get("popen_raises"):
            raise RuntimeError("boom")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if any(part == "sglang.launch_server" for part in cmd):
            proc = FakeProc(self.script.get("server_rc"))
        else:
            proc = FakeProc(self.script.get("bench_rc", 0))
            if "--output-file" in cmd:
                out = Path(cmd[cmd.index("--output-file") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    '{"output_throughput": 123.4, "p99_ttft_ms": 800}\n',
                    encoding="utf-8",
                )
        self.procs.append(proc)
        return proc

    def killpg(self, pgid, sig):
        self.kill_signals.append(sig)
        for proc in self.procs:
            if proc.pid == pgid and proc._rc is None:
                proc._rc = -sig

    def http_ok(self, url, timeout):
        value = self.script.get("http", True)
        if isinstance(value, list):
            value = bool(value.pop(0)) if value else False
        return value


def make_runner(tmp_path, script=None, probe=lambda: {0: 100}, cfg_text=CFG):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    cell = expand_cells(cfg)[0]
    runner = Runner(cfg, tmp_path / "run", procs=FakeProcs(script), hbm_probe=probe)
    return runner, cell


def statuses(manifest_path):
    return [
        json.loads(line)["status"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_happy_path_reaches_done(tmp_path):
    runner, cell = make_runner(tmp_path)
    status = runner.run_cell(cell)
    assert status == "done"
    log = statuses(runner.manifest.path)
    assert log[0] == "launching"
    assert "healthy" in log and "benching" in log
    assert log[-1] == "done"
    final = json.loads(
        runner.manifest.path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert final["metrics"]["output_throughput"] == 123.4


def test_spawn_failure_maps_to_failed_launch(tmp_path):
    runner, cell = make_runner(tmp_path, script={"popen_raises": True})
    assert runner.run_cell(cell) == "failed_launch"


def test_early_server_exit_maps_to_failed_health(tmp_path):
    runner, cell = make_runner(tmp_path, script={"server_rc": 1})
    assert runner.run_cell(cell) == "failed_health_timeout"


def test_health_never_ok_times_out(tmp_path):
    zero_timeout = CFG.replace("health_timeout_s: 5", "health_timeout_s: 0")
    runner, cell = make_runner(tmp_path, script={"http": False}, cfg_text=zero_timeout)
    assert runner.run_cell(cell) == "failed_health_timeout"


def test_bench_failure_maps_to_failed_bench(tmp_path):
    runner, cell = make_runner(tmp_path, script={"bench_rc": 1})
    assert runner.run_cell(cell) == "failed_bench"


def test_stuck_hbm_upgrades_done_to_failed_hbm(tmp_path):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return {0: 100} if calls["n"] == 1 else {0: 9000}

    runner, cell = make_runner(tmp_path, probe=probe)
    assert runner.run_cell(cell) == "failed_hbm"


def test_manifest_roundtrip(tmp_path):
    runner, cell = make_runner(tmp_path)
    runner.run_cell(cell)
    last = Manifest.last_status_by_cell(runner.manifest.path)
    assert last[cell.cell_id] == "done"
    rows = Manifest.result_rows(runner.manifest.path)
    assert rows[-1]["status"] == "done"
    assert rows[-1]["cell_hash"] == cell.cell_hash
