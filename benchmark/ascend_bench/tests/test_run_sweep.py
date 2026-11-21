import run_sweep
from asc_bench.config import ConfigError, load_config

TWO_CELL = """
name: resume
model:
  path: Qwen/Qwen3-8B
server:
  args:
    --device: npu
workload:
  dataset_name: random
  axes:
    --max-concurrency: [1, 2]
  num_prompts_mult: 2
sla:
  thresholds: {}
run:
  repeats: 1
"""

VALID = """
name: cli
model:
  path: Qwen/Qwen3-8B
server:
  args:
    --device: npu
  axes:
    --mem-fraction-static: [0.8]
workload:
  dataset_name: random
  axes:
    --max-concurrency: [4]
  num_prompts_mult: 2
sla:
  thresholds:
    p99_ttft_ms: 2000
run:
  repeats: 1
"""


def _stub_runner(monkeypatch, calls):
    monkeypatch.setattr(run_sweep, "capture", lambda *a, **k: {"git_sha": "x"})
    monkeypatch.setattr(run_sweep, "hbm_used_mb", lambda: {0: 100})

    def fake_run_cell(self, cell):
        calls.append(cell.cell_id)
        self.manifest.append(
            cell.cell_id,
            cell.cell_hash,
            "done",
            metrics={"output_throughput": 10.0},
        )
        return "done"

    monkeypatch.setattr(run_sweep.Runner, "run_cell", fake_run_cell)


def test_resume_reuses_run_id_and_skips_done(tmp_path, monkeypatch):
    calls: list[str] = []
    _stub_runner(monkeypatch, calls)
    cfg = write_cfg(tmp_path, TWO_CELL)
    workdir = tmp_path / "runs"

    code = run_sweep.main(
        ["--config", str(cfg), "--workdir", str(workdir), "--run-id", "r1"]
    )
    assert code == 0
    assert sorted(calls) == ["qwen3-8b__s00__w00__r0", "qwen3-8b__s00__w01__r0"]
    assert (workdir / "r1" / "report.md").exists()

    code = run_sweep.main(
        ["--config", str(cfg), "--workdir", str(workdir), "--run-id", "r1"]
    )
    assert code == 0
    assert len(calls) == 2  # nothing re-ran


def test_filter_runs_only_matching_cells(tmp_path, monkeypatch):
    calls: list[str] = []
    _stub_runner(monkeypatch, calls)
    cfg = write_cfg(tmp_path, TWO_CELL)

    code = run_sweep.main(
        [
            "--config",
            str(cfg),
            "--workdir",
            str(tmp_path / "runs"),
            "--run-id",
            "r2",
            "--filter",
            "*w01*",
        ]
    )
    assert code == 0
    assert calls == ["qwen3-8b__s00__w01__r0"]


def write_cfg(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_dry_run_lists_cells_and_exits_zero(tmp_path, capsys):
    cfg = write_cfg(tmp_path, VALID)
    code = run_sweep.main(["--config", str(cfg), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "1 cell(s)" in out
    assert "qwen3-8b__s00__w00__r0" in out
    assert "conc=4 num_prompts=8" in out


def test_cli_rejects_invalid_config_with_exit_2(tmp_path, capsys):
    cfg = write_cfg(tmp_path, VALID.replace("p99_ttft_ms: 2000", "bogus: 1"))
    code = run_sweep.main(["--config", str(cfg), "--dry-run"])
    assert code == 2
    assert "config error" in capsys.readouterr().err


def test_dict_axis_entry_graph_bs_is_validated(tmp_path):
    bad = VALID.replace(
        "    --mem-fraction-static: [0.8]",
        """    --speculative-algorithm: [EAGLE]
    --cuda-graph-bs:
      - [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256]""",
    )
    expect_config_error(write_cfg(tmp_path, bad), "cuda-graph-bs")


def test_hbm_poll_must_be_positive(tmp_path):
    bad = VALID.replace("repeats: 1", "repeats: 1\n  hbm_poll_s: 0")
    expect_config_error(write_cfg(tmp_path, bad), "hbm_poll_s")


def expect_config_error(path, needle):
    try:
        load_config(path)
    except ConfigError as exc:
        assert needle in str(exc)
        return
    raise AssertionError(f"expected ConfigError mentioning {needle!r}")
