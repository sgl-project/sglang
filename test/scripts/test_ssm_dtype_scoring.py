"""Unit tests for scripts/ssm_dtype/{scoring,summarize_per_model,summarize_matrix}.py.

Run from repo root:
    pytest test/scripts/test_ssm_dtype_scoring.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ssm_dtype"))

import scoring  # noqa: E402
import summarize_matrix  # noqa: E402
import summarize_per_model  # noqa: E402
from write_run_config import build_config_from_env  # noqa: E402

# ---- scoring.extract_score -------------------------------------------------


def test_extract_score_prefers_score_key():
    assert (
        scoring.extract_score({"score": 0.9, "mean_score": 0.8, "accuracy": 0.7}) == 0.9
    )


def test_extract_score_falls_back_to_mean_score():
    assert scoring.extract_score({"mean_score": 0.8, "accuracy": 0.7}) == 0.8


def test_extract_score_falls_back_to_accuracy():
    assert scoring.extract_score({"accuracy": 0.7}) == 0.7


def test_extract_score_missing_returns_none():
    assert scoring.extract_score({"latency": 1.0}) is None


def test_extract_score_zero_is_returned_not_none():
    # Regression guard: ``metrics.get("score", metrics.get(...))`` in the old
    # heredoc returned the right thing for 0.0, but if anyone refactors with
    # ``or``-chaining, 0.0 would be falsy-coerced. Lock the contract.
    assert scoring.extract_score({"score": 0.0}) == 0.0


# ---- scoring.select_baseline_dtype -----------------------------------------


def test_select_baseline_prefers_float32():
    assert (
        scoring.select_baseline_dtype(["bfloat16", "float32", "float16"]) == "float32"
    )


def test_select_baseline_falls_back_to_first():
    assert scoring.select_baseline_dtype(["bfloat16", "float16"]) == "bfloat16"


def test_select_baseline_empty_raises():
    with pytest.raises(ValueError):
        scoring.select_baseline_dtype([])


# ---- scoring.compute_deltas ------------------------------------------------


def test_compute_deltas_per_model_shape():
    scores = {
        ("float32", "mmlu"): 0.80,
        ("bfloat16", "mmlu"): 0.78,
        ("float32", "gsm8k"): 0.50,
        ("bfloat16", "gsm8k"): 0.49,
    }
    deltas = scoring.compute_deltas(
        scores, ["float32", "bfloat16"], ["mmlu", "gsm8k"], "float32"
    )
    assert len(deltas) == 2
    assert deltas[0] == {
        "eval": "mmlu",
        "baseline_dtype": "float32",
        "dtype": "bfloat16",
        "baseline_score": 0.80,
        "score": 0.78,
        "delta": pytest.approx(-0.02),
    }
    assert deltas[1]["delta"] == pytest.approx(-0.01)


def test_compute_deltas_missing_score_yields_none_delta():
    scores: dict = {("float32", "mmlu"): 0.8}  # bfloat16 missing
    deltas = scoring.compute_deltas(
        scores, ["float32", "bfloat16"], ["mmlu"], "float32"
    )
    assert deltas[0]["delta"] is None
    assert deltas[0]["score"] is None


def test_compute_deltas_skips_baseline_dtype():
    scores = {("float32", "mmlu"): 0.8}
    deltas = scoring.compute_deltas(scores, ["float32"], ["mmlu"], "float32")
    assert deltas == []


def test_compute_deltas_matrix_form_uses_three_tuple_keys():
    scores = {
        ("modelA", "float32", "mmlu"): 0.80,
        ("modelA", "bfloat16", "mmlu"): 0.79,
    }
    deltas = scoring.compute_deltas(
        scores, ["float32", "bfloat16"], ["mmlu"], "float32", model="modelA"
    )
    assert deltas[0]["delta"] == pytest.approx(-0.01)


# ---- scoring formatting ----------------------------------------------------


def test_format_score_handles_none_and_value():
    assert scoring.format_score(None) == "N/A"
    assert scoring.format_score(0.5) == "0.5000000000"


def test_format_delta_sign_and_precision():
    assert scoring.format_delta(None) == "N/A"
    assert scoring.format_delta(0.01) == "+0.0100000000"
    assert scoring.format_delta(-0.01) == "-0.0100000000"


def test_render_score_table_layout():
    scores = {("float32", "mmlu"): 0.8, ("bfloat16", "mmlu"): None}
    lines = scoring.render_score_table(scores, ["float32", "bfloat16"], ["mmlu"])
    assert lines[0] == "| Eval | float32 | bfloat16 |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| mmlu | 0.8000000000 | N/A |"


def test_sort_dtypes_float32_first():
    assert scoring.sort_dtypes_float32_first(["bfloat16", "float32", "float16"]) == [
        "float32",
        "bfloat16",
        "float16",
    ]


# ---- summarize_per_model end-to-end ----------------------------------------


def _write_metrics(dir_: Path, dtype: str, eval_name: str, payload: dict) -> None:
    (dir_ / f"{dtype}_{eval_name}.metrics.json").write_text(json.dumps(payload))


def test_summarize_per_model_writes_expected_files(tmp_path: Path):
    _write_metrics(
        tmp_path,
        "float32",
        "mmlu",
        {"score": 0.80, "latency": 1.0, "output_throughput": 100.0},
    )
    _write_metrics(
        tmp_path,
        "bfloat16",
        "mmlu",
        {"score": 0.78, "latency": 0.9, "output_throughput": 110.0},
    )

    rc = summarize_per_model.main(
        [
            "--output-dir",
            str(tmp_path),
            "--model-label",
            "test_model",
            "--dtypes",
            "float32 bfloat16",
            "--evals",
            "mmlu",
        ]
    )
    assert rc == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {(r["dtype"], r["eval"]) for r in summary["rows"]} == {
        ("float32", "mmlu"),
        ("bfloat16", "mmlu"),
    }
    assert summary["deltas"][0]["delta"] == pytest.approx(-0.02)

    md = (tmp_path / "summary.md").read_text()
    assert "# test_model SSM State Dtype Accuracy" in md
    assert "| float32 | bfloat16 |" in md
    assert "-0.0200000000" in md  # delta row

    with (tmp_path / "summary.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 2
    assert {r["dtype"] for r in csv_rows} == {"float32", "bfloat16"}


def test_summarize_per_model_skips_missing_metrics(tmp_path: Path):
    # Only float32 present; bfloat16 row should be absent and delta None.
    _write_metrics(tmp_path, "float32", "mmlu", {"score": 0.8})

    summarize_per_model.main(
        [
            "--output-dir",
            str(tmp_path),
            "--model-label",
            "m",
            "--dtypes",
            "float32 bfloat16",
            "--evals",
            "mmlu",
        ]
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["rows"]) == 1
    assert summary["deltas"][0]["score"] is None
    assert summary["deltas"][0]["delta"] is None


# ---- summarize_matrix end-to-end -------------------------------------------


def _make_per_model_run(root: Path, model_label: str, rows: list[dict]) -> Path:
    """Create a per-model output_dir containing the summary.json that
    summarize_matrix consumes."""
    run_dir = root / model_label
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"rows": rows, "deltas": []}))
    return run_dir


def test_summarize_matrix_handles_ok_and_missing_runs(tmp_path: Path):
    runs_tsv = tmp_path / "runs.tsv"
    run_a_dir = _make_per_model_run(
        tmp_path,
        "model_a",
        [
            {
                "dtype": "float32",
                "eval": "mmlu",
                "score": 0.8,
                "latency": None,
                "output_throughput": None,
                "metrics_file": "float32_mmlu.metrics.json",
            },
            {
                "dtype": "bfloat16",
                "eval": "mmlu",
                "score": 0.78,
                "latency": None,
                "output_throughput": None,
                "metrics_file": "bfloat16_mmlu.metrics.json",
            },
        ],
    )
    missing_dir = tmp_path / "model_missing"
    # Don't write summary.json — this run should be flagged.

    with runs_tsv.open("w") as f:
        f.write("model_key\tmodel_label\tmodel_path\toutput_dir\n")
        f.write(f"a\tmodel_a\t/p/a\t{run_a_dir}\n")
        f.write(f"m\tmodel_missing\t/p/m\t{missing_dir}\n")

    matrix_out = tmp_path / "matrix"
    matrix_out.mkdir()

    rc = summarize_matrix.main(
        ["--runs-tsv", str(runs_tsv), "--matrix-output-root", str(matrix_out)]
    )
    assert rc == 0

    rows = json.loads((matrix_out / "matrix_summary.json").read_text())
    statuses = {r["status"] for r in rows}
    assert "ok" in statuses and "missing_summary" in statuses

    md = (matrix_out / "matrix_summary.md").read_text()
    assert "# Linear Attention SSM Dtype Accuracy Matrix" in md
    assert "## model_a" in md
    assert "Delta vs float32" in md
    assert "model_missing: missing_summary" in md
    # float32 must appear before bfloat16 in the table.
    assert md.index("float32") < md.index("bfloat16")


# ---- write_run_config ------------------------------------------------------


def _baseline_env(output_dir: Path) -> dict:
    return {
        "OUTPUT_DIR": str(output_dir),
        "MODEL_LABEL": "lab",
        "MODEL_PATH": "/m",
        "PYTHON_BIN": "/p",
        "PYTHONPATH_BASE": "/pp",
        "BASE_URL": "http://x",
        "DTYPES": "float32 bfloat16",
        "EVALS": "mmlu gsm8k",
        "MAMBA_SCHEDULER_STRATEGY": "extra_buffer",
        "ENABLE_DTYPE_PROBE": "1",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "LOCAL_NO_PROXY": "127.0.0.1",
        "TP_SIZE": "4",
        "PORT": "30001",
        "NUM_THREADS": "512",
        "MAX_TOKENS": "2048",
        "NUM_SHOTS": "5",
        "CHUNKED_PREFILL_SIZE": "2048",
        "MAMBA_TRACK_INTERVAL": "128",
        "TEMPERATURE": "0.0",
        "TOP_P": "1.0",
        "EXTRA_SERVER_ARGS": "--foo",
        "EXTRA_EVAL_ARGS": "",
    }


def test_write_run_config_typed_fields(tmp_path: Path):
    env = _baseline_env(tmp_path)
    config = build_config_from_env(env)
    assert config["tp_size"] == 4 and isinstance(config["tp_size"], int)
    assert config["temperature"] == 0.0 and isinstance(config["temperature"], float)
    assert config["dtypes"] == ["float32", "bfloat16"]
    assert config["evals"] == ["mmlu", "gsm8k"]
    assert config["extra_server_args"] == "--foo"
    assert config["extra_eval_args"] == ""


def test_write_run_config_optional_extras_default_to_empty(tmp_path: Path):
    env = _baseline_env(tmp_path)
    del env["EXTRA_SERVER_ARGS"]
    del env["EXTRA_EVAL_ARGS"]
    config = build_config_from_env(env)
    assert config["extra_server_args"] == ""
    assert config["extra_eval_args"] == ""
