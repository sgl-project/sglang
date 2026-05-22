"""Nightly precision regression CI test: dump per-layer hidden states and compare day-over-day.

Launches GLM-5.1 (or configurable models) on 8x H200 with the non-intrusive
dumper in "all" mode, sends a deterministic inference request, dumps sampled
per-layer tensors, then compares against a persistent baseline from the
previous successful run.

If the diff exceeds the threshold, the test fails — catching numerical
regressions introduced by code changes across commits.

Baseline management:
- Baseline stored at SGLANG_PRECISION_BASELINE_DIR (default: /tmp/sglang_precision_baselines)
- First run: establishes baseline, no comparison.
- Subsequent runs: compare with baseline, update baseline on success.
- SGLANG_PRECISION_FORCE_UPDATE=1 skips comparison and unconditionally updates baseline.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=3600, suite="nightly-precision-8-gpu-h200", nightly=True)

DEFAULT_MODELS_FOR_NIGHTLY_PRECISION = "zai-org/GLM-5.1-FP8"
DEFAULT_DIFF_THRESHOLD = 1e-3
DUMPER_FILTER = "layer_id in [0, 19, 39, 58, 77]"
EXP_NAME = "nightly_precision"
PROMPT = "The capital of France is"
NIGHTLY_PRECISION_SERVER_TIMEOUT = 1800


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class TestNightlyPrecisionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models_str = os.environ.get(
            "SGLANG_PRECISION_MODELS", DEFAULT_MODELS_FOR_NIGHTLY_PRECISION
        )
        cls.models = [
            ModelLaunchSettings(m, tp_size=8) for m in parse_models(models_str)
        ]
        cls.baseline_dir = Path(
            os.environ.get(
                "SGLANG_PRECISION_BASELINE_DIR", "/tmp/sglang_precision_baselines"
            )
        )
        cls.baseline_dir.mkdir(parents=True, exist_ok=True)
        cls.diff_threshold = float(
            os.environ.get(
                "SGLANG_PRECISION_DIFF_THRESHOLD", str(DEFAULT_DIFF_THRESHOLD)
            )
        )
        cls.force_update = os.environ.get("SGLANG_PRECISION_FORCE_UPDATE", "0") == "1"
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_precision_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        all_results = []
        for model_setup in self.models:
            with self.subTest(model=model_setup.model_path):
                try:
                    result = _test_one_model(
                        model_setup=model_setup,
                        baseline_dir=self.baseline_dir,
                        diff_threshold=self.diff_threshold,
                        force_update=self.force_update,
                        base_url=self.base_url,
                    )
                    all_results.append(result)
                except Exception as e:
                    all_results.append((model_setup.model_path, "ERROR", str(e)))

        _report_summary(all_results)

        failed = [r for r in all_results if r[1] == "FAILED"]
        errored = [r for r in all_results if r[1] == "ERROR"]
        if failed or errored:
            msg = "Nightly precision regression failures:\n"
            for model, status, details in failed + errored:
                msg += f"  {model}: {status} - {details}\n"
            self.fail(msg)


def _test_one_model(
    *,
    model_setup: ModelLaunchSettings,
    baseline_dir: Path,
    diff_threshold: float,
    force_update: bool,
    base_url: str,
):
    model = model_setup.model_path
    model_dir_name = _sanitize_model_name(model)
    model_baseline_dir = baseline_dir / model_dir_name
    baseline_exp_dir = model_baseline_dir / EXP_NAME

    with tempfile.TemporaryDirectory() as today_tmp:
        today_dump_dir = Path(today_tmp)
        _run_server_and_dump(
            model_setup=model_setup,
            dump_dir=today_dump_dir,
            base_url=base_url,
        )
        today_exp_dir = today_dump_dir / EXP_NAME

        has_baseline = baseline_exp_dir.exists() and any(baseline_exp_dir.glob("*.pt"))

        if has_baseline and not force_update:
            result = _run_comparator(
                baseline=baseline_exp_dir,
                target=today_exp_dir,
                threshold=diff_threshold,
            )
            debug_file = _save_comparator_output(
                stdout=result.stdout, stderr=result.stderr, prefix=model_dir_name
            )
            print(f"Comparator output for {model}: {debug_file}")

            if result.returncode == 0:
                _update_baseline(model_baseline_dir, today_exp_dir)
                return (model, "PASSED", "comparison ok, baseline updated")
            else:
                summary = _extract_diff_summary(result.stdout)
                return (model, "FAILED", summary)
        else:
            _update_baseline(model_baseline_dir, today_exp_dir)
            reason = "forced update" if force_update else "first run"
            return (model, "BASELINE_ESTABLISHED", reason)


def _run_server_and_dump(
    *, model_setup: ModelLaunchSettings, dump_dir: Path, base_url: str
):
    env: dict[str, str] = {
        **os.environ,
        "DUMPER_DIR": str(dump_dir),
        "DUMPER_EXP_NAME": EXP_NAME,
        "DUMPER_SERVER_PORT": "reuse",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
    }

    server_args: list[str] = list(model_setup.extra_args) + [
        "--max-total-tokens",
        "128",
        "--mem-fraction-static",
        "0.9",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
    ]

    proc = popen_launch_server(
        model_setup.model_path,
        base_url,
        timeout=NIGHTLY_PRECISION_SERVER_TIMEOUT,
        other_args=server_args,
        env=env,
    )
    try:
        requests.post(
            f"{base_url}/dumper/configure",
            json={
                "enable": True,
                "filter": DUMPER_FILTER,
                "cleanup_previous": True,
            },
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_setup.model_path,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, f"Chat completions failed: {resp.text}"
    finally:
        kill_process_tree(proc.pid)


def _run_comparator(
    *, baseline: Path, target: Path, threshold: float
) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = [
        "python",
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline),
        "--target-path",
        str(target),
        "--diff-threshold",
        str(threshold),
        "--output-format",
        "json",
        "--allow-skipped-pattern",
        "input_ids|positions|seq_lens|req_pool_indices|rids",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _update_baseline(model_baseline_dir: Path, today_exp_dir: Path):
    final_dir = model_baseline_dir / EXP_NAME
    staging_dir = model_baseline_dir / "_staging"
    old_dir = model_baseline_dir / "_old_baseline"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    shutil.copytree(today_exp_dir, staging_dir)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _get_git_commit(),
    }
    (staging_dir.parent / "baseline_meta.json").write_text(json.dumps(meta, indent=2))

    if final_dir.exists():
        if old_dir.exists():
            shutil.rmtree(old_dir)
        final_dir.rename(old_dir)

    staging_dir.rename(final_dir)

    if old_dir.exists():
        shutil.rmtree(old_dir)


def _extract_diff_summary(stdout: str) -> str:
    try:
        for line in stdout.strip().splitlines():
            record = json.loads(line)
            if record.get("type") == "comparison_tensor" and not record.get(
                "passed", True
            ):
                name = record.get("name", "unknown")
                rel_diff = record.get("rel_diff", "N/A")
                return f"tensor={name} rel_diff={rel_diff}"
    except (json.JSONDecodeError, KeyError):
        pass
    return stdout[-200:] if stdout else "no output"


def _save_comparator_output(
    *, stdout: str, stderr: str, prefix: str
) -> Path:
    fd, path_str = tempfile.mkstemp(
        prefix=f"nightly_precision_{prefix}_", suffix=".log", dir="/tmp"
    )
    with os.fdopen(fd, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)
    return Path(path_str)


def _report_summary(results):
    lines = ["\n" + "=" * 60]
    lines.append("Nightly Precision Regression Summary")
    lines.append("=" * 60)
    lines.append(f"{'Model':<45} {'Status':<25} Details")
    lines.append("-" * 60)

    for model, status, details in results:
        lines.append(f"{model:<45} {status:<25} {details[:80]}")

    lines.append("=" * 60)
    summary = "\n".join(lines)
    print(summary, flush=True)

    if is_in_ci():
        md = "## Nightly Precision Regression\n\n"
        md += "| Model | Status | Details |\n"
        md += "|-------|--------|--------|\n"
        for model, status, details in results:
            md += f"| {model} | {status} | {details[:100]} |\n"
        write_github_step_summary(md)


if __name__ == "__main__":
    unittest.main()
