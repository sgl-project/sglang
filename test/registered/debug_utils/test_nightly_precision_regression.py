"""Nightly precision regression CI test: dump per-layer hidden states and
compare day-over-day against a rolling baseline.

Env knobs:
  SGLANG_PRECISION_MODELS         comma-separated model ids (default GLM-5.2-FP8)
  SGLANG_PRECISION_BASELINE_DIR   local baseline dir
  SGLANG_PRECISION_DIFF_THRESHOLD per-tensor rel_diff cutoff (default 1e-3)
  SGLANG_PRECISION_FORCE_UPDATE=1 skip comparison, refresh baseline
  SGLANG_PRECISION_COMMIT         override sglang sha (7-40 hex) tagged on push
  SGLANG_PRECISION_HF_REPO        required HF dataset repo for cross-runner
                                  baseline storage; see precision_baseline_store
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

# Soft dep: missing huggingface_hub → import fails loudly in setUpClass.
try:
    from sglang.test import precision_baseline_store as _hfs
except Exception:  # pragma: no cover
    _hfs = None

register_cuda_ci(est_time=3600, suite="nightly-precision-8-gpu-h200", nightly=True)

DEFAULT_MODELS_FOR_NIGHTLY_PRECISION = "zai-org/GLM-5.2-FP8"
DEFAULT_DIFF_THRESHOLD = 1e-3
# Fallback when the layer count can't be resolved: never silently shrink coverage.
DUMPER_FILTER_ALL_LAYERS = (
    r"match(r'^non_intrusive__model\.layers\.\d+\.inputs\.1$', name)"
)
LAYER_CAPTURE_STRIDE = 8
MAX_TOKENS = 2
SCHEMA_VERSION = 3
EXP_NAME = "nightly_precision"
PROMPT = "The capital of France is"
NIGHTLY_PRECISION_SERVER_TIMEOUT = 3600
# Pin fusion ON: captured inputs.1 must be TP-partial (the comparator's tp:partial
# contract), and SM90 auto-enable was dropped in #23402.
PRECISION_FUSION_BACKEND = "trtllm"


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def _select_capture_layers(
    num_layers: int, stride: int = LAYER_CAPTURE_STRIDE
) -> list[int]:
    # First + last + every stride-th: the residual stream is cumulative, so the
    # last layer still reflects drift originating in any earlier layer.
    layers = set(range(0, num_layers, stride))
    layers.add(0)
    layers.add(num_layers - 1)
    return sorted(i for i in layers if 0 <= i < num_layers)


def _build_dumper_filter(capture_layers: Optional[list[int]]) -> str:
    if not capture_layers:
        return DUMPER_FILTER_ALL_LAYERS
    # The dumper evals this with builtins stripped (no int()/arithmetic), so the
    # layer subset is baked into the regex; longest-first avoids ambiguous matches.
    alt = "|".join(
        str(i) for i in sorted(capture_layers, key=lambda x: (-len(str(x)), x))
    )
    return rf"match(r'^non_intrusive__model\.layers\.({alt})\.inputs\.1$', name)"


def _resolve_num_layers(model_path: str) -> Optional[int]:
    # None on any failure -> caller captures every layer rather than a wrong subset.
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"could not load config for {model_path}: {e}")
        return None
    text_config = getattr(config, "text_config", None)  # VL configs nest LM dims here
    for obj in (text_config, config):
        if obj is None:
            continue
        for attr in ("num_hidden_layers", "num_layers"):
            n = getattr(obj, attr, None)
            if isinstance(n, int) and n > 0:
                return n
    warnings.warn(f"could not read layer count from config for {model_path}")
    return None


def _assert_decode_captured(exp_dir: Path, *, tp_size: int) -> None:
    # One dump per (layer, rank) == prefill only: decode never ran, which would
    # pass the comparison while silently halving coverage. Fail loudly instead.
    pts = list(exp_dir.glob("*.pt"))
    if not pts:
        raise AssertionError(f"no .pt dumps produced in {exp_dir}")
    layers: set[str] = set()
    steps: set[str] = set()
    for p in pts:
        for kv in p.stem.split("___"):
            if kv.startswith("layer_id="):
                layers.add(kv[len("layer_id=") :])
            elif kv.startswith("step="):
                steps.add(kv[len("step=") :])
    prefill_only = len(layers) * tp_size
    if len(pts) <= prefill_only:
        raise AssertionError(
            f"decode path not captured: {len(pts)} .pt files for {len(layers)} "
            f"layers x {tp_size} tp (== {prefill_only}, prefill-only); "
            f"steps={sorted(steps)}. The model generated no decode tokens "
            f"(check --max-total-tokens vs the decode reservation, max_tokens, "
            f"and ignore_eos)."
        )


def _capture_signature(dump_cfg: dict[str, Any], tp_size: int) -> str:
    # Identifies the dump shape; fetch only compares against baselines with the
    # same signature, so changing layers/forwards/tp re-establishes cleanly
    # instead of erroring against incompatible tensors.
    raw = "|".join(
        str(x)
        for x in (
            SCHEMA_VERSION,
            dump_cfg["max_tokens"],
            dump_cfg["ignore_eos"],
            tp_size,
            dump_cfg["dumper_filter"],
            dump_cfg["fusion_backend"],
        )
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _get_git_commit() -> str:
    val = os.environ.get("SGLANG_PRECISION_COMMIT", "").strip()
    if _SHA_RE.match(val):
        return val
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], timeout=10)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


_NVIDIA_SMI_NAME_RE = re.compile(r"NVIDIA\s+([A-Za-z0-9]+)")


def _collect_runtime_context() -> dict[str, Any]:
    # Each probe is independently guarded — missing nvidia-smi/torch never blocks.
    ctx: dict[str, Any] = {}
    try:
        import torch

        ctx["torch_version"] = torch.__version__
        if torch.version.cuda:
            ctx["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            ctx["num_gpus"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        import sglang

        ctx["sglang_version"] = getattr(sglang, "__version__", None)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        m = _NVIDIA_SMI_NAME_RE.search(out)
        if m:
            ctx["hardware"] = m.group(1)
    except Exception:
        pass
    return ctx


def _collect_ci_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        ctx["ci_run_id"] = run_id
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if repo:
            ctx["ci_run_url"] = f"{server}/{repo}/actions/runs/{run_id}"
    for env_key, meta_key in (
        ("GITHUB_ACTOR", "ci_actor"),
        ("GITHUB_REF", "git_ref"),
        ("GITHUB_WORKFLOW", "ci_workflow"),
    ):
        v = os.environ.get(env_key)
        if v:
            ctx[meta_key] = v
    return ctx


def _parse_comparator_stats(stdout: str) -> dict[str, Any]:
    # Only passing records contribute to max/mean — failed records' rel_diff
    # would corrupt the headline. NaN/inf is dropped before aggregation.
    n_total = 0
    n_passed = 0
    n_failed = 0
    rel_diffs: list[float] = []
    abs_diffs: list[float] = []
    failing: list[str] = []

    def _record_failure(name: str) -> None:
        nonlocal n_failed
        n_failed += 1
        if len(failing) < 10:
            failing.append(name)

    def _safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    for line in stdout.strip().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "comparison_tensor":
            continue
        n_total += 1
        name = rec.get("name", "?")
        diff = rec.get("diff") or {}
        if rec.get("errors"):
            _record_failure(name)
            continue
        if not diff.get("passed", True):
            _record_failure(name)
            continue
        bad_rep = next(
            (c for c in rec.get("replicated_checks", []) if not c.get("passed", True)),
            None,
        )
        if bad_rep is not None:
            _record_failure(name)
            continue
        rel = _safe_float(diff.get("rel_diff"))
        if rel is not None:
            rel_diffs.append(rel)
        abs_ = _safe_float(diff.get("abs_diff"))
        if abs_ is not None:
            abs_diffs.append(abs_)
        n_passed += 1

    out: dict[str, Any] = {
        "num_layers_compared": n_total,
        "num_layers_passed": n_passed,
        "num_layers_failed": n_failed,
    }
    if rel_diffs:
        out["max_rel_diff"] = max(rel_diffs)
        out["mean_rel_diff"] = sum(rel_diffs) / len(rel_diffs)
    if abs_diffs:
        out["max_abs_diff"] = max(abs_diffs)
    if failing:
        out["failing_layers"] = failing
    return out


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

        if _hfs is None:
            raise RuntimeError(
                "precision baseline store unavailable: could not import "
                "sglang.test.precision_baseline_store"
            )
        # Raises if SGLANG_PRECISION_HF_REPO is unset — the test requires a
        # remote baseline store, there is no local-only mode.
        cls.hf_cfg = _hfs.HfStoreConfig.from_env()

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
                        hf_cfg=self.hf_cfg,
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
    hf_cfg,
):
    model = model_setup.model_path
    model_dir_name = _sanitize_model_name(model)
    model_baseline_dir = baseline_dir / model_dir_name
    baseline_exp_dir = model_baseline_dir / EXP_NAME

    # Resolve the capture shape once and reuse it for the dump request and every
    # meta push, so the manifest always reflects exactly what was dumped.
    num_layers = _resolve_num_layers(model)
    capture_layers = _select_capture_layers(num_layers) if num_layers else None
    dump_cfg = {
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "dumper_filter": _build_dumper_filter(capture_layers),
        "num_hidden_layers": num_layers,
        "capture_layers": capture_layers,
        "fusion_backend": PRECISION_FUSION_BACKEND,
    }
    dump_cfg["capture_signature"] = _capture_signature(dump_cfg, model_setup.tp_size)

    _maybe_hf_fetch(
        hf_cfg=hf_cfg,
        model=model,
        baseline_exp_dir=baseline_exp_dir,
        capture_signature=dump_cfg["capture_signature"],
    )

    with tempfile.TemporaryDirectory() as today_tmp:
        today_dump_dir = Path(today_tmp)
        _run_server_and_dump(
            model_setup=model_setup,
            dump_dir=today_dump_dir,
            base_url=base_url,
            dumper_filter=dump_cfg["dumper_filter"],
            max_tokens=dump_cfg["max_tokens"],
            ignore_eos=dump_cfg["ignore_eos"],
        )
        today_exp_dir = today_dump_dir / EXP_NAME
        _assert_decode_captured(today_exp_dir, tp_size=model_setup.tp_size)

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
            report_path = model_baseline_dir / "comparator_report.jsonl"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result.stdout, encoding="utf-8")
            comparator_stats = _parse_comparator_stats(result.stdout)
            if comparator_stats.get("num_layers_compared", 0) == 0:
                # A clean returncode with nothing compared means the baseline
                # and target tensor names never lined up — fail loudly rather
                # than pass on an empty comparison.
                return (
                    model,
                    "FAILED",
                    "comparator compared 0 layers (baseline/target name mismatch?)",
                )

            if result.returncode == 0:
                _update_baseline(model_baseline_dir, today_exp_dir)
                _maybe_hf_push(
                    hf_cfg=hf_cfg,
                    model=model,
                    model_setup=model_setup,
                    tensors_dir=baseline_exp_dir,
                    pass_label="passed",
                    diff_threshold=diff_threshold,
                    dump_cfg=dump_cfg,
                    comparator_report=report_path,
                    comparator_stats=comparator_stats,
                )
                return (model, "PASSED", "comparison ok, baseline updated")
            # FAILED: push today's tensors as pass_label="failed" so the
            # diagnostic diff survives on HF without rerunning the comparator.
            _maybe_hf_push(
                hf_cfg=hf_cfg,
                model=model,
                model_setup=model_setup,
                tensors_dir=today_exp_dir,
                pass_label="failed",
                diff_threshold=diff_threshold,
                dump_cfg=dump_cfg,
                comparator_report=report_path,
                comparator_stats=comparator_stats,
            )
            summary = _extract_diff_summary(result.stdout)
            return (model, "FAILED", summary)
        else:
            _update_baseline(model_baseline_dir, today_exp_dir)
            _maybe_hf_push(
                hf_cfg=hf_cfg,
                model=model,
                model_setup=model_setup,
                tensors_dir=baseline_exp_dir,
                pass_label="baseline_established",
                diff_threshold=diff_threshold,
                dump_cfg=dump_cfg,
                comparator_report=None,
                comparator_stats=None,
            )
            reason = "forced update" if force_update else "first run"
            return (model, "BASELINE_ESTABLISHED", reason)


def _maybe_hf_fetch(
    *, hf_cfg, model: str, baseline_exp_dir: Path, capture_signature: str
) -> None:
    if baseline_exp_dir.exists() and any(baseline_exp_dir.glob("*.pt")):
        return
    try:
        src = _hfs.fetch_latest_baseline(
            config=hf_cfg,
            model=model,
            target_tensors_dir=baseline_exp_dir,
            capture_signature=capture_signature,
        )
        if src is not None:
            print(f"[hf-store] restored baseline for {model} from {src}", flush=True)
    except Exception as e:
        msg = f"[hf-store] fetch failed for {model}: {e}"
        if os.environ.get("GITHUB_RUN_ID"):
            raise RuntimeError(msg) from e
        warnings.warn(msg)


def _maybe_hf_push(
    *,
    hf_cfg,
    model: str,
    model_setup: ModelLaunchSettings,
    tensors_dir: Path,
    pass_label: str,
    diff_threshold: float,
    dump_cfg: dict[str, Any],
    comparator_report: Optional[Path] = None,
    comparator_stats: Optional[dict[str, Any]] = None,
) -> None:
    # Under CI a push failure raises rather than warns, so a misconfigured
    # store can't quietly mask the regression-detection guarantee.
    pt_files = list(tensors_dir.glob("*.pt"))
    if not pt_files:
        return
    try:
        meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "sglang_commit": _get_git_commit(),
            "tp_size": model_setup.tp_size,
            "prompt": PROMPT,
            "max_tokens": dump_cfg["max_tokens"],
            "ignore_eos": dump_cfg["ignore_eos"],
            "temperature": 0,
            "diff_threshold": diff_threshold,
            "dumper_filter": dump_cfg["dumper_filter"],
            "num_hidden_layers": dump_cfg.get("num_hidden_layers"),
            "capture_layers": dump_cfg.get("capture_layers"),
            "capture_signature": dump_cfg.get("capture_signature"),
            "fusion_backend": dump_cfg.get("fusion_backend"),
            "num_tensor_files": len(pt_files),
            "pass_label": pass_label,
            "source": "test_nightly_precision_regression.py",
        }
        meta.update(_collect_runtime_context())
        meta.update(_collect_ci_context())
        if comparator_stats:
            meta.update(comparator_stats)
        run_path = _hfs.push_run(
            config=hf_cfg,
            model=model,
            sglang_commit=meta["sglang_commit"],
            today_tensors_dir=tensors_dir,
            meta=meta,
            comparator_report=comparator_report,
        )
        print(
            f"[hf-store] {pass_label} {len(pt_files)} tensors for {model} -> {run_path}",
            flush=True,
        )
    except Exception as e:
        msg = f"[hf-store] push failed for {model}: {e}"
        if os.environ.get("GITHUB_RUN_ID"):
            raise RuntimeError(msg) from e
        warnings.warn(msg)


def _run_server_and_dump(
    *,
    model_setup: ModelLaunchSettings,
    dump_dir: Path,
    base_url: str,
    dumper_filter: str,
    max_tokens: int,
    ignore_eos: bool,
):
    env: dict[str, str] = {
        **os.environ,
        "DUMPER_DIR": str(dump_dir),
        "DUMPER_EXP_NAME": EXP_NAME,
        "DUMPER_SERVER_PORT": "reuse",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
    }

    server_args: list[str] = list(model_setup.extra_args or []) + [
        # Below the scheduler's decode-token reservation (default 512) the KV
        # pool clamps max_new_tokens to 0, so decode never runs and max_tokens>1
        # has no effect. Keep it well above that.
        "--max-total-tokens",
        "4096",
        "--mem-fraction-static",
        "0.9",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
        # Explicit `trtllm`, not `auto` (which resolves to mnnvl on SM90).
        "--flashinfer-allreduce-fusion-backend",
        PRECISION_FUSION_BACKEND,
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
                "filter": dumper_filter,
                "cleanup_previous": True,
            },
            timeout=60,
        ).raise_for_status()

        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_setup.model_path,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0,
                # Without this the model may EOS on the first token and never
                # enter the decode loop, leaving the decode path uncaptured.
                "ignore_eos": ignore_eos,
            },
            timeout=600,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Chat completions failed: {resp.text}")
    finally:
        kill_process_tree(proc.pid)


def _run_comparator(
    *, baseline: Path, target: Path, threshold: float
) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = [
        sys.executable,
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
        # inputs.1 is hidden_states entering the layer (inputs.0 = positions).
        # LayerCommunicator defers the cross-layer allreduce, so layer N sees
        # per-tp partial sums; bs h[tp:partial] sums across tp on the h axis
        # before diffing.
        "--override-dims",
        r"^non_intrusive__model\.layers\.\d+\.inputs\.1$:bs h[tp:partial]",
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
    (staging_dir.parent / "baseline_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if final_dir.exists():
        if old_dir.exists():
            shutil.rmtree(old_dir)
        final_dir.rename(old_dir)

    staging_dir.rename(final_dir)

    if old_dir.exists():
        shutil.rmtree(old_dir)


def _extract_diff_summary(stdout: str) -> str:
    for line in stdout.strip().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "comparison_tensor":
            continue
        if record.get("errors"):
            name = record.get("name", "unknown")
            return f"tensor={name} errored"
        diff = record.get("diff") or {}
        if diff and not diff.get("passed", True):
            name = record.get("name", "unknown")
            rel_diff = diff.get("rel_diff", "N/A")
            return f"tensor={name} rel_diff={rel_diff}"
        bad_replicated = next(
            (
                c
                for c in record.get("replicated_checks", [])
                if not c.get("passed", True)
            ),
            None,
        )
        if bad_replicated is not None:
            name = record.get("name", "unknown")
            axis = bad_replicated.get("axis", "?")
            return f"tensor={name} replicated_check_failed axis={axis}"
    return stdout[-200:] if stdout else "no output"


def _save_comparator_output(*, stdout: str, stderr: str, prefix: str) -> Path:
    fd, path_str = tempfile.mkstemp(
        prefix=f"nightly_precision_{prefix}_", suffix=".log", dir="/tmp"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
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
