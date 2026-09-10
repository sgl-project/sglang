"""YAML benchmark configuration model, validation, and loading.

A config describes one model spec: the server argument grid, the workload
grid, SLA thresholds, and run policy.  The schema is deliberately flat —
every key in ``server.args`` / ``server.axes`` / ``workload.args`` /
``workload.axes`` is a literal ``sglang.launch_server`` or
``sglang.bench_serving`` CLI flag, so the YAML stays verifiable against
upstream documentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SLA_KEY_RE = re.compile(r"^(p\d+|mean|median)_(ttft|tpot|itl|e2e_latency)_ms$")
MAX_GRAPH_BS_ENTRIES = 10
DATASET_NAMES = {
    "agentic-trace",
    "sharegpt",
    "custom",
    "openai",
    "random",
    "random-ids",
    "generated-shared-prefix",
    "mmmu",
    "image",
    "mooncake",
    "longbench_v2",
    "speed-bench",
}


class ConfigError(ValueError):
    """Raised with every collected validation error at once."""


def argv_from_args(args: dict[str, Any]) -> list[str]:
    """Render a ``{flag: value}`` mapping as CLI argv tokens.

    ``None`` / ``False`` omit the flag, ``True`` emits the bare flag,
    lists/tuples emit space-separated values after one flag.
    """
    argv: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        argv.append(key)
        if value is True:
            continue
        if isinstance(value, (list, tuple)):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


@dataclass
class ModelSpec:
    path: str
    dtype: str = "bfloat16"
    quantization: str | None = None
    trust_remote_code: bool = True

    def to_argv(self) -> list[str]:
        argv = ["--model-path", self.path, "--dtype", self.dtype]
        if self.quantization:
            argv += ["--quantization", self.quantization]
        if self.trust_remote_code:
            argv.append("--trust-remote-code")
        return argv


@dataclass
class ServerSpec:
    args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    axes: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class Gsm8kSpec:
    num_questions: int = 200
    num_shots: int = 5
    data_path: str | None = None
    accuracy_floor: float = 0.30


@dataclass
class WorkloadSpec:
    dataset_name: str = "random"
    args: dict[str, Any] = field(default_factory=dict)
    axes: dict[str, list[Any]] = field(default_factory=dict)
    num_prompts_mult: int | None = None


@dataclass
class SLASpec:
    thresholds: dict[str, float] = field(default_factory=dict)
    cv_max: float = 0.15


@dataclass
class RunSpec:
    repeats: int = 1
    seed: int = 42
    warmup_requests: int = 0
    health_timeout_s: int = 1800
    bench_timeout_s: int = 3600
    hbm_budget_mb: int = 500
    hbm_timeout_s: int = 240
    hbm_poll_s: float = 5.0
    python: str | None = None
    gsm8k: Gsm8kSpec | None = None


@dataclass
class BenchConfig:
    name: str
    model: ModelSpec
    server: ServerSpec
    workload: WorkloadSpec
    sla: SLASpec
    run: RunSpec
    source_path: str | None = None


def _require_flag_dict(
    where: str, data: Any, key: str, errors: list[str]
) -> dict[str, Any]:
    value = data.get(key) if isinstance(data, dict) else None
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{where}.{key} must be a mapping of flags to values")
        return {}
    for flag in value:
        if not isinstance(flag, str) or not flag.startswith("--"):
            errors.append(f"{where}.{key} flag {flag!r} must start with '--'")
    return value


def _merge_axis_dict(
    where: str, axes: dict[str, Any], errors: list[str]
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for flag, values in axes.items():
        if not isinstance(values, list) or not values:
            errors.append(f"{where}.axes[{flag}] must be a non-empty list")
            continue
        out[flag] = values
    return out


def _check_graph_bs(where: str, args: dict[str, Any], errors: list[str]) -> None:
    value = args.get("--cuda-graph-bs")
    if isinstance(value, (list, tuple)) and len(value) > MAX_GRAPH_BS_ENTRIES:
        errors.append(
            f"{where}: --cuda-graph-bs has {len(value)} entries; on Ascend NPU "
            f"graph capture is limited to {MAX_GRAPH_BS_ENTRIES} batch sizes "
            "(stream-conflict crash 507000)"
        )


def _parse(raw: dict[str, Any], source: str | None) -> BenchConfig:
    errors: list[str] = []

    name = raw.get("name")
    if not name or not isinstance(name, str):
        errors.append("name must be a non-empty string")
        name = "unnamed"

    model_raw = raw.get("model") or {}
    model = ModelSpec(
        path=str(model_raw.get("path", "")),
        dtype=str(model_raw.get("dtype", "bfloat16")),
        quantization=model_raw.get("quantization"),
        trust_remote_code=bool(model_raw.get("trust_remote_code", True)),
    )
    if not model.path:
        errors.append("model.path is required")

    server_raw = raw.get("server") or {}
    server_args = _require_flag_dict("server", server_raw, "args", errors)
    server_axes_raw = _require_flag_dict("server", server_raw, "axes", errors)
    server_axes = _merge_axis_dict("server", server_axes_raw, errors)
    overlap = set(server_args) & set(server_axes)
    if overlap:
        errors.append(f"server: flags in both args and axes: {sorted(overlap)}")
    env = server_raw.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in env.items()
    ):
        errors.append("server.env must be a mapping of str to scalar")
        env = {}
    _check_graph_bs("server.args", server_args, errors)
    for combo in server_axes.values():
        _check_graph_bs("server.axes", dict(zip(server_axes, combo)), errors)

    workload_raw = raw.get("workload") or {}
    workload_args = _require_flag_dict("workload", workload_raw, "args", errors)
    workload_axes_raw = _require_flag_dict("workload", workload_raw, "axes", errors)
    workload_axes = _merge_axis_dict("workload", workload_axes_raw, errors)
    overlap = set(workload_args) & set(workload_axes)
    if overlap:
        errors.append(f"workload: flags in both args and axes: {sorted(overlap)}")

    dataset = str(workload_raw.get("dataset_name", "random"))
    if dataset not in DATASET_NAMES:
        errors.append(
            f"workload.dataset_name {dataset!r} not in bench_serving choices "
            f"{sorted(DATASET_NAMES)}"
        )
    mult = workload_raw.get("num_prompts_mult")
    if mult is not None and (not isinstance(mult, int) or mult < 1):
        errors.append("workload.num_prompts_mult must be a positive integer")
        mult = None

    sla_raw = raw.get("sla") or {}
    thresholds: dict[str, float] = {}
    for key, value in (sla_raw.get("thresholds") or {}).items():
        if not SLA_KEY_RE.match(key):
            errors.append(
                f"sla.thresholds key {key!r} must match "
                f"(pNN|mean|median)_(ttft|tpot|itl|e2e_latency)_ms"
            )
            continue
        try:
            thresholds[key] = float(value)
        except (TypeError, ValueError):
            errors.append(f"sla.thresholds[{key}] must be numeric")
    cv_max = float(sla_raw.get("cv_max", 0.15))
    if not 0 < cv_max <= 1:
        errors.append("sla.cv_max must be in (0, 1]")

    run_raw = raw.get("run") or {}
    gsm8k_raw = run_raw.get("gsm8k")
    gsm8k = None
    if gsm8k_raw:
        gsm8k = Gsm8kSpec(
            num_questions=int(gsm8k_raw.get("num_questions", 200)),
            num_shots=int(gsm8k_raw.get("num_shots", 5)),
            data_path=gsm8k_raw.get("data_path"),
            accuracy_floor=float(gsm8k_raw.get("accuracy_floor", 0.30)),
        )
    run = RunSpec(
        repeats=int(run_raw.get("repeats", 1)),
        seed=int(run_raw.get("seed", 42)),
        warmup_requests=int(run_raw.get("warmup_requests", 0)),
        health_timeout_s=int(run_raw.get("health_timeout_s", 1800)),
        bench_timeout_s=int(run_raw.get("bench_timeout_s", 3600)),
        hbm_budget_mb=int(run_raw.get("hbm_budget_mb", 500)),
        hbm_timeout_s=int(run_raw.get("hbm_timeout_s", 240)),
        python=run_raw.get("python"),
        gsm8k=gsm8k,
    )
    if run.repeats < 1:
        errors.append("run.repeats must be >= 1")
    if "--max-concurrency" not in workload_axes and mult is not None:
        errors.append(
            "workload.num_prompts_mult requires --max-concurrency in workload.axes"
        )

    if errors:
        raise ConfigError(
            f"{len(errors)} config error(s) in {source or 'config'}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return BenchConfig(
        name=name,
        model=model,
        server=ServerSpec(args=server_args, env=dict(env), axes=server_axes),
        workload=WorkloadSpec(
            dataset_name=dataset,
            args=workload_args,
            axes=workload_axes,
            num_prompts_mult=mult,
        ),
        sla=SLASpec(thresholds=thresholds, cv_max=cv_max),
        run=run,
        source_path=source,
    )


def load_config(path: str | Path) -> BenchConfig:
    """Load and validate a YAML benchmark config; raises ConfigError."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    return _parse(raw, source=str(path))
