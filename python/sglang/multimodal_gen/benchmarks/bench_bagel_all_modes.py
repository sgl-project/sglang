# SPDX-License-Identifier: Apache-2.0
"""Benchmark every supported BAGEL capability through its explicit pipeline.

The four BAGEL pipeline variants are loaded one at a time so their model
weights never need to coexist on the same GPU. Each benchmark case has its own
warmup and produces a machine-readable JSON report.

Example:
    python -m sglang.multimodal_gen.benchmarks.bench_bagel_all_modes \
        --model-path ByteDance-Seed/BAGEL-7B-MoT \
        --image-path input.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL_REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
DEFAULT_PROMPT = "A beautiful sunset over a mountain lake with snow-capped peaks"
DEFAULT_UNDERSTANDING_PROMPT = "Describe this image in detail."
EDIT_WORKLOADS = (
    ("sunset_sky", "Change the background to a sunset sky."),
    ("snowy_mountains", "Change the background to snowy mountains."),
    ("tropical_beach", "Change the background to a tropical beach."),
)


class _Generator(Protocol):
    def generate(self, sampling_params_kwargs: dict[str, Any] | None = None) -> Any: ...

    def shutdown(self) -> None: ...


GeneratorFactory = Callable[..., _Generator]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the BAGEL all-pipeline benchmark."""

    model_path: str
    output_dir: Path
    output_json: Path | None = None
    perf_dump_dir: Path | None = None
    revision: str = DEFAULT_MODEL_REVISION
    image_path: Path | None = None
    prompt: str = DEFAULT_PROMPT
    understanding_prompt: str = DEFAULT_UNDERSTANDING_PROMPT
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    true_cfg_scale: float = 2.0
    flow_shift: float = 3.0
    seed: int = 42
    warmup: int = 2
    runs: int = 3
    editing_warmup: int = 1
    editing_runs: int = 1
    max_think_tokens: int = 100
    think_do_sample: bool = False
    think_temperature: float = 0.3
    max_new_tokens: int = 200
    num_gpus: int = 1
    tp_size: int | None = None
    skip_edit: bool = False
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate benchmark values before any pipeline is started."""
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        if self.image_path is not None and not Path(self.image_path).is_file():
            raise ValueError(f"image_path is not a file: {self.image_path}")
        if (
            self.perf_dump_dir is not None
            and self.perf_dump_dir.exists()
            and not self.perf_dump_dir.is_dir()
        ):
            raise ValueError(f"perf_dump_dir is not a directory: {self.perf_dump_dir}")
        if not self.prompt.strip() or not self.understanding_prompt.strip():
            raise ValueError("benchmark prompts must not be empty")
        for field_name in ("height", "width", "num_inference_steps", "runs"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("warmup", "editing_warmup"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if self.editing_runs <= 0:
            raise ValueError("editing_runs must be positive")
        if self.max_think_tokens <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token limits must be positive")
        for field_name in ("guidance_scale", "true_cfg_scale", "flow_shift"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.think_do_sample and (
            not math.isfinite(self.think_temperature) or self.think_temperature <= 0
        ):
            raise ValueError(
                "think_temperature must be finite and positive when sampling"
            )
        if self.num_gpus not in (1, 2):
            raise ValueError("num_gpus must be 1 or 2 for BAGEL")
        if self.tp_size not in (None, 1, 2):
            raise ValueError("tp_size must be 1 or 2 for BAGEL")
        if self.tp_size is not None and self.tp_size != self.num_gpus:
            raise ValueError("BAGEL requires tp_size to equal num_gpus; use 1/1 or 2/2")


@dataclass(frozen=True)
class _Workload:
    id: str
    prompt: str


@dataclass(frozen=True)
class _CaseSpec:
    id: str
    name: str
    pipeline_class_name: str
    result_kind: str
    parameters: dict[str, Any]
    workloads: tuple[_Workload, ...]
    warmup: int
    runs: int
    skip_reason: str | None = None


def _default_generator_factory(**kwargs: Any) -> _Generator:
    # Import lazily so --help and CPU-only unit tests do not initialize runtime state.
    from sglang.multimodal_gen import DiffGenerator

    return DiffGenerator.from_pretrained(**kwargs)


def _build_case_groups(config: BenchmarkConfig) -> tuple[tuple[_CaseSpec, ...], ...]:
    image_path = str(config.image_path) if config.image_path is not None else None
    prompt_workload = (_Workload("default", config.prompt),)

    generation_params = {
        "height": config.height,
        "width": config.width,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "flow_shift": config.flow_shift,
        "negative_prompt": "",
        "seed": config.seed,
        "generator_device": "cpu",
    }
    thinking_params = {
        **generation_params,
        "max_think_tokens": config.max_think_tokens,
        "think_do_sample": config.think_do_sample,
        "think_temperature": config.think_temperature,
    }

    missing_image_reason = "--image-path was not provided"
    understanding_skip = None if image_path is not None else missing_image_reason
    if image_path is None:
        editing_skip = missing_image_reason
    elif config.skip_edit:
        editing_skip = "--skip-edit was set"
    else:
        editing_skip = None

    return (
        (
            _CaseSpec(
                id="t2i",
                name="Standard T2I",
                pipeline_class_name="BagelPipeline",
                result_kind="image",
                parameters={**generation_params, "enable_taylorseer": False},
                workloads=prompt_workload,
                warmup=config.warmup,
                runs=config.runs,
            ),
            _CaseSpec(
                id="t2i_taylorseer",
                name="T2I + TaylorSeer",
                pipeline_class_name="BagelPipeline",
                result_kind="image",
                parameters={**generation_params, "enable_taylorseer": True},
                workloads=prompt_workload,
                warmup=config.warmup,
                runs=config.runs,
            ),
        ),
        (
            _CaseSpec(
                id="thinking",
                name="Thinking",
                pipeline_class_name="BagelThinkingPipeline",
                result_kind="thinking",
                parameters={**thinking_params, "enable_taylorseer": False},
                workloads=prompt_workload,
                warmup=config.warmup,
                runs=config.runs,
            ),
            _CaseSpec(
                id="thinking_taylorseer",
                name="Thinking + TaylorSeer",
                pipeline_class_name="BagelThinkingPipeline",
                result_kind="thinking",
                parameters={**thinking_params, "enable_taylorseer": True},
                workloads=prompt_workload,
                warmup=config.warmup,
                runs=config.runs,
            ),
        ),
        (
            _CaseSpec(
                id="understanding",
                name="Understanding",
                pipeline_class_name="BagelUnderstandingPipeline",
                result_kind="text",
                parameters={
                    "image_path": image_path,
                    "max_new_tokens": config.max_new_tokens,
                    "do_sample": False,
                    "enable_thinking": False,
                    "seed": config.seed,
                },
                workloads=(_Workload("describe_image", config.understanding_prompt),),
                warmup=config.warmup,
                runs=config.runs,
                skip_reason=understanding_skip,
            ),
        ),
        (
            _CaseSpec(
                id="editing",
                name="Editing",
                pipeline_class_name="BagelEditPipeline",
                result_kind="image",
                parameters={
                    "image_path": image_path,
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "true_cfg_scale": config.true_cfg_scale,
                    "flow_shift": config.flow_shift,
                    "negative_prompt": "",
                    "seed": config.seed,
                    "generator_device": "cpu",
                },
                workloads=tuple(
                    _Workload(workload_id, prompt)
                    for workload_id, prompt in EDIT_WORKLOADS
                ),
                warmup=config.editing_warmup,
                runs=config.editing_runs,
                skip_reason=editing_skip,
            ),
        ),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported benchmark report value: {type(value).__name__}")


def _case_record(spec: _CaseSpec, status: str) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "pipeline_class_name": spec.pipeline_class_name,
        "status": status,
        "parameters": _json_safe(spec.parameters),
        "warmup": spec.warmup,
        "runs": spec.runs,
        "workloads": [],
        "summary": None,
        "error": None,
        "skip_reason": spec.skip_reason if status == "skipped" else None,
    }


def _request_params(
    config: BenchmarkConfig,
    spec: _CaseSpec,
    workload: _Workload,
    *,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    params = {**spec.parameters, "prompt": workload.prompt}
    if spec.result_kind == "text":
        params.update(save_output=False, return_file_paths_only=False)
    elif warmup:
        # Avoid benchmark artifacts while still materializing output for validation.
        params.update(save_output=False, return_file_paths_only=False)
    else:
        output_file_name = f"{spec.id}_{workload.id}_run_{run_index + 1:02d}.png"
        params.update(
            save_output=True,
            return_file_paths_only=True,
            output_path=str(config.output_dir),
            output_file_name=output_file_name,
        )

    if not warmup and config.perf_dump_dir is not None:
        perf_dump_path = (
            config.perf_dump_dir
            / spec.id
            / workload.id
            / f"run_{run_index + 1:02d}.json"
        ).resolve()
        params["perf_dump_path"] = str(perf_dump_path)
    return params


def _single_result(raw_result: Any) -> Any:
    if raw_result is None:
        raise RuntimeError("DiffGenerator returned None instead of a generation result")
    if isinstance(raw_result, list):
        if len(raw_result) != 1:
            raise RuntimeError(
                f"Expected one generation result, received {len(raw_result)}"
            )
        return raw_result[0]
    return raw_result


def _validate_result(result: Any, result_kind: str, *, warmup: bool) -> None:
    if result_kind == "text":
        text = getattr(result, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("BAGEL Understanding returned no text")
        return

    if result_kind == "thinking":
        revised_prompt = getattr(result, "revised_prompt", None)
        if not isinstance(revised_prompt, str) or not revised_prompt.strip():
            raise RuntimeError("BAGEL Thinking returned no revised prompt")

    if warmup:
        has_materialized_output = getattr(result, "samples", None) is not None or bool(
            getattr(result, "frames", None)
        )
        if not has_materialized_output:
            raise RuntimeError("BAGEL image warmup returned no materialized output")
    else:
        output_file_path = getattr(result, "output_file_path", None)
        if not output_file_path:
            raise RuntimeError("BAGEL image request returned no output file path")
        if not Path(output_file_path).is_file():
            raise RuntimeError(
                f"BAGEL image artifact does not exist: {output_file_path}"
            )


def _sample_record(
    result: Any,
    run_index: int,
    wall_time_ms: float,
    worker_perf_dump_path: str | None,
) -> dict[str, Any]:
    metrics = _json_safe(getattr(result, "metrics", {}) or {})
    assert isinstance(metrics, dict)
    generation_time = float(getattr(result, "generation_time", 0.0) or 0.0)
    sample: dict[str, Any] = {
        "run_index": run_index,
        "wall_time_ms": wall_time_ms,
        "generation_time_ms": generation_time * 1000.0,
        "scheduler_total_duration_ms": metrics.get("total_duration_ms"),
        "peak_memory_mb": float(getattr(result, "peak_memory_mb", 0.0) or 0.0),
        "metrics": metrics,
        "artifact_path": getattr(result, "output_file_path", None),
        "worker_perf_dump_path": worker_perf_dump_path,
    }
    revised_prompt = getattr(result, "revised_prompt", None)
    if revised_prompt:
        sample["revised_prompt"] = revised_prompt
    text = getattr(result, "text", None)
    if text:
        sample.update(
            text=text,
            finish_reason=getattr(result, "finish_reason", None),
            prompt_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(result, "completion_tokens", 0) or 0),
        )
    return sample


def _summarize(samples: list[dict[str, Any]], workload_count: int) -> dict[str, Any]:
    wall_times = [float(sample["wall_time_ms"]) for sample in samples]
    generation_times = [float(sample["generation_time_ms"]) for sample in samples]
    peak_memory = [float(sample["peak_memory_mb"]) for sample in samples]
    scheduler_times = [
        float(sample["scheduler_total_duration_ms"])
        for sample in samples
        if sample["scheduler_total_duration_ms"] is not None
    ]
    return {
        "sample_count": len(samples),
        "workload_count": workload_count,
        "mean_wall_time_ms": statistics.fmean(wall_times),
        "median_wall_time_ms": statistics.median(wall_times),
        "min_wall_time_ms": min(wall_times),
        "max_wall_time_ms": max(wall_times),
        "stdev_wall_time_ms": (
            statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
        ),
        "mean_generation_time_ms": statistics.fmean(generation_times),
        "mean_scheduler_total_duration_ms": (
            statistics.fmean(scheduler_times) if scheduler_times else None
        ),
        "max_peak_memory_mb": max(peak_memory),
    }


def _run_case(
    generator: _Generator, config: BenchmarkConfig, spec: _CaseSpec
) -> dict[str, Any]:
    record = _case_record(spec, "passed")
    all_samples: list[dict[str, Any]] = []
    workload_records: list[dict[str, Any]] = []
    active_workload_record: dict[str, Any] | None = None

    try:
        for workload in spec.workloads:
            workload_record: dict[str, Any] = {
                "id": workload.id,
                "prompt": workload.prompt,
                "status": "not_run",
                "samples": [],
                "error": None,
            }
            record["workloads"].append(workload_record)
            workload_records.append(workload_record)

        # Warm every workload before measuring any of them so Editing's three
        # variants all receive the same benchmark treatment.
        for workload, workload_record in zip(
            spec.workloads, workload_records, strict=True
        ):
            active_workload_record = workload_record
            for warmup_index in range(spec.warmup):
                params = _request_params(
                    config,
                    spec,
                    workload,
                    run_index=warmup_index,
                    warmup=True,
                )
                result = _single_result(
                    generator.generate(sampling_params_kwargs=params)
                )
                _validate_result(result, spec.result_kind, warmup=True)

        for workload, workload_record in zip(
            spec.workloads, workload_records, strict=True
        ):
            active_workload_record = workload_record
            for run_index in range(spec.runs):
                params = _request_params(
                    config,
                    spec,
                    workload,
                    run_index=run_index,
                    warmup=False,
                )
                worker_perf_dump_path = params.get("perf_dump_path")
                if worker_perf_dump_path is not None:
                    # Remove an older run at the deterministic destination so
                    # existence checks cannot accept a stale profiler report.
                    Path(worker_perf_dump_path).unlink(missing_ok=True)
                start = time.perf_counter()
                result = _single_result(
                    generator.generate(sampling_params_kwargs=params)
                )
                wall_time_ms = (time.perf_counter() - start) * 1000.0
                _validate_result(result, spec.result_kind, warmup=False)
                if (
                    worker_perf_dump_path is not None
                    and not Path(worker_perf_dump_path).is_file()
                ):
                    raise RuntimeError(
                        "BAGEL worker perf dump does not exist: "
                        f"{worker_perf_dump_path}"
                    )
                sample = _sample_record(
                    result,
                    run_index,
                    wall_time_ms,
                    worker_perf_dump_path,
                )
                workload_record["samples"].append(sample)
                all_samples.append(sample)
            workload_record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        if active_workload_record is not None:
            active_workload_record["status"] = "failed"
            active_workload_record["error"] = record["error"]
        return record

    record["summary"] = _summarize(all_samples, len(spec.workloads))
    return record


def _comparison(
    cases_by_id: dict[str, dict[str, Any]], baseline: str, candidate: str
) -> dict[str, Any] | None:
    baseline_case = cases_by_id.get(baseline)
    candidate_case = cases_by_id.get(candidate)
    if not baseline_case or not candidate_case:
        return None
    if baseline_case["status"] != "passed" or candidate_case["status"] != "passed":
        return None
    # Worker inference metrics stop before image encoding and filesystem I/O,
    # unlike the parent wall clock used for end-to-end latency reporting.
    metric_name = "mean_scheduler_total_duration_ms"
    baseline_ms = baseline_case["summary"].get(metric_name)
    candidate_ms = candidate_case["summary"].get(metric_name)
    if baseline_ms is None or candidate_ms is None:
        return None
    baseline_ms = float(baseline_ms)
    candidate_ms = float(candidate_ms)
    if candidate_ms <= 0:
        return None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": "scheduler_total_duration_ms",
        "speedup": baseline_ms / candidate_ms,
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    generator_factory: GeneratorFactory = _default_generator_factory,
) -> dict[str, Any]:
    """Run all BAGEL benchmark cases and return a serializable report.

    Args:
        config: Validated benchmark configuration.
        generator_factory: DiffGenerator-compatible factory, injectable for tests.

    Returns:
        A JSON-serializable report containing pipeline and request metrics.

    Raises:
        OSError: If the output directory cannot be created.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.perf_dump_dir is not None:
        config.perf_dump_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "metadata": {
            "benchmark": "bagel_all_modes",
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "model": {"path": config.model_path, "revision": config.revision},
        "configuration": _json_safe(config_dict),
        "pipelines": [],
        "cases": [],
        "comparisons": [],
    }

    case_groups = _build_case_groups(config)
    stop = False
    for group in case_groups:
        runnable = [spec for spec in group if spec.skip_reason is None]
        skipped = [spec for spec in group if spec.skip_reason is not None]
        if not runnable:
            report["cases"].extend(_case_record(spec, "skipped") for spec in skipped)
            continue

        pipeline_name = runnable[0].pipeline_class_name
        pipeline_record: dict[str, Any] = {
            "pipeline_class_name": pipeline_name,
            "status": "passed",
            "startup_ms": None,
            "shutdown_ms": None,
            "error": None,
        }
        report["pipelines"].append(pipeline_record)

        generator: _Generator | None = None
        startup_start = time.perf_counter()
        try:
            factory_kwargs: dict[str, Any] = {
                "model_path": config.model_path,
                "revision": config.revision,
                "pipeline_class_name": pipeline_name,
                "num_gpus": config.num_gpus,
                # BAGEL supports only TP=1 and TP=2; never leave extra GPUs for
                # ServerArgs to reinterpret as unsupported parallel dimensions.
                "tp_size": config.tp_size or config.num_gpus,
            }
            generator = generator_factory(**factory_kwargs)
            pipeline_record["startup_ms"] = (
                time.perf_counter() - startup_start
            ) * 1000.0

            for spec in runnable:
                case_record = _run_case(generator, config, spec)
                report["cases"].append(case_record)
                if case_record["status"] == "failed":
                    pipeline_record["status"] = "failed"
                    pipeline_record["error"] = case_record["error"]
                    if not config.continue_on_error:
                        stop = True
                        break
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            if pipeline_record["startup_ms"] is None:
                pipeline_record["startup_ms"] = (
                    time.perf_counter() - startup_start
                ) * 1000.0
            pipeline_record["status"] = "failed"
            pipeline_record["error"] = error_message
            for spec in runnable:
                case_record = _case_record(spec, "failed")
                case_record["error"] = f"Pipeline startup failed: {error_message}"
                report["cases"].append(case_record)
            # A factory exception may occur after it launches workers but before
            # it returns a handle, leaving no safe way to prove cleanup.
            stop = True
        finally:
            if generator is not None:
                shutdown_start = time.perf_counter()
                try:
                    generator.shutdown()
                except Exception as error:
                    error_message = f"{type(error).__name__}: {error}"
                    pipeline_record["status"] = "failed"
                    pipeline_record["error"] = (
                        f"{pipeline_record['error']}; shutdown failed: {error_message}"
                        if pipeline_record["error"]
                        else f"Shutdown failed: {error_message}"
                    )
                    # A failed shutdown may leave model workers and their GPU
                    # allocations alive, so starting another pipeline is unsafe.
                    stop = True
                finally:
                    pipeline_record["shutdown_ms"] = (
                        time.perf_counter() - shutdown_start
                    ) * 1000.0

        report["cases"].extend(_case_record(spec, "skipped") for spec in skipped)
        if stop:
            break

    cases_by_id = {case["id"]: case for case in report["cases"]}
    ordered_cases: list[dict[str, Any]] = []
    for spec in (spec for group in case_groups for spec in group):
        if spec.id in cases_by_id:
            ordered_cases.append(cases_by_id[spec.id])
            continue
        if spec.skip_reason is not None:
            ordered_cases.append(_case_record(spec, "skipped"))
            continue
        case_record = _case_record(spec, "not_run")
        case_record["skip_reason"] = "Benchmark stopped after an earlier failure"
        ordered_cases.append(case_record)
    report["cases"] = ordered_cases
    cases_by_id = {case["id"]: case for case in ordered_cases}
    for baseline, candidate in (
        ("t2i", "t2i_taylorseer"),
        ("thinking", "thinking_taylorseer"),
    ):
        comparison = _comparison(cases_by_id, baseline, candidate)
        if comparison is not None:
            report["comparisons"].append(comparison)
    return report


def write_report_atomic(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a benchmark report using an atomic same-directory replacement.

    Args:
        report: JSON-serializable benchmark report.
        output_path: Destination JSON path.

    Returns:
        None.

    Raises:
        OSError: If the destination directory or file cannot be written.
        TypeError: If the report contains a non-serializable value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(report, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark all six BAGEL cases across four explicit pipelines."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("BAGEL_MODEL_PATH"),
        help="BAGEL model path or Hugging Face ID (or BAGEL_MODEL_PATH)",
    )
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--image-path",
        type=Path,
        default=os.environ.get("BAGEL_EDIT_IMAGE_PATH"),
        help="Source image for Understanding and Editing (or BAGEL_EDIT_IMAGE_PATH)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--perf-dump-dir",
        type=Path,
        help="Write one worker stage/step timing JSON file per timed request",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--understanding-prompt", default=DEFAULT_UNDERSTANDING_PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--true-cfg-scale", type=float, default=2.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--editing-warmup", type=int, default=1)
    parser.add_argument("--editing-runs", type=int, default=1)
    parser.add_argument("--max-think-tokens", type=int, default=100)
    parser.add_argument("--think-do-sample", action="store_true")
    parser.add_argument("--think-temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, choices=(1, 2))
    parser.add_argument("--skip-edit", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _config_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> BenchmarkConfig:
    if not args.model_path:
        parser.error("--model-path is required (or set BAGEL_MODEL_PATH)")
    output_json = args.output_json or args.output_dir / "bagel_all_modes.json"
    try:
        return BenchmarkConfig(
            model_path=args.model_path,
            revision=args.revision,
            image_path=args.image_path,
            output_dir=args.output_dir,
            output_json=output_json,
            perf_dump_dir=args.perf_dump_dir,
            prompt=args.prompt,
            understanding_prompt=args.understanding_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            flow_shift=args.flow_shift,
            seed=args.seed,
            warmup=args.warmup,
            runs=args.runs,
            editing_warmup=args.editing_warmup,
            editing_runs=args.editing_runs,
            max_think_tokens=args.max_think_tokens,
            think_do_sample=args.think_do_sample,
            think_temperature=args.think_temperature,
            max_new_tokens=args.max_new_tokens,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            skip_edit=args.skip_edit,
            continue_on_error=args.continue_on_error,
        )
    except ValueError as error:
        parser.error(str(error))


def _print_summary(report: Mapping[str, Any], output_json: Path) -> None:
    print("\nBAGEL all-pipeline benchmark")
    print(f"{'Case':<28} {'Status':<9} {'Mean latency':>14}")
    print(f"{'-' * 28} {'-' * 9} {'-' * 14}")
    for case in report["cases"]:
        summary = case.get("summary") or {}
        mean_ms = summary.get("mean_wall_time_ms")
        latency = f"{mean_ms / 1000.0:.2f}s" if mean_ms is not None else "-"
        print(f"{case['name']:<28} {case['status']:<9} {latency:>14}")
    for comparison in report["comparisons"]:
        print(
            f"{comparison['candidate']} vs {comparison['baseline']}: "
            f"{comparison['speedup']:.2f}x"
        )
    print(f"Report: {output_json}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the BAGEL benchmark command-line interface.

    Args:
        argv: Optional argument sequence. Uses process arguments when omitted.

    Returns:
        Zero when all executed cases and pipeline lifecycles pass, otherwise one.

    Raises:
        OSError: If benchmark artifacts or the report cannot be written.
    """
    parser = _build_parser()
    config = _config_from_args(parser, parser.parse_args(argv))
    report = run_benchmark(config)
    assert config.output_json is not None
    write_report_atomic(report, config.output_json)
    _print_summary(report, config.output_json)
    failed = any(case["status"] == "failed" for case in report["cases"]) or any(
        pipeline["status"] == "failed" for pipeline in report["pipelines"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
