# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sglang.multimodal_gen.benchmarks.bench_bagel_all_modes import (
    BenchmarkConfig,
    run_benchmark,
    write_report_atomic,
)

PIPELINE_ORDER = (
    "BagelPipeline",
    "BagelThinkingPipeline",
    "BagelUnderstandingPipeline",
    "BagelEditPipeline",
)
LEGACY_MODE_FIELDS = {
    "enable_think",
    "enable_understanding",
    "enable_editing",
}


class FakeGeneratorFactory:
    """Create observable, CPU-only generators for benchmark lifecycle tests."""

    def __init__(
        self,
        *,
        return_none_when: Callable[[str, dict[str, object]], bool] | None = None,
        startup_error_pipeline: str | None = None,
        shutdown_error_pipeline: str | None = None,
    ) -> None:
        self.return_none_when = return_none_when
        self.startup_error_pipeline = startup_error_pipeline
        self.shutdown_error_pipeline = shutdown_error_pipeline
        self.created: list[FakeGenerator] = []
        self.created_kwargs: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0

    def __call__(self, *args: object, **kwargs: object) -> "FakeGenerator":
        """Create one fake generator from DiffGenerator-compatible kwargs."""
        del args
        pipeline_class_name = str(kwargs.get("pipeline_class_name") or "BagelPipeline")
        self.created_kwargs.append(dict(kwargs))
        if pipeline_class_name == self.startup_error_pipeline:
            raise RuntimeError(f"failed to start {pipeline_class_name}")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        generator = FakeGenerator(self, pipeline_class_name)
        self.created.append(generator)
        return generator


class FakeGenerator:
    """Mimic the small DiffGenerator surface exercised by the benchmark."""

    def __init__(self, factory: FakeGeneratorFactory, pipeline_class_name: str) -> None:
        self.factory = factory
        self.pipeline_class_name = pipeline_class_name
        self.shutdown_calls = 0
        self._closed = False

    def generate(
        self,
        sampling_params_kwargs: dict[str, object] | None = None,
        external_trace_header: dict[str, str] | None = None,
    ) -> SimpleNamespace | None:
        """Record one request and return a serializable GenerationResult stand-in."""
        del external_trace_header
        params = dict(sampling_params_kwargs or {})
        self.factory.generate_calls.append(
            {
                "pipeline_class_name": self.pipeline_class_name,
                "params": params,
            }
        )
        if self.factory.return_none_when is not None and self.factory.return_none_when(
            self.pipeline_class_name, params
        ):
            return None

        is_understanding = self.pipeline_class_name == "BagelUnderstandingPipeline"
        is_thinking = self.pipeline_class_name == "BagelThinkingPipeline"
        enable_taylorseer = bool(params.get("enable_taylorseer", False))
        generation_time = {
            ("BagelPipeline", False): 4.0,
            ("BagelPipeline", True): 2.0,
            ("BagelThinkingPipeline", False): 6.0,
            ("BagelThinkingPipeline", True): 3.0,
        }.get((self.pipeline_class_name, enable_taylorseer), 5.0)
        frame_value = len(self.factory.generate_calls) % 255
        frame = np.full((4, 4, 3), frame_value, dtype=np.uint8)
        perf_dump_path = params.get("perf_dump_path")
        if perf_dump_path is not None:
            perf_path = Path(str(perf_dump_path))
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            perf_path.write_text(
                json.dumps({"request_id": len(self.factory.generate_calls)}),
                encoding="utf-8",
            )
        output_file_path = None
        if params.get("save_output"):
            artifact_path = Path(str(params["output_path"])) / str(
                params["output_file_name"]
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(artifact_path)
            output_file_path = str(artifact_path)
        return SimpleNamespace(
            samples=None if is_understanding else frame,
            frames=[] if is_understanding else [frame],
            text="A tiny test image." if is_understanding else None,
            finish_reason="stop" if is_understanding else None,
            prompt_tokens=7 if is_understanding else 0,
            completion_tokens=5 if is_understanding else 0,
            prompt=params.get("prompt"),
            revised_prompt=(
                f"{params.get('prompt')}\nA concise plan." if is_thinking else None
            ),
            size=(params.get("height"), params.get("width"), 1),
            generation_time=generation_time,
            peak_memory_mb=1024.0,
            metrics={
                "request_id": f"fake-{len(self.factory.generate_calls)}",
                "stages": {"fake_stage": generation_time * 1000.0},
                "steps": [generation_time * 1000.0],
                "total_duration_ms": generation_time * 1000.0,
                "memory_snapshots": {},
            },
            output_file_path=output_file_path,
        )

    def shutdown(self) -> None:
        """Release the fake lifecycle exactly once, matching DiffGenerator."""
        if self._closed:
            return
        self._closed = True
        self.shutdown_calls += 1
        self.factory.active -= 1
        if self.pipeline_class_name == self.factory.shutdown_error_pipeline:
            raise RuntimeError(f"failed to stop {self.pipeline_class_name}")


def _write_input_image(path: Path) -> None:
    """Write a valid source image for input-dependent benchmark cases."""
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)


def _config(tmp_path: Path, *, image_path: Path | None) -> BenchmarkConfig:
    """Build a one-run configuration so each recorded call is a timed workload."""
    return BenchmarkConfig(
        model_path="test-bagel",
        revision="test-revision",
        image_path=image_path,
        output_dir=tmp_path / "outputs",
        output_json=tmp_path / "report.json",
        prompt="A blue robot holding a flower.",
        height=256,
        width=320,
        num_inference_steps=2,
        guidance_scale=4.0,
        true_cfg_scale=2.25,
        seed=17,
        warmup=0,
        runs=1,
        editing_warmup=0,
        editing_runs=1,
        max_think_tokens=23,
        think_do_sample=False,
        think_temperature=0.3,
        max_new_tokens=37,
    )


def _case(report: dict[str, object], case_id: str) -> dict[str, object]:
    """Return one normalized case from the public report schema."""
    cases = report["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        if case.get("id") == case_id:
            return case
    raise AssertionError(f"case {case_id!r} is missing from report")


def _calls_for(
    factory: FakeGeneratorFactory, pipeline_class_name: str
) -> list[dict[str, object]]:
    """Return recorded sampling dictionaries for one pipeline."""
    calls: list[dict[str, object]] = []
    for call in factory.generate_calls:
        if call["pipeline_class_name"] != pipeline_class_name:
            continue
        params = call["params"]
        assert isinstance(params, dict)
        calls.append(params)
    return calls


def test_all_modes_use_sequential_pipeline_lifecycles_and_current_params(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory()

    report = run_benchmark(
        _config(tmp_path, image_path=image_path), generator_factory=factory
    )

    assert [generator.pipeline_class_name for generator in factory.created] == list(
        PIPELINE_ORDER
    )
    assert factory.max_active == 1
    assert factory.active == 0
    assert all(generator.shutdown_calls == 1 for generator in factory.created)

    case_ids = {
        case_id
        for case_id in (
            "t2i",
            "t2i_taylorseer",
            "thinking",
            "thinking_taylorseer",
            "understanding",
            "editing",
        )
        if _case(report, case_id)["status"] == "passed"
    }
    assert case_ids == {
        "t2i",
        "t2i_taylorseer",
        "thinking",
        "thinking_taylorseer",
        "understanding",
        "editing",
    }

    for call in factory.generate_calls:
        params = call["params"]
        assert isinstance(params, dict)
        assert LEGACY_MODE_FIELDS.isdisjoint(params)
        assert "perf_dump_path" not in params

    t2i_calls = _calls_for(factory, "BagelPipeline")
    assert len(t2i_calls) == 2
    assert t2i_calls[0].get("enable_taylorseer", False) is False
    assert t2i_calls[1]["enable_taylorseer"] is True
    assert all(params["height"] == 256 for params in t2i_calls)
    assert all(params["width"] == 320 for params in t2i_calls)

    thinking_calls = _calls_for(factory, "BagelThinkingPipeline")
    assert len(thinking_calls) == 2
    assert thinking_calls[0].get("enable_taylorseer", False) is False
    assert thinking_calls[1]["enable_taylorseer"] is True
    assert all(params["max_think_tokens"] == 23 for params in thinking_calls)
    assert all(params["think_do_sample"] is False for params in thinking_calls)
    assert all(params["think_temperature"] == 0.3 for params in thinking_calls)

    understanding_calls = _calls_for(factory, "BagelUnderstandingPipeline")
    assert len(understanding_calls) == 1
    assert understanding_calls[0]["image_path"] == str(image_path)
    assert understanding_calls[0]["max_new_tokens"] == 37
    assert "max_understanding_tokens" not in understanding_calls[0]

    editing_calls = _calls_for(factory, "BagelEditPipeline")
    assert len(editing_calls) == 3
    assert len({str(params["prompt"]) for params in editing_calls}) == 3
    assert all(params["image_path"] == str(image_path) for params in editing_calls)
    assert all(params["true_cfg_scale"] == 2.25 for params in editing_calls)
    assert all(
        "height" not in params and "width" not in params for params in editing_calls
    )


def test_missing_image_skips_understanding_and_editing_without_starting_pipelines(
    tmp_path: Path,
) -> None:
    factory = FakeGeneratorFactory()

    report = run_benchmark(
        _config(tmp_path, image_path=None), generator_factory=factory
    )

    assert [generator.pipeline_class_name for generator in factory.created] == [
        "BagelPipeline",
        "BagelThinkingPipeline",
    ]
    assert factory.max_active == 1
    assert factory.active == 0
    assert _case(report, "understanding")["status"] == "skipped"
    assert _case(report, "editing")["status"] == "skipped"
    assert _calls_for(factory, "BagelUnderstandingPipeline") == []
    assert _calls_for(factory, "BagelEditPipeline") == []


def test_none_generation_result_never_leaks_an_active_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory(
        return_none_when=lambda pipeline, _params: pipeline
        == "BagelUnderstandingPipeline"
    )

    report = run_benchmark(
        _config(tmp_path, image_path=image_path), generator_factory=factory
    )

    assert _case(report, "understanding")["status"] == "failed"
    assert _case(report, "editing")["status"] == "not_run"
    assert len(report["cases"]) == 6
    assert factory.max_active == 1
    assert factory.active == 0
    assert all(generator.shutdown_calls == 1 for generator in factory.created)


def test_shutdown_failure_never_starts_the_next_pipeline(tmp_path: Path) -> None:
    factory = FakeGeneratorFactory(shutdown_error_pipeline="BagelPipeline")
    config = replace(
        _config(tmp_path, image_path=None),
        continue_on_error=True,
    )

    report = run_benchmark(config, generator_factory=factory)

    assert [generator.pipeline_class_name for generator in factory.created] == [
        "BagelPipeline"
    ]
    assert report["pipelines"][0]["status"] == "failed"
    assert "Shutdown failed" in report["pipelines"][0]["error"]


def test_startup_failure_stops_even_when_continue_is_requested(tmp_path: Path) -> None:
    factory = FakeGeneratorFactory(startup_error_pipeline="BagelPipeline")
    config = replace(
        _config(tmp_path, image_path=None),
        continue_on_error=True,
    )

    report = run_benchmark(config, generator_factory=factory)

    assert factory.created == []
    assert len(report["pipelines"]) == 1
    assert report["pipelines"][0]["status"] == "failed"
    assert _case(report, "t2i")["status"] == "failed"
    assert _case(report, "thinking")["status"] == "not_run"
    assert len(report["cases"]) == 6


def test_continue_on_error_requires_successful_cleanup(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory(
        return_none_when=lambda pipeline, _params: pipeline
        == "BagelUnderstandingPipeline"
    )
    config = replace(
        _config(tmp_path, image_path=image_path),
        continue_on_error=True,
    )

    report = run_benchmark(config, generator_factory=factory)

    assert _case(report, "understanding")["status"] == "failed"
    assert _case(report, "editing")["status"] == "passed"
    assert [generator.pipeline_class_name for generator in factory.created] == list(
        PIPELINE_ORDER
    )
    assert all(generator.shutdown_calls == 1 for generator in factory.created)


def test_unattempted_editing_workloads_are_never_reported_as_passed(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory(
        return_none_when=lambda pipeline, _params: pipeline == "BagelEditPipeline"
    )

    report = run_benchmark(
        _config(tmp_path, image_path=image_path), generator_factory=factory
    )

    editing = _case(report, "editing")
    assert editing["status"] == "failed"
    assert [workload["status"] for workload in editing["workloads"]] == [
        "failed",
        "not_run",
        "not_run",
    ]


def test_parallelism_is_limited_to_valid_bagel_tp_layouts(tmp_path: Path) -> None:
    base_config = _config(tmp_path, image_path=None)
    invalid_perf_dir = tmp_path / "perf.json"
    invalid_perf_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="perf_dump_dir is not a directory"):
        replace(base_config, perf_dump_dir=invalid_perf_dir)
    with pytest.raises(ValueError, match="image_path is not a file"):
        replace(base_config, image_path=tmp_path / "missing.png")
    with pytest.raises(ValueError, match="num_gpus must be 1 or 2"):
        replace(base_config, num_gpus=4)
    with pytest.raises(ValueError, match="tp_size to equal num_gpus"):
        replace(base_config, num_gpus=2, tp_size=1)

    factory = FakeGeneratorFactory()
    run_benchmark(
        replace(base_config, num_gpus=2, tp_size=None), generator_factory=factory
    )

    assert factory.created_kwargs
    assert all(kwargs["num_gpus"] == 2 for kwargs in factory.created_kwargs)
    assert all(kwargs["tp_size"] == 2 for kwargs in factory.created_kwargs)


def test_report_contains_only_comparable_speedups_and_aggregates_editing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)

    report = run_benchmark(
        _config(tmp_path, image_path=image_path),
        generator_factory=FakeGeneratorFactory(),
    )

    comparisons = report["comparisons"]
    assert isinstance(comparisons, list)
    assert {
        (comparison["baseline"], comparison["candidate"]) for comparison in comparisons
    } == {
        ("t2i", "t2i_taylorseer"),
        ("thinking", "thinking_taylorseer"),
    }
    assert all(
        comparison["metric"] == "scheduler_total_duration_ms"
        for comparison in comparisons
    )
    assert all(float(comparison["speedup"]) == 2.0 for comparison in comparisons)

    editing = _case(report, "editing")
    workloads = editing.get("workloads", editing.get("variants"))
    assert isinstance(workloads, list)
    assert len(workloads) == 3
    assert all(workload["status"] == "passed" for workload in workloads)
    summary = editing["summary"]
    assert isinstance(summary, dict)
    assert summary.get("workload_count", len(workloads)) == 3


def test_each_case_warms_up_before_its_timed_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory()
    config = replace(
        _config(tmp_path, image_path=image_path),
        warmup=1,
        editing_warmup=1,
    )

    report = run_benchmark(config, generator_factory=factory)

    assert all(
        _case(report, case_id)["status"] == "passed"
        for case_id in (
            "t2i",
            "t2i_taylorseer",
            "thinking",
            "thinking_taylorseer",
            "understanding",
            "editing",
        )
    )
    assert len(_calls_for(factory, "BagelPipeline")) == 4
    assert len(_calls_for(factory, "BagelThinkingPipeline")) == 4
    assert len(_calls_for(factory, "BagelUnderstandingPipeline")) == 2
    assert len(_calls_for(factory, "BagelEditPipeline")) == 6

    for pipeline_class_name in PIPELINE_ORDER[:-1]:
        calls = _calls_for(factory, pipeline_class_name)
        for warmup, timed in zip(calls[::2], calls[1::2], strict=True):
            assert warmup["save_output"] is False
            if pipeline_class_name == "BagelUnderstandingPipeline":
                assert timed["save_output"] is False
            else:
                assert timed["save_output"] is True

    editing_calls = _calls_for(factory, "BagelEditPipeline")
    assert all(params["save_output"] is False for params in editing_calls[:3])
    assert all(params["save_output"] is True for params in editing_calls[3:])


def test_perf_dump_dir_profiles_only_timed_requests_with_unique_paths(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    _write_input_image(image_path)
    factory = FakeGeneratorFactory()
    perf_dump_dir = tmp_path / "worker-perf"
    config = replace(
        _config(tmp_path, image_path=image_path),
        perf_dump_dir=perf_dump_dir,
        warmup=1,
        editing_warmup=1,
    )

    report = run_benchmark(config, generator_factory=factory)

    profiled_calls = [
        call["params"]
        for call in factory.generate_calls
        if "perf_dump_path" in call["params"]
    ]
    assert len(profiled_calls) == 8
    perf_paths = [Path(str(params["perf_dump_path"])) for params in profiled_calls]
    assert len(set(perf_paths)) == 8
    assert all(path.is_absolute() and path.is_file() for path in perf_paths)
    assert all(perf_dump_dir.resolve() in path.parents for path in perf_paths)
    for pipeline_class_name in PIPELINE_ORDER[:-1]:
        calls = _calls_for(factory, pipeline_class_name)
        assert ["perf_dump_path" in params for params in calls] == [
            False,
            True,
        ] * (len(calls) // 2)
    editing_calls = _calls_for(factory, "BagelEditPipeline")
    assert ["perf_dump_path" in params for params in editing_calls] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    reported_paths: list[Path] = []
    for case in report["cases"]:
        for workload in case["workloads"]:
            for sample in workload["samples"]:
                worker_path = sample["worker_perf_dump_path"]
                assert worker_path is not None
                reported_paths.append(Path(worker_path))
    assert set(reported_paths) == set(perf_paths)


def test_write_report_atomic_round_trips_json_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    report: dict[str, object] = {
        "schema_version": 1,
        "cases": [{"id": "t2i", "status": "passed"}],
        "comparisons": [],
    }
    output_path = tmp_path / "nested" / "report.json"

    write_report_atomic(report, output_path)

    with output_path.open(encoding="utf-8") as report_file:
        assert json.load(report_file) == report
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []
