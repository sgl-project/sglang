"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.conftest import _GLOBAL_PERF_RESULTS
from sglang.multimodal_gen.test.server.test_server_utils import (
    VALIDATOR_REGISTRY,
    PerformanceValidator,
    ServerContext,
    ServerManager,
    WarmupRunner,
    download_image_from_url,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DIFFUSION_CASES,
    DiffusionTestCase,
    PerformanceSummary,
    ScenarioConfig,
)
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    read_perf_records,
    validate_image,
    validate_openai_video,
    wait_for_perf_record,
    wait_for_stage_metrics,
)

logger = init_logger(__name__)


@pytest.fixture(params=DIFFUSION_CASES, ids=lambda c: c.id)
def case(request) -> DiffusionTestCase:
    """Provide a DiffusionTestCase for each test."""
    return request.param


@pytest.fixture
def diffusion_server(case: DiffusionTestCase) -> ServerContext:
    """Start a diffusion server for a single case and tear it down afterwards."""
    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))

    # start server
    manager = ServerManager(
        model=case.model_path,
        port=port,
        wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
        extra_args=os.environ.get("SGLANG_TEST_SERVE_ARGS", ""),
    )
    ctx = manager.start()

    if case.startup_grace_seconds > 0:
        logger.info(
            "[server-test] Waiting %.1fs for %s to settle",
            case.startup_grace_seconds,
            case.id,
        )
        time.sleep(case.startup_grace_seconds)

    try:
        warmup = WarmupRunner(
            port=ctx.port,
            model=case.model_path,
            prompt=case.prompt or "A colorful raccoon icon",
            output_size=case.output_size,
        )
        warmup.run_text_warmups(case.warmup_text)

        if case.warmup_edit > 0 and case.edit_prompt and case.image_path:
            # Handle URL or local path
            image_path = case.image_path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_path))
            else:
                image_path = Path(case.image_path)

            warmup.run_edit_warmups(
                count=case.warmup_edit,
                edit_prompt=case.edit_prompt,
                image_path=image_path,
            )
    except Exception as exc:
        logger.error("Warm-up failed for %s: %s", case.id, exc)
        ctx.cleanup()
        raise

    try:
        yield ctx
    finally:
        ctx.cleanup()


class TestDiffusionPerformance:
    """Performance tests for all diffusion models/scenarios.

    This single test class runs against all cases defined in DIFFUSION_CASES.
    Each case gets its own server instance via the parametrized fixture.
    """

    _perf_results: list[dict[str, Any]] = []
    _improved_baselines: list[dict[str, Any]] = []

    @classmethod
    def setup_class(cls):
        cls._perf_results = []
        cls._improved_baselines = []

    @classmethod
    def teardown_class(cls):
        for result in cls._perf_results:
            result["class_name"] = cls.__name__
            _GLOBAL_PERF_RESULTS.append(result)

        if cls._improved_baselines:
            import json

            output = """
--- POTENTIAL BASELINE IMPROVEMENTS DETECTED ---
The following test cases performed significantly better than their baselines.
Consider updating perf_baselines.json with the snippets below:
"""
            for item in cls._improved_baselines:
                output += (
                    f'\n"{item["id"]}": {json.dumps(item["baseline"], indent=4)},\n'
                )
            print(output)

    def _client(self, ctx: ServerContext) -> OpenAI:
        """Get OpenAI client for the server."""
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
        )

    def _run_and_collect(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[], None],
    ) -> tuple[dict, dict]:
        """Run generation and collect performance records."""
        log_path = ctx.perf_log_path
        prev_len = len(read_perf_records(log_path))
        log_wait_timeout = 1200

        generate_fn()

        perf_record, _ = wait_for_perf_record(
            "total_inference_time",
            prev_len,
            log_path,
            timeout=log_wait_timeout,
        )

        stage_metrics = {}
        if perf_record:
            stage_metrics, _ = wait_for_stage_metrics(
                perf_record.get("request_id", ""),
                prev_len,
                log_path,
                timeout=log_wait_timeout,
            )

        return perf_record, stage_metrics

    def _generate_for_case(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
    ) -> Callable[[], None]:
        """Return appropriate generation function for the case."""
        client = self._client(ctx)

        def _create_and_download_video(
            *,
            model: str,
            size: str,
            prompt: str | None = None,
            seconds: int | None = None,
            input_reference: Any | None = None,
        ) -> bytes:
            """
            Create a video job via /v1/videos, poll until completion,
            then download the binary content and validate it.
            """
            create_kwargs: dict[str, Any] = {
                "model": model,
                "size": size,
            }
            if prompt is not None:
                create_kwargs["prompt"] = prompt
            if seconds is not None:
                create_kwargs["seconds"] = seconds
            if input_reference is not None:
                create_kwargs["input_reference"] = input_reference  # triggers multipart

            # create video job
            job = client.videos.create(**create_kwargs)  # type: ignore[attr-defined]
            video_id = job.id

            job_completed = False
            is_baseline_generation_mode = (
                os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"
            )
            timeout = 3600.0 if is_baseline_generation_mode else 600.0
            deadline = time.time() + timeout
            while True:
                page = client.videos.list()  # type: ignore[attr-defined]
                item = next((v for v in page.data if v.id == video_id), None)

                if item and getattr(item, "status", None) == "completed":
                    job_completed = True
                    break

                if time.time() > deadline:
                    break

                time.sleep(5)

            if not job_completed:
                if is_baseline_generation_mode:
                    logger.warning(
                        f"{case.id}: video job {video_id} timed out during baseline generation. "
                        "Attempting to collect performance data anyway."
                    )
                    return b""

                pytest.fail(f"{case.id}: video job {video_id} did not complete in time")

            # download video
            resp = client.videos.download_content(video_id=video_id)  # type: ignore[attr-defined]
            content = resp.read()
            validate_openai_video(content)
            return content

        # for all tests, seconds = case.seconds or fallback 4 seconds
        video_seconds = case.seconds or 4

        # -------------------------
        # IMAGE MODE
        # -------------------------

        def generate_image():
            """T2I: Text to Image generation."""
            if not case.prompt:
                pytest.skip(f"{case.id}: no text prompt configured")
            result = client.images.generate(
                model=case.model_path,
                prompt=case.prompt,
                n=1,
                size=case.output_size,
                response_format="b64_json",
            )
            validate_image(result.data[0].b64_json)

        def generate_image_edit():
            """TI2I: Text + Image ? Image edit."""
            if not case.edit_prompt or not case.image_path:
                pytest.skip(f"{case.id}: no edit config")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_path))
            else:
                image_path = Path(case.image_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                result = client.images.edit(
                    model=case.model_path,
                    image=fh,
                    prompt=case.edit_prompt,
                    n=1,
                    size=case.output_size,
                    response_format="b64_json",
                )
            validate_image(result.data[0].b64_json)

        # -------------------------
        # VIDEO MODE
        # -------------------------

        def generate_video():
            """T2V: Text ? Video."""
            if not case.prompt:
                pytest.skip(f"{case.id}: no text prompt configured")

            _create_and_download_video(
                model=case.model_path,
                prompt=case.prompt,
                size=case.output_size,
                seconds=video_seconds,
            )

        def generate_image_to_video():
            """I2V: Image ? Video (optional prompt)."""
            if not case.image_path:
                pytest.skip(f"{case.id}: no input image configured")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_path))
            else:
                image_path = Path(case.image_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                _create_and_download_video(
                    model=case.model_path,
                    prompt=case.edit_prompt,
                    size=case.output_size,
                    seconds=video_seconds,
                    input_reference=fh,
                )

        def generate_text_image_to_video():
            """TI2V: Text + Image ? Video."""
            if not case.edit_prompt or not case.image_path:
                pytest.skip(f"{case.id}: no edit config")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_path))
            else:
                image_path = Path(case.image_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                _create_and_download_video(
                    model=case.model_path,
                    prompt=case.edit_prompt,
                    size=case.output_size,
                    seconds=video_seconds,
                    input_reference=fh,
                )

        if case.modality == "video":
            if case.image_path and case.edit_prompt:
                return generate_text_image_to_video
            elif case.image_path:
                return generate_image_to_video
            else:
                return generate_video

        # Image modality
        if case.edit_prompt and case.image_path:
            return generate_image_edit

        return generate_image

    def _validate_and_record(
        self,
        case: DiffusionTestCase,
        perf_record: dict,
        stage_metrics: dict,
    ) -> None:
        """Validate metrics and record results."""
        is_baseline_generation_mode = os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"

        scenario = BASELINE_CONFIG.scenarios.get(case.id)
        missing_scenario = False
        if scenario is None:
            # Create dummy scenario to allow metric collection
            scenario = type(
                "DummyScenario",
                (),
                {
                    "expected_e2e_ms": 0,
                    "expected_avg_denoise_ms": 0,
                    "expected_median_denoise_ms": 0,
                    "stages_ms": {},
                    "denoise_step_ms": {},
                },
            )()
            if not is_baseline_generation_mode:
                missing_scenario = True

        validator_name = case.custom_validator or "default"
        validator_class = VALIDATOR_REGISTRY.get(validator_name, PerformanceValidator)

        validator = validator_class(
            scenario=scenario,
            tolerances=BASELINE_CONFIG.tolerances,
            step_fractions=BASELINE_CONFIG.step_fractions,
        )

        summary = validator.collect_metrics(perf_record, stage_metrics)

        if is_baseline_generation_mode or missing_scenario:
            self._dump_baseline_for_testcase(case, summary)
            if missing_scenario:
                pytest.fail(f"Testcase '{case.id}' not found in perf_baselines.json")
            return

        self._check_for_improvement(case, summary, scenario)

        try:
            validator.validate(perf_record, stage_metrics, case.num_frames)
        except AssertionError as e:
            logger.error(f"Performance validation failed for {case.id}:\n{e}")
            self._dump_baseline_for_testcase(case, summary)
            raise

        if case.modality == "video" and summary.frames_per_second:
            logger.info(
                "[Perf] %s: E2E %.2f ms; Avg %.2f ms; FPS %.2f; Frames %d",
                case.id,
                summary.e2e_ms,
                summary.avg_denoise_ms,
                summary.frames_per_second,
                summary.total_frames or 0,
            )
        else:
            logger.info(
                "[Perf] %s: E2E %.2f ms; Avg %.2f ms; Median %.2f ms",
                case.id,
                summary.e2e_ms,
                summary.avg_denoise_ms,
                summary.median_denoise_ms,
            )

        result = {
            "test_name": case.id,
            "modality": case.modality,
            "e2e_ms": summary.e2e_ms,
            "avg_denoise_ms": summary.avg_denoise_ms,
            "median_denoise_ms": summary.median_denoise_ms,
            "stage_metrics": summary.stage_metrics,
            "sampled_steps": summary.sampled_steps,
        }

        # video-specific metrics
        if summary.frames_per_second:
            result.update(
                {
                    "frames_per_second": summary.frames_per_second,
                    "total_frames": summary.total_frames,
                    "avg_frame_time_ms": summary.avg_frame_time_ms,
                }
            )

        self.__class__._perf_results.append(result)

        logger.info("[BASELINE] %s expected_e2e_ms = %.2f", case.id, summary.e2e_ms)
        logger.info(
            "[BASELINE] %s expected_avg_denoise_ms = %.2f",
            case.id,
            summary.avg_denoise_ms,
        )
        logger.info(
            "[BASELINE] %s expected_median_denoise_ms = %.2f",
            case.id,
            summary.median_denoise_ms,
        )
        logger.info("[BASELINE] %s stages_ms = %r", case.id, summary.stage_metrics)
        logger.info(
            "[BASELINE] %s denoise_step_ms = %r", case.id, summary.sampled_steps
        )

        # Only log video-specific metrics when they exist
        if summary.frames_per_second is not None:
            logger.info(
                "[BASELINE] %s frames_per_second = %.2f",
                case.id,
                summary.frames_per_second,
            )
        if summary.total_frames is not None:
            logger.info(
                "[BASELINE] %s total_frames = %d", case.id, summary.total_frames
            )
        if summary.avg_frame_time_ms is not None:
            logger.info(
                "[BASELINE] %s avg_frame_time_ms = %.2f",
                case.id,
                summary.avg_frame_time_ms,
            )

    def _check_for_improvement(
        self,
        case: DiffusionTestCase,
        summary: PerformanceSummary,
        scenario: "ScenarioConfig",
    ) -> None:
        """Check for potential significant performance improvements and record them."""
        is_improved = False
        threshold = BASELINE_CONFIG.improvement_threshold

        def is_sig_faster(actual, expected):
            if expected == 0 or expected is None:
                return False
            return actual < expected * (1 - threshold)

        def safe_get_metric(metric_dict, key):
            val = metric_dict.get(key)
            return val if val is not None else float("inf")

        # Check for any significant improvement
        if (
            is_sig_faster(summary.e2e_ms, scenario.expected_e2e_ms)
            or is_sig_faster(summary.avg_denoise_ms, scenario.expected_avg_denoise_ms)
            or is_sig_faster(
                summary.median_denoise_ms, scenario.expected_median_denoise_ms
            )
        ):
            is_improved = True

        # Combine metrics, always taking the better (lower) value
        new_stages = {
            stage: min(
                safe_get_metric(summary.stage_metrics, stage),
                safe_get_metric(scenario.stages_ms, stage),
            )
            for stage in set(summary.stage_metrics) | set(scenario.stages_ms)
        }
        new_denoise_steps = {
            step: min(
                safe_get_metric(summary.all_denoise_steps, step),
                safe_get_metric(scenario.denoise_step_ms, step),
            )
            for step in set(summary.all_denoise_steps) | set(scenario.denoise_step_ms)
        }

        # Check for stage-level improvements
        if not is_improved:
            for stage, new_val in new_stages.items():
                if is_sig_faster(new_val, scenario.stages_ms.get(stage, float("inf"))):
                    is_improved = True
                    break
        if not is_improved:
            for step, new_val in new_denoise_steps.items():
                if is_sig_faster(
                    new_val, scenario.denoise_step_ms.get(step, float("inf"))
                ):
                    is_improved = True
                    break

        if is_improved:
            new_baseline = {
                "stages_ms": {k: round(v, 2) for k, v in new_stages.items()},
                "denoise_step_ms": {
                    str(k): round(v, 2) for k, v in new_denoise_steps.items()
                },
                "expected_e2e_ms": round(
                    min(summary.e2e_ms, scenario.expected_e2e_ms), 2
                ),
                "expected_avg_denoise_ms": round(
                    min(summary.avg_denoise_ms, scenario.expected_avg_denoise_ms), 2
                ),
                "expected_median_denoise_ms": round(
                    min(summary.median_denoise_ms, scenario.expected_median_denoise_ms),
                    2,
                ),
            }
            self._improved_baselines.append({"id": case.id, "baseline": new_baseline})

    def _dump_baseline_for_testcase(
        self, case: DiffusionTestCase, summary: "PerformanceSummary"
    ) -> None:
        """Dump performance metrics as a JSON scenario for baselines."""
        import json

        denoise_steps_formatted = {
            str(k): round(v, 2) for k, v in summary.all_denoise_steps.items()
        }
        stages_formatted = {k: round(v, 2) for k, v in summary.stage_metrics.items()}

        baseline = {
            "stages_ms": stages_formatted,
            "denoise_step_ms": denoise_steps_formatted,
            "expected_e2e_ms": round(summary.e2e_ms, 2),
            "expected_avg_denoise_ms": round(summary.avg_denoise_ms, 2),
            "expected_median_denoise_ms": round(summary.median_denoise_ms, 2),
        }

        # Video-specific metrics
        if case.modality == "video":
            if "per_frame_generation" not in baseline["stages_ms"]:
                baseline["stages_ms"]["per_frame_generation"] = (
                    round(summary.avg_frame_time_ms, 2)
                    if summary.avg_frame_time_ms
                    else None
                )

        output = f"""
To add this baseline, copy the following JSON snippet into
the "scenarios" section of perf_baselines.json:

"{case.id}": {json.dumps(baseline, indent=4)}

"""
        logger.error(output)

    def test_diffusion_perf(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        """Single parametrized test that runs for all cases.

        Pytest will execute this test once per case in DIFFUSION_CASES,
        with test IDs like:
        - test_diffusion_perf[qwen_image_text]
        - test_diffusion_perf[qwen_image_edit]
        - etc.
        """
        generate_fn = self._generate_for_case(diffusion_server, case)
        perf_record, stage_metrics = self._run_and_collect(
            diffusion_server,
            case,
            generate_fn,
        )
        self._validate_and_record(case, perf_record, stage_metrics)
