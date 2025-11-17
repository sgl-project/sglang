"""
Config-driven diffusion performance test with pytest parametrization.
Adding a new model/scenario = adding one DiffusionCase entry in diffusion_config.py.
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
from sglang.multimodal_gen.test.server.diffusion_config import (
    BASELINE_CONFIG,
    DIFFUSION_CASES,
    DiffusionCase,
)
from sglang.multimodal_gen.test.server.diffusion_server import (
    VALIDATOR_REGISTRY,
    PerformanceValidator,
    ServerContext,
    ServerManager,
    VideoPerformanceValidator,
    WarmupRunner,
    download_image_from_url,
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
def case(request) -> DiffusionCase:
    """Provide a DiffusionCase for each test."""
    return request.param


@pytest.fixture
def diffusion_server(case: DiffusionCase) -> ServerContext:
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

        if case.warmup_edit > 0 and case.image_edit_prompt and case.image_edit_path:
            # Handle URL or local path
            image_path = case.image_edit_path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_edit_path))
            else:
                image_path = Path(case.image_edit_path)

            warmup.run_edit_warmups(
                count=case.warmup_edit,
                edit_prompt=case.image_edit_prompt,
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

    @classmethod
    def setup_class(cls):
        cls._perf_results = []

    @classmethod
    def teardown_class(cls):
        for result in cls._perf_results:
            result["class_name"] = cls.__name__
            _GLOBAL_PERF_RESULTS.append(result)

    def _client(self, ctx: ServerContext) -> OpenAI:
        """Get OpenAI client for the server."""
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
        )

    def _run_and_collect(
        self,
        ctx: ServerContext,
        case: DiffusionCase,
        generate_fn: Callable[[], None],
    ) -> tuple[dict, dict]:
        """Run generation and collect performance records."""
        log_path = ctx.perf_log_path
        prev_len = len(read_perf_records(log_path))

        generate_fn()

        perf_record, _ = wait_for_perf_record(
            "total_inference_time",
            prev_len,
            log_path,
        )

        scenario = BASELINE_CONFIG.scenarios[case.scenario_name]
        stage_metrics, _ = wait_for_stage_metrics(
            perf_record.get("request_id", ""),
            prev_len,
            len(scenario.stages_ms),
            log_path,
        )

        return perf_record, stage_metrics

    def _generate_for_case(
        self,
        ctx: ServerContext,
        case: DiffusionCase,
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

            deadline = time.time() + 600
            while True:
                page = client.videos.list()  # type: ignore[attr-defined]
                item = next((v for v in page.data if v.id == video_id), None)

                if item and getattr(item, "status", None) == "completed":
                    break

                if time.time() > deadline:
                    pytest.fail(
                        f"{case.id}: video job {video_id} did not complete in time"
                    )

                time.sleep(5)

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
            if not case.image_edit_prompt or not case.image_edit_path:
                pytest.skip(f"{case.id}: no edit config")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_edit_path))
            else:
                image_path = Path(case.image_edit_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                result = client.images.edit(
                    model=case.model_path,
                    image=fh,
                    prompt=case.image_edit_prompt,
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
            if not case.image_edit_path:
                pytest.skip(f"{case.id}: no input image configured")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_edit_path))
            else:
                image_path = Path(case.image_edit_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                _create_and_download_video(
                    model=case.model_path,
                    prompt=case.image_edit_prompt,
                    size=case.output_size,
                    seconds=video_seconds,
                    input_reference=fh,
                )

        def generate_text_image_to_video():
            """TI2V: Text + Image ? Video."""
            if not case.image_edit_prompt or not case.image_edit_path:
                pytest.skip(f"{case.id}: no edit config")

            # Handle URL or local path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_edit_path))
            else:
                image_path = Path(case.image_edit_path)
                if not image_path.exists():
                    pytest.skip(f"{case.id}: file missing: {image_path}")

            with image_path.open("rb") as fh:
                _create_and_download_video(
                    model=case.model_path,
                    prompt=case.image_edit_prompt,
                    size=case.output_size,
                    seconds=video_seconds,
                    input_reference=fh,
                )

        if case.modality == "video":
            if case.image_edit_path and case.image_edit_prompt:
                return generate_text_image_to_video
            elif case.image_edit_path:
                return generate_image_to_video
            else:
                return generate_video

        # Image modality
        if case.image_edit_prompt and case.image_edit_path:
            return generate_image_edit

        return generate_image

    def _validate_and_record(
        self,
        case: DiffusionCase,
        perf_record: dict,
        stage_metrics: dict,
    ) -> None:
        """Validate metrics and record results."""
        scenario = BASELINE_CONFIG.scenarios[case.scenario_name]

        validator_name = case.custom_validator or "default"
        validator_class = VALIDATOR_REGISTRY.get(validator_name, PerformanceValidator)

        validator = validator_class(
            scenario=scenario,
            tolerances=BASELINE_CONFIG.tolerances,
            step_fractions=BASELINE_CONFIG.step_fractions,
        )

        if isinstance(validator, VideoPerformanceValidator):
            summary = validator.validate(perf_record, stage_metrics, case.num_frames)
        else:
            summary = validator.validate(perf_record, stage_metrics)

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

    def test_diffusion_perf(
        self,
        case: DiffusionCase,
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
