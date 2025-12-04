"""
Server management and performance validation for diffusion tests.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import urlopen

import pytest
from openai import Client, OpenAI

from sglang.multimodal_gen.benchmarks.compare_perf import calculate_upper_bound
from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    PerformanceSummary,
    ScenarioConfig,
    ToleranceConfig,
)
from sglang.multimodal_gen.test.slack_utils import upload_file_to_slack
from sglang.multimodal_gen.test.test_utils import (
    is_image_url,
    prepare_perf_log,
    validate_image,
    validate_openai_video,
)

logger = init_logger(__name__)


def download_image_from_url(url: str) -> Path:
    """Download an image from a URL to a temporary file.

    Args:
        url: The URL of the image to download

    Returns:
        Path to the downloaded temporary file
    """
    logger.info(f"Downloading image from URL: {url}")

    # Determine file extension from URL
    ext = ".jpg"  # default
    if url.lower().endswith((".png", ".jpeg", ".jpg", ".webp", ".gif")):
        ext = url[url.rfind(".") :]

    # Create temporary file
    temp_file = (
        Path(tempfile.gettempdir()) / f"diffusion_test_image_{int(time.time())}{ext}"
    )

    try:
        with urlopen(url, timeout=30) as response:
            temp_file.write_bytes(response.read())
        logger.info(f"Downloaded image to: {temp_file}")
        return temp_file
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


@dataclass
class ServerContext:
    """Context for a running diffusion server."""

    port: int
    process: subprocess.Popen
    model: str
    stdout_file: Path
    perf_log_path: Path
    log_dir: Path
    _stdout_fh: Any = field(repr=False)
    _log_thread: threading.Thread | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Clean up server resources."""
        try:
            kill_process_tree(self.process.pid)
        except Exception:
            pass
        try:
            self._stdout_fh.flush()
            self._stdout_fh.close()
        except Exception:
            pass


class ServerManager:
    """Manages diffusion server lifecycle."""

    def __init__(
        self,
        model: str,
        port: int,
        wait_deadline: float = 1200.0,
        extra_args: str = "",
    ):
        self.model = model
        self.port = port
        self.wait_deadline = wait_deadline
        self.extra_args = extra_args

    def start(self) -> ServerContext:
        """Start the diffusion server and wait for readiness."""
        log_dir, perf_log_path = prepare_perf_log()

        safe_model_name = self.model.replace("/", "_")
        stdout_path = (
            Path(tempfile.gettempdir())
            / f"sgl_server_{self.port}_{safe_model_name}.log"
        )
        stdout_path.unlink(missing_ok=True)

        command = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--port",
            str(self.port),
            "--log-level=debug",
        ]
        if self.extra_args.strip():
            command.extend(self.extra_args.strip().split())

        env = os.environ.copy()
        env["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
        env["SGLANG_PERF_LOG_DIR"] = log_dir.as_posix()

        # TODO: unify with run_command
        logger.info(f"Running command: {shlex.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        log_thread = None
        stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
        if process.stdout:

            def _log_pipe(pipe: Any, file: Any) -> None:
                """Read from pipe and write to file and stdout."""
                try:
                    with pipe:
                        for line in iter(pipe.readline, ""):
                            sys.stdout.write(line)
                            file.write(line)
                            file.flush()
                except Exception as e:
                    logger.error("Log pipe thread error: %s", e)
                finally:
                    file.close()
                    logger.debug("Log pipe thread finished.")

            log_thread = threading.Thread(
                target=_log_pipe, args=(process.stdout, stdout_fh)
            )
            log_thread.daemon = True
            log_thread.start()

        logger.info(
            "[server-test] Starting server pid=%s, model=%s, log=%s",
            process.pid,
            self.model,
            stdout_path,
        )

        self._wait_for_ready(process, stdout_path)

        return ServerContext(
            port=self.port,
            process=process,
            model=self.model,
            stdout_file=stdout_path,
            perf_log_path=perf_log_path,
            log_dir=log_dir,
            _stdout_fh=stdout_fh,
            _log_thread=log_thread,
        )

    def _wait_for_ready(self, process: subprocess.Popen, stdout_path: Path) -> None:
        """Wait for server to become ready."""
        start = time.time()
        ready_message = "Application startup complete."

        while time.time() - start < self.wait_deadline:
            if process.poll() is not None:
                tail = self._get_log_tail(stdout_path)
                raise RuntimeError(
                    f"Server exited early (code {process.returncode}).\n{tail}"
                )

            if stdout_path.exists():
                try:
                    content = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    if ready_message in content:
                        logger.info("[server-test] Server ready")
                        return
                except Exception as e:
                    logger.debug("Could not read log yet: %s", e)

            elapsed = int(time.time() - start)
            logger.info("[server-test] Waiting for server... elapsed=%ss", elapsed)
            time.sleep(5)

        tail = self._get_log_tail(stdout_path)
        raise TimeoutError(f"Server not ready within {self.wait_deadline}s.\n{tail}")

    @staticmethod
    def _get_log_tail(path: Path, lines: int = 200) -> str:
        """Get the last N lines from a log file."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return "\n".join(content.splitlines()[-lines:])
        except Exception:
            return ""


class WarmupRunner:
    """Handles warmup requests for a server."""

    def __init__(
        self,
        port: int,
        model: str,
        prompt: str,
        output_size: str,
    ):
        self.client = OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{port}/v1",
        )
        self.model = model
        self.prompt = prompt
        self.output_size = output_size

    def run_text_warmups(self, count: int) -> None:
        """Run text-to-image warmup requests."""
        if count <= 0:
            return

        logger.info("[server-test] Running %s text warm-up(s)", count)
        for _ in range(count):
            result = self.client.images.generate(
                model=self.model,
                prompt=self.prompt,
                n=1,
                size=self.output_size,
                response_format="b64_json",
            )
            validate_image(result.data[0].b64_json)

    def run_edit_warmups(
        self,
        count: int,
        edit_prompt: str,
        image_path: Path,
    ) -> None:
        """Run image-edit warmup requests."""
        if count <= 0:
            return

        if not image_path.exists():
            logger.warning(
                "[server-test] Skipping edit warmup: image missing at %s", image_path
            )
            return

        logger.info("[server-test] Running %s edit warm-up(s)", count)
        for _ in range(count):
            with image_path.open("rb") as fh:
                result = self.client.images.edit(
                    model=self.model,
                    image=fh,
                    prompt=edit_prompt,
                    n=1,
                    size=self.output_size,
                    response_format="b64_json",
                )
            validate_image(result.data[0].b64_json)


class PerformanceValidator:
    """Validates performance metrics against expectations."""

    is_video_gen: bool = False

    def __init__(
        self,
        scenario: ScenarioConfig,
        tolerances: ToleranceConfig,
        step_fractions: Sequence[float],
    ):
        self.scenario = scenario
        self.tolerances = tolerances
        self.step_fractions = step_fractions
        self.is_baseline_generation_mode = (
            os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"
        )

    def _assert_le(
        self,
        name: str,
        actual: float,
        expected: float,
        tolerance: float,
        min_abs_tolerance_ms: float = 20.0,
    ):
        """Assert that actual is less than or equal to expected within a tolerance.

        Uses the larger of relative tolerance or absolute tolerance to prevent
        flaky failures on very fast operations.
        """
        upper_bound = calculate_upper_bound(expected, tolerance, min_abs_tolerance_ms)
        assert actual <= upper_bound, (
            f"Validation failed for '{name}'.\n"
            f"  Actual:   {actual:.4f}ms\n"
            f"  Expected: {expected:.4f}ms\n"
            f"  Limit:    {upper_bound:.4f}ms "
            f"(rel_tol: {tolerance:.1%}, abs_pad: {min_abs_tolerance_ms}ms)"
        )

    def validate(
        self, perf_record: RequestPerfRecord, *args, **kwargs
    ) -> PerformanceSummary:
        """Validate all performance metrics and return summary."""
        summary = self.collect_metrics(perf_record)
        if self.is_baseline_generation_mode:
            return summary

        self._validate_e2e(summary)
        self._validate_denoise_agg(summary)
        self._validate_denoise_steps(summary)
        self._validate_stages(summary)

        return summary

    def collect_metrics(
        self,
        perf_record: RequestPerfRecord,
    ) -> PerformanceSummary:
        return PerformanceSummary.from_req_perf_record(perf_record, self.step_fractions)

    def _validate_e2e(self, summary: PerformanceSummary) -> None:
        """Validate end-to-end performance."""
        assert summary.e2e_ms > 0, "E2E duration missing"
        self._assert_le(
            "E2E Latency",
            summary.e2e_ms,
            self.scenario.expected_e2e_ms,
            self.tolerances.e2e,
        )

    def _validate_denoise_agg(self, summary: PerformanceSummary) -> None:
        """Validate aggregate denoising metrics."""
        assert summary.avg_denoise_ms > 0, "Denoising step timings missing"

        self._assert_le(
            "Average Denoise Step",
            summary.avg_denoise_ms,
            self.scenario.expected_avg_denoise_ms,
            self.tolerances.denoise_agg,
        )
        self._assert_le(
            "Median Denoise Step",
            summary.median_denoise_ms,
            self.scenario.expected_median_denoise_ms,
            self.tolerances.denoise_agg,
        )

    def _validate_denoise_steps(self, summary: PerformanceSummary) -> None:
        """Validate individual denoising steps."""
        for idx, actual in summary.sampled_steps.items():
            expected = self.scenario.denoise_step_ms.get(idx)
            if expected is None:
                continue
            # FIXME: hardcode, looser for first step
            tolerance = 0.4 if idx == 0 else self.tolerances.denoise_step

            self._assert_le(
                f"Denoise Step {idx}",
                actual,
                expected,
                tolerance,
            )

    def _validate_stages(self, summary: PerformanceSummary) -> None:
        """Validate stage-level metrics."""
        assert summary.stage_metrics, "Stage metrics missing"

        for stage, expected in self.scenario.stages_ms.items():
            if stage == "per_frame_generation" and self.is_video_gen:
                continue
            actual = summary.stage_metrics.get(stage)
            assert actual is not None, f"Stage {stage} timing missing"
            tolerance = (
                self.tolerances.denoise_stage
                if stage == "DenoisingStage"
                else self.tolerances.non_denoise_stage
            )
            self._assert_le(
                f"Stage '{stage}'",
                actual,
                expected,
                tolerance,
                min_abs_tolerance_ms=120.0,  # relax absolute tolerance for non-denoising stages
            )


class VideoPerformanceValidator(PerformanceValidator):
    """Extended validator for video diffusion with frame-level metrics."""

    is_video_gen = True

    def validate(
        self,
        perf_record: RequestPerfRecord,
        num_frames: int | None = None,
    ) -> PerformanceSummary:
        """Validate video metrics including frame generation rates."""
        summary = super().validate(perf_record)

        if num_frames and summary.e2e_ms > 0:
            summary.total_frames = num_frames
            summary.avg_frame_time_ms = summary.e2e_ms / num_frames
            summary.frames_per_second = 1000.0 / summary.avg_frame_time_ms

            if not self.is_baseline_generation_mode:
                self._validate_frame_rate(summary)

        return summary

    def _validate_frame_rate(self, summary: PerformanceSummary) -> None:
        """Validate frame generation performance."""
        expected_frame_time = self.scenario.stages_ms.get("per_frame_generation")
        if expected_frame_time and summary.avg_frame_time_ms:
            self._assert_le(
                "Average Frame Time",
                summary.avg_frame_time_ms,
                expected_frame_time,
                self.tolerances.denoise_stage,
            )


# Registry of validators by name
VALIDATOR_REGISTRY = {
    "default": PerformanceValidator,
    "video": VideoPerformanceValidator,
}


def get_generate_fn(
    model_path: str,
    modality: str,
    sampling_params: DiffusionSamplingParams,
) -> Callable[[str, Client], str]:
    """Return appropriate generation function for the case."""

    def _create_and_download_video(
        client,
        case_id,
        *,
        model: str,
        size: str,
        prompt: str | None = None,
        seconds: int | None = None,
        input_reference: Any | None = None,
    ) -> str:
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

        job = client.videos.create(**create_kwargs)  # type: ignore[attr-defined]
        video_id = job.id

        job_completed = False
        is_baseline_generation_mode = os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"
        timeout = 3600.0 if is_baseline_generation_mode else 1200.0
        deadline = time.time() + timeout
        while True:
            page = client.videos.list()  # type: ignore[attr-defined]
            item = next((v for v in page.data if v.id == video_id), None)

            if item and getattr(item, "status", None) == "completed":
                job_completed = True
                break

            if time.time() > deadline:
                break

            time.sleep(1)

        if not job_completed:
            if is_baseline_generation_mode:
                logger.warning(
                    f"{id}: video job {video_id} timed out during baseline generation. "
                    "Attempting to collect performance data anyway."
                )
                return video_id

            pytest.fail(f"{id}: video job {video_id} did not complete in time")

        # download video
        resp = client.videos.download_content(video_id=video_id)  # type: ignore[attr-defined]
        content = resp.read()
        validate_openai_video(content)

        tmp_path = f"{video_id}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(content)
        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
            origin_file_path=sampling_params.image_path,
        )
        os.remove(tmp_path)

        return video_id

    video_seconds = sampling_params.seconds or 4

    def generate_image(case_id, client) -> str:
        """T2I: Text to Image generation."""
        if not sampling_params.prompt:
            pytest.skip(f"{id}: no text prompt configured")

        response = client.images.with_raw_response.generate(
            model=model_path,
            prompt=sampling_params.prompt,
            n=1,
            size=sampling_params.output_size,
            response_format="b64_json",
        )
        result = response.parse()
        validate_image(result.data[0].b64_json)

        img_data = base64.b64decode(result.data[0].b64_json)
        tmp_path = f"{result.created}.png"
        with open(tmp_path, "wb") as f:
            f.write(img_data)
        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
        )
        os.remove(tmp_path)

        return str(result.created)

    def generate_image_edit(case_id, client) -> str:
        """TI2I: Text + Image ? Image edit."""
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")

        if is_image_url(sampling_params.image_path):
            image_path = download_image_from_url(str(sampling_params.image_path))
        else:
            image_path = Path(sampling_params.image_path)
            if not image_path.exists():
                pytest.skip(f"{id}: file missing: {image_path}")

        with image_path.open("rb") as fh:
            response = client.images.with_raw_response.edit(
                model=model_path,
                image=fh,
                prompt=sampling_params.prompt,
                n=1,
                size=sampling_params.output_size,
                response_format="b64_json",
            )
        rid = response.headers.get("x-request-id", "")

        result = response.parse()
        validate_image(result.data[0].b64_json)

        img_data = base64.b64decode(result.data[0].b64_json)
        tmp_path = f"{rid}.png"
        with open(tmp_path, "wb") as f:
            f.write(img_data)
        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
            origin_file_path=sampling_params.image_path,
        )
        os.remove(tmp_path)

        return rid

    def generate_video(case_id, client) -> str:
        """T2V: Text ? Video."""
        if not sampling_params.prompt:
            pytest.skip(f"{id}: no text prompt configured")

        return _create_and_download_video(
            client,
            case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            size=sampling_params.output_size,
            seconds=video_seconds,
        )

    def generate_image_to_video(case_id, client) -> str:
        """I2V: Image ? Video (optional prompt)."""
        if not sampling_params.image_path:
            pytest.skip(f"{id}: no input image configured")

        if is_image_url(sampling_params.image_path):
            image_path = download_image_from_url(str(sampling_params.image_path))
        else:
            image_path = Path(sampling_params.image_path)
            if not image_path.exists():
                pytest.skip(f"{id}: file missing: {image_path}")

        with image_path.open("rb") as fh:
            return _create_and_download_video(
                client,
                case_id,
                model=model_path,
                prompt=sampling_params.prompt,
                size=sampling_params.output_size,
                seconds=video_seconds,
                input_reference=fh,
            )

    def generate_text_image_to_video(case_id, client) -> str:
        """TI2V: Text + Image ? Video."""
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")

        if is_image_url(sampling_params.image_path):
            image_path = download_image_from_url(str(sampling_params.image_path))
        else:
            image_path = Path(sampling_params.image_path)
            if not image_path.exists():
                pytest.skip(f"{id}: file missing: {image_path}")

        with image_path.open("rb") as fh:
            return _create_and_download_video(
                client,
                case_id,
                model=model_path,
                prompt=sampling_params.prompt,
                size=sampling_params.output_size,
                seconds=video_seconds,
                input_reference=fh,
            )

    if modality == "video":
        if sampling_params.image_path and sampling_params.prompt:
            fn = generate_text_image_to_video
        elif sampling_params.image_path:
            fn = generate_image_to_video
        else:
            fn = generate_video
    elif sampling_params.prompt and sampling_params.image_path:
        fn = generate_image_edit
    else:
        fn = generate_image

    return fn
