"""
Server management and performance validation for diffusion tests.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.diffusion_config import (
    PerformanceSummary,
    ScenarioConfig,
    ToleranceConfig,
)
from sglang.multimodal_gen.test.test_utils import (
    prepare_perf_log,
    sample_step_indices,
    validate_image,
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
        env["SGL_DIFFUSION_STAGE_LOGGING"] = "1"
        env["SGLANG_PERF_LOG_DIR"] = log_dir.as_posix()

        stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            stdout=stdout_fh,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

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

    def __init__(
        self,
        scenario: ScenarioConfig,
        tolerances: ToleranceConfig,
        step_fractions: Sequence[float],
    ):
        self.scenario = scenario
        self.tolerances = tolerances
        self.step_fractions = step_fractions

    def validate(
        self,
        perf_record: dict,
        stage_metrics: dict,
    ) -> PerformanceSummary:
        """Validate all performance metrics and return summary."""
        self._validate_e2e(perf_record)
        avg_denoise, median_denoise = self._validate_denoise_agg(perf_record)
        sampled_steps = self._validate_denoise_steps(perf_record)
        self._validate_stages(stage_metrics)

        return PerformanceSummary(
            e2e_ms=float(perf_record["total_duration_ms"]),
            avg_denoise_ms=avg_denoise,
            median_denoise_ms=median_denoise,
            stage_metrics=stage_metrics,
            sampled_steps=sampled_steps,
        )

    def _validate_e2e(self, perf_record: dict) -> None:
        """Validate end-to-end performance."""
        e2e_ms = float(perf_record.get("total_duration_ms", 0.0))
        assert e2e_ms > 0, "E2E duration missing"

        upper = self.scenario.expected_e2e_ms * (1 + self.tolerances.e2e)
        assert e2e_ms <= upper, f"E2E {e2e_ms:.2f}ms exceeds {upper:.2f}ms"

    def _validate_denoise_agg(self, perf_record: dict) -> tuple[float, float]:
        """Validate aggregate denoising metrics."""
        steps = [
            s
            for s in perf_record.get("steps", []) or []
            if s.get("name") == "denoising_step_guided" and "duration_ms" in s
        ]
        assert steps, "Denoising step timings missing"

        durations = [float(s["duration_ms"]) for s in steps]
        avg = sum(durations) / len(durations)
        median = statistics.median(durations)

        avg_upper = self.scenario.expected_avg_denoise_ms * (
            1 + self.tolerances.denoise_agg
        )
        med_upper = self.scenario.expected_median_denoise_ms * (
            1 + self.tolerances.denoise_agg
        )

        assert avg <= avg_upper, f"Avg denoise {avg:.2f}ms exceeds {avg_upper:.2f}ms"
        assert (
            median <= med_upper
        ), f"Median denoise {median:.2f}ms exceeds {med_upper:.2f}ms"

        return avg, median

    def _validate_denoise_steps(self, perf_record: dict) -> dict[int, float]:
        """Validate individual denoising steps."""
        steps = [
            s
            for s in perf_record.get("steps", []) or []
            if s.get("name") == "denoising_step_guided" and "duration_ms" in s
        ]

        per_step = {
            int(s["index"]): float(s["duration_ms"])
            for s in steps
            if s.get("index") is not None
        }

        sample_indices = sample_step_indices(per_step, self.step_fractions)
        sampled = {idx: per_step[idx] for idx in sample_indices}

        for idx in sample_indices:
            expected = self.scenario.denoise_step_ms.get(idx)
            if expected is None:
                continue

            actual = per_step[idx]
            upper = expected * (1 + self.tolerances.denoise_step)
            assert actual <= upper, f"Step {idx}: {actual:.2f}ms > {upper:.2f}ms"

        return sampled

    def _validate_stages(self, stage_metrics: dict) -> None:
        """Validate stage-level metrics."""
        assert stage_metrics, "Stage metrics missing"

        for stage, expected in self.scenario.stages_ms.items():
            actual = stage_metrics.get(stage)
            assert actual is not None, f"Stage {stage} timing missing"

            upper = expected * (1 + self.tolerances.stage)
            assert actual <= upper, f"Stage {stage}: {actual:.2f}ms > {upper:.2f}ms"


class VideoPerformanceValidator(PerformanceValidator):
    """Extended validator for video diffusion with frame-level metrics."""

    def validate(
        self,
        perf_record: dict,
        stage_metrics: dict,
        num_frames: int | None = None,
    ) -> PerformanceSummary:
        """Validate video metrics including frame generation rates."""
        summary = super().validate(perf_record, stage_metrics)

        if num_frames and summary.e2e_ms > 0:
            summary.total_frames = num_frames
            summary.avg_frame_time_ms = summary.e2e_ms / num_frames
            summary.frames_per_second = 1000.0 / summary.avg_frame_time_ms

            self._validate_frame_rate(summary)

        return summary

    def _validate_frame_rate(self, summary: PerformanceSummary) -> None:
        """Validate frame generation performance."""
        expected_frame_time = self.scenario.stages_ms.get("per_frame_generation")
        if expected_frame_time and summary.avg_frame_time_ms:
            upper = expected_frame_time * (1 + self.tolerances.stage)
            assert (
                summary.avg_frame_time_ms <= upper
            ), f"Avg frame time {summary.avg_frame_time_ms:.2f}ms exceeds {upper:.2f}ms"

    def _validate_stages(self, stage_metrics: dict) -> None:
        """Validate video-specific stages."""
        assert stage_metrics, "Stage metrics missing"

        for stage, expected in self.scenario.stages_ms.items():
            if stage == "per_frame_generation":
                continue

            actual = stage_metrics.get(stage)
            assert actual is not None, f"Stage {stage} timing missing"

            upper = expected * (1 + self.tolerances.stage)
            assert actual <= upper, f"Stage {stage}: {actual:.2f}ms > {upper:.2f}ms"


# Registry of validators by name
VALIDATOR_REGISTRY = {
    "default": PerformanceValidator,
    "video": VideoPerformanceValidator,
}
