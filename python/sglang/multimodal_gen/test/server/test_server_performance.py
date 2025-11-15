"""
Config-driven diffusion performance test with pytest parametrization.
Adding a new model/scenario = adding one DiffusionCase entry.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.conftest import _GLOBAL_PERF_RESULTS
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    prepare_perf_log,
    read_perf_records,
    sample_step_indices,
    validate_image,
    validate_openai_video,
    wait_for_perf_record,
    wait_for_stage_metrics,
)

logger = init_logger(__name__)


# ============================================================================
# Configuration Loading
# ============================================================================


@dataclass
class ToleranceConfig:
    """Tolerance ratios for performance validation."""

    e2e: float
    stage: float
    denoise_step: float
    denoise_agg: float


@dataclass
class ScenarioConfig:
    """Expected performance metrics for a test scenario."""

    stages_ms: dict[str, float]
    denoise_step_ms: dict[int, float]
    expected_e2e_ms: float
    expected_avg_denoise_ms: float
    expected_median_denoise_ms: float


@dataclass
class BaselineConfig:
    """Full baseline configuration."""

    scenarios: dict[str, ScenarioConfig]
    step_fractions: Sequence[float]
    warmup_defaults: dict[str, int]
    tolerances: ToleranceConfig

    @classmethod
    def load(cls, path: Path) -> BaselineConfig:
        """Load baseline configuration from JSON file."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        tol_data = data["tolerances"]
        tolerances = ToleranceConfig(
            e2e=float(os.getenv("SGLANG_E2E_TOLERANCE", tol_data["e2e"])),
            stage=float(os.getenv("SGLANG_STAGE_TIME_TOLERANCE", tol_data["stage"])),
            denoise_step=float(
                os.getenv("SGLANG_DENOISE_STEP_TOLERANCE", tol_data["denoise_step"])
            ),
            denoise_agg=float(
                os.getenv("SGLANG_DENOISE_AGG_TOLERANCE", tol_data["denoise_agg"])
            ),
        )

        scenarios = {}
        for name, cfg in data["scenarios"].items():
            scenarios[name] = ScenarioConfig(
                stages_ms=cfg["stages_ms"],
                denoise_step_ms={int(k): v for k, v in cfg["denoise_step_ms"].items()},
                expected_e2e_ms=float(cfg["expected_e2e_ms"]),
                expected_avg_denoise_ms=float(cfg["expected_avg_denoise_ms"]),
                expected_median_denoise_ms=float(cfg["expected_median_denoise_ms"]),
            )

        return cls(
            scenarios=scenarios,
            step_fractions=tuple(data["sampling"]["step_fractions"]),
            warmup_defaults=data["sampling"].get("warmup_requests", {}),
            tolerances=tolerances,
        )


# load global configuration
BASELINE_CONFIG = BaselineConfig.load(Path(__file__).with_name("perf_baselines.json"))


# ============================================================================
# Test Case Definition
# ============================================================================


@dataclass(frozen=True)
class DiffusionCase:
    """Configuration for a single model/scenario test case."""

    id: str  # pytest test id
    model_path: str  # HF repo or local path
    scenario_name: str  # key into BASELINE_CONFIG.scenarios
    modality: str = "image"  # "image" or "video" or "3d"
    prompt: str | None = None  # text prompt for generation
    output_size: str = "1024x1024"  # output image dimensions (or video resolution)
    num_frames: int | None = None  # for video: number of frames
    fps: int | None = None  # for video: frames per second
    warmup_text: int = 1  # number of text-to-image/video warmups
    warmup_edit: int = 0  # number of image/video-edit warmups
    image_edit_prompt: str | None = None  # prompt for editing
    image_edit_path: Path | None = None  # input image/video for editing
    startup_grace_seconds: float = 0.0  # wait time after server starts
    custom_validator: str | None = None  # optional custom validator name
    seconds: int = 4  # for video: duration in seconds


# Common paths
IMAGE_INPUT_FILE = Path(__file__).resolve().parents[1] / "test_files" / "girl.jpg"
VIDEO_INPUT_FILE = Path(__file__).resolve().parents[1] / "test_files" / "i2v_input.jpg"


# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
DIFFUSION_CASES: list[DiffusionCase] = [
    # === Text to Image (T2I) ===
    DiffusionCase(
        id="qwen_image_t2i",
        model_path="Qwen/Qwen-Image",
        scenario_name="text_to_image",
        modality="image",
        prompt="A futuristic cityscape at sunset with flying cars",
        output_size="1024x1024",
        warmup_text=1,
        warmup_edit=0,
        startup_grace_seconds=30.0,
    ),
    DiffusionCase(
        id="flux_image_t2i",
        model_path="black-forest-labs/FLUX.1-dev",
        scenario_name="text_to_image",
        modality="image",
        prompt="A futuristic cityscape at sunset with flying cars",
        output_size="1024x1024",
        warmup_text=1,
        warmup_edit=0,
        startup_grace_seconds=30.0,
    ),
    # === Text and Image to Image (TI2I) ===
    DiffusionCase(
        id="qwen_image_edit_ti2i",
        model_path="Qwen/Qwen-Image-Edit",
        scenario_name="image_edit",
        modality="image",
        prompt=None,  # not used for editing
        output_size="1024x1536",
        warmup_text=0,
        warmup_edit=1,
        image_edit_prompt="Convert 2D style to 3D style",
        image_edit_path=IMAGE_INPUT_FILE,
        startup_grace_seconds=30.0,
    ),
    # === Text to Video (T2V) ===
    DiffusionCase(
        id="fastwan2_1_t2v",
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        scenario_name="text_to_video",
        modality="video",
        prompt="A curious raccoon",
        output_size="848x480",
        seconds=4,
        warmup_text=0,
        warmup_edit=0,
        startup_grace_seconds=30.0,
        custom_validator="video",
    ),
    # # === Image to Video (I2V) ===
    # DiffusionCase(
    #     id="wan2_1_i2v_480p",
    #     model_path="Wan-AI/Wan2.1-I2V-14B-Diffusers",
    #     scenario_name="image_to_video",
    #     modality="video",
    #     prompt="generate", # passing in something since failing if no prompt is passed
    #     warmup_text=0, # warmups only for image gen models
    #     warmup_edit=0,
    #     output_size="1024x1536",
    #     image_edit_prompt="generate",
    #     image_edit_path=VIDEO_INPUT_FILE,
    #     startup_grace_seconds=30.0,
    #     custom_validator="video",
    #     seconds=4,
    # ),
    # === Text and Image to Video (TI2V) ===
    DiffusionCase(
        id="wan2_2_ti2v_5b",
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        scenario_name="text_image_to_video",
        modality="video",
        prompt="Animate this image",
        output_size="832x1104",
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        image_edit_prompt="Add dynamic motion to the scene",
        image_edit_path=VIDEO_INPUT_FILE,
        startup_grace_seconds=30.0,
        custom_validator="video",
        seconds=4,
    ),
]


# ============================================================================
# Server Management
# ============================================================================


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
        log_dir, perf_log_path = prepare_perf_log(Path(__file__))

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


@dataclass
class PerformanceSummary:
    """Summary of performance metrics."""

    e2e_ms: float
    avg_denoise_ms: float
    median_denoise_ms: float
    stage_metrics: dict[str, float]
    sampled_steps: dict[int, float]
    frames_per_second: float | None = None
    total_frames: int | None = None
    avg_frame_time_ms: float | None = None


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


# registry of validators by name
VALIDATOR_REGISTRY = {
    "default": PerformanceValidator,
    "video": VideoPerformanceValidator,
}


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
            warmup.run_edit_warmups(
                count=case.warmup_edit,
                edit_prompt=case.image_edit_prompt,
                image_path=case.image_edit_path,
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
            """TI2I: Text + Image → Image edit."""
            if not case.image_edit_prompt or not case.image_edit_path:
                pytest.skip(f"{case.id}: no edit config")
            if not case.image_edit_path.exists():
                pytest.skip(f"{case.id}: file missing: {case.image_edit_path}")
            with case.image_edit_path.open("rb") as fh:
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
            """T2V: Text → Video."""
            if not case.prompt:
                pytest.skip(f"{case.id}: no text prompt configured")

            _create_and_download_video(
                model=case.model_path,
                prompt=case.prompt,
                size=case.output_size,
                seconds=video_seconds,
            )

        def generate_image_to_video():
            """I2V: Image → Video (optional prompt)."""
            if not case.image_edit_path:
                pytest.skip(f"{case.id}: no input image configured")
            if not case.image_edit_path.exists():
                pytest.skip(f"{case.id}: file missing: {case.image_edit_path}")

            with case.image_edit_path.open("rb") as fh:
                _create_and_download_video(
                    model=case.model_path,
                    prompt=case.image_edit_prompt,
                    size=case.output_size,
                    seconds=video_seconds,
                    input_reference=fh,
                )

        def generate_text_image_to_video():
            """TI2V: Text + Image → Video."""
            if not case.image_edit_prompt or not case.image_edit_path:
                pytest.skip(f"{case.id}: no edit config")
            if not case.image_edit_path.exists():
                pytest.skip(f"{case.id}: file missing: {case.image_edit_path}")

            with case.image_edit_path.open("rb") as fh:
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

        # video-specific metrics: TODO
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
