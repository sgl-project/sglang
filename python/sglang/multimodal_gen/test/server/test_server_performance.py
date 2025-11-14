# Server-based diffusion performance test:
# - Launches an sglang diffusion server via the CLI.
# - Issues an OpenAI-compatible Images API request.
# - Extracts all performance metrics from performance.log (no stdout parsing).
# - Verifies E2E, stage-level, and denoising-step latencies with configurable buffers.

from __future__ import annotations

import base64
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import pytest
from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.conftest import _GLOBAL_PERF_RESULTS
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    is_jpeg,
    is_png,
    prepare_perf_log,
    read_perf_records,
    sample_step_indices,
    wait_for_perf_record,
    wait_for_stage_metrics,
)

logger = init_logger(__name__)


_BASELINE_PATH = Path(__file__).with_name("perf_baselines.json")
with _BASELINE_PATH.open("r", encoding="utf-8") as _fh:
    _BASELINE_CONFIG = json.load(_fh)

_SCENARIOS = _BASELINE_CONFIG["scenarios"]
_TEXT_SCENARIO = _SCENARIOS["text_to_image"]
_IMAGE_EDIT_SCENARIO = _SCENARIOS["image_edit"]

STEP_SAMPLE_FRACTIONS: Sequence[float] = tuple(
    _BASELINE_CONFIG["sampling"]["step_fractions"]
)

_WARMUP_DEFAULTS = _BASELINE_CONFIG["sampling"].get("warmup_requests", {})
_DEFAULT_WARMUP_TEXT = int(_WARMUP_DEFAULTS.get("text", 1))
_DEFAULT_WARMUP_EDIT = int(_WARMUP_DEFAULTS.get("image_edit", 0))

_TOLERANCES = _BASELINE_CONFIG["tolerances"]


def _tolerance_from_env(var_name: str, default: float) -> float:
    override = os.environ.get(var_name)
    if override is not None:
        return float(override)
    return float(default)


E2E_TOLERANCE_RATIO = _tolerance_from_env("SGLANG_E2E_TOLERANCE", _TOLERANCES["e2e"])
STAGE_TOLERANCE_RATIO = _tolerance_from_env(
    "SGLANG_STAGE_TIME_TOLERANCE", _TOLERANCES["stage"]
)
DENOISE_STEP_TOLERANCE_RATIO = _tolerance_from_env(
    "SGLANG_DENOISE_STEP_TOLERANCE", _TOLERANCES["denoise_step"]
)
DENOISE_AGG_TOLERANCE_RATIO = _tolerance_from_env(
    "SGLANG_DENOISE_AGG_TOLERANCE", _TOLERANCES["denoise_agg"]
)


def _decode_and_validate_image(b64_json: str) -> None:
    image_bytes = base64.b64decode(b64_json)
    assert is_png(image_bytes) or is_jpeg(
        image_bytes
    ), "Warm-up image must be PNG or JPEG"


def _run_warmup_requests(cls, port: int) -> None:
    warmup_text_requests = int(getattr(cls, "WARMUP_TEXT_REQUESTS", 1))
    warmup_edit_requests = int(getattr(cls, "WARMUP_IMAGE_EDIT_REQUESTS", 0))
    if warmup_text_requests <= 0 and warmup_edit_requests <= 0:
        return

    client = OpenAI(
        api_key="sglang-anything",
        base_url=f"http://localhost:{port}/v1",
    )
    prompt = getattr(cls, "PROMPT", "A colorful raccoon icon")
    output_size = getattr(cls, "OUTPUT_SIZE", "1024x1024")

    logger.info(
        "[server-test] Running %s text warm-up(s) and %s edit warm-up(s)",
        warmup_text_requests,
        warmup_edit_requests,
    )

    for _ in range(warmup_text_requests):
        result = client.images.generate(
            model=getattr(cls, "MODEL_PATH"),
            prompt=prompt,
            n=1,
            size=output_size,
            response_format="b64_json",
        )
        _decode_and_validate_image(result.data[0].b64_json)

    if warmup_edit_requests > 0:
        edit_prompt = getattr(cls, "IMAGE_EDIT_PROMPT", None)
        edit_path: Path | None = getattr(cls, "IMAGE_EDIT_PATH", None)
        if not edit_prompt or not edit_path or not edit_path.exists():
            logger.warning(
                "[server-test] Skipping image-edit warm-up: prompt=%s path=%s exists=%s",
                bool(edit_prompt),
                edit_path,
                edit_path.exists() if edit_path else False,
            )
            return
        for _ in range(warmup_edit_requests):
            with edit_path.open("rb") as fh:
                result = client.images.edit(
                    model=getattr(cls, "MODEL_PATH"),
                    image=fh,
                    prompt=edit_prompt,
                    n=1,
                    size=output_size,
                    response_format="b64_json",
                )
            _decode_and_validate_image(result.data[0].b64_json)


@pytest.fixture(scope="class")
def diffusion_server(request):
    cls = request.cls

    log_dir, perf_log_path = prepare_perf_log(Path(__file__))

    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))
    port = getattr(cls, "SERVER_PORT", port)

    model = getattr(cls, "MODEL_PATH")
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))
    serve_extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")

    safe_model_name = model.replace("/", "_")
    stdout_path = (
        Path(tempfile.gettempdir()) / f"sgl_server_{port}_{safe_model_name}.log"
    )
    stdout_path.unlink(missing_ok=True)

    base_command = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--port",
        str(port),
        "--log-level=debug",
    ]
    if serve_extra_args.strip():
        base_command += serve_extra_args.strip().split()

    env = os.environ.copy()
    env["SGL_DIFFUSION_STAGE_LOGGING"] = "1"
    env["SGLANG_PERF_LOG_DIR"] = log_dir.as_posix()

    stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        base_command,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    logger.info(
        "[server-test] Starting diffusion server pid=%s, model=%s, log=%s",
        process.pid,
        model,
        stdout_path.as_posix(),
    )

    start = time.time()
    server_ready_message = "Application startup complete."
    server_ready = False

    while time.time() - start < wait_deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(
                    stdout_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()[-200:]
                )
            except Exception:
                pass
            raise RuntimeError(
                f"Server exited early (code {process.returncode}). Last logs:\n{tail}"
            )

        if stdout_path.exists():
            try:
                log_content = stdout_path.read_text(encoding="utf-8", errors="ignore")
                if server_ready_message in log_content:
                    logger.info("[server-test] Server is fully loaded and ready.")
                    server_ready = True
                    break
            except Exception as e:
                logger.debug("Could not read server log file yet: %s", e)

        logger.info(
            "[server-test] Waiting for server to initialize... elapsed=%ss",
            int(time.time() - start),
        )
        time.sleep(5)

    if not server_ready:
        tail = ""
        try:
            tail = "\n".join(
                stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines()[
                    -200:
                ]
            )
        except Exception:
            pass
        raise TimeoutError(
            f"Server did not become ready within {wait_deadline}s. Last logs:\n{tail}"
        )

    ctx = {
        "port": port,
        "stdout_file": stdout_path,
        "process": process,
        "model": model,
        "fh": stdout_fh,
        "perf_log_path": perf_log_path,
        "log_dir": log_dir,
    }
    request.cls.server_ctx = ctx
    request.cls.perf_log_path = perf_log_path

    grace = float(getattr(cls, "STARTUP_GRACE_SECONDS", 0.0) or 0.0)
    if grace > 0:
        logger.info(
            "[server-test] Waiting %.1fs before warm-ups to let model settle", grace
        )
        time.sleep(grace)

    try:
        _run_warmup_requests(cls, port)
    except Exception as exc:
        logger.error("Warm-up requests failed: %s", exc)
        kill_process_tree(process.pid)
        raise

    yield ctx

    try:
        kill_process_tree(process.pid)
    except Exception:
        pass
    try:
        stdout_fh.flush()
        stdout_fh.close()
    except Exception:
        pass


@pytest.mark.usefixtures("diffusion_server")
class DiffusionPerfTestBase:
    MODEL_PATH: str
    # SERVER_PORT = int(os.environ.get("SGLANG_TEST_SERVER_PORT", "30100"))
    PROMPT = "A Logo With Bold Large Text: SGL Diffusion"
    IMAGE_EDIT_PROMPT: str | None = None
    IMAGE_EDIT_PATH = Path(__file__).resolve().parents[1] / "test_files" / "girl.jpg"
    OUTPUT_SIZE = "1024x1024"
    WARMUP_TEXT_REQUESTS = _DEFAULT_WARMUP_TEXT
    WARMUP_IMAGE_EDIT_REQUESTS = _DEFAULT_WARMUP_EDIT
    STARTUP_GRACE_SECONDS = 0.0

    STAGE_EXPECTATIONS: dict
    STEP_EXPECTATIONS: dict
    EXPECTED_E2E_MS: float
    EXPECTED_AVG_DENOISE_MS: float
    EXPECTED_MEDIAN_DENOISE_MS: float

    _perf_results: list[dict[str, Any]] = []

    @classmethod
    def setup_class(cls):
        cls._perf_results = []

    @classmethod
    def teardown_class(cls):
        for result in cls._perf_results:
            result["class_name"] = cls.__name__
            _GLOBAL_PERF_RESULTS.append(result)

    def _client(self) -> OpenAI:
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{self.server_ctx['port']}/v1",
        )

    def _perf_log_path(self) -> Path:
        return self.server_ctx["perf_log_path"]

    def _record_result(self, test_name: str, summary: dict[str, Any]) -> None:
        if not summary:
            return
        entry = {"test_name": test_name, **summary}
        self.__class__._perf_results.append(entry)

    def _run_and_collect_records(self, generate_fn) -> tuple[dict, dict]:
        log_path = self._perf_log_path()
        prev_len = len(read_perf_records(log_path))
        generate_fn()
        perf_record, _ = wait_for_perf_record(
            "total_inference_time",
            prev_len,
            log_path,
        )
        stage_metrics, _ = wait_for_stage_metrics(
            perf_record.get("request_id", ""),
            prev_len,
            len(self.STAGE_EXPECTATIONS),
            log_path,
        )
        return perf_record, stage_metrics

    def _generate_image(self):
        client = self._client()
        result = client.images.generate(
            model=self.MODEL_PATH,
            prompt=self.PROMPT,
            n=1,
            size=self.OUTPUT_SIZE,
            response_format="b64_json",
        )
        image_bytes = base64.b64decode(result.data[0].b64_json)
        assert is_png(image_bytes) or is_jpeg(
            image_bytes
        ), "Generated image must be PNG or JPEG"

    def _generate_image_edit(self):
        if not self.IMAGE_EDIT_PROMPT:
            pytest.skip("Image edit prompt not configured")
        if not self.IMAGE_EDIT_PATH.exists():
            pytest.skip(f"Image edit file missing: {self.IMAGE_EDIT_PATH}")
        client = self._client()
        with self.IMAGE_EDIT_PATH.open("rb") as fh:
            result = client.images.edit(
                model=self.MODEL_PATH,
                image=fh,
                prompt=self.IMAGE_EDIT_PROMPT,
                n=1,
                size=self.OUTPUT_SIZE,
                response_format="b64_json",
            )
        image_bytes = base64.b64decode(result.data[0].b64_json)
        assert is_png(image_bytes) or is_jpeg(
            image_bytes
        ), "Edited image must be PNG or JPEG"

    def _assert_metrics(self, perf_record: dict, stage_metrics: dict):
        e2e_ms = float(perf_record.get("total_duration_ms", 0.0))
        assert e2e_ms > 0, "E2E duration missing from perf log"
        e2e_upper = self.EXPECTED_E2E_MS * (1 + E2E_TOLERANCE_RATIO)
        assert (
            e2e_ms <= e2e_upper
        ), f"E2E time {e2e_ms:.2f}ms exceeds allowed {e2e_upper:.2f}ms"

        steps = [
            step
            for step in perf_record.get("steps", []) or []
            if step.get("name") == "denoising_step_guided" and "duration_ms" in step
        ]
        assert steps, "Denoising step timings missing from perf log"

        durations = [float(step["duration_ms"]) for step in steps]
        avg_duration = sum(durations) / len(durations)
        median_duration = statistics.median(durations)

        avg_upper = self.EXPECTED_AVG_DENOISE_MS * (1 + DENOISE_AGG_TOLERANCE_RATIO)
        med_upper = self.EXPECTED_MEDIAN_DENOISE_MS * (1 + DENOISE_AGG_TOLERANCE_RATIO)
        assert (
            avg_duration <= avg_upper
        ), f"Avg denoise {avg_duration:.2f}ms exceeds {avg_upper:.2f}ms"
        assert (
            median_duration <= med_upper
        ), f"Median denoise {median_duration:.2f}ms exceeds {med_upper:.2f}ms"

        avg_per_step = {
            int(step.get("index")): float(step["duration_ms"])
            for step in steps
            if step.get("index") is not None
        }
        sample_indices = sample_step_indices(avg_per_step, STEP_SAMPLE_FRACTIONS)
        sampled_steps = {idx: avg_per_step[idx] for idx in sample_indices}
        for idx in sample_indices:
            expected = self.STEP_EXPECTATIONS.get(idx)
            if expected is None:
                continue
            actual = avg_per_step[idx]
            upper_bound = expected * (1 + DENOISE_STEP_TOLERANCE_RATIO)
            assert (
                actual <= upper_bound
            ), f"Denoise step {idx} took {actual:.2f}ms > allowed {upper_bound:.2f}ms"

        assert stage_metrics, "Stage metrics missing from performance log"
        for stage, expected in self.STAGE_EXPECTATIONS.items():
            actual = stage_metrics.get(stage)
            assert actual is not None, f"Stage {stage} timing missing"
            upper_bound = expected * (1 + STAGE_TOLERANCE_RATIO)
            assert (
                actual <= upper_bound
            ), f"Stage {stage} took {actual:.2f}ms > allowed {upper_bound:.2f}ms"

        # Log to pytest console during the run for immediate feedback
        logger.info(
            "[Perf] %s/%s: E2E %.2f ms; Avg denoise %.2f ms; Median %.2f ms",
            self.__class__.__name__,
            perf_record.get("test_name", "test"),
            e2e_ms,
            avg_duration,
            median_duration,
        )

        return {
            "e2e_ms": e2e_ms,
            "avg_denoise_ms": avg_duration,
            "median_denoise_ms": median_duration,
            "stage_metrics": stage_metrics,
            "sampled_steps": sampled_steps,
        }


class TestQwenImageGeneration(DiffusionPerfTestBase):
    """Performance tests for the Qwen/Qwen-image model."""

    MODEL_PATH = "Qwen/Qwen-Image"
    STARTUP_GRACE_SECONDS = 30.0
    WARMUP_IMAGE_EDIT_REQUESTS = 0
    STAGE_EXPECTATIONS = _TEXT_SCENARIO["stages_ms"]
    STEP_EXPECTATIONS = {
        int(k): v for k, v in _TEXT_SCENARIO["denoise_step_ms"].items()
    }
    EXPECTED_E2E_MS = float(_TEXT_SCENARIO["expected_e2e_ms"])
    EXPECTED_AVG_DENOISE_MS = float(_TEXT_SCENARIO["expected_avg_denoise_ms"])
    EXPECTED_MEDIAN_DENOISE_MS = float(_TEXT_SCENARIO["expected_median_denoise_ms"])

    def test_text_to_image_performance(self):
        perf_record, stage_metrics = self._run_and_collect_records(self._generate_image)
        summary = self._assert_metrics(perf_record, stage_metrics)
        self._record_result("text_to_image", summary)


class TestQwenImageEdit(DiffusionPerfTestBase):
    """Performance tests for the Qwen/Qwen-Image-Edit model."""

    MODEL_PATH = "Qwen/Qwen-Image-Edit"
    IMAGE_EDIT_PROMPT = "Convert 2D style to 3D style"
    OUTPUT_SIZE = "1024x1536"
    STARTUP_GRACE_SECONDS = 30.0
    WARMUP_TEXT_REQUESTS = 0
    WARMUP_IMAGE_EDIT_REQUESTS = 1
    STAGE_EXPECTATIONS = _IMAGE_EDIT_SCENARIO["stages_ms"]
    STEP_EXPECTATIONS = {
        int(k): v for k, v in _IMAGE_EDIT_SCENARIO["denoise_step_ms"].items()
    }
    EXPECTED_E2E_MS = float(_IMAGE_EDIT_SCENARIO["expected_e2e_ms"])
    EXPECTED_AVG_DENOISE_MS = float(_IMAGE_EDIT_SCENARIO["expected_avg_denoise_ms"])
    EXPECTED_MEDIAN_DENOISE_MS = float(
        _IMAGE_EDIT_SCENARIO["expected_median_denoise_ms"]
    )

    def test_image_edit_performance(self):
        perf_record, stage_metrics = self._run_and_collect_records(
            self._generate_image_edit
        )
        summary = self._assert_metrics(perf_record, stage_metrics)
        self._record_result("image_edit", summary)
