# Server-based diffusion performance test:
# - Launches an sglang diffusion server via the CLI.
# - Issues an OpenAI-compatible Images API request.
# - Extracts all performance metrics from performance.log (no stdout parsing).
# - Verifies E2E, stage-level, and denoising-step latencies with configurable buffers.

from __future__ import annotations

import base64
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
from sglang.multimodal_gen.test.test_utils import (
    is_jpeg,
    is_png,
    prepare_perf_log,
    read_perf_records,
    sample_step_indices,
    wait_for_perf_record,
    wait_for_port,
    wait_for_stage_metrics,
)

logger = init_logger(__name__)


EXPECTED_STAGE_MS = {
    "InputValidationStage": 0.10,
    "TextEncodingStage": 834.20,
    "ConditioningStage": 0.10,
    "TimestepPreparationStage": 10.60,
    "LatentPreparationStage": 5.20,
    "DenoisingStage": 21202.60,
    "DecodingStage": 327.60,
}

EXPECTED_STAGE_MS_IMAGE_EDIT = {
    "InputValidationStage": 9.5,
    "TextEncodingStage": 1084.46,
    "ConditioningStage": 0.13,
    "TimestepPreparationStage": 13.78,
    "LatentPreparationStage": 6.76,
    "DenoisingStage": 27563.38,
    "DecodingStage": 425.88,
}

EXPECTED_DENOISE_STEP_MS = {
    0: 1077.77,
    1: 345.13,
    2: 413.80,
    3: 405.49,
    4: 408.14,
    5: 409.06,
    6: 408.85,
    7: 410.53,
    8: 407.51,
    9: 409.44,
    10: 408.65,
    11: 410.14,
    12: 411.74,
    13: 409.59,
    14: 409.17,
    15: 410.78,
    16: 410.66,
    17: 410.58,
    18: 411.27,
    19: 410.51,
    20: 409.03,
    21: 410.16,
    22: 409.42,
    23: 411.03,
    24: 410.18,
    25: 409.72,
    26: 410.26,
    27: 410.21,
    28: 410.71,
    29: 410.76,
    30: 411.06,
    31: 410.10,
    32: 410.55,
    33: 410.77,
    34: 410.74,
    35: 411.75,
    36: 410.78,
    37: 411.56,
    38: 410.85,
    39: 411.08,
    40: 411.12,
    41: 411.10,
    42: 411.09,
    43: 410.87,
    44: 411.37,
    45: 411.68,
    46: 411.00,
    47: 410.09,
    48: 412.72,
    49: 410.42,
}

DEFAULT_EXPECTED_E2E_MS = 22383.71
DEFAULT_EXPECTED_AVG_DENOISE_MS = 422.42
DEFAULT_EXPECTED_MEDIAN_DENOISE_MS = 410.62

STEP_SAMPLE_FRACTIONS: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

E2E_TOLERANCE_RATIO = float(os.environ.get("SGLANG_E2E_TOLERANCE", "0.25"))
STAGE_TOLERANCE_RATIO = float(os.environ.get("SGLANG_STAGE_TIME_TOLERANCE", "0.30"))
DENOISE_STEP_TOLERANCE_RATIO = float(
    os.environ.get("SGLANG_DENOISE_STEP_TOLERANCE", "0.10")
)
DENOISE_AGG_TOLERANCE_RATIO = float(
    os.environ.get("SGLANG_DENOISE_AGG_TOLERANCE", "0.10")
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
            model=getattr(cls, "MODEL_PATH", "Qwen/Qwen-Image"),
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
                    model=getattr(cls, "MODEL_PATH", "Qwen/Qwen-Image"),
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

    port = getattr(
        cls,
        "SERVER_PORT",
        int(os.environ.get("SGLANG_TEST_SERVER_PORT", "30100")),
    )
    model = getattr(
        cls, "MODEL_PATH", os.environ.get("SGLANG_TEST_IMAGE_MODEL", "Qwen/Qwen-Image")
    )
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))
    serve_extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")

    stdout_path = Path(tempfile.gettempdir()) / f"sgl_server_{port}.log"
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
    last_size = 0
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
        try:
            if wait_for_port(host="127.0.0.1", port=port, deadline=1.5, interval=0.3):
                break
        except Exception:
            pass
        if time.time() - start > 10:
            size = stdout_path.stat().st_size
            if size != last_size:
                logger.info(
                    "[server-test] Waiting for port %s... log size=%s bytes, elapsed=%ss",
                    port,
                    size,
                    int(time.time() - start),
                )
                last_size = size
    else:
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
            f"Port 127.0.0.1:{port} not ready within {wait_deadline}s.\nLast logs:\n{tail}"
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
class DiffusionServerPerfTestBase:
    MODEL_PATH = os.environ.get("SGLANG_TEST_IMAGE_MODEL", "Qwen/Qwen-Image")
    SERVER_PORT = int(os.environ.get("SGLANG_TEST_SERVER_PORT", "30100"))
    PROMPT = "A minimal colorful icon of a raccoon face, flat style"
    IMAGE_EDIT_PROMPT: str | None = None
    IMAGE_EDIT_PATH = Path(__file__).resolve().parents[1] / "test_files" / "rabbit.jpg"
    OUTPUT_SIZE = "1024x1024"
    WARMUP_TEXT_REQUESTS = 1
    WARMUP_IMAGE_EDIT_REQUESTS = 0
    STAGE_EXPECTATIONS = EXPECTED_STAGE_MS
    STEP_EXPECTATIONS = EXPECTED_DENOISE_STEP_MS
    EXPECTED_E2E_MS = DEFAULT_EXPECTED_E2E_MS
    EXPECTED_AVG_DENOISE_MS = DEFAULT_EXPECTED_AVG_DENOISE_MS
    EXPECTED_MEDIAN_DENOISE_MS = DEFAULT_EXPECTED_MEDIAN_DENOISE_MS
    _perf_results: list[dict[str, Any]] = []

    @classmethod
    def setup_class(cls):
        cls._perf_results = []

    @classmethod
    def teardown_class(cls):
        results = getattr(cls, "_perf_results", [])
        if not results:
            return
        lines = [
            "",
            f"[server-test] Perf summary for {cls.__name__}",
            "Test Name             |   E2E (ms) | Avg Denoise (ms) | Median (ms)",
            "----------------------+-----------+------------------+-------------",
        ]
        for entry in results:
            lines.append(
                f"{entry['test_name']:<22} | "
                f"{entry['e2e_ms']:>9.2f} | "
                f"{entry['avg_denoise_ms']:>16.2f} | "
                f"{entry['median_denoise_ms']:>11.2f}"
            )
            stage_report = ", ".join(
                f"{name}:{duration:.2f}ms"
                for name, duration in entry.get("stage_metrics", {}).items()
            )
            if stage_report:
                lines.append(f"    stages: {stage_report}")
            sampled_steps = entry.get("sampled_steps") or {}
            if sampled_steps:
                step_report = ", ".join(
                    f"{idx}:{duration:.2f}ms" for idx, duration in sampled_steps.items()
                )
                lines.append(f"    sampled steps: {step_report}")
        lines.append("")
        summary_text = "\n".join(lines)
        logger.info(summary_text)
        print(summary_text)

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

        stage_report = ", ".join(f"{k}:{v:.2f}ms" for k, v in stage_metrics.items())
        per_step_report = ", ".join(
            f"step_{idx}:{avg_per_step[idx]:.2f}ms" for idx in sample_indices
        )
        logger.info(
            "[Perf] E2E %.2f ms; Avg denoise %.2f ms; Median %.2f ms; "
            "Stages: %s; Sampled steps: %s",
            e2e_ms,
            avg_duration,
            median_duration,
            stage_report,
            per_step_report,
        )

        return {
            "e2e_ms": e2e_ms,
            "avg_denoise_ms": avg_duration,
            "median_denoise_ms": median_duration,
            "stage_metrics": stage_metrics,
            "sampled_steps": sampled_steps,
        }

    def test_text_to_image_performance(self):
        perf_record, stage_metrics = self._run_and_collect_records(self._generate_image)
        summary = self._assert_metrics(perf_record, stage_metrics)
        self._record_result("text_to_image", summary)

    def test_image_edit_performance(self):
        if not self.IMAGE_EDIT_PROMPT:
            pytest.skip("Image edit prompt not configured")
        perf_record, stage_metrics = self._run_and_collect_records(
            self._generate_image_edit
        )
        summary = self._assert_metrics(perf_record, stage_metrics)
        self._record_result("image_edit", summary)


class TestQwenImagePerformance(DiffusionServerPerfTestBase):
    IMAGE_EDIT_PROMPT = "Change the rabbit's color to purple."

    def test_image_edit_performance(self):
        if not self.IMAGE_EDIT_PROMPT:
            pytest.skip("Image edit prompt not configured")
        original_stage_expectations = self.STAGE_EXPECTATIONS
        try:
            self.STAGE_EXPECTATIONS = EXPECTED_STAGE_MS_IMAGE_EDIT
            perf_record, stage_metrics = self._run_and_collect_records(
                self._generate_image_edit
            )
            summary = self._assert_metrics(perf_record, stage_metrics)
            self._record_result("image_edit", summary)
        finally:
            self.STAGE_EXPECTATIONS = original_stage_expectations
