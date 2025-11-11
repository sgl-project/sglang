# Server-based performance test for SGLang Diffusion:
# - Launches `sglang serve` with an image diffusion model.
# - Issues a small generation via the OpenAI-compatible Images API.
# - Verifies that performance metrics are emitted:
#     * End-to-end time from performance.log
#     * Denoising step average duration from performance.log
#     * Pipeline stage timings from stdout

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import pytest
from openai import InternalServerError, OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import is_jpeg, is_png, wait_for_port

logger = init_logger(__name__)


# Qwen-Image on 1xH100 80GB assertions
EXPECTED_STAGE_MS = {
    "InputValidationStage": 0.10,
    "TextEncodingStage": 834.20,
    "ConditioningStage": 0.10,
    "TimestepPreparationStage": 10.60,
    "LatentPreparationStage": 5.20,
    "DenoisingStage": 21202.60,
    "DecodingStage": 327.60,
}

# Qwen-Image on 1xH100 80GB assertions
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

STAGE_TOLERANCE_RATIO = float(os.environ.get("SGLANG_STAGE_TIME_TOLERANCE", "0.25"))
DENOISE_STEP_TOLERANCE_RATIO = float(
    os.environ.get("SGLANG_DENOISE_STEP_TOLERANCE", "0.10")
)


def _logs_dir() -> Path:
    """
    `PerformanceLogger` writes under python/sglang/logs. Keep a legacy fallback to
    python/sglang/multimodal_gen/logs if that directory already exists locally.
    """
    this_file = Path(__file__).resolve()
    sglang_root = this_file.parents[3] / "logs"
    multimodal_root = this_file.parents[2] / "logs"
    if sglang_root.exists() or not multimodal_root.exists():
        return sglang_root
    return multimodal_root


def _perf_log_path() -> Path:
    return _logs_dir() / "performance.log"


def _clear_perf_log() -> None:
    logs_dir = _logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    perf_log = _perf_log_path()
    try:
        if perf_log.exists():
            perf_log.unlink()
    except Exception:
        pass
    print(f"[server-test] Monitoring perf log at: {perf_log.as_posix()}")


def _read_perf_records() -> List[dict]:
    path = _perf_log_path()
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # ignore malformed lines to avoid flakiness
                pass
    return records


def _wait_for_perf_records(timeout: float = 60.0) -> List[dict]:
    """Poll performance.log until at least one record is observed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = _read_perf_records()
        if records:
            return records
        time.sleep(0.5)
    raise AssertionError(
        f"No performance records found in {_perf_log_path().as_posix()} within {timeout}s"
    )


def _extract_e2e_ms(records: List[dict]) -> float | None:
    for entry in records:
        if entry.get("tag") == "total_inference_time" and "total_duration_ms" in entry:
            try:
                return float(entry["total_duration_ms"])
            except Exception:
                continue
    return None


def _collect_denoise_step_stats(
    records: List[dict],
) -> tuple[List[float], Dict[int, float]]:
    """
    Returns:
        durations: flat list of every denoising step duration (ms).
        avg_per_step: mapping of step index -> average duration across all records.
    """
    durations: List[float] = []
    per_index: Dict[int, List[float]] = {}

    for entry in records:
        for step in entry.get("steps", []) or []:
            if step.get("name") != "denoising_step_guided" or "duration_ms" not in step:
                continue
            try:
                duration_ms = float(step["duration_ms"])
            except Exception:
                continue
            durations.append(duration_ms)
            if step.get("index") is None:
                continue
            idx = int(step["index"])
            per_index.setdefault(idx, []).append(duration_ms)

    avg_per_step: Dict[int, float] = {}
    for idx, values in per_index.items():
        if values:
            avg_per_step[idx] = sum(values) / len(values)

    return durations, avg_per_step


def _extract_avg_denoise_ms(records: List[dict]) -> float | None:
    durations, _ = _collect_denoise_step_stats(records)
    if not durations:
        return None
    return sum(durations) / len(durations)


_STAGE_PATTERN = re.compile(
    r"\[(?P<stage>[^\]]+)\]\s+(?:finished in\s+(?P<sec>[0-9.]+)\s+seconds|"
    r"Execution completed in\s+(?P<ms>[0-9.]+)\s+ms)"
)


def _parse_stage_times(stdout_text: str) -> Dict[str, float]:
    """Return stage timings in milliseconds."""
    timings: Dict[str, float] = {}
    for match in _STAGE_PATTERN.finditer(stdout_text):
        try:
            if match.group("ms") is not None:
                timings[match.group("stage")] = float(match.group("ms"))
            else:
                timings[match.group("stage")] = float(match.group("sec")) * 1000.0
        except Exception:
            pass
    return timings


@pytest.fixture(scope="module")
def server():
    _clear_perf_log()

    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", "30100"))
    model = os.environ.get("SGLANG_TEST_IMAGE_MODEL", "Qwen/Qwen-Image")
    serve_extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))

    stdout_path = Path(tempfile.gettempdir()) / f"sgl_server_{port}.log"
    stdout_path.unlink(missing_ok=True)

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--port",
        str(port),
        "--log-level=debug",
    ]
    if serve_extra_args.strip():
        cmd += serve_extra_args.strip().split()

    env = os.environ.copy()

    stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        cmd,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    pid = process.pid
    print(f"[server-test] Starting server pid={pid}, logging to {stdout_path}")

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
            if wait_for_port("127.0.0.1", port, deadline=1.5, interval=0.3):
                break
        except Exception:
            pass

        try:
            size = stdout_path.stat().st_size
            if size != last_size and (time.time() - start) > 10:
                print(
                    f"[server-test] Waiting for port {port}… "
                    f"log size={size} bytes, elapsed={int(time.time()-start)}s"
                )
                last_size = size
        except Exception:
            pass
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

    try:
        yield {
            "port": port,
            "stdout_file": stdout_path,
            "pid": pid,
            "process": process,
            "fh": stdout_fh,
            "model": model,
        }
    finally:
        try:
            kill_process_tree(pid)
        except Exception:
            pass
        try:
            process.wait(timeout=30)
        except Exception:
            pass
        try:
            stdout_fh.flush()
            stdout_fh.close()
        except Exception:
            pass


def test_server_perf_metrics_image(server):
    client = OpenAI(
        api_key="sglang-anything",
        base_url=f"http://localhost:{server['port']}/v1",
    )
    model_name = server.get("model", "Qwen/Qwen-Image")

    try:
        result = client.images.generate(
            model=model_name,
            prompt="A minimal colorful icon of a raccoon face, flat style",
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
    except InternalServerError:
        tail = ""
        try:
            tail = "\n".join(
                server["stdout_file"]
                .read_text(encoding="utf-8", errors="ignore")
                .splitlines()[-200:]
            )
        except Exception:
            pass
        print("=== SERVER LOG TAIL ===")
        print(tail)
        raise

    image_bytes = base64.b64decode(result.data[0].b64_json)
    assert is_png(image_bytes) or is_jpeg(image_bytes), "Output must be PNG or JPEG"

    try:
        perf_records = _wait_for_perf_records()
    except AssertionError as err:
        pytest.fail(str(err))

    e2e_ms = _extract_e2e_ms(perf_records)
    assert e2e_ms and e2e_ms > 0, "E2E time missing or non-positive"

    denoise_durations, avg_per_step = _collect_denoise_step_stats(perf_records)
    assert denoise_durations, "Denoising step timings not found in performance.log"
    avg_denoise_ms = sum(denoise_durations) / len(denoise_durations)
    assert (
        avg_per_step
    ), "Per-step denoising timings missing; enable step index logging."

    try:
        stdout_text = server["stdout_file"].read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stdout_text = ""
    stage_times = _parse_stage_times(stdout_text)
    assert (
        stage_times
    ), "No PipelineStage timing entries found�enable stage logging in the server run"

    for stage, expected in EXPECTED_STAGE_MS.items():
        actual = stage_times.get(stage)
        assert actual is not None, f"Stage {stage} timing missing from logs"
        upper_bound = expected * (1 + STAGE_TOLERANCE_RATIO)
        assert (
            actual <= upper_bound
        ), f"Stage {stage} took {actual:.2f}ms > allowed {upper_bound:.2f}ms"

    for idx, expected in EXPECTED_DENOISE_STEP_MS.items():
        actual = avg_per_step.get(idx)
        assert actual is not None, f"Denoise step {idx} timing missing"
        upper_bound = expected * (1 + DENOISE_STEP_TOLERANCE_RATIO)
        assert (
            actual <= upper_bound
        ), f"Denoise step {idx} took {actual:.2f}ms > allowed {upper_bound:.2f}ms"

    stage_report = ", ".join(f"{name}:{ms:.2f}ms" for name, ms in stage_times.items())
    per_step_report = ", ".join(
        f"step_{idx}:{ms:.2f}ms" for idx, ms in sorted(avg_per_step.items())
    )
    print(
        f"[Perf] E2E: {e2e_ms:.2f} ms; "
        f"Avg denoise step: {avg_denoise_ms:.2f} ms; "
        f"Stages: {stage_report}; "
        f"Per-step denoise avg: {per_step_report}"
    )

    print(
        f"[Perf] E2E: {e2e_ms:.2f} ms; "
        f"Avg denoise step: {avg_denoise_ms:.2f} ms; "
        f"Stages: {stage_report}; "
        f"Per-step denoise avg: {per_step_report}"
    )
