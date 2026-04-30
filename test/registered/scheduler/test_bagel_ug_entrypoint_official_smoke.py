# SPDX-License-Identifier: Apache-2.0
"""Opt-in true-weight BAGEL UG entrypoint smoke.

This test intentionally does not compare pixels with the official BAGEL runner.
That parity is covered by ``test_bagel_g_official_parity.py``.  This smoke proves
the explicit UG modes are reachable through the experimental Python API, CLI,
and HTTP entrypoints.

Usage:
CUDA_VISIBLE_DEVICES=2 \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
SGLANG_TEST_BAGEL_OFFICIAL_REPO=/data/BAGEL \
SGLANG_TEST_BAGEL_UG_ENTRYPOINT_OUTPUT=/tmp/ug-entrypoint-smoke \
python3 test/registered/scheduler/test_bagel_ug_entrypoint_official_smoke.py -v
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=1200,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL UG entrypoint true-weight smoke; requires "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL and a BAGEL input image"
    ),
)

_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
_OFFICIAL_REPO_ENV = "SGLANG_TEST_BAGEL_OFFICIAL_REPO"
_IMAGE_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_IMAGE"
_OUTPUT_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_OUTPUT"
_MODES_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_MODES"
_CASES_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_CASES"
_PROMPT_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_PROMPT"
_STEPS_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_NUM_STEPS"
_MEM_FRACTION_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION"
_CHUNKED_PREFILL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL"
_U_DECODE_TOKENS_ENV = "SGLANG_TEST_BAGEL_UG_U_DECODE_MAX_NEW_TOKENS"
_ATTENTION_BACKEND_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_ATTENTION_BACKEND"
_TIMEOUT_ENV = "SGLANG_TEST_BAGEL_UG_ENTRYPOINT_TIMEOUT_SEC"
_THINK_TOKENS_ENV = "SGLANG_TEST_BAGEL_UG_THINK_MAX_NEW_TOKENS"


def _has_live_env() -> bool:
    return bool(os.getenv(_MODEL_ENV))


@unittest.skipUnless(
    _has_live_env(),
    f"Set {_MODEL_ENV} for BAGEL UG entrypoint true-weight smoke",
)
class TestBAGELUGEntrypointOfficialSmoke(CustomTestCase):
    def test_selected_entrypoints_run_explicit_ug_modes(self):
        checkpoint_dir = Path(os.environ[_MODEL_ENV]).expanduser()
        self.assertTrue(checkpoint_dir.exists(), checkpoint_dir)

        image_path = _resolve_image_path()
        self.assertTrue(image_path.exists(), image_path)

        output_dir = Path(
            os.getenv(_OUTPUT_ENV) or tempfile.mkdtemp(prefix="ug-entrypoint-smoke-")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        modes = _selected_modes()
        cases = _selected_cases()
        for case in cases:
            for mode in modes:
                with self.subTest(case=case, mode=mode):
                    payload = _build_payload(
                        case,
                        image_path,
                        include_think=mode != "cli",
                    )
                    payload_path = output_dir / f"{case}.{mode}.payload.json"
                    payload_path.write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    result_path = output_dir / f"{case}.{mode}.json"
                    if mode == "python":
                        _run_python_api_smoke(
                            checkpoint_dir=checkpoint_dir,
                            payload_path=payload_path,
                            result_path=result_path,
                            output_dir=output_dir,
                            case=case,
                        )
                    elif mode == "cli":
                        _run_cli_smoke(
                            checkpoint_dir=checkpoint_dir,
                            payload_path=payload_path,
                            result_path=result_path,
                            output_dir=output_dir,
                            case=case,
                        )
                    elif mode == "http":
                        _run_http_smoke(
                            checkpoint_dir=checkpoint_dir,
                            payload_path=payload_path,
                            result_path=result_path,
                            output_dir=output_dir,
                            case=case,
                        )
                    else:
                        raise AssertionError(f"Unsupported mode: {mode}")

                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    _assert_entrypoint_result(
                        self,
                        result,
                        case=case,
                        expected_velocity_count=_expected_velocity_count(payload),
                    )


def _resolve_image_path() -> Path:
    if image := os.getenv(_IMAGE_ENV):
        return Path(image).expanduser()
    official_repo = os.getenv(_OFFICIAL_REPO_ENV)
    if not official_repo:
        raise AssertionError(
            f"Set {_IMAGE_ENV}, or set {_OFFICIAL_REPO_ENV} so the default "
            "test_images/women.jpg can be used"
        )
    return Path(official_repo).expanduser() / "test_images" / "women.jpg"


def _build_payload(
    case: str,
    image_path: Path,
    *,
    include_think: bool = True,
) -> dict[str, Any]:
    prompt = os.getenv(
        _PROMPT_ENV,
        "Turn the scene into a warm cinematic portrait.",
    )
    mode = _generation_mode_for_case(case)
    messages: list[dict[str, Any]]
    if mode in {"edit", "interleave", "vlm"}:
        messages = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]
    else:
        messages = [{"type": "text", "text": prompt}]

    sampling_params = {
        "height": int(os.getenv("SGLANG_TEST_BAGEL_UG_ENTRYPOINT_HEIGHT", "512")),
        "width": int(os.getenv("SGLANG_TEST_BAGEL_UG_ENTRYPOINT_WIDTH", "512")),
        "seed": 123,
        "num_inference_steps": int(os.getenv(_STEPS_ENV, "4")),
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.5,
        "cfg_interval": [0.4, 1.0],
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
        "timestep_shift": 3.0,
        "suppress_logs": True,
    }
    if mode in {"edit", "interleave"}:
        # BAGEL edit/interleave should use the resized source-image shape,
        # matching the official parity harness.
        sampling_params["height"] = None
        sampling_params["width"] = None

    payload: dict[str, Any] = {
        "messages": messages,
        "mode": mode,
        "metadata": {"case": case},
        "sampling_params": sampling_params,
    }
    if mode == "vlm":
        payload["max_new_tokens"] = int(os.getenv(_U_DECODE_TOKENS_ENV, "4"))
    if case == "think_t2i" and include_think:
        payload["think"] = True
        payload["think_max_new_tokens"] = _think_max_new_tokens()
    return payload


def _generation_mode_for_case(case: str) -> str:
    if case == "think_t2i":
        return "t2i"
    return case


def _selected_modes() -> tuple[str, ...]:
    raw_modes = os.getenv(_MODES_ENV, "python,cli,http")
    modes = tuple(mode.strip() for mode in raw_modes.split(",") if mode.strip())
    if not modes:
        raise ValueError(f"{_MODES_ENV} must select at least one mode")
    allowed = {"python", "cli", "http"}
    unsupported = [mode for mode in modes if mode not in allowed]
    if unsupported:
        raise ValueError(f"{_MODES_ENV} has unsupported modes: {unsupported}")
    return modes


def _selected_cases() -> tuple[str, ...]:
    raw_cases = os.getenv(_CASES_ENV, "interleave")
    cases = tuple(case.strip() for case in raw_cases.split(",") if case.strip())
    if not cases:
        raise ValueError(f"{_CASES_ENV} must select at least one case")
    allowed = {"t2i", "edit", "interleave", "vlm", "think_t2i"}
    unsupported = [case for case in cases if case not in allowed]
    if unsupported:
        raise ValueError(f"{_CASES_ENV} has unsupported cases: {unsupported}")
    return cases


def _think_max_new_tokens() -> int:
    return int(os.getenv(_THINK_TOKENS_ENV, "2"))


def _expected_velocity_count(payload: dict[str, Any]) -> int:
    if payload.get("mode") == "vlm":
        return 0
    return max(0, int(payload["sampling_params"]["num_inference_steps"]) - 1)


def _runtime_kwargs() -> dict[str, Any]:
    return {
        "mem_fraction": float(os.getenv(_MEM_FRACTION_ENV, "0.35")),
        "chunked_prefill": int(os.getenv(_CHUNKED_PREFILL_ENV, "256")),
        "u_decode_max_new_tokens": int(os.getenv(_U_DECODE_TOKENS_ENV, "4")),
        "attention_backend": os.getenv(_ATTENTION_BACKEND_ENV) or "",
    }


def _run_python_api_smoke(
    *,
    checkpoint_dir: Path,
    payload_path: Path,
    result_path: Path,
    output_dir: Path,
    case: str,
) -> None:
    scheduler_port, master_port = _pick_ports(2)
    env = _subprocess_env(
        {
            "UG_ENTRYPOINT_MODEL": str(checkpoint_dir),
            "UG_ENTRYPOINT_PAYLOAD": str(payload_path),
            "UG_ENTRYPOINT_RESULT": str(result_path),
            "UG_ENTRYPOINT_CASE": case,
            "UG_ENTRYPOINT_SCHEDULER_PORT": str(scheduler_port),
            "UG_ENTRYPOINT_MASTER_PORT": str(master_port),
            **{
                f"UG_ENTRYPOINT_{key.upper()}": str(value)
                for key, value in _runtime_kwargs().items()
            },
        }
    )
    code = textwrap.dedent("""
        import json
        import os
        from pathlib import Path

        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        payload = json.loads(Path(os.environ["UG_ENTRYPOINT_PAYLOAD"]).read_text())
        generator = DiffGenerator.from_pretrained(
            model_path=os.environ["UG_ENTRYPOINT_MODEL"],
            pipeline_class_name="UGPipeline",
            num_gpus=1,
            enable_cfg_parallel=False,
            log_level="error",
            ug_srt_log_level="error",
            ug_srt_mem_fraction_static=float(
                os.environ["UG_ENTRYPOINT_MEM_FRACTION"]
            ),
            ug_srt_chunked_prefill_size=int(
                os.environ["UG_ENTRYPOINT_CHUNKED_PREFILL"]
            ),
            ug_srt_u_decode_max_new_tokens=int(
                os.environ["UG_ENTRYPOINT_U_DECODE_MAX_NEW_TOKENS"]
            ),
            ug_srt_attention_backend=os.environ.get(
                "UG_ENTRYPOINT_ATTENTION_BACKEND"
            )
            or None,
            scheduler_port=int(os.environ["UG_ENTRYPOINT_SCHEDULER_PORT"]),
            master_port=int(os.environ["UG_ENTRYPOINT_MASTER_PORT"]),
            local_mode=True,
        )
        try:
            if os.environ["UG_ENTRYPOINT_CASE"] == "vlm":
                result = generator.generate_vlm_serializable(payload)
            else:
                result = generator.generate_interleave_serializable(payload)
            Path(os.environ["UG_ENTRYPOINT_RESULT"]).write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        finally:
            generator.shutdown()
        """)
    _run_subprocess(
        [sys.executable, "-c", code],
        env=env,
        log_path=output_dir / "python.log",
    )


def _run_cli_smoke(
    *,
    checkpoint_dir: Path,
    payload_path: Path,
    result_path: Path,
    output_dir: Path,
    case: str,
) -> None:
    runtime_kwargs = _runtime_kwargs()
    scheduler_port, master_port = _pick_ports(2)
    cmd = [
        sys.executable,
        "-m",
        "sglang.cli.main",
        "generate",
        "--model-path",
        str(checkpoint_dir),
        "--pipeline-class-name",
        "UGPipeline",
        "--num-gpus",
        "1",
        "--log-level",
        "error",
        "--scheduler-port",
        str(scheduler_port),
        "--master-port",
        str(master_port),
        "--ug-srt-log-level",
        "error",
        "--ug-srt-mem-fraction-static",
        str(runtime_kwargs["mem_fraction"]),
        "--ug-srt-chunked-prefill-size",
        str(runtime_kwargs["chunked_prefill"]),
        "--ug-srt-u-decode-max-new-tokens",
        str(runtime_kwargs["u_decode_max_new_tokens"]),
    ]
    if case == "vlm":
        cmd.extend(
            ["--ug-vlm-input", str(payload_path), "--ug-vlm-output", str(result_path)]
        )
    else:
        cmd.extend(
            [
                "--ug-interleave-input",
                str(payload_path),
                "--ug-interleave-output",
                str(result_path),
            ]
        )
    if case == "think_t2i":
        cmd.extend(["--think", "--think-max-new-tokens", str(_think_max_new_tokens())])
    if runtime_kwargs["attention_backend"]:
        cmd.extend(
            [
                "--ug-srt-attention-backend",
                str(runtime_kwargs["attention_backend"]),
            ]
        )
    _run_subprocess(cmd, env=_subprocess_env({}), log_path=output_dir / "cli.log")


def _run_http_smoke(
    *,
    checkpoint_dir: Path,
    payload_path: Path,
    result_path: Path,
    output_dir: Path,
    case: str,
) -> None:
    port, scheduler_port, master_port = _pick_ports(3)
    runtime_kwargs = _runtime_kwargs()
    cmd = [
        sys.executable,
        "-m",
        "sglang.cli.main",
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        str(checkpoint_dir),
        "--pipeline-class-name",
        "UGPipeline",
        "--num-gpus",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--scheduler-port",
        str(scheduler_port),
        "--master-port",
        str(master_port),
        "--log-level",
        "error",
        "--ug-srt-log-level",
        "error",
        "--ug-srt-mem-fraction-static",
        str(runtime_kwargs["mem_fraction"]),
        "--ug-srt-chunked-prefill-size",
        str(runtime_kwargs["chunked_prefill"]),
        "--ug-srt-u-decode-max-new-tokens",
        str(runtime_kwargs["u_decode_max_new_tokens"]),
    ]
    if runtime_kwargs["attention_backend"]:
        cmd.extend(
            [
                "--ug-srt-attention-backend",
                str(runtime_kwargs["attention_backend"]),
            ]
        )
    log_path = output_dir / "http.log"
    timeout_sec = int(os.getenv(_TIMEOUT_ENV, "1800"))
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=_subprocess_env({}),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_http_health(port, process, timeout_sec=timeout_sec)
            payload = payload_path.read_bytes()
            endpoint = "vlm" if case == "vlm" else "interleave"
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/ug/{endpoint}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                result_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AssertionError(
                f"HTTP UG {case} request failed: {exc.code} {body}\n"
                f"log tail:\n{_tail(log_path)}"
            ) from exc
        finally:
            _terminate_process_group(process)


def _assert_entrypoint_result(
    test_case: CustomTestCase,
    result: dict[str, Any],
    *,
    case: str,
    expected_velocity_count: int,
) -> None:
    test_case.assertIsInstance(result, dict)
    segments = result.get("segments")
    test_case.assertIsInstance(segments, list)
    segment_types = [segment.get("type") for segment in segments]
    metadata = result.get("metadata")
    test_case.assertIsInstance(metadata, dict)
    test_case.assertEqual(metadata.get("mode"), _generation_mode_for_case(case))
    if case == "think_t2i":
        test_case.assertTrue(metadata.get("think"))

    if case == "vlm":
        test_case.assertEqual(segment_types, ["text"])
        test_case.assertIsInstance(segments[0].get("text"), str)
    else:
        test_case.assertIn("image", segment_types)
        image_index = segment_types.index("image")
        image_payload = segments[image_index].get("image")
        test_case.assertIsInstance(image_payload, str)
        test_case.assertTrue(image_payload.startswith("data:image/png;base64,"))
        if case == "interleave":
            trailing_text = [
                segment
                for segment in segments[image_index + 1 :]
                if segment.get("type") == "text"
            ]
            test_case.assertTrue(
                trailing_text,
                f"Expected a post-image text segment, got segment types {segment_types}",
            )
            test_case.assertIsInstance(trailing_text[0].get("text"), str)
        else:
            test_case.assertEqual(segment_types, ["image"])

    stats = result.get("stats")
    test_case.assertIsInstance(stats, dict)
    test_case.assertEqual(stats.get("prefill_count"), 1)
    test_case.assertEqual(stats.get("velocity_count"), expected_velocity_count)
    test_case.assertGreater(stats.get("srt_request_count", 0), 0)
    test_case.assertGreater(stats.get("srt_executed_request_count", 0), 0)
    if case == "vlm":
        test_case.assertEqual(stats.get("append_image_count"), 0)
        test_case.assertGreater(stats.get("srt_u_decode_request_count", 0), 0)
    elif case == "interleave":
        test_case.assertEqual(stats.get("append_image_count"), 1)
        test_case.assertGreaterEqual(stats.get("decode_count", 0), 2)
        test_case.assertGreater(stats.get("srt_sidecar_request_count", 0), 0)
        test_case.assertGreater(stats.get("srt_u_decode_request_count", 0), 0)
    else:
        test_case.assertEqual(stats.get("append_image_count"), 0)
        if case == "edit":
            test_case.assertGreater(stats.get("srt_sidecar_request_count", 0), 0)
        else:
            test_case.assertEqual(stats.get("srt_sidecar_request_count", 0), 0)
        if case == "think_t2i":
            test_case.assertGreater(stats.get("srt_u_decode_request_count", 0), 0)


def _subprocess_env(updates: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = _repo_root()
    python_path = str(repo_root / "python")
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    env.update(updates)
    return env


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pick_ports(count: int) -> tuple[int, ...]:
    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return tuple(ports)


def _run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> None:
    timeout_sec = int(os.getenv(_TIMEOUT_ENV, "1800"))
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise AssertionError(
                f"Command timed out after {timeout_sec}s: {cmd}\n"
                f"log tail:\n{_tail(log_path)}"
            ) from exc
    if process.returncode != 0:
        raise AssertionError(
            f"Command failed with code {process.returncode}: {cmd}\n"
            f"log tail:\n{_tail(log_path)}"
        )


def _wait_for_http_health(
    port: int,
    process: subprocess.Popen,
    *,
    timeout_sec: int,
) -> None:
    deadline = time.time() + timeout_sec
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"HTTP server exited early with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise AssertionError(f"Timed out waiting for {url}")


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def _tail(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


if __name__ == "__main__":
    unittest.main()
