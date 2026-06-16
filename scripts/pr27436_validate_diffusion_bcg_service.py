#!/usr/bin/env python3
"""Service-level BCG prompt-shape validation for PR 27436.

Starts `sglang serve` for each selected diffusion preset, sends two requests
with the same shape but different prompt text, and checks whether the second
request adds a new BCG capture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT_DIR = (
    ROOT
    / "python/sglang/multimodal_gen/.claude/skills"
    / "sglang-diffusion-benchmark-profile/scripts"
)
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(BENCH_SCRIPT_DIR))

from bench_diffusion_denoise import MODELS, required_gpus_for_model  # noqa: E402

CAPTURE_RE = re.compile(r"\[Diffusion BCG\] captured")
CAPTURE_FAILED_RE = re.compile(r"\[Diffusion BCG\] capture failed")
PEAK_MEMORY_RE = re.compile(r"Peak memory usage:\s*([0-9.]+)\s*MB")
SERVER_WARMUP_READY_RE = re.compile(r"The server is fired up and ready to roll!")
SERVER_WARMUP_FAILED_RE = re.compile(r"Server warmup failed")
FALLBACK_SIGNALS = (
    "falling back to diffusers backend",
    "using diffusers backend",
    "loaded diffusers pipeline",
)

IMAGE_MODELS = {
    "flux",
    "flux2",
    "qwen",
    "zimage",
    "qwen-image",
    "zimage-base",
    "flux2-klein",
    "flux2-klein-base",
    "cosmos3-nano-t2i",
    "ideogram4-fp8",
    "ernie-image-turbo",
    "glm-image",
    "sana-1.5-1.6b",
}
IMAGE_EDIT_MODELS = {
    "qwen-edit",
    "qwen-edit-2509",
    "joyai-edit",
    "firered-edit-1.0",
    "firered-edit-1.1",
}
MESH_MODELS = {"hunyuan3d-shape"}

SERVER_ARG_FLAGS = {
    "attention-backend",
    "backend",
    "cfg-parallel-size",
    "component-attention-backends",
    "dit-cpu-offload",
    "dit-layerwise-offload",
    "enable-cfg-parallel",
    "ltx2-two-stage-device-mode",
    "num-gpus",
    "pin-cpu-memory",
    "pipeline-class-name",
    "ring-degree",
    "text-encoder-cpu-offload",
    "ulysses-degree",
    "vae-cpu-offload",
}
REQUEST_ARG_FLAGS = {
    "adjust-frames",
    "flow-shift",
    "fps",
    "guidance-scale",
    "guidance-scale-2",
    "height",
    "max-sequence-length",
    "negative-prompt",
    "num-frames",
    "num-inference-steps",
    "true-cfg-scale",
    "width",
}


def parse_cli_args(items: list[str]) -> list[tuple[str, str | bool]]:
    parsed: list[tuple[str, str | bool]] = []
    i = 0
    while i < len(items):
        item = str(items[i])
        if not item.startswith("--"):
            i += 1
            continue
        if "=" in item:
            key, value = item[2:].split("=", 1)
        elif i + 1 < len(items) and not str(items[i + 1]).startswith("--"):
            key, value = item[2:], str(items[i + 1])
            i += 1
        else:
            key, value = item[2:], True
        parsed.append((key, value))
        i += 1
    return parsed


def cli_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def request_params(cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg["path"],
        "prompt": prompt,
        "seed": cfg.get("seed", 42),
    }
    if "negative_prompt" in cfg:
        params["negative_prompt"] = cfg["negative_prompt"]

    width = None
    height = None
    for key, value in parse_cli_args(list(cfg.get("extra_args", []))):
        if key == "width":
            width = int(cli_value(value))
        elif key == "height":
            height = int(cli_value(value))
        elif key in REQUEST_ARG_FLAGS:
            api_key = key.replace("-", "_")
            raw = cli_value(value)
            if key in {
                "fps",
                "max-sequence-length",
                "num-frames",
                "num-inference-steps",
            }:
                params[api_key] = int(raw)
            elif key in {
                "flow-shift",
                "guidance-scale",
                "guidance-scale-2",
                "true-cfg-scale",
            }:
                params[api_key] = float(raw)
            elif key == "adjust-frames":
                params[api_key] = raw.lower() == "true"
            elif key == "negative-prompt":
                params[api_key] = raw

    if width and height:
        params["size"] = f"{width}x{height}"

    if cfg.get("config_overrides", {}).get("paint_enable") is False:
        params["paint_enable"] = False
    if "ernie" in cfg["path"].lower():
        params["use_pe"] = False

    return params


def second_prompt(model_key: str) -> str:
    if model_key in IMAGE_EDIT_MODELS:
        return (
            "Make the cat wear a small blue raincoat and a bright yellow scarf "
            "while keeping the pose and background unchanged."
        )
    if model_key in MESH_MODELS:
        return "generate a detailed 3d mesh with smooth rounded ears and a clean base"
    if "video" in model_key or "wan" in model_key or "ltx" in model_key:
        return (
            "A calm cinematic shot where a tiny robot walks across a polished "
            "studio floor, pauses, and waves at the camera under soft lights."
        )
    return (
        "A bright glass greenhouse filled with rare blue flowers, brass tools, "
        "and warm afternoon sunlight reflected in small water droplets."
    )


def _model_resolution(cfg: dict[str, Any]) -> str | None:
    """The WxH this model is requested at, used as the BCG warmup resolution."""
    width = height = None
    for key, value in parse_cli_args(list(cfg.get("extra_args", []))):
        if key == "width":
            width = int(cli_value(value))
        elif key == "height":
            height = int(cli_value(value))
    if width and height:
        return f"{width}x{height}"
    return None


def build_server_cmd(
    model_key: str,
    port: int,
    *,
    runtime_model_path: str,
    output_dir: Path,
    no_warmup: bool,
    enable_bcg: bool,
    performance_mode: str | None,
    text_buckets: str | None = None,
) -> list[str]:
    cfg = MODELS[model_key]
    cmd = [
        "sglang",
        "serve",
        "--backend=sglang",
        f"--model-path={runtime_model_path}",
        "--host=127.0.0.1",
        f"--port={port}",
        "--strict-ports",
        f"--scheduler-port={port + 1000}",
        f"--master-port={port + 2000}",
        f"--output-path={output_dir / model_key / 'outputs'}",
        f"--input-save-path={output_dir / model_key / 'inputs'}",
    ]
    if enable_bcg:
        cmd.append("--enable-breakable-cuda-graph")
        # BCG requires explicit resolutions; capture every text bucket at warmup
        # so serving never records a fresh graph.
        resolution = _model_resolution(cfg)
        if resolution is not None:
            cmd.extend(["--warmup-resolutions", resolution])
        if text_buckets:
            cmd.extend(["--bcg-text-buckets", *text_buckets.replace(",", " ").split()])
    if performance_mode:
        cmd.extend(["--performance-mode", performance_mode])
    if no_warmup:
        cmd.extend(["--warmup", "false", "--server-warmup", "false"])
    else:
        cmd.append("--warmup")

    for key, value in parse_cli_args(list(cfg.get("extra_args", []))):
        if key not in SERVER_ARG_FLAGS:
            continue
        if isinstance(value, bool):
            cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    if "config_overrides" in cfg:
        config_path = output_dir / model_key / "server_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg["config_overrides"], indent=2))
        cmd.extend(["--config", str(config_path)])

    return cmd


def hf_hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def is_hf_model_id(model_path: str) -> bool:
    return "/" in model_path and not model_path.startswith(("/", "."))


def has_model_entrypoint(path: Path) -> bool:
    return (path / "model_index.json").exists() or (path / "config.json").exists()


def cached_snapshot_for_model(model_path: str) -> str | None:
    if not is_hf_model_id(model_path):
        return None

    repo_dir = hf_hub_cache_dir() / ("models--" + model_path.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref = repo_dir / "refs" / "main"
    candidates: list[Path] = []
    if ref.exists():
        revision = ref.read_text().strip()
        if revision:
            candidates.append(snapshots_dir / revision)

    candidates.extend(
        sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        if has_model_entrypoint(candidate):
            return str(candidate)
    return None


def runtime_model_path(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    path = str(cfg["path"])
    if args.offline or args.prefer_local_cache:
        cached = cached_snapshot_for_model(path)
        if cached:
            return cached
    return path


def wait_for_server(base_url: str, proc: subprocess.Popen, timeout: int) -> None:
    start = time.time()
    last = None
    while time.time() - start < timeout:
        ret = proc.poll()
        if ret is not None:
            raise RuntimeError(f"server exited with code {ret}")
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            last = f"HTTP {resp.status_code}"
            if resp.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout}s: {last}")


def wait_for_server_warmup(
    log_path: Path, proc: subprocess.Popen, timeout: int
) -> None:
    start = time.time()
    while time.time() - start < timeout:
        ret = proc.poll()
        if ret is not None:
            raise RuntimeError(f"server exited with code {ret}")
        text = log_text(log_path)
        if SERVER_WARMUP_READY_RE.search(text):
            return
        if SERVER_WARMUP_FAILED_RE.search(text):
            raise RuntimeError("server warmup failed")
        time.sleep(2)
    raise TimeoutError(f"server warmup did not finish within {timeout}s")


def log_text(log_path: Path) -> str:
    try:
        return log_path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def capture_count(log_path: Path) -> int:
    return len(CAPTURE_RE.findall(log_text(log_path)))


def has_bcg_capture_failed(log_path: Path) -> bool:
    return bool(CAPTURE_FAILED_RE.search(log_text(log_path)))


def has_diffusers_fallback(log_path: Path) -> bool:
    text = log_text(log_path).lower()
    return any(signal in text for signal in FALLBACK_SIGNALS)


def peak_memory_values(log_path: Path) -> list[float]:
    return [float(x) for x in PEAK_MEMORY_RE.findall(log_text(log_path))]


def output_files(model_dir: Path) -> list[str]:
    outputs = model_dir / "outputs"
    if not outputs.exists():
        return []
    files = [p for p in outputs.rglob("*") if p.is_file()]
    return [str(p) for p in sorted(files, key=lambda p: (p.stat().st_mtime_ns, str(p)))]


def new_output_files(before: list[str], after: list[str]) -> list[str]:
    before_set = set(before)
    return [path for path in after if path not in before_set]


def post_image(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    payload.update({"n": 1, "response_format": "url"})
    resp = requests.post(
        f"{base_url}/v1/images/generations", json=payload, timeout=3600
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
    return resp.json()


def post_image_edit(base_url: str, params: dict[str, Any], image_path: str) -> dict:
    data = {k: str(v) for k, v in params.items() if k != "model"}
    data.update({"model": params["model"], "n": "1", "response_format": "url"})
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/v1/images/edits",
            data=data,
            files={"image": (Path(image_path).name, f, "application/octet-stream")},
            timeout=3600,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
    return resp.json()


def poll_job(url: str, timeout: int = 3600) -> dict[str, Any]:
    start = time.time()
    last_error = None
    first_error_at = None
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=30)
            last_error = None
            first_error_at = None
        except requests.RequestException as exc:
            last_error = exc
            if first_error_at is None:
                first_error_at = time.time()
            if time.time() - first_error_at > 120:
                raise RuntimeError(f"poll connection failed: {last_error}") from exc
            time.sleep(2)
            continue
        if resp.status_code != 200:
            last_error = RuntimeError(
                f"poll HTTP {resp.status_code}: {resp.text[:1000]}"
            )
            if first_error_at is None:
                first_error_at = time.time()
            if time.time() - first_error_at > 120:
                raise last_error
            time.sleep(2)
            continue
        data = resp.json()
        last_error = None
        first_error_at = None
        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"job failed: {data.get('error')}")
        time.sleep(2)
    if last_error is not None:
        raise TimeoutError(f"job did not complete within {timeout}s: {last_error}")
    raise TimeoutError(f"job did not complete within {timeout}s")


def post_video(base_url: str, params: dict[str, Any], image_path: str | None) -> dict:
    if image_path:
        data = {
            k: str(v)
            for k, v in params.items()
            if k not in {"model", "paint_enable", "use_pe"}
        }
        data["model"] = params["model"]
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{base_url}/v1/videos",
                data=data,
                files={
                    "input_reference": (
                        Path(image_path).name,
                        f,
                        "application/octet-stream",
                    )
                },
                timeout=120,
            )
    else:
        resp = requests.post(f"{base_url}/v1/videos", json=params, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"submit HTTP {resp.status_code}: {resp.text[:1000]}")
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"no job id in response: {resp.text[:1000]}")
    return poll_job(f"{base_url}/v1/videos/{job_id}")


def post_mesh(base_url: str, params: dict[str, Any], image_path: str) -> dict:
    data = {k: str(v) for k, v in params.items() if k != "model"}
    data["model"] = params["model"]
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/v1/meshes",
            data=data,
            files={"image": (Path(image_path).name, f, "application/octet-stream")},
            timeout=120,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"submit HTTP {resp.status_code}: {resp.text[:1000]}")
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"no job id in response: {resp.text[:1000]}")
    return poll_job(f"{base_url}/v1/meshes/{job_id}")


def send_request(model_key: str, base_url: str, params: dict[str, Any]) -> dict:
    cfg = MODELS[model_key]
    image_path = cfg.get("image_path")
    if model_key in IMAGE_MODELS:
        return post_image(base_url, params)
    if model_key in IMAGE_EDIT_MODELS:
        if not image_path:
            raise RuntimeError("image edit preset is missing image_path")
        return post_image_edit(base_url, params, image_path)
    if model_key in MESH_MODELS:
        if not image_path:
            raise RuntimeError("mesh preset is missing image_path")
        return post_mesh(base_url, params, image_path)
    return post_video(base_url, params, image_path)


def run_one(
    model_key: str,
    *,
    args: argparse.Namespace,
    result_dir: Path,
) -> dict[str, Any]:
    cfg = MODELS[model_key]
    gpus = args.gpu_pool[: required_gpus_for_model(model_key)]
    if len(gpus) < required_gpus_for_model(model_key):
        return {
            "model": model_key,
            "status": "skipped",
            "reason": "not enough GPUs in gpu pool",
        }

    port = args.port_start + args.index
    base_url = f"http://127.0.0.1:{port}"
    model_dir = result_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "server.log"
    resolved_model_path = runtime_model_path(cfg, args)
    cmd = build_server_cmd(
        model_key,
        port,
        runtime_model_path=resolved_model_path,
        output_dir=result_dir,
        no_warmup=args.no_warmup,
        enable_bcg=not args.disable_bcg,
        performance_mode=args.performance_mode,
        text_buckets=args.text_buckets,
    )

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["PYTHONPATH"] = "python"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"

    result: dict[str, Any] = {
        "model": model_key,
        "model_path": cfg["path"],
        "runtime_model_path": resolved_model_path,
        "gpus": gpus,
        "port": port,
        "mode": "eager" if args.disable_bcg else "bcg",
        "cmd": " ".join(shlex.quote(x) for x in cmd),
        "log": str(log_path),
    }

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        try:
            wait_for_server(base_url, proc, args.startup_timeout)
            if has_diffusers_fallback(log_path):
                result.update(
                    {"status": "failed", "reason": "diffusers fallback detected"}
                )
                return result

            if not args.no_warmup:
                wait_for_server_warmup(log_path, proc, args.startup_timeout)
            captures_after_warmup = capture_count(log_path)

            request_cfg = dict(cfg)
            request_cfg["path"] = resolved_model_path
            first = request_params(request_cfg, cfg["prompt"])
            second = request_params(request_cfg, second_prompt(model_key))
            files_before_first = output_files(model_dir)
            t0 = time.time()
            send_request(model_key, base_url, first)
            first_latency = time.time() - t0
            time.sleep(2)
            captures_after_first = capture_count(log_path)
            files_after_first = output_files(model_dir)

            t0 = time.time()
            send_request(model_key, base_url, second)
            second_latency = time.time() - t0
            time.sleep(2)
            captures_after_second = capture_count(log_path)
            files_after_second = output_files(model_dir)
            if args.disable_bcg:
                pass_capture_check = not has_diffusers_fallback(log_path)
            elif args.no_warmup:
                pass_capture_check = (
                    not has_bcg_capture_failed(log_path)
                    and captures_after_first > 0
                    and captures_after_second == captures_after_first
                )
            else:
                # Strict: warmup must capture everything, so neither the first
                # nor the second serving request may add a new BCG capture.
                pass_capture_check = (
                    captures_after_warmup > 0
                    and not has_bcg_capture_failed(log_path)
                    and captures_after_first == captures_after_warmup
                    and captures_after_second == captures_after_warmup
                )

            result.update(
                {
                    "status": "passed" if pass_capture_check else "failed",
                    "captures_after_warmup": captures_after_warmup,
                    "captures_after_first": captures_after_first,
                    "captures_after_second": captures_after_second,
                    "first_latency_s": round(first_latency, 3),
                    "second_latency_s": round(second_latency, 3),
                    "peak_memory_mb": peak_memory_values(log_path),
                    "first_output_files": new_output_files(
                        files_before_first, files_after_first
                    ),
                    "second_output_files": new_output_files(
                        files_after_first, files_after_second
                    ),
                    "output_files": files_after_second,
                    "bcg_capture_failed": has_bcg_capture_failed(log_path),
                    "diffusers_fallback": has_diffusers_fallback(log_path),
                    "reason": (
                        "eager run completed"
                        if pass_capture_check and args.disable_bcg
                        else (
                            "warmup captured all graphs; both serving prompts "
                            "added no new capture"
                            if pass_capture_check and not args.no_warmup
                            else (
                                "no second-request capture"
                                if pass_capture_check
                                else (
                                    "BCG capture failed"
                                    if has_bcg_capture_failed(log_path)
                                    else (
                                        "warmup did not capture a BCG graph"
                                        if not args.no_warmup
                                        and captures_after_warmup == 0
                                        else (
                                            "request added BCG capture after warmup"
                                            if not args.no_warmup
                                            else "second request added BCG capture"
                                        )
                                    )
                                )
                            )
                        )
                    ),
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001
            result.update(
                {
                    "status": "failed",
                    "reason": str(exc),
                    "captures": capture_count(log_path),
                    "peak_memory_mb": peak_memory_values(log_path),
                    "output_files": output_files(model_dir),
                    "bcg_capture_failed": has_bcg_capture_failed(log_path),
                    "diffusers_fallback": has_diffusers_fallback(log_path),
                    "log_tail": log_text(log_path)[-4000:],
                }
            )
            return result
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(args.cooldown)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--result-dir", default="/tmp/pr27436_bcg_service_validation")
    parser.add_argument("--gpu-pool", default="0")
    parser.add_argument("--port-start", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prefer-local-cache", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--text-buckets", default="256,512,1024,2048")
    parser.add_argument("--disable-bcg", action="store_true")
    parser.add_argument("--performance-mode")
    args = parser.parse_args()
    args.gpu_pool = [x.strip() for x in args.gpu_pool.split(",") if x.strip()]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, model_key in enumerate(args.models):
        args.index = index
        print(f"=== {model_key} ===", flush=True)
        if model_key not in MODELS:
            result = {"model": model_key, "status": "skipped", "reason": "unknown"}
        else:
            result = run_one(model_key, args=args, result_dir=result_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        (result_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
    return 0 if all(r.get("status") == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
