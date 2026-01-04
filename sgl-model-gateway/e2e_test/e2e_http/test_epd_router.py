"""
E2E tests for EPD (Encode-Prefill-Decode) disaggregated serving mode.

EPD mode enables multimodal LLM inference with separate encode, prefill, and decode workers:
- Encode worker: Processes multimodal inputs (images/video) via HTTP REST API (--encoder-only)
- Prefill worker: Receives embeddings via ZMQ, computes KV cache (--disaggregation-mode prefill --language-only)
- Decode worker: Generates output tokens using KV cache from prefill (--disaggregation-mode decode)

Text-only requests automatically fall back to PD (Prefill-Decode) mode.
"""

import logging
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Optional

import grpc
import pytest
import requests
from grpc_health.v1 import health_pb2, health_pb2_grpc

from sglang.test.run_eval import run_eval

logger = logging.getLogger(__name__)


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(
    url: str, timeout: float = 180.0, proc: Optional[subprocess.Popen] = None
) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            if proc is not None and proc.poll() is not None:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                raise RuntimeError(
                    f"Process died during health check. Output:\n{stdout}"
                )
            try:
                r = session.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
    if proc is not None and proc.stdout:
        try:
            proc.terminate()
            stdout, _ = proc.communicate(timeout=5)
            raise TimeoutError(
                f"Service at {url} failed to become healthy in time. Output:\n{stdout.decode()}"
            )
        except Exception:
            pass
    raise TimeoutError(f"Service at {url} failed to become healthy in time")


def _wait_workers_ready(
    url: str, expected_count: int, timeout: float = 120.0, proc: Optional[subprocess.Popen] = None
) -> None:
    """Wait for workers to register in the router by attempting a test request."""
    start = time.perf_counter()
    with requests.Session() as session:
        # First wait for /v1/models to return a model
        while time.perf_counter() - start < timeout:
            if proc is not None and proc.poll() is not None:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                raise RuntimeError(
                    f"Process died waiting for workers. Output:\n{stdout}"
                )
            try:
                r = session.get(f"{url}/v1/models", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("data") and len(data["data"]) > 0:
                        logger.info(
                            f"Models available after {time.perf_counter() - start:.1f}s - "
                            f"models: {[m.get('id') for m in data['data']]}"
                        )
                        break
            except requests.RequestException:
                pass
            time.sleep(3)

        # Now try actual requests until they succeed (workers fully registered)
        model_id = None
        try:
            r = session.get(f"{url}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("data") and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id")
        except:
            pass

        while time.perf_counter() - start < timeout:
            if proc is not None and proc.poll() is not None:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                raise RuntimeError(
                    f"Process died waiting for workers. Output:\n{stdout}"
                )
            try:
                r = session.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": model_id or "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    logger.info(f"Workers ready after {time.perf_counter() - start:.1f}s")
                    return
                elif r.status_code == 503:
                    # Workers not ready yet
                    pass
            except requests.RequestException:
                pass
            time.sleep(3)
    logger.warning(f"Workers may not be fully ready after {timeout}s timeout")


def _wait_health_grpc(
    host: str, port: int, timeout: float = 180.0, proc: Optional[subprocess.Popen] = None
) -> None:
    """Wait for gRPC server to become healthy using standard gRPC health protocol."""
    start = time.perf_counter()
    target = f"{host}:{port}"
    while time.perf_counter() - start < timeout:
        if proc is not None and proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(
                f"Process died during gRPC health check. Output:\n{stdout}"
            )
        try:
            with grpc.insecure_channel(target) as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service="")
                response = stub.Check(request, timeout=5)
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    return
        except grpc.RpcError:
            pass
        except Exception:
            pass
        time.sleep(1)
    if proc is not None and proc.stdout:
        try:
            proc.terminate()
            stdout, _ = proc.communicate(timeout=5)
            raise TimeoutError(
                f"gRPC service at {target} failed to become healthy in time. Output:\n{stdout.decode()}"
            )
        except Exception:
            pass
    raise TimeoutError(f"gRPC service at {target} failed to become healthy in time")


def _detect_ib_device() -> Optional[str]:
    """Return first active IB device name (e.g., mlx5_0) or None if unavailable."""
    try:
        subprocess.run(
            ["ibv_devinfo", "-l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for i in range(12):
        dev = f"mlx5_{i}"
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and ("state:" in res.stdout):
                for line in res.stdout.splitlines():
                    if "state:" in line and "PORT_ACTIVE" in line:
                        return dev
        except Exception:
            pass
    return None


def _popen_launch_encode_worker(
    model: str,
    ib_device: Optional[str] = None,
    base_gpu_id: int = 0,
) -> SimpleNamespace:
    """Launch an encode worker that processes multimodal inputs (gRPC mode for Rust router)."""
    port = _find_available_port()
    url = f"grpc://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--encoder-only",
        "--grpc-mode",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(base_gpu_id),
    ]
    if ib_device:
        cmd += ["--disaggregation-ib-device", ib_device]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_health_grpc("127.0.0.1", port, timeout=300.0, proc=proc)
    return SimpleNamespace(proc=proc, url=url)


def _popen_launch_prefill_worker(
    model: str,
    bootstrap_port: int,
    encoder_urls: list[str],
    ib_device: Optional[str] = None,
    base_gpu_id: int = 0,
    mem_fraction: float = 0.9,
    chunk_size: int = 8192,
) -> SimpleNamespace:
    """Launch a prefill worker that computes KV cache (gRPC mode for Rust router)."""
    import os

    port = _find_available_port()
    url = f"grpc://127.0.0.1:{port}"
    # Use ZMQ by default, allow override via environment variable
    transfer_backend = os.getenv("EPD_TRANSFER_BACKEND", "mooncake")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--grpc-mode",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        transfer_backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disaggregation-bootstrap-port",
        str(bootstrap_port),
        "--base-gpu-id",
        str(base_gpu_id),
        "--mem-fraction-static",
        str(mem_fraction),
        "--disable-radix-cache",
        "--chunked-prefill-size",
        str(chunk_size),
        "--language-only",
        "--skip-server-warmup",
        "--chat-template",
        "chatml",
        "--encoder-urls",
    ] + encoder_urls
    if ib_device and transfer_backend == "mooncake":
        cmd += ["--disaggregation-ib-device", ib_device]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_health_grpc("127.0.0.1", port, timeout=300.0, proc=proc)
    return SimpleNamespace(proc=proc, url=url, bootstrap_port=bootstrap_port)


def _popen_launch_decode_worker(
    model: str,
    ib_device: Optional[str] = None,
    base_gpu_id: int = 0,
    mem_fraction: float = 0.9,
) -> SimpleNamespace:
    """Launch a decode worker that generates output tokens (gRPC mode for Rust router)."""
    import os

    port = _find_available_port()
    url = f"grpc://127.0.0.1:{port}"
    # Use ZMQ by default, allow override via environment variable
    transfer_backend = os.getenv("EPD_TRANSFER_BACKEND", "mooncake")
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        transfer_backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(base_gpu_id),
        "--mem-fraction-static",
        str(mem_fraction),
        "--skip-server-warmup",
        "--chat-template",
        "chatml",
    ]
    if ib_device and transfer_backend == "mooncake":
        cmd += ["--disaggregation-ib-device", ib_device]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_health_grpc("127.0.0.1", port, timeout=300.0, proc=proc)
    return SimpleNamespace(proc=proc, url=url)


def _terminate(proc: subprocess.Popen, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


@pytest.fixture(scope="module")
def epd_vlm_model() -> str:
    """VLM model to use for EPD tests.

    Supports:
    - EPD_VLM_MODEL env var with local path or HF model ID
    - Auto-detection of cached HF models for offline use
    """
    import os
    from pathlib import Path

    model = os.getenv("EPD_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

    # If it's already a local path, use it directly
    if os.path.isdir(model):
        return model

    # Try to find cached HF model for offline use
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        # Convert HF model ID to cache directory format
        model_dir_name = f"models--{model.replace('/', '--')}"
        model_cache_dir = hf_cache / model_dir_name / "snapshots"
        if model_cache_dir.exists():
            snapshots = list(model_cache_dir.iterdir())
            if snapshots:
                # Use the most recent snapshot
                cached_path = str(snapshots[-1])
                logger.info(f"Using cached model: {cached_path}")
                return cached_path

    return model


@pytest.fixture(scope="module")
def epd_cluster(epd_vlm_model: str):
    """Start 1 encode + 1 prefill + 1 decode workers and one EPD router."""
    try:
        import sgl_kernel  # noqa: F401
    except Exception as e:
        pytest.fail(f"EPD e2e requires sgl_kernel but it is not available: {e}")

    try:
        import torch
    except Exception as e:
        pytest.fail(f"EPD e2e requires torch: {e}")

    if not torch.cuda.is_available():
        pytest.fail("EPD e2e requires CUDA backend")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 3:
        pytest.skip(f"EPD e2e requires at least 3 GPUs, but only {gpu_count} available")

    workers: list[SimpleNamespace] = []
    router_proc = None
    try:
        ib_device = _detect_ib_device()

        # 1. Encode Worker (GPU 0)
        logger.info("Launching encode worker on GPU 0...")
        enc1 = _popen_launch_encode_worker(
            epd_vlm_model, ib_device=ib_device, base_gpu_id=0
        )
        encodes = [enc1]
        workers.extend(encodes)

        # 2. Prefill Worker (GPU 1)
        encoder_urls = [enc.url for enc in encodes]
        logger.info(f"Launching prefill worker on GPU 1...")
        pf1 = _popen_launch_prefill_worker(
            epd_vlm_model,
            bootstrap_port=_find_available_port(),
            encoder_urls=encoder_urls,
            ib_device=ib_device,
            base_gpu_id=1,
        )
        prefills = [pf1]
        workers.extend(prefills)

        # 3. Decode Worker (GPU 2)
        logger.info("Launching decode worker on GPU 2...")
        dc1 = _popen_launch_decode_worker(
            epd_vlm_model, ib_device=ib_device, base_gpu_id=2
        )
        decodes = [dc1]
        workers.extend(decodes)

        # 4. EPD Router
        rport = _find_available_port()
        router_url = f"http://127.0.0.1:{rport}"
        prom_port = _find_available_port()

        # Build command
        encode = [(enc.url, None) for enc in encodes]
        prefill = [(pf.url, pf.bootstrap_port) for pf in prefills]
        decode = [dc.url for dc in decodes]

        cmd = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(rport),
            "--policy",
            "round_robin",
            "--epd-disaggregation",
            "--model-path",
            epd_vlm_model,
            "--log-level",
            "info",
        ]
        for url, _ in encode:
            cmd += ["--encode", url]
        for url, bport in prefill:
            cmd += ["--prefill", url, str(bport)]
        for url in decode:
            cmd += ["--decode", url]
        cmd += [
            "--prometheus-port",
            str(prom_port),
            "--prometheus-host",
            "127.0.0.1",
        ]

        logger.info(f"Launching EPD router: {' '.join(cmd)}")
        router_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _wait_health(router_url, timeout=180.0, proc=router_proc)

        # Wait for workers to register in the router (they register asynchronously)
        _wait_workers_ready(router_url, expected_count=3, timeout=120.0, proc=router_proc)

        yield SimpleNamespace(
            router_url=router_url,
            workers=workers,
            router_proc=router_proc,
            encodes=encodes,
            prefills=prefills,
            decodes=decodes,
        )
    finally:
        if router_proc is not None:
            _terminate(router_proc)
        for w in workers:
            _terminate(w.proc)


@pytest.fixture(scope="module")
def epd_cluster_multiworker(epd_vlm_model: str):
    """Start a larger EPD cluster (6 encode + 1 prefill + 1 decode)."""
    try:
        import torch
    except Exception as e:
        pytest.fail(f"EPD e2e requires torch: {e}")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 8:
        pytest.skip(f"Multi-worker EPD test requires 8 GPUs, found {gpu_count}")

    workers: list[SimpleNamespace] = []
    router_proc = None
    try:
        ib_device = _detect_ib_device()

        # 1. Encode Workers (GPU 0-5)
        encodes = []
        for i in range(6):
            logger.info(f"Launching encode worker {i + 1}/6 on GPU {i}...")
            enc = _popen_launch_encode_worker(
                epd_vlm_model, ib_device=ib_device, base_gpu_id=i
            )
            encodes.append(enc)
        workers.extend(encodes)

        # 2. Prefill Worker (GPU 6)
        encoder_urls = [enc.url for enc in encodes]
        logger.info(f"Launching prefill worker on GPU 6...")
        pf1 = _popen_launch_prefill_worker(
            epd_vlm_model,
            bootstrap_port=_find_available_port(),
            encoder_urls=encoder_urls,
            ib_device=ib_device,
            base_gpu_id=6,
        )
        prefills = [pf1]
        workers.extend(prefills)

        # 3. Decode Worker (GPU 7)
        logger.info("Launching decode worker on GPU 7...")
        dc1 = _popen_launch_decode_worker(
            epd_vlm_model, ib_device=ib_device, base_gpu_id=7
        )
        decodes = [dc1]
        workers.extend(decodes)

        # 4. EPD Router
        rport = _find_available_port()
        router_url = f"http://127.0.0.1:{rport}"
        prom_port = _find_available_port()

        encode = [(enc.url, None) for enc in encodes]
        prefill = [(pf.url, pf.bootstrap_port) for pf in prefills]
        decode = [dc.url for dc in decodes]

        cmd = [
            sys.executable,
            "-m",
            "sglang_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(rport),
            "--policy",
            "round_robin",
            "--epd-disaggregation",
            "--model-path",
            epd_vlm_model,
            "--log-level",
            "warn",
        ]
        for url, _ in encode:
            cmd += ["--encode", url]
        for url, bport in prefill:
            cmd += ["--prefill", url, str(bport)]
        for url in decode:
            cmd += ["--decode", url]
        cmd += [
            "--prometheus-port",
            str(prom_port),
            "--prometheus-host",
            "127.0.0.1",
        ]

        router_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        _wait_health(router_url, timeout=180.0, proc=router_proc)

        # Wait for workers to register in the router (they register asynchronously)
        _wait_workers_ready(router_url, expected_count=8, timeout=120.0, proc=router_proc)

        yield SimpleNamespace(
            router_url=router_url,
            workers=workers,
            router_proc=router_proc,
            encodes=encodes,
            prefills=prefills,
            decodes=decodes,
        )
    finally:
        if router_proc is not None:
            _terminate(router_proc)
        for w in workers:
            _terminate(w.proc)


@pytest.mark.e2e
@pytest.mark.epd
@pytest.mark.quick
def test_epd_simple_text(epd_vlm_model: str, epd_cluster):
    """Simple text request to verify EPD pipeline works (PD fallback path)."""
    response = requests.post(
        f"{epd_cluster.router_url}/v1/chat/completions",
        json={
            "model": epd_vlm_model,
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60,
    )
    assert response.status_code == 200, f"Request failed: {response.text}"
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    content = data["choices"][0]["message"]["content"]
    assert "4" in content, f"Expected '4' in response, got: {content}"
    logger.info(f"Simple text test passed. Response: {content}")


@pytest.mark.e2e
@pytest.mark.epd
def test_epd_mmlu(epd_vlm_model: str, epd_cluster):
    """Run MMLU (Text-only fallback test)."""
    args = SimpleNamespace(
        base_url=epd_cluster.router_url,
        model=epd_vlm_model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.50


@pytest.mark.e2e
@pytest.mark.epd
def test_epd_mmmu(epd_vlm_model: str, epd_cluster):
    """Run MMMU (Multimodal test)."""
    args = SimpleNamespace(
        base_url=epd_cluster.router_url,
        model=epd_vlm_model,
        eval_name="mmmu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.45


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


@pytest.mark.e2e
@pytest.mark.epd
def test_epd_genai_bench(epd_vlm_model: str, epd_cluster, genai_bench_runner):
    """Run genai-bench."""
    if not _which("genai-bench"):
        pytest.skip("genai-bench CLI not found")

    genai_bench_runner(
        router_url=epd_cluster.router_url,
        model_path=epd_vlm_model,
        experiment_folder="benchmark_round_robin_epd",
        thresholds={
            "ttft_mean_max": 15,
            "e2e_latency_mean_max": 20,
            "input_throughput_mean_min": 200,
            "output_throughput_mean_min": 10,
            "gpu_util_p50_min": 90,
        },
        kill_procs=epd_cluster.workers,
    )


@pytest.mark.e2e
@pytest.mark.epd
def test_epd_mmmu_bench(epd_vlm_model: str, epd_cluster_multiworker):
    """Run MMMU on multi-worker cluster."""
    args = SimpleNamespace(
        base_url=epd_cluster_multiworker.router_url,
        model=epd_vlm_model,
        eval_name="mmmu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.45


@pytest.mark.e2e
@pytest.mark.epd
def test_epd_bench_serving_mmmu(epd_vlm_model: str, epd_cluster_multiworker):
    """Run sglang.bench_serving."""
    import os
    import subprocess

    router_url = epd_cluster_multiworker.router_url
    port = router_url.split(":")[-1]

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--tokenizer",
        epd_vlm_model,
        "--model",
        epd_vlm_model,
        "--num-prompts",
        "32",
        "--dataset-name",
        "mmmu",
        "--port",
        port,
        "--backend",
        "vllm-chat",
        "--request-rate",
        "0.1",
    ]

    logger.info(f"Running bench_serving: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "SGLANG_HOST": "127.0.0.1"},
    )

    if result.returncode != 0:
        logger.error(
            f"bench_serving failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        logger.error(
            f"bench_serving failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        pytest.fail(f"bench_serving returned non-zero exit code: {result.returncode}")

    logger.info(f"bench_serving output:\n{result.stdout}")
