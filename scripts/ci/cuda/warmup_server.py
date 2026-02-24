"""
Full server warmup to pre-warm Triton autotuning and CUDA graph capture.

On cold H200 nodes (new nodes or after container recreation), CUDA graph capture
triggers Triton autotuning which takes ~330s per server launch. This script
launches actual servers with CUDA graphs enabled to cache the autotuned kernels,
so subsequent test launches are fast (~30-60s).

Uses marker files to skip warmup on already-warm nodes. Marker files are
invalidated when Python, Triton, or PyTorch versions change.

Usage:
    python3 scripts/ci/cuda/warmup_server.py \
        deepseek-ai/DeepSeek-V3-0324:8 \
        inclusionAI/Ring-2.5-1T:8
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Reuse helpers from warmup_deep_gemm (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from warmup_deep_gemm import get_architecture_key, get_config_json

MARKER_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sglang", "warmup_markers")
HEALTH_POLL_INTERVAL = 10  # seconds between health checks
SERVER_STARTUP_TIMEOUT = 900  # 15 min max to wait for server ready
DEFAULT_PORT = 39876


def get_version_key():
    """Hash of Python + Triton + PyTorch versions to invalidate markers on upgrades."""
    parts = [sys.version]
    try:
        import triton

        parts.append(f"triton={triton.__version__}")
    except ImportError:
        parts.append("triton=none")
    try:
        import torch

        parts.append(f"torch={torch.__version__}")
    except ImportError:
        parts.append("torch=none")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def get_marker_path(model, tp):
    """Get the marker file path for a model:tp pair."""
    version_key = get_version_key()
    safe_model = model.replace("/", "--")
    return os.path.join(
        MARKER_DIR, f"server_warmup_{safe_model}_tp{tp}_{version_key}.done"
    )


def check_marker(model, tp):
    """Check if warmup marker exists (node already warm)."""
    marker = get_marker_path(model, tp)
    return os.path.exists(marker)


def write_marker(model, tp):
    """Write warmup marker after successful warmup."""
    marker = get_marker_path(model, tp)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    Path(marker).write_text(
        json.dumps(
            {
                "model": model,
                "tp": tp,
                "version_key": get_version_key(),
                "timestamp": time.time(),
            }
        )
    )
    print(f"  Wrote marker: {marker}")


def kill_server(proc):
    """Kill server process tree."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def wait_for_server(base_url, proc, timeout):
    """Poll /health_generate until server is ready or timeout."""
    import requests

    start = time.time()
    while time.time() - start < timeout:
        ret = proc.poll()
        if ret is not None:
            return False, f"Server exited with code {ret}"
        try:
            resp = requests.get(f"{base_url}/health_generate", timeout=5)
            if resp.status_code == 200:
                return True, None
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return False, "Timed out waiting for server"


def send_generate_request(base_url):
    """Send one /generate request to exercise the full inference path."""
    import requests

    payload = {
        "input_ids": [0, 1, 2, 3],
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0,
        },
    }
    try:
        resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
        if resp.status_code == 200:
            print("  Generate request succeeded")
        else:
            print(f"  Warning: generate request returned {resp.status_code}")
    except requests.RequestException as e:
        print(f"  Warning: generate request failed: {e}")


def warmup_one_model(model, tp, port):
    """Launch server, wait for ready, send one request, then kill."""
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--tp",
        str(tp),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    # Use a temp file for server output to avoid pipe buffer deadlock
    # (server logs can exceed the 64KB pipe buffer during CUDA graph capture)
    log_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="warmup_server_", suffix=".log", delete=False
    )
    log_path = log_file.name

    print(f"  Launching server: {' '.join(cmd)}")
    print(f"  Server log: {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        # Wait for server to be ready (includes CUDA graph capture)
        print(
            f"  Waiting for server (timeout={SERVER_STARTUP_TIMEOUT}s, "
            f"polling every {HEALTH_POLL_INTERVAL}s)..."
        )
        ok, err = wait_for_server(base_url, proc, SERVER_STARTUP_TIMEOUT)
        if not ok:
            print(f"  Warning: server not ready: {err}")
            # Dump last lines of server log for debugging
            try:
                log_file.flush()
                with open(log_path) as f:
                    lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    | {line.rstrip()}")
            except Exception:
                pass
            return False

        print("  Server ready, sending generate request...")
        send_generate_request(base_url)
        return True

    finally:
        print("  Killing server...")
        kill_server(proc)
        log_file.close()
        try:
            os.unlink(log_path)
        except OSError:
            pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: warmup_server.py model1:tp1 [model2:tp2 ...]")
        print(
            "\nLaunches full servers with CUDA graphs enabled to pre-warm"
            " Triton autotuning."
        )
        print("Skips instantly on warm nodes (marker file exists).")
        sys.exit(0)

    # Parse model:tp pairs
    model_tp_pairs = []
    for arg in sys.argv[1:]:
        if ":" not in arg:
            print(f"Error: expected model:tp format, got '{arg}'")
            sys.exit(1)
        model, tp_str = arg.rsplit(":", 1)
        model_tp_pairs.append((model, int(tp_str)))

    print(f"=== Server CUDA Graph Warmup ({len(model_tp_pairs)} model(s)) ===")
    print(f"    Marker dir: {MARKER_DIR}")
    print(f"    Version key: {get_version_key()}\n")

    # Deduplicate by architecture and check markers
    seen_keys = {}
    to_warmup = []

    for model, tp in model_tp_pairs:
        # Check marker first (fast path)
        if check_marker(model, tp):
            print(f"  SKIP   {model} (tp={tp}): already warm (marker exists)")
            continue

        # Architecture dedup
        config = get_config_json(model)
        if config is not None:
            key = get_architecture_key(config, tp)
            if key in seen_keys:
                print(
                    f"  DEDUP  {model} (tp={tp}): same architecture as {seen_keys[key]}"
                )
                continue
            seen_keys[key] = model

        to_warmup.append((model, tp))
        print(f"  QUEUE  {model} (tp={tp}): needs warmup")

    if not to_warmup:
        print("\nAll models already warm. Done.")
        return

    print(f"\n{len(to_warmup)} model(s) to warm up.\n")

    port = DEFAULT_PORT
    for i, (model, tp) in enumerate(to_warmup, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(to_warmup)}] {model} (tp={tp})")
        print(f"{'=' * 60}")

        t0 = time.time()
        success = warmup_one_model(model, tp, port)
        elapsed = time.time() - t0

        if success:
            print(f"  Completed in {elapsed:.0f}s")
            write_marker(model, tp)
            # Also write markers for dedup'd models that share this architecture
            config = get_config_json(model)
            if config is not None:
                key = get_architecture_key(config, tp)
                for other_model, other_tp in model_tp_pairs:
                    if (other_model, other_tp) == (model, tp):
                        continue
                    other_config = get_config_json(other_model)
                    if other_config is not None:
                        other_key = get_architecture_key(other_config, other_tp)
                        if other_key == key and not check_marker(other_model, other_tp):
                            write_marker(other_model, other_tp)
                            print(
                                f"  Also marked {other_model} (tp={other_tp}) as warm (same arch)"
                            )
        else:
            print(
                f"  Warning: warmup failed after {elapsed:.0f}s (non-fatal, tests will still work)"
            )

        # Use a different port for the next model to avoid bind conflicts
        port += 100

    print("\nServer CUDA graph warmup complete.")


if __name__ == "__main__":
    main()
