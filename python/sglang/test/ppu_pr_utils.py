"""Bounded small-model fixtures shared only by PPU PR tests."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_MODEL = "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-0.5B-Instruct"


def weight_files(path):
    weights = list(path.glob("*.safetensors"))
    for index in path.glob("*.safetensors.index.json"):
        # Index shards may live in subdirectories; resolve checkpoint-relative.
        weights.extend(
            path / name for name in json.loads(index.read_text())["weight_map"].values()
        )
    return weights


def small_model():
    path = Path(os.environ.get("PPU_SMALL_MODEL_PATH", DEFAULT_MODEL)).resolve()
    required = [
        path / "config.json",
        path / "tokenizer_config.json",
        path / "tokenizer.json",
    ]
    weights = weight_files(path)
    if not weights:
        raise FileNotFoundError(f"No local model weights: {path}")
    for file in required + weights:
        if (
            not file.is_file()
            or file.stat().st_size == 0
            or path not in file.resolve().parents
        ):
            raise FileNotFoundError(f"Missing/invalid local model file: {file}")
    return str(path)


def model_identity():
    path = Path(small_model())
    # Hash small metadata; record shard size/mtime without scanning a GB per job.
    # Key shards by checkpoint-relative path so subdirectory shards are included.
    return {
        "path": str(path),
        "metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.glob("*.json")
        },
        "weights": {
            str(p.relative_to(path)): {
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in weight_files(path)
        },
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(base, method, path, **kwargs):
    import requests

    return requests.request(method, base + path, timeout=(5, 45), **kwargs)


@contextmanager
def server(tp=1, startup_timeout=150, bounded=True, *, compact_json=False):
    from sglang.srt.utils import kill_process_tree

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        small_model(),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(tp),
        "--attention-backend",
        "fa3",
    ]
    if bounded:
        args += [
            "--sampling-backend",
            "pytorch",
            "--grammar-backend",
            "xgrammar",
            "--chunked-prefill-size",
            "128",
            "--max-running-requests",
            "4",
            "--cuda-graph-max-bs",
            "4",
            "--mem-fraction-static",
            "0.6",
            "--disable-custom-all-reduce",
            "--dist-timeout",
            "60",
            "--watchdog-timeout",
            "90",
        ]
    if compact_json:
        args.append("--constrained-json-disable-any-whitespace")
    process = subprocess.Popen(
        args, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    )
    try:
        import requests

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"PPU server exited: {process.returncode}")
            try:
                if requests.get(base + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("PPU server startup timed out")
        yield base
    finally:
        kill_process_tree(process.pid)
        process.wait(timeout=15)


def prompt_ids(length):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(small_model(), local_files_only=True)
    seed = tokenizer.encode(
        "The quick brown fox visits a quiet village. ", add_special_tokens=False
    )
    assert seed
    return (seed * (length // len(seed) + 1))[:length]


def generate(base, ids, **extra):
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        **extra,
    }
    response = request(base, "POST", "/generate", json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(json.dumps(model_identity(), indent=2))
