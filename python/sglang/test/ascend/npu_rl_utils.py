"""Shared harness for NPU RLHF lifecycle tests.

The NPU test suites are plain ``unittest`` files executed as
``python3 <test_file> -f`` by ``test/run_suite.py``; there is no pytest
collection, so this module (not a ``conftest.py``) is the place for common
server, memory, and weight-signature helpers.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests
import torch

try:
    import torch_npu  # noqa: F401

    _NPU_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only off NPU hosts
    _NPU_IMPORT_ERROR = exc

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _npu_smi(*args: str) -> str:
    return subprocess.check_output(
        ["npu-smi", *args], text=True, stderr=subprocess.DEVNULL
    )


def npu_available() -> bool:
    return _NPU_IMPORT_ERROR is None and shutil.which("npu-smi") is not None


def npu_device_count() -> int:
    if not npu_available():
        return 0
    out = _npu_smi("info", "-l")
    npus = int(re.search(r"Total Count\s*:\s*(\d+)", out).group(1))
    chips = int(re.search(r"Chip Count\s*:\s*(\d+)", out).group(1))
    return npus * chips


def warmup_npu(device: int = 0) -> None:
    """No-op: the test process must not create an NPU context.

    Starting a server after the parent has initialized an NPU context resets
    the device and kills the parent on some CANN/driver combinations.
    """


def _physical_device_id(device: int) -> int:
    """Map a process-visible logical device index to the host device id."""
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not visible_devices:
        return device

    devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if device < 0 or device >= len(devices):
        raise ValueError(
            f"logical NPU device {device} is outside ASCEND_RT_VISIBLE_DEVICES="
            f"{visible_devices!r}"
        )
    return int(devices[device])


def _npu_smi_card_and_chip(physical_device: int) -> list[str]:
    """Translate a host device id into npu-smi's card/chip addressing."""
    out = _npu_smi("info", "-l")
    chips_per_card = int(re.search(r"Chip Count\s*:\s*(\d+)", out).group(1))
    if chips_per_card <= 1:
        return ["-i", str(physical_device)]
    return [
        "-i",
        str(physical_device // chips_per_card),
        "-c",
        str(physical_device % chips_per_card),
    ]


def _hbm_capacity_and_rate(device: int = 0):
    physical_device = _physical_device_id(device)
    out = _npu_smi("info", "-t", "usages", *_npu_smi_card_and_chip(physical_device))
    capacity = int(re.search(r"HBM Capacity\(MB\)\s*:\s*(\d+)", out).group(1))
    rate = int(re.search(r"HBM Usage Rate\(%\)\s*:\s*(\d+)", out).group(1))
    return capacity * 1024**2, rate


def npu_total_bytes(device: int = 0) -> int:
    return _hbm_capacity_and_rate(device)[0]


def npu_free_bytes(device: int = 0) -> int:
    total, rate = _hbm_capacity_and_rate(device)
    return total - total * rate // 100


def wait_for_free_memory(
    device: int = 0,
    min_free_bytes: int = 2 * 1024**3,
    *,
    timeout: float = 60,
    interval: float = 0.5,
) -> None:
    """Wait for a previous server process to hand device memory back.

    ``device`` is a process-visible logical index, matching torch device
    numbering under ``ASCEND_RT_VISIBLE_DEVICES``.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if npu_free_bytes(device) >= min_free_bytes:
            return
        time.sleep(interval)
    raise RuntimeError(
        f"device {device} still has less than {min_free_bytes} free bytes "
        f"after {timeout}s"
    )


def _wait_for_hbm_baseline(
    device: int,
    baseline_free: int,
    *,
    timeout: float = 60,
    interval: float = 0.5,
) -> None:
    """Best-effort wait for HBM to return to its pre-launch level.

    SIGTERM/SIGKILL only starts driver-side teardown; the next server should
    not race that cleanup. Teardown must not fail the test, so a timeout is
    ignored.
    """
    if not npu_available():
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if npu_free_bytes(device) >= baseline_free:
            return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def npu_server(
    model_path: str,
    *,
    extra_args: Sequence[str] = (),
    port: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
) -> Iterator[str]:
    """Launch an NPU server and yield its base URL.

    ``popen_launch_server`` auto-detects the device and appends ``--device npu``.
    """
    port = port or find_available_port(30000)
    base_url = f"http://127.0.0.1:{port}"
    baseline_free = 0
    if npu_available():
        # The previous test's server may still be handing HBM back; a partial
        # free is not enough for the next full KV-cache allocation. Fail instead
        # of launching a second server on a device that is still occupied.
        wait_for_free_memory(0, int(0.65 * npu_total_bytes(0)), timeout=300)
        baseline_free = npu_free_bytes(0)
    process = popen_launch_server(
        model_path,
        base_url,
        timeout=timeout,
        other_args=list(extra_args),
        env=env,
    )
    try:
        yield base_url
    finally:
        # SIGTERM first so the server can release its NPU mempool in userspace,
        # then SIGKILL the remaining tree. A bare SIGKILL leaves driver-side
        # mempool/shared-handle cleanup racing the next launch.
        if process.poll() is None:
            terminate_and_kill_process_tree(process, terminate_timeout=30)
        else:
            process.wait()
        _wait_for_hbm_baseline(0, baseline_free)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def post(
    base_url: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    expect: int = 200,
) -> requests.Response:
    response = requests.post(base_url + path, json=payload or {}, timeout=timeout)
    if expect is not None and response.status_code != expect:
        raise RuntimeError(
            f"POST {path} returned {response.status_code}: {response.text[:800]}"
        )
    return response


def get(
    base_url: str,
    path: str,
    *,
    timeout: float = 30,
    expect: int = 200,
) -> requests.Response:
    response = requests.get(base_url + path, timeout=timeout)
    if expect is not None and response.status_code != expect:
        raise RuntimeError(
            f"GET {path} returned {response.status_code}: {response.text[:800]}"
        )
    return response


def health(base_url: str) -> bool:
    try:
        return requests.get(base_url + "/health", timeout=5).status_code == 200
    except Exception:
        return False


def release_memory(
    base_url: str,
    tags: Sequence[str],
    *,
    retries: int = 30,
    interval: float = 0.5,
    timeout: float = 120,
) -> None:
    """Release memory, retrying while in-flight requests drain.

    SGLang refuses the call with a 400 while requests are still running or
    queued; retry briefly instead of sleeping for a guessed duration.
    """
    last_error = ""
    for _ in range(retries):
        response = requests.post(
            base_url + "/release_memory_occupation",
            json={"tags": list(tags)},
            timeout=timeout,
        )
        if response.status_code == 200:
            return
        last_error = response.text
        if "still running or queued" not in last_error:
            break
        time.sleep(interval)
    raise RuntimeError(f"release_memory_occupation{list(tags)} failed: {last_error}")


def resume_memory(base_url: str, tags: Sequence[str], *, timeout: float = 120) -> None:
    post(
        base_url,
        "/resume_memory_occupation",
        {"tags": list(tags)},
        timeout=timeout,
    )


def pause_generation(
    base_url: str, mode: str = "abort", *, timeout: float = 120
) -> None:
    post(base_url, "/pause_generation", {"mode": mode}, timeout=timeout)


def continue_generation(base_url: str, *, timeout: float = 120) -> None:
    post(base_url, "/continue_generation", {}, timeout=timeout)


def flush_cache(base_url: str, *, timeout: float = 120) -> None:
    post(base_url, "/flush_cache", {}, timeout=timeout)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    timeout: float = 180,
) -> Dict[str, Any]:
    response = post(
        base_url,
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "ignore_eos": ignore_eos,
            },
        },
        timeout=timeout,
    )
    return response.json()


def stream_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    timeout: float = 300,
) -> Iterator[Dict[str, Any]]:
    with requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "ignore_eos": ignore_eos,
            },
            "stream": True,
        },
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                break
            yield json.loads(data)


def generation_signature(
    base_url: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 8,
) -> Tuple[Tuple[int, ...], ...]:
    signature = []
    for prompt in prompts:
        result = generate(base_url, prompt, max_new_tokens=max_new_tokens)
        signature.append(tuple(result["output_ids"]))
    return tuple(signature)


def assert_signature_equal(
    actual: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    message: str,
) -> None:
    actual = tuple(tuple(ids) for ids in actual)
    expected = tuple(tuple(ids) for ids in expected)
    if actual != expected:
        raise AssertionError(f"{message}\n  actual:   {actual}\n  expected: {expected}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def checkpoint_files(model_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint under {model_path}")
    return files


def checkpoint_weight_bytes(model_path: str) -> int:
    return sum(os.path.getsize(path) for path in checkpoint_files(model_path))


_SAFETENSORS_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def checkpoint_manifest(model_path: str) -> List[Tuple[str, str, List[int]]]:
    """Return ``(name, dtype, shape)`` for every tensor in the checkpoint."""
    from safetensors import safe_open

    manifest: List[Tuple[str, str, List[int]]] = []
    for path in checkpoint_files(model_path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                raw_dtype = str(tensor.dtype).replace("torch.", "")
                dtype = _SAFETENSORS_DTYPE.get(raw_dtype.upper(), raw_dtype)
                manifest.append((name, dtype, list(tensor.shape)))
    return manifest


def checkpoint_tensor_index(model_path: str) -> Dict[str, str]:
    from safetensors import safe_open

    index: Dict[str, str] = {}
    for path in checkpoint_files(model_path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                index[name] = path
    return index


def load_checkpoint_tensor(index: Dict[str, str], name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(index[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def iter_checkpoint_tensors(model_path: str) -> Iterator[Tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    for path in checkpoint_files(model_path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


# ---------------------------------------------------------------------------
# Weight-update helpers
# ---------------------------------------------------------------------------


def begin_weight_update(base_url: str, selector: str = "all") -> None:
    post(base_url, "/begin_weight_update", {"selector": selector})


def end_weight_update(base_url: str) -> None:
    post(base_url, "/end_weight_update", {})


def serialize_named_tensors(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> str:
    from sglang.srt.utils import MultiprocessingSerializer

    return MultiprocessingSerializer.serialize(named_tensors, output_str=True)


def update_weights_from_tensor(
    base_url: str,
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    *,
    load_format: Optional[str] = None,
    flush_cache_after: bool = False,
    weight_version: Optional[str] = None,
) -> None:
    payload = serialize_named_tensors(named_tensors)
    body: Dict[str, Any] = {
        "serialized_named_tensors": [payload],
        "load_format": load_format,
        "flush_cache": flush_cache_after,
    }
    if weight_version is not None:
        body["weight_version"] = weight_version
    post(base_url, "/update_weights_from_tensor", body, timeout=900)


def update_full_checkpoint(
    base_url: str,
    model_path: str,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
    weight_version: Optional[str] = None,
) -> None:
    """Stream the whole checkpoint into a running engine from tensor updates."""
    begin_weight_update(base_url)
    try:
        buffer: List[Tuple[str, torch.Tensor]] = []
        buffer_bytes = 0
        for name, cpu_tensor in iter_checkpoint_tensors(model_path):
            tensor = cpu_tensor.to("npu").contiguous()
            buffer.append((name, tensor))
            buffer_bytes += tensor.numel() * tensor.element_size()
            if buffer_bytes >= chunk_bytes:
                update_weights_from_tensor(base_url, buffer, flush_cache_after=False)
                buffer = []
                buffer_bytes = 0
        if buffer:
            update_weights_from_tensor(
                base_url,
                buffer,
                flush_cache_after=True,
                weight_version=weight_version,
            )
        else:
            flush_cache(base_url)
    finally:
        end_weight_update(base_url)


def update_weights_from_flattened_bucket(
    base_url: str,
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    *,
    weight_version: Optional[str] = None,
) -> None:
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    bucket = FlattenedTensorBucket(named_tensors=list(named_tensors))
    payload = MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        },
        output_str=True,
    )
    body: Dict[str, Any] = {
        "serialized_named_tensors": [payload],
        "load_format": "flattened_bucket",
        "flush_cache": True,
    }
    if weight_version is not None:
        body["weight_version"] = weight_version
    post(base_url, "/update_weights_from_tensor", body, timeout=900)


def read_weights_by_name(
    base_url: str,
    name: str,
    *,
    truncate_size: int = 8,
) -> List[float]:
    response = post(
        base_url,
        "/get_weights_by_name",
        {"name": name, "truncate_size": truncate_size},
    )
    parameter = response.json()
    if isinstance(parameter, dict):
        parameter = parameter.get("parameter")
    return list(parameter or [])
