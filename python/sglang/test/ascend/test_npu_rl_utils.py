"""Shared harness for NPU RLHF lifecycle tests.

The suites run as plain ``unittest`` files via ``python3 <test_file> -f``;
this module holds the common server, memory, request, and weight helpers.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests
import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - exercised only off NPU hosts
    torch_npu = None

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
A5_SPEC_ENV = {
    "ASCEND_USE_FIA": "1",
    # A5 only allows aclnn operators, so online weight updates cannot copy into
    # FRACTAL_NZ parameters; keep MoE weights in ND for these servers.
    "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
}


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


MEMORY_WAIT_MARGIN_BYTES = 2 * 1024**3


def _torch_mem(device: int = 0) -> Tuple[int, int]:
    """Device-wide (free, total) HBM in bytes, read in-process."""
    free, total = torch.npu.mem_get_info(device)
    return int(free), int(total)


def _is_ascend_950() -> bool:
    """Return whether the current NPU is an Ascend 950 (A5).

    SOC versions 250-256 are A3; Ascend 950 reports 260.
    """
    if torch_npu is None:
        return False
    try:
        return int(torch_npu.npu.get_soc_version()) >= 260
    except Exception:
        return False


def npu_rl_env() -> Dict[str, str]:
    """Return the extra server env required by the RL lifecycle tests."""
    return dict(A5_SPEC_ENV) if _is_ascend_950() else {}


def npu_free_bytes(device: int = 0) -> int:
    return _torch_mem(device)[0]


def npu_total_bytes(device: int = 0) -> int:
    return _torch_mem(device)[1]


def wait_for_free_memory(
    device: int = 0,
    min_free_bytes: int = 2 * 1024**3,
    *,
    timeout: float = 60,
    interval: float = 0.5,
) -> None:
    """Wait for a previous server process to hand device memory back.

    ``device`` is a process-visible logical index; ``mem_get_info`` reports
    device-wide free bytes, so this also sees other tenants on shared hosts.
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        if npu_free_bytes(device) >= baseline_free:
            return
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _server_arg(extra_args: Sequence[str], name: str, default: str) -> str:
    for index, arg in enumerate(extra_args):
        if arg == name and index + 1 < len(extra_args):
            return extra_args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


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
    # Warm the test process's NPU context while the card is idle; the first
    # in-process mem_get_info() call is otherwise paid while the server is
    # starting. Then wait for the previous server to hand HBM back, using the
    # memory fraction this server will reserve plus a fixed margin.
    npu_free_bytes(0)
    mem_fraction = float(_server_arg(extra_args, "--mem-fraction-static", "0.9"))
    tp_size = max(1, int(_server_arg(extra_args, "--tp-size", "1")))
    min_free = int(mem_fraction * npu_total_bytes(0)) + MEMORY_WAIT_MARGIN_BYTES
    for device in range(tp_size):
        wait_for_free_memory(device, min_free, timeout=300)
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
    return_logprob: bool = False,
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
            "return_logprob": return_logprob,
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


@dataclass
class StreamRequest:
    """A streaming /generate request running in a background thread."""

    thread: threading.Thread
    events: List[Dict[str, Any]]
    errors: List[BaseException]


def start_stream(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    timeout: float = 300,
) -> StreamRequest:
    events: List[Dict[str, Any]] = []
    errors: List[BaseException] = []

    def _run():
        try:
            for event in stream_generate(
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
            ):
                events.append(event)
        except Exception as exc:  # pragma: no cover - surfaced in the test
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return StreamRequest(thread=thread, events=events, errors=errors)


def wait_for_first_token(
    request: StreamRequest,
    *,
    label: str,
    timeout: float = 30,
) -> None:
    """Wait until the request decoded one token and is still running."""
    deadline = time.time() + timeout
    while (
        request.thread.is_alive()
        and not request.events
        and not request.errors
        and time.time() < deadline
    ):
        time.sleep(0.02)

    if request.errors:
        raise AssertionError(f"{label} failed before the first token: {request.errors}")
    if not request.events:
        raise AssertionError(f"{label} produced no token within {timeout}s")
    if not request.thread.is_alive():
        raise AssertionError(f"{label} finished before it could be paused")


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


def logprob_signature(
    base_url: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 8,
) -> Tuple[Tuple[Tuple[int, ...], Tuple[float, ...]], ...]:
    """Return ``(output_ids, output_token_logprobs)`` for each prompt.

    Logprobs catch a partially or silently loaded payload that still produces
    the same greedy tokens, which plain output ids would miss.
    """
    signature = []
    for prompt in prompts:
        result = generate(
            base_url, prompt, max_new_tokens=max_new_tokens, return_logprob=True
        )
        meta = result.get("meta_info", {})
        token_logprobs = tuple(
            round(float(item[0]), 3)
            for item in (meta.get("output_token_logprobs") or ())
        )
        if not token_logprobs:
            raise AssertionError(
                f"/generate returned no output_token_logprobs for {prompt!r}"
            )
        signature.append((tuple(result["output_ids"]), token_logprobs))
    return tuple(signature)


def assert_logprob_signature_equal(
    actual: Sequence[Tuple[Sequence[int], Sequence[float]]],
    expected: Sequence[Tuple[Sequence[int], Sequence[float]]],
    message: str,
    *,
    logprob_tolerance: float = 5e-2,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            f"{message}: signature lengths differ ({len(actual)} != {len(expected)})"
        )
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        actual_ids = tuple(actual_item[0])
        expected_ids = tuple(expected_item[0])
        if actual_ids != expected_ids:
            raise AssertionError(
                f"{message}: token ids differ for prompt {index}\n"
                f"  actual:   {actual_ids}\n  expected: {expected_ids}"
            )
        actual_logprobs = tuple(actual_item[1])
        expected_logprobs = tuple(expected_item[1])
        if len(actual_logprobs) != len(expected_logprobs):
            raise AssertionError(
                f"{message}: logprob count differs for prompt {index} "
                f"({len(actual_logprobs)} != {len(expected_logprobs)})"
            )
        for position, (actual_lp, expected_lp) in enumerate(
            zip(actual_logprobs, expected_logprobs)
        ):
            if abs(actual_lp - expected_lp) > logprob_tolerance:
                raise AssertionError(
                    f"{message}: logprob differs for prompt {index}, token {position} "
                    f"({actual_lp} vs {expected_lp}, tolerance {logprob_tolerance})"
                )


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


# Dtypes are ordered so that wider tensors start 4-byte aligned, which is what
# FlattenedTensorBucket.empty() validates on the receiving side.
_FLATTEN_ORDER = {"float32": 0, "float16": 1, "bfloat16": 2}


def flattened_checkpoint_manifest(
    model_path: str,
) -> List[Tuple[str, str, List[int]]]:
    """Manifest ordered for a single flattened_bucket transfer buffer."""
    manifest = checkpoint_manifest(model_path)
    return sorted(manifest, key=lambda item: (_FLATTEN_ORDER.get(item[1], 3), item[0]))


def flattened_tensor_bucket(model_path: str, device: str = "npu"):
    """Materialize the whole checkpoint as one FlattenedTensorBucket on ``device``.

    Tensors are copied one at a time into the bucket buffer, so peak memory is
    the flattened payload plus one source tensor rather than both full sets.
    """
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    manifest = flattened_checkpoint_manifest(model_path)
    bucket = FlattenedTensorBucket.empty(
        [name for name, _, _ in manifest],
        [dtype for _, dtype, _ in manifest],
        [shape for _, _, shape in manifest],
        device,
    )
    index = checkpoint_tensor_index(model_path)
    flat = bucket.get_flattened_tensor()
    for meta in bucket.metadata:
        source = load_checkpoint_tensor(index, meta.name)
        flat[meta.start_idx : meta.end_idx].view(meta.dtype).reshape(meta.shape).copy_(
            source
        )
    return bucket


def serialize_flattened_bucket(bucket) -> str:
    from sglang.srt.utils import MultiprocessingSerializer

    return MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        },
        output_str=True,
    )


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


def update_weights_from_tensor_per_rank(
    base_url: str,
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
    tp_size: int,
    *,
    flattened_bucket: bool = False,
    weight_version: Optional[str] = None,
) -> None:
    """Push the same named tensors as one payload per TP rank.

    Each rank's payload is built on that rank's device; the engine shards the
    full tensors inside ``load_weights``. ``flattened_bucket=True`` sends each
    rank's payload as one flattened buffer (``load_format="flattened_bucket"``).
    """
    load_format = "flattened_bucket" if flattened_bucket else None
    from sglang.srt.utils import MultiprocessingSerializer

    keepalive = []
    payloads = []
    try:
        for rank in range(tp_size):
            device = f"npu:{rank}"
            tensors = [
                (name, tensor.to(device).contiguous()) for name, tensor in named_tensors
            ]
            keepalive.append(tensors)
            if load_format == "flattened_bucket":
                from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

                bucket = FlattenedTensorBucket(named_tensors=tensors)
                keepalive.append(bucket)
                payloads.append(serialize_flattened_bucket(bucket))
            else:
                payloads.append(
                    MultiprocessingSerializer.serialize(tensors, output_str=True)
                )
        body = {
            "serialized_named_tensors": payloads,
            "load_format": load_format,
            "flush_cache": True,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        begin_weight_update(base_url)
        try:
            post(base_url, "/update_weights_from_tensor", body, timeout=1800)
        finally:
            end_weight_update(base_url)
    finally:
        del keepalive, payloads


def update_full_checkpoint_per_rank(
    base_url: str,
    model_path: str,
    tp_size: int,
    *,
    flattened_bucket: bool = False,
    weight_version: Optional[str] = None,
) -> None:
    """Push one payload per TP rank, like an SPMD RL trainer.

    Each rank's payload holds full (unsharded) tensors created on that rank's
    device; the engine shards them inside ``load_weights``. The payloads are
    serialized as NPU IPC handles, so the HTTP body stays small.
    """
    from sglang.srt.utils import MultiprocessingSerializer

    keepalive = []
    payloads = []
    try:
        for rank in range(tp_size):
            device = f"npu:{rank}"
            if flattened_bucket:
                bucket = flattened_tensor_bucket(model_path, device)
                keepalive.append(bucket)
                payloads.append(serialize_flattened_bucket(bucket))
            else:
                tensors = [
                    (name, cpu_tensor.to(device).contiguous())
                    for name, cpu_tensor in iter_checkpoint_tensors(model_path)
                ]
                keepalive.append(tensors)
                payloads.append(
                    MultiprocessingSerializer.serialize(tensors, output_str=True)
                )
        body = {
            "serialized_named_tensors": payloads,
            "load_format": "flattened_bucket" if flattened_bucket else None,
            "flush_cache": True,
        }
        if weight_version is not None:
            body["weight_version"] = weight_version
        begin_weight_update(base_url)
        try:
            post(base_url, "/update_weights_from_tensor", body, timeout=1800)
        finally:
            end_weight_update(base_url)
    finally:
        del keepalive, payloads
