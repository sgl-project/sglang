#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark official HF and native SGLang SenseNova U1 interleave."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_MODEL_PATH = "/mnt/afs/fanyijiat/models/SenseNova-U1-8B-MoT-Interleaved-bd39"
DEFAULT_VENDOR_ROOT = "/mnt/afs/fanyijiat/sensenova_interleave/vendor/SenseNova-U1"
DEFAULT_OUT_DIR = "outputs/sensenova_u1_main/interleave_perf_20260816"
DEFAULT_PROMPT = (
    "Say 'Here is the image:' then generate an image of a red apple. After the "
    "image, write at least 100 words describing it."
)
DEFAULT_TEXT_TOKEN_BUDGET = 64
DEFAULT_NATIVE_MAX_NEW_TOKENS = DEFAULT_TEXT_TOKEN_BUDGET + 1
DEFAULT_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> "
    "block and place all reasoning inside it. You MUST interleave text with "
    "generated images using tags like <image1>, <image2>. Images can ONLY be "
    "generated between <think> and </think>, and may be referenced in the final "
    "answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer "
    "without reasoning. Do not use tags like <image1>, <image2>; present any "
    "images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in "
    "both reasoning and the final answer."
)

IMG_START_TOKEN = "<img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_END_TOKEN = "</img>"
IMAGE_PLACEHOLDER = "<image>"
MIB = 1024**2


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile over no values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


def _synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GPUMemorySampler:
    def __init__(self, gpu_index: int, interval_s: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples_mib: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            self.samples_mib.append(int(output.strip().splitlines()[0]))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)
        self._sample()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return {
            "sample_count": len(self.samples_mib),
            "peak_used_mib": max(self.samples_mib) if self.samples_mib else None,
            "minimum_used_mib": min(self.samples_mib) if self.samples_mib else None,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    model_path: str
    vendor_root: str
    out_dir: Path
    prompt: str
    system_message: str
    width: int
    height: int
    num_steps: int
    text_token_budget: int
    native_max_new_tokens: int
    max_images: int
    seed: int
    concurrency_levels: tuple[int, ...]
    warmup_repeats: int
    measurement_repeats: int
    gpu_index: int

    def shared_settings(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "prompt": self.prompt,
            "system_message_sha256": _sha256(self.system_message.encode("utf-8")),
            "image_geometry": [self.width, self.height],
            "num_steps": self.num_steps,
            "effective_generated_text_token_budget": self.text_token_budget,
            "native_api_max_new_tokens": self.native_max_new_tokens,
            "native_budget_note": (
                "Native max_new_tokens is one larger because the model-emitted "
                "<img> boundary is counted by SGLang; synthetic image-context "
                "tokens are added outside the effective text budget."
            ),
            "max_images": self.max_images,
            "seed": self.seed,
            "cfg_scale": 1.0,
            "img_cfg_scale": 1.0,
            "cfg_norm": "none",
            "timestep_shift": 1.0,
            "enable_timestep_shift": True,
            "cfg_interval": [0.0, 1.0],
            "t_eps": 0.05,
            "think_mode": False,
            "concurrency_levels": list(self.concurrency_levels),
            "warmup_repeats_per_level": self.warmup_repeats,
            "measurement_repeats_per_level": self.measurement_repeats,
        }


@dataclass(frozen=True)
class RequestInvocation:
    concurrency: int
    batch_kind: str
    repeat_index: int
    request_ordinal: int

    @property
    def logical_request_id(self) -> str:
        return (
            f"{self.batch_kind}-repeat-{self.repeat_index}-"
            f"ordinal-{self.request_ordinal}"
        )


def _raw_native_prompt(config: BenchmarkConfig) -> str:
    return (
        f"<|im_start|>system\n{config.system_message}<|im_end|>\n"
        f"<|im_start|>user\n{config.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def _build_contract(
    *,
    pre_image_token_ids: list[int],
    post_image_token_ids: list[int],
    image_bytes: bytes,
    image_shape: list[int],
    generated_text_tokens: int,
    text: str,
) -> dict[str, Any]:
    public_contract = {
        "text": text,
        "pre_image_token_ids": [int(token) for token in pre_image_token_ids],
        "post_image_token_ids": [int(token) for token in post_image_token_ids],
        "generated_text_tokens": int(generated_text_tokens),
        "image_turns": 1,
        "image_shape": [int(value) for value in image_shape],
        "image_dtype": "float16",
        "image_sha256": _sha256(image_bytes),
    }
    return {
        **public_contract,
        "contract_sha256": _canonical_digest(public_contract),
        "real_text_image_text_turn": bool(
            pre_image_token_ids and post_image_token_ids and image_bytes
        ),
    }


def _validate_contract(
    contract: dict[str, Any],
    *,
    expected_text_tokens: int,
    backend: str,
) -> None:
    text_tokens = int(contract["generated_text_tokens"])
    token_ids = [
        *contract["pre_image_token_ids"],
        *contract["post_image_token_ids"],
    ]
    if not contract["real_text_image_text_turn"]:
        raise RuntimeError(f"{backend} did not produce a real text-image-text turn")
    if int(contract["image_turns"]) != 1:
        raise RuntimeError(f"{backend} did not produce exactly one image turn")
    if text_tokens != expected_text_tokens or len(token_ids) != expected_text_tokens:
        raise RuntimeError(
            f"{backend} produced {text_tokens} generated text tokens "
            f"({len(token_ids)} contract token ids), expected "
            f"{expected_text_tokens}"
        )


def _serialize_contract(contract: dict[str, Any], image_bytes: bytes) -> int:
    payload = {
        **contract,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    json.loads(encoded)
    return len(encoded.encode("utf-8"))


def _summarize_level(
    *,
    concurrency: int,
    expected_text_tokens: int,
    warmups: list[dict[str, Any]],
    repeats: list[dict[str, Any]],
    gpu_memory: dict[str, Any],
    torch_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    requests = [row for repeat in repeats for row in repeat["requests"]]
    batch_walls = [float(repeat["batch_wall_s"]) for repeat in repeats]
    total_requests = sum(len(repeat["requests"]) for repeat in repeats)
    total_batch_wall = sum(batch_walls)
    total_text_tokens = sum(
        int(row["contract"]["generated_text_tokens"]) for row in requests
    )
    total_image_turns = sum(int(row["contract"]["image_turns"]) for row in requests)
    signatures = [row["contract"]["contract_sha256"] for row in requests]
    cross_repeat_batch_qps = [
        len(repeat["requests"]) / float(repeat["batch_wall_s"]) for repeat in repeats
    ]

    result = {
        "concurrency": concurrency,
        "warmup_batches": len(warmups),
        "measurement_batches": len(repeats),
        "measured_requests": total_requests,
        "per_request_latency_s": _stats(
            [float(row["request_latency_s"]) for row in requests]
        ),
        "queue_wait_s": _stats([float(row["queue_wait_s"]) for row in requests]),
        "model_execution_s": _stats(
            [float(row["model_execution_s"]) for row in requests]
        ),
        "serialization_or_client_overhead_s": _stats(
            [float(row["serialization_or_client_overhead_s"]) for row in requests]
        ),
        "batch_wall_s": _stats(batch_walls),
        "batch_requests_per_s": _stats(cross_repeat_batch_qps),
        "requests_per_s": total_requests / total_batch_wall,
        "text_tokens_per_s": total_text_tokens / total_batch_wall,
        "image_turns_per_s": total_image_turns / total_batch_wall,
        "gpu_memory": gpu_memory,
        "determinism": {
            "unique_contract_digests": sorted(set(signatures)),
            "deterministic": len(set(signatures)) == 1,
        },
        "correctness": {
            "all_real_text_image_text_turns": all(
                row["contract"]["real_text_image_text_turn"] for row in requests
            ),
            "all_effective_text_budgets_match": {
                int(row["contract"]["generated_text_tokens"]) for row in requests
            }
            == {expected_text_tokens},
            "generated_text_token_counts": sorted(
                {int(row["contract"]["generated_text_tokens"]) for row in requests}
            ),
            "image_turn_counts": sorted(
                {int(row["contract"]["image_turns"]) for row in requests}
            ),
        },
        "raw_repeats": repeats,
    }
    if torch_memory is not None:
        result["torch_memory"] = torch_memory
    return result


def _run_concurrent_batch(
    *,
    concurrency: int,
    batch_kind: str,
    repeat_index: int,
    request_fn: Callable[[RequestInvocation], dict[str, Any]],
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)

    def task(index: int) -> dict[str, Any]:
        barrier.wait()
        return request_fn(
            RequestInvocation(
                concurrency=concurrency,
                batch_kind=batch_kind,
                repeat_index=repeat_index,
                request_ordinal=index,
            )
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, index) for index in range(concurrency)]
        started = time.perf_counter()
        barrier.wait()
        requests = [future.result() for future in futures]
        batch_wall_s = time.perf_counter() - started
    return {
        "batch_wall_s": batch_wall_s,
        "requests": requests,
    }


def _run_levels(
    *,
    config: BenchmarkConfig,
    request_fn: Callable[[RequestInvocation], dict[str, Any]],
    before_measurement: Callable[[], None] | None = None,
    after_measurement: Callable[[], dict[str, Any] | None] | None = None,
    progress_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    levels: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    for concurrency in config.concurrency_levels:
        warmups = [
            _run_concurrent_batch(
                concurrency=concurrency,
                batch_kind="warmup",
                repeat_index=repeat_index,
                request_fn=request_fn,
            )
            for repeat_index in range(config.warmup_repeats)
        ]
        if before_measurement is not None:
            before_measurement()
        sampler = GPUMemorySampler(config.gpu_index)
        sampler.start()
        repeats = [
            _run_concurrent_batch(
                concurrency=concurrency,
                batch_kind="measurement",
                repeat_index=repeat_index,
                request_fn=request_fn,
            )
            for repeat_index in range(config.measurement_repeats)
        ]
        gpu_memory = sampler.stop()
        torch_memory = after_measurement() if after_measurement is not None else None
        if reference is None:
            reference = repeats[0]["requests"][0]["reference"]
        for repeat in repeats:
            for row in repeat["requests"]:
                row.pop("reference", None)
        level = _summarize_level(
            concurrency=concurrency,
            expected_text_tokens=config.text_token_budget,
            warmups=warmups,
            repeats=repeats,
            gpu_memory=gpu_memory,
            torch_memory=torch_memory,
        )
        levels[str(concurrency)] = level
        _write_json(progress_path, {"levels": levels})
        print(
            f"[level {concurrency}] wall median={level['batch_wall_s']['median']:.6f}s "
            f"req/s={level['requests_per_s']:.6f}",
            flush=True,
        )
    assert reference is not None
    return levels, reference


def _run_hf(
    config: BenchmarkConfig,
    hf_attn_backend: str,
    hf_compat_root: str | None,
) -> int:
    vendor_root = Path(config.vendor_root)
    vendor_src = vendor_root / "src"
    sys.path.insert(0, str(vendor_src))
    sys.path.insert(0, str(vendor_root / "examples/interleave"))

    import torch
    from inference import DEFAULT_SYSTEM_MESSAGE as OFFICIAL_SYSTEM_MESSAGE

    if OFFICIAL_SYSTEM_MESSAGE != config.system_message:
        raise RuntimeError("official interleave system prompt changed")
    load_started = time.perf_counter()
    if hf_compat_root:
        sys.path.insert(0, str(Path(hf_compat_root)))
        from sglang_omni.models.sensenova_u1.flow_matching import (
            SenseNovaU1FlowMatchingFallbackRunner,
        )
        from sglang_omni.models.sensenova_u1.hf_runner import (
            _official_block_mask_scope,
        )

        runner = SenseNovaU1FlowMatchingFallbackRunner(
            config.model_path,
            vendor_root=config.vendor_root,
            device=f"cuda:{config.gpu_index}",
            dtype="bfloat16",
            attn_backend=hf_attn_backend,
        )
        assert runner.model is not None and runner.tokenizer is not None
        model = runner.model
        tokenizer = runner.tokenizer
        official_scope: Callable[[], Any] = _official_block_mask_scope
        hf_loader = "official_hf_with_transformers_compatibility_loader"
    else:
        from sensenova_u1 import set_attn_backend
        from sensenova_u1.utils import load_model_and_tokenizer

        set_attn_backend(hf_attn_backend)
        model, tokenizer = load_model_and_tokenizer(
            config.model_path,
            dtype=torch.bfloat16,
            device=f"cuda:{config.gpu_index}",
        )
        official_scope = nullcontext
        hf_loader = "official_vendor_loader"
    model_load_s = time.perf_counter() - load_started
    model_lock = threading.Lock()
    generation_config = SimpleNamespace(max_new_tokens=config.text_token_budget)

    def request_fn(invocation: RequestInvocation) -> dict[str, Any]:
        request_started = time.perf_counter()
        with model_lock:
            model_acquired = time.perf_counter()
            _synchronize_cuda()
            model_started = time.perf_counter()
            with official_scope():
                text, image_tensors = model.interleave_gen(
                    tokenizer,
                    config.prompt,
                    images=[],
                    generation_config=generation_config,
                    cfg_scale=1.0,
                    img_cfg_scale=1.0,
                    cfg_norm="none",
                    max_images=config.max_images,
                    enable_timestep_shift=True,
                    timestep_shift=1.0,
                    image_size=(config.width, config.height),
                    num_steps=config.num_steps,
                    cfg_interval=(0.0, 1.0),
                    t_eps=0.05,
                    verbose=False,
                    system_message=config.system_message,
                    think_mode=False,
                    seed=config.seed,
                )
            _synchronize_cuda()
            model_finished = time.perf_counter()
            if len(image_tensors) != 1:
                raise RuntimeError(
                    f"official HF generated {len(image_tensors)} images, expected 1"
                )
            if IMAGE_PLACEHOLDER not in text:
                raise RuntimeError("official HF output has no <image> continuation")
            pre_text, post_text = text.split(IMAGE_PLACEHOLDER, maxsplit=1)
            pre_ids = tokenizer(pre_text, add_special_tokens=False)["input_ids"]
            post_ids = tokenizer(post_text, add_special_tokens=False)["input_ids"]
            image = (
                image_tensors[0]
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
            )
            image_bytes = image.numpy().tobytes()
            generated_tokens = int(
                getattr(model, "last_interleave_generated_tokens", 0)
            )
            contract = _build_contract(
                pre_image_token_ids=list(pre_ids),
                post_image_token_ids=list(post_ids),
                image_bytes=image_bytes,
                image_shape=list(image.shape),
                generated_text_tokens=generated_tokens,
                text=str(text),
            )
            _validate_contract(
                contract,
                expected_text_tokens=config.text_token_budget,
                backend="official HF",
            )
            serialized_bytes = _serialize_contract(contract, image_bytes)
            request_finished = time.perf_counter()
            image_elapsed = list(getattr(model, "last_interleave_image_elapsed_s", []))
            stop_reason = str(getattr(model, "last_interleave_stop_reason", "unknown"))

        return {
            "logical_request_id": invocation.logical_request_id,
            "request_ordinal": invocation.request_ordinal,
            "request_seed": config.seed,
            "request_latency_s": request_finished - request_started,
            "queue_wait_s": model_acquired - request_started,
            "model_execution_s": model_finished - model_started,
            "serialization_or_client_overhead_s": request_finished - model_finished,
            "serialized_response_bytes": serialized_bytes,
            "flow_or_image_execution_s": sum(float(value) for value in image_elapsed),
            "stop_reason": stop_reason,
            "contract": contract,
            "reference": {
                "contract": contract,
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            },
        }

    def before_measurement() -> None:
        torch.cuda.reset_peak_memory_stats()

    def after_measurement() -> dict[str, Any]:
        return {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        }

    out_dir = config.out_dir / "hf"
    levels, reference = _run_levels(
        config=config,
        request_fn=request_fn,
        before_measurement=before_measurement,
        after_measurement=after_measurement,
        progress_path=out_dir / "progress.json",
    )
    result = {
        "ok": True,
        "backend": "official_hf_interleave_gen",
        "loader": hf_loader,
        "compatibility_root": hf_compat_root,
        "concurrency_methodology": (
            "The official interleave_gen implementation is single-request and "
            "mutates shared KV/current-index state. Concurrent clients therefore "
            "enter a bounded queue guarded by one model lock; request latency and "
            "batch wall include queueing, while model_execution_s excludes it."
        ),
        "model_load_s": model_load_s,
        "settings": config.shared_settings(),
        "levels": levels,
        "reference": reference,
    }
    _write_json(out_dir / "results.json", result)
    return 0


def _native_contract(
    *,
    response: dict[str, Any],
    img_start_id: int,
    img_context_id: int,
    img_end_id: int,
    tokenizer,
) -> tuple[dict[str, Any], bytes]:
    output_ids = [int(token) for token in response["output_ids"]]
    try:
        image_start = output_ids.index(img_start_id)
        image_end = output_ids.index(img_end_id, image_start + 1)
    except ValueError as error:
        raise RuntimeError(
            "native output has no complete generated image span"
        ) from error
    image_context_ids = output_ids[image_start + 1 : image_end]
    if not image_context_ids or any(
        token != img_context_id for token in image_context_ids
    ):
        raise RuntimeError("native generated image span has invalid context tokens")
    pre_ids = output_ids[:image_start]
    post_ids = output_ids[image_end + 1 :]
    meta = response["meta_info"]
    image_values = [
        value
        for value in meta["sensenova_u1_interleave_image_b64"]
        if value is not None
    ]
    shape_values = [
        value
        for value in meta["sensenova_u1_interleave_image_shape"]
        if value is not None
    ]
    if len(image_values) != 1 or len(shape_values) != 1:
        raise RuntimeError("native response did not return exactly one image")
    image_bytes = base64.b64decode(image_values[0])
    normalized_text = (
        tokenizer.decode(pre_ids, skip_special_tokens=True)
        + IMAGE_PLACEHOLDER
        + tokenizer.decode(post_ids, skip_special_tokens=True)
    )
    contract = _build_contract(
        pre_image_token_ids=pre_ids,
        post_image_token_ids=post_ids,
        image_bytes=image_bytes,
        image_shape=shape_values[0],
        generated_text_tokens=len(pre_ids) + len(post_ids),
        text=normalized_text,
    )
    return contract, image_bytes


def _run_native(config: BenchmarkConfig, server_url: str) -> int:
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )
    img_start_id = int(tokenizer.convert_tokens_to_ids(IMG_START_TOKEN))
    img_context_id = int(tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN))
    img_end_id = int(tokenizer.convert_tokens_to_ids(IMG_END_TOKEN))
    prompt = _raw_native_prompt(config)
    endpoint = server_url.rstrip("/") + "/generate"
    benchmark_run_id = time.time_ns()

    def request_fn(invocation: RequestInvocation) -> dict[str, Any]:
        request_started = time.perf_counter()
        rid = (
            f"interleave-perf-{benchmark_run_id}-"
            f"{invocation.batch_kind}-c{invocation.concurrency}-"
            f"r{invocation.repeat_index}-o{invocation.request_ordinal}"
        )
        payload = {
            "rid": rid,
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": config.native_max_new_tokens,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "custom_params": {
                    "sensenova_u1_interleave": {
                        "width": config.width,
                        "height": config.height,
                        "num_steps": config.num_steps,
                        "max_images": config.max_images,
                        "seed": config.seed,
                        "timestep_shift": 1.0,
                        "enable_timestep_shift": True,
                        "return_images": True,
                    }
                },
            },
        }
        http_response = requests.post(endpoint, json=payload, timeout=600)
        http_response.raise_for_status()
        response = http_response.json()
        contract, image_bytes = _native_contract(
            response=response,
            img_start_id=img_start_id,
            img_context_id=img_context_id,
            img_end_id=img_end_id,
            tokenizer=tokenizer,
        )
        _validate_contract(
            contract,
            expected_text_tokens=config.text_token_budget,
            backend="native SGLang",
        )
        meta = response["meta_info"]
        dispatch_finish = float(meta.get("api_server_dispatch_finish_ts", 0.0))
        forward_entry = float(meta.get("forward_entry_time", 0.0))
        request_finish = float(meta.get("request_finished_ts", 0.0))
        reported_queue_wait_s = float(meta.get("queue_time", 0.0))
        if (
            dispatch_finish > 0
            and forward_entry >= dispatch_finish
            and request_finish >= forward_entry
        ):
            model_inference_s = request_finish - dispatch_finish
            queue_wait_s = forward_entry - dispatch_finish
            active_execution_s = request_finish - forward_entry
            timing_source = "server_dispatch_forward_finish_timestamps"
        else:
            model_inference_s = float(meta["e2e_latency"])
            queue_wait_s = (
                reported_queue_wait_s
                if 0.0 <= reported_queue_wait_s <= model_inference_s
                else 0.0
            )
            active_execution_s = model_inference_s - queue_wait_s
            timing_source = "server_e2e_latency_fallback"
        request_finished = time.perf_counter()
        client_overhead_s = max(
            (request_finished - request_started) - model_inference_s,
            0.0,
        )
        flow_values = [
            float(value)
            for value in meta.get("sensenova_u1_flow_compute_seconds", [])
            if value is not None
        ]
        serialized_bytes = len(http_response.content)
        return {
            "rid": rid,
            "logical_request_id": invocation.logical_request_id,
            "request_ordinal": invocation.request_ordinal,
            "request_seed": config.seed,
            "request_latency_s": request_finished - request_started,
            "queue_wait_s": queue_wait_s,
            "model_execution_s": active_execution_s,
            "server_model_inference_s": model_inference_s,
            "reported_queue_wait_s": reported_queue_wait_s,
            "timing_source": timing_source,
            "serialization_or_client_overhead_s": client_overhead_s,
            "serialized_response_bytes": serialized_bytes,
            "flow_or_image_execution_s": sum(flow_values) if flow_values else None,
            "stop_reason": meta["finish_reason"],
            "contract": contract,
            "reference": {
                "contract": contract,
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            },
        }

    out_dir = config.out_dir / "native"
    levels, reference = _run_levels(
        config=config,
        request_fn=request_fn,
        progress_path=out_dir / "progress.json",
    )
    result = {
        "ok": True,
        "backend": "main_sglang_scheduler_owned_interleave",
        "server_url": server_url,
        "benchmark_run_id": benchmark_run_id,
        "concurrency_methodology": (
            "All clients issue simultaneous HTTP requests. SGLang owns parent "
            "decode batching, parent parking, isolated internal flow children, "
            "image re-encoding, and resumed text decode."
        ),
        "settings": config.shared_settings(),
        "levels": levels,
        "reference": reference,
    }
    _write_json(out_dir / "results.json", result)
    return 0


def _image_metrics(
    hf_bytes: bytes, native_bytes: bytes, shape: list[int]
) -> dict[str, Any]:
    import numpy as np

    hf = np.frombuffer(hf_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
    native = (
        np.frombuffer(native_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
    )
    hf_rgb = np.clip(hf[0] * 0.5 + 0.5, 0.0, 1.0)
    native_rgb = np.clip(native[0] * 0.5 + 0.5, 0.0, 1.0)
    diff = native_rgb - hf_rgb
    mse = float(np.mean(diff**2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    x = hf_rgb.reshape(-1).astype(np.float64)
    y = native_rgb.reshape(-1).astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    ux = float(x.mean())
    uy = float(y.mean())
    vx = float(((x - ux) ** 2).mean())
    vy = float(((y - uy) ** 2).mean())
    covariance = float(((x - ux) * (y - uy)).mean())
    ssim = ((2 * ux * uy + c1) * (2 * covariance + c2)) / (
        (ux**2 + uy**2 + c1) * (vx + vy + c2)
    )
    return {
        "mse_float_rgb": mse,
        "psnr_db": psnr,
        "ssim_global_rgb": float(ssim),
        "max_abs_diff_float_rgb": float(np.max(np.abs(diff))),
        "mean_abs_diff_float_rgb": float(np.mean(np.abs(diff))),
        "exact_float16_tensor": hf_bytes == native_bytes,
        "hf_image_sha256": _sha256(hf_bytes),
        "native_image_sha256": _sha256(native_bytes),
    }


def _speedup(hf_value: float, native_value: float) -> float:
    return float(hf_value) / float(native_value)


def _aggregated_level(level: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in level.items() if key != "raw_repeats"}


def _determinism_summary(
    result: dict[str, Any],
    concurrency_levels: tuple[int, ...],
) -> dict[str, Any]:
    per_level_contract_digests: dict[str, list[str]] = {}
    all_contract_digests: set[str] = set()
    all_text_digests: set[str] = set()
    all_image_digests: set[str] = set()
    logical_ids_by_level: dict[str, set[str]] = {}
    logical_digests: dict[str, dict[str, set[str]]] = {}
    repeat_deterministic = True
    for concurrency in concurrency_levels:
        key = str(concurrency)
        requests = [
            request
            for repeat in result["levels"][key]["raw_repeats"]
            for request in repeat["requests"]
        ]
        contract_digests = {
            str(request["contract"]["contract_sha256"]) for request in requests
        }
        text_digests = {
            _canonical_digest(
                {
                    "pre_image_token_ids": request["contract"]["pre_image_token_ids"],
                    "post_image_token_ids": request["contract"]["post_image_token_ids"],
                }
            )
            for request in requests
        }
        image_digests = {
            str(request["contract"]["image_sha256"]) for request in requests
        }
        logical_ids = {
            str(request["logical_request_id"])
            for request in requests
            if request.get("logical_request_id") is not None
        }
        logical_ids_by_level[key] = logical_ids
        for request in requests:
            logical_request_id = request.get("logical_request_id")
            if logical_request_id is None:
                continue
            logical_digests.setdefault(str(logical_request_id), {}).setdefault(
                key, set()
            ).add(str(request["contract"]["contract_sha256"]))
        per_level_contract_digests[key] = sorted(contract_digests)
        repeat_deterministic = repeat_deterministic and len(contract_digests) == 1
        all_contract_digests.update(contract_digests)
        all_text_digests.update(text_digests)
        all_image_digests.update(image_digests)
    common_logical_ids = (
        set.intersection(*logical_ids_by_level.values())
        if logical_ids_by_level
        else set()
    )
    common_logical_request_digests = {
        logical_request_id: {
            key: sorted(logical_digests[logical_request_id][key])
            for key in sorted(logical_digests[logical_request_id])
        }
        for logical_request_id in sorted(common_logical_ids)
    }
    logical_request_batch_size_invariant = bool(common_logical_ids) and all(
        len(
            {
                digest
                for per_level in common_logical_request_digests[
                    logical_request_id
                ].values()
                for digest in per_level
            }
        )
        == 1
        for logical_request_id in common_logical_ids
    )
    return {
        "repeat_deterministic_per_level": repeat_deterministic,
        "contract_batch_size_invariant": len(all_contract_digests) == 1,
        "text_batch_size_invariant": len(all_text_digests) == 1,
        "image_batch_size_invariant": len(all_image_digests) == 1,
        "logical_request_batch_size_invariant": (logical_request_batch_size_invariant),
        "common_logical_request_digests": common_logical_request_digests,
        "unique_contract_digests_across_levels": sorted(all_contract_digests),
        "unique_text_digests_across_levels": sorted(all_text_digests),
        "unique_image_digests_across_levels": sorted(all_image_digests),
        "per_level_contract_digests": per_level_contract_digests,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SenseNova U1 End-to-End Interleave Benchmark",
        "",
        (
            "Headline metric: total client-observed end-to-end wall latency. "
            "All rows are measured; no batch/concurrency value is estimated."
        ),
        "",
        "## Headline comparison",
        "",
        "| Concurrency | HF request median / P95 (s) | Native request median / P95 (s) | E2E latency speedup | HF batch wall median / P95 (s) | Native batch wall median / P95 (s) | Batch wall speedup | HF req/s | Native req/s | Throughput speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, row in summary["levels"].items():
        hf = row["hf"]
        native = row["native"]
        speed = row["speedup"]
        lines.append(
            f"| {level} | "
            f"{hf['per_request_latency_s']['median']:.4f} / "
            f"{hf['per_request_latency_s']['p95']:.4f} | "
            f"{native['per_request_latency_s']['median']:.4f} / "
            f"{native['per_request_latency_s']['p95']:.4f} | "
            f"{speed['request_latency_median']:.3f}x | "
            f"{hf['batch_wall_s']['median']:.4f} / "
            f"{hf['batch_wall_s']['p95']:.4f} | "
            f"{native['batch_wall_s']['median']:.4f} / "
            f"{native['batch_wall_s']['p95']:.4f} | "
            f"{speed['batch_wall_median']:.3f}x | "
            f"{hf['requests_per_s']:.4f} | {native['requests_per_s']:.4f} | "
            f"{speed['requests_per_s']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Execution and resource detail",
            "",
            "| Backend | Concurrency | Model execution median / P95 (s) | Queue median / P95 (s) | Serialization/client overhead median / P95 (s) | Text tokens/s | Image turns/s | Peak GPU used (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level, row in summary["levels"].items():
        for backend in ("hf", "native"):
            data = row[backend]
            peak = data["gpu_memory"]["peak_used_mib"]
            lines.append(
                f"| {backend.upper()} | {level} | "
                f"{data['model_execution_s']['median']:.4f} / "
                f"{data['model_execution_s']['p95']:.4f} | "
                f"{data['queue_wait_s']['median']:.4f} / "
                f"{data['queue_wait_s']['p95']:.4f} | "
                f"{data['serialization_or_client_overhead_s']['median']:.4f} / "
                f"{data['serialization_or_client_overhead_s']['p95']:.4f} | "
                f"{data['text_tokens_per_s']:.3f} | "
                f"{data['image_turns_per_s']:.4f} | "
                f"{peak if peak is not None else 'n/a'} |"
            )

    correctness = summary["correctness"]
    lines.extend(
        [
            "",
            "## Correctness and methodology",
            "",
            f"- Real text → `<img>` → flow → re-encode → continued-text turn: **{correctness['real_text_image_text_turn']}**",
            (
                f"- Effective generated text tokens: "
                f"**{correctness['generated_text_tokens']}** "
                f"({correctness['pre_image_tokens']} before image, "
                f"{correctness['post_image_tokens']} after image)."
            ),
            (
                "- Cross-backend pre-image / post-image text token exact: "
                f"**{correctness['pre_image_text_token_exact']}** / "
                f"**{correctness['post_image_text_token_exact']}**."
            ),
            (
                "- Shared output contract satisfied: "
                f"**{correctness['output_contract_match']}**."
            ),
            (
                f"- Image PSNR: "
                f"**{correctness['image_metrics']['psnr_db']:.4f} dB**; "
                f"SSIM: "
                f"**{correctness['image_metrics']['ssim_global_rgb']:.6f}**."
            ),
            f"- HF deterministic across all measured concurrency levels: **{correctness['hf_deterministic']}**.",
            f"- Native deterministic across all measured concurrency levels: **{correctness['native_deterministic']}**.",
            (
                "- HF output contract batch-size invariant: "
                f"**{correctness['hf_batch_size_invariant']}**."
            ),
            (
                "- Native output contract batch-size invariant: "
                f"**{correctness['native_batch_size_invariant']}** "
                f"(text: **{correctness['native_text_batch_size_invariant']}**; "
                f"image: **{correctness['native_image_batch_size_invariant']}**)."
            ),
            (
                "- Native same-logical-request invariant across BS1/2/4/8: "
                f"**{correctness['native_logical_request_batch_size_invariant']}**."
            ),
            (
                "- HF concurrency methodology: simultaneous clients enter a "
                "bounded single-model queue because official `interleave_gen` "
                "mutates shared KV/current-index state and has no native "
                "batched/concurrent API."
            ),
            (
                "- Native methodology: simultaneous HTTP clients use main-SGLang "
                "continuous batching with scheduler-owned parent/child interleave."
            ),
            (
                "- Each concurrency level uses the configured warmup batches "
                "followed by measured repeats; model load is excluded."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _summarize(config: BenchmarkConfig) -> int:
    hf = json.loads((config.out_dir / "hf/results.json").read_text(encoding="utf-8"))
    native = json.loads(
        (config.out_dir / "native/results.json").read_text(encoding="utf-8")
    )
    if hf["settings"] != native["settings"]:
        raise RuntimeError("HF and native benchmark settings differ")

    hf_ref = hf["reference"]
    native_ref = native["reference"]
    hf_contract = hf_ref["contract"]
    native_contract = native_ref["contract"]
    hf_image = base64.b64decode(hf_ref["image_b64"])
    native_image = base64.b64decode(native_ref["image_b64"])
    if hf_contract["image_shape"] != native_contract["image_shape"]:
        raise RuntimeError("HF and native image shapes differ")
    image_metrics = _image_metrics(
        hf_image,
        native_image,
        hf_contract["image_shape"],
    )
    hf_determinism = _determinism_summary(hf, config.concurrency_levels)
    native_determinism = _determinism_summary(native, config.concurrency_levels)

    levels: dict[str, Any] = {}
    for concurrency in config.concurrency_levels:
        key = str(concurrency)
        hf_level = hf["levels"][key]
        native_level = native["levels"][key]
        levels[key] = {
            "hf": _aggregated_level(hf_level),
            "native": _aggregated_level(native_level),
            "speedup": {
                "request_latency_median": _speedup(
                    hf_level["per_request_latency_s"]["median"],
                    native_level["per_request_latency_s"]["median"],
                ),
                "request_latency_p95": _speedup(
                    hf_level["per_request_latency_s"]["p95"],
                    native_level["per_request_latency_s"]["p95"],
                ),
                "batch_wall_median": _speedup(
                    hf_level["batch_wall_s"]["median"],
                    native_level["batch_wall_s"]["median"],
                ),
                "batch_wall_p95": _speedup(
                    hf_level["batch_wall_s"]["p95"],
                    native_level["batch_wall_s"]["p95"],
                ),
                "model_execution_median": _speedup(
                    hf_level["model_execution_s"]["median"],
                    native_level["model_execution_s"]["median"],
                ),
                "requests_per_s": _speedup(
                    native_level["requests_per_s"],
                    hf_level["requests_per_s"],
                ),
                "text_tokens_per_s": _speedup(
                    native_level["text_tokens_per_s"],
                    hf_level["text_tokens_per_s"],
                ),
                "image_turns_per_s": _speedup(
                    native_level["image_turns_per_s"],
                    hf_level["image_turns_per_s"],
                ),
            },
        }

    hf_deterministic = hf_determinism["repeat_deterministic_per_level"]
    native_deterministic = native_determinism["repeat_deterministic_per_level"]
    pre_image_text_token_exact = (
        hf_contract["pre_image_token_ids"] == native_contract["pre_image_token_ids"]
    )
    post_image_text_token_exact = (
        hf_contract["post_image_token_ids"] == native_contract["post_image_token_ids"]
    )
    output_contract_match = bool(
        hf_contract["generated_text_tokens"]
        == native_contract["generated_text_tokens"]
        == config.text_token_budget
        and hf_contract["image_turns"] == native_contract["image_turns"] == 1
        and hf_contract["image_shape"] == native_contract["image_shape"]
        and hf_contract["pre_image_token_ids"]
        and native_contract["pre_image_token_ids"]
        and hf_contract["post_image_token_ids"]
        and native_contract["post_image_token_ids"]
    )
    correctness = {
        "real_text_image_text_turn": bool(
            hf_contract["real_text_image_text_turn"]
            and native_contract["real_text_image_text_turn"]
        ),
        "generated_text_tokens": hf_contract["generated_text_tokens"],
        "native_generated_text_tokens": native_contract["generated_text_tokens"],
        "pre_image_tokens": len(hf_contract["pre_image_token_ids"]),
        "post_image_tokens": len(hf_contract["post_image_token_ids"]),
        "output_contract_match": output_contract_match,
        "pre_image_text_token_exact": pre_image_text_token_exact,
        "post_image_text_token_exact": post_image_text_token_exact,
        "text_token_exact": bool(
            pre_image_text_token_exact and post_image_text_token_exact
        ),
        "image_metrics": image_metrics,
        "hf_deterministic": hf_deterministic,
        "native_deterministic": native_deterministic,
        "hf_batch_size_invariant": hf_determinism["contract_batch_size_invariant"],
        "native_batch_size_invariant": native_determinism[
            "contract_batch_size_invariant"
        ],
        "native_text_batch_size_invariant": native_determinism[
            "text_batch_size_invariant"
        ],
        "native_image_batch_size_invariant": native_determinism[
            "image_batch_size_invariant"
        ],
        "hf_logical_request_batch_size_invariant": hf_determinism[
            "logical_request_batch_size_invariant"
        ],
        "native_logical_request_batch_size_invariant": native_determinism[
            "logical_request_batch_size_invariant"
        ],
        "determinism_detail": {
            "hf": hf_determinism,
            "native": native_determinism,
        },
        "all_hf_levels_real_turn": all(
            hf["levels"][str(level)]["correctness"]["all_real_text_image_text_turns"]
            for level in config.concurrency_levels
        ),
        "all_native_levels_real_turn": all(
            native["levels"][str(level)]["correctness"][
                "all_real_text_image_text_turns"
            ]
            for level in config.concurrency_levels
        ),
    }
    ok = bool(
        correctness["real_text_image_text_turn"]
        and correctness["output_contract_match"]
        and correctness["generated_text_tokens"] == config.text_token_budget
        and correctness["native_generated_text_tokens"] == config.text_token_budget
        and correctness["pre_image_tokens"] > 0
        and correctness["post_image_tokens"] > 0
        and correctness["hf_deterministic"]
        and correctness["native_deterministic"]
        and correctness["native_batch_size_invariant"]
        and correctness["native_logical_request_batch_size_invariant"]
        and correctness["all_hf_levels_real_turn"]
        and correctness["all_native_levels_real_turn"]
        and image_metrics["psnr_db"] >= 46.0
        and image_metrics["ssim_global_rgb"] >= 0.999
    )
    summary = {
        "ok": ok,
        "date": "2026-08-16",
        "repository": "sgl-project/sglang",
        "commit": _git_commit(),
        "headline_metric": "total_end_to_end_interleave_client_wall",
        "settings": config.shared_settings(),
        "methodology": {
            "percentile": "linear interpolation over measured samples",
            "model_load_excluded": True,
            "hf": hf["concurrency_methodology"],
            "hf_loader": hf["loader"],
            "hf_compatibility_root": hf["compatibility_root"],
            "hf_model_load_s": hf["model_load_s"],
            "native": native["concurrency_methodology"],
            "logical_request_mapping": (
                "For each measured repeat, request ordinal N has the same "
                "logical_request_id and explicit seed at every concurrency level. "
                "Transport RIDs remain unique only to prevent cross-run Radix reuse."
            ),
            "timing": {
                "hf_model_execution": (
                    "CUDA-synchronized official interleave_gen call, excluding "
                    "lock queueing and output-contract serialization."
                ),
                "native_model_execution": (
                    "Server request_finished_ts minus first forward_entry_time; "
                    "this includes scheduler-owned text decode, flow, re-encode, "
                    "and resumed decode, while excluding initial server queueing."
                ),
                "serialization_or_client_overhead": (
                    "Client-observed request wall minus server inference wall; "
                    "includes HTTP transfer, JSON/base64 handling, and contract "
                    "materialization."
                ),
            },
            "no_estimation_or_extrapolation": True,
        },
        "correctness": correctness,
        "levels": levels,
        "artifacts": {
            "hf_raw": "hf/results.json",
            "native_raw": "native/results.json",
            "markdown": "summary.md",
        },
    }
    _write_json(config.out_dir / "summary.json", summary)
    (config.out_dir / "summary.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    return 0 if ok else 1


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not levels or any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("concurrency levels must be positive")
    return levels


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_path=args.model_path,
        vendor_root=args.vendor_root,
        out_dir=Path(args.out_dir),
        prompt=args.prompt,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        text_token_budget=args.text_token_budget,
        native_max_new_tokens=args.native_max_new_tokens,
        max_images=1,
        seed=args.seed,
        concurrency_levels=args.concurrency_levels,
        warmup_repeats=args.warmup_repeats,
        measurement_repeats=args.measurement_repeats,
        gpu_index=args.gpu_index,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("hf", "native", "summarize"), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vendor-root", default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument(
        "--text-token-budget", type=int, default=DEFAULT_TEXT_TOKEN_BUDGET
    )
    parser.add_argument(
        "--native-max-new-tokens",
        type=int,
        default=DEFAULT_NATIVE_MAX_NEW_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--concurrency-levels", type=_parse_levels, default=(1, 2, 4, 8)
    )
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--hf-attn-backend", default="auto")
    parser.add_argument("--hf-compat-root")
    parser.add_argument("--server-url", default="http://127.0.0.1:31001")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("image dimensions must be positive")
    if args.width % 32 or args.height % 32:
        parser.error("image dimensions must be divisible by 32")
    if args.num_steps <= 0:
        parser.error("num_steps must be positive")
    if args.text_token_budget <= 0:
        parser.error("text token budget must be positive")
    if args.native_max_new_tokens != args.text_token_budget + 1:
        parser.error("native max_new_tokens must equal text token budget + 1")
    if args.warmup_repeats <= 0 or args.measurement_repeats < 5:
        parser.error("require at least one warmup and five measured repeats")
    return args


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.out_dir / "config.json",
        {
            "phase": args.phase,
            "settings": config.shared_settings(),
            "server_url": args.server_url,
            "hf_attn_backend": args.hf_attn_backend,
        },
    )
    if args.phase == "hf":
        return _run_hf(config, args.hf_attn_backend, args.hf_compat_root)
    if args.phase == "native":
        return _run_native(config, args.server_url)
    return _summarize(config)


if __name__ == "__main__":
    raise SystemExit(main())
