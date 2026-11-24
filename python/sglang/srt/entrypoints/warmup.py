from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import math
import multiprocessing
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import psutil
import tqdm

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}
# Warmups that only send generate requests, which Rust-server mode runs through
# its listener (`rust_server.warmups`); the others need the TokenizerManager.
_rust_server_warmups = set()
_current_app_state: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_current_app_state", default=None
)
_DEFAULT_MAX_PREFILL_SIZE = 32768
_DEFAULT_BARRIER_TIMEOUT_S = 600.0


def warmup(name: str, *, rust_server: bool = False):
    def decorator(fn):
        _warmup_registry[name] = fn
        if rust_server:
            _rust_server_warmups.add(name)
        else:
            _rust_server_warmups.discard(name)
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
    app_state: Optional[Any] = None,
):
    token = _current_app_state.set(app_state)
    try:
        for raw in warmup_names:
            warmup_name = raw.strip() if raw else ""
            if not warmup_name:
                continue
            if warmup_name not in _warmup_registry:
                logger.warning(f"Could not find custom warmup {warmup_name}")
                continue
            logger.info(f"Running warmup {warmup_name}")
            await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)
    finally:
        _current_app_state.reset(token)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _multi_worker_info(tm: Any) -> Optional[tuple[str, int, int]]:
    """Return `(path_prefix, main_pid, num_workers)` when in multi-tokenizer mode."""
    parent_pid = getattr(multiprocessing.current_process(), "_parent_pid", None)
    server_args = getattr(tm, "server_args", None)
    num_workers = _positive_int(getattr(server_args, "tokenizer_worker_num", 1))
    if parent_pid is None or num_workers is None or num_workers <= 1:
        return None
    port = getattr(server_args, "port", 0)
    tag = hashlib.sha256(
        f"{parent_pid}:{port}:{getattr(tm, 'startup_time', None)!r}".encode()
    ).hexdigest()[:12]
    prefix = os.path.join(
        tempfile.gettempdir(), f"sglang_warmup_{parent_pid}_{port}_{tag}"
    )
    return prefix, parent_pid, num_workers


def sync_multi_worker_ready_barrier(tm: Any, timeout_s: float = 0) -> bool:
    """Wait until all tokenizer workers finish warmup; return True on the leader."""
    info = _multi_worker_info(tm)
    if info is None:
        return True
    prefix, main_pid, num_workers = info
    barrier_dir = f"{prefix}_ready"
    my_pid = os.getpid()
    os.makedirs(barrier_dir, exist_ok=True)
    with open(os.path.join(barrier_dir, str(my_pid)), "w"):
        pass

    effective_timeout = (
        timeout_s
        if isinstance(timeout_s, (int, float))
        and not isinstance(timeout_s, bool)
        and math.isfinite(timeout_s)
        and timeout_s > 0
        else _DEFAULT_BARRIER_TIMEOUT_S
    )
    deadline = time.monotonic() + effective_timeout
    live_pids = [my_pid]
    while time.monotonic() < deadline:
        try:
            live_pids = [
                int(entry)
                for entry in os.listdir(barrier_dir)
                if entry.isdigit() and psutil.pid_exists(int(entry))
            ]
        except OSError:
            live_pids = []
        main_alive = psutil.pid_exists(main_pid)
        if len(live_pids) >= num_workers or not main_alive:
            return main_alive and bool(live_pids) and my_pid == min(live_pids)
        time.sleep(0.05)

    logger.warning(
        "Timed out waiting for %d tokenizer workers (%d ready).",
        num_workers,
        len(live_pids),
    )
    return bool(live_pids) and my_pid == min(live_pids)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes."""
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_data_uri = (
        f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"
    )

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat", rust_server=True)
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST
        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


def _max_warmup_input_len(tm: Any, max_new_tokens: int = 1) -> Optional[int]:
    """Compute the largest safe input token length for a warmup request."""
    server_args = getattr(tm, "server_args", None)
    model_config = getattr(tm, "model_config", None)
    reserved_tokens = max(0, int(getattr(tm, "num_reserved_tokens", 0) or 0))
    output_tokens = max(1, int(max_new_tokens or 1))

    limits: List[int] = []
    for ctx in (
        getattr(tm, "context_len", None),
        getattr(model_config, "context_len", None),
        getattr(server_args, "context_length", None),
    ):
        ctx_len = _positive_int(ctx)
        if ctx_len is not None:
            limits.append(ctx_len - reserved_tokens - output_tokens)

    # Scheduler sets `max_req_input_len = min(context_len - 1, kv_capacity - 1) - 5`
    # and requires `len(input_ids) < max_req_input_len`.
    max_req_input_len = _positive_int(getattr(tm, "max_req_input_len", None))
    if max_req_input_len is not None:
        limits.append(
            min(
                max_req_input_len - 1,
                max_req_input_len + 5 - reserved_tokens - output_tokens,
            )
        )

    chunked_prefill_size = getattr(server_args, "chunked_prefill_size", None)
    max_prefill_tokens = _positive_int(getattr(server_args, "max_prefill_tokens", None))
    if (
        isinstance(chunked_prefill_size, int)
        and not isinstance(chunked_prefill_size, bool)
        and chunked_prefill_size <= 0
        and max_prefill_tokens is not None
    ):
        limits.append(max_prefill_tokens)

    return max(0, min(limits)) if limits else None


def _prefill_shape_sizes(max_size: int = _DEFAULT_MAX_PREFILL_SIZE) -> List[int]:
    sizes, base = set(), 64
    while base <= max_size:
        sizes.add(base)
        mid = (base * 3 // 2 + 63) // 64 * 64
        if mid <= max_size:
            sizes.add(mid)
        base *= 2
    return sorted(sizes)


@warmup("prefill_shapes", rust_server=True)
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    input_limit = _max_warmup_input_len(tokenizer_manager, max_new_tokens=1)
    max_size = (
        min(_DEFAULT_MAX_PREFILL_SIZE, input_limit)
        if input_limit is not None
        else _DEFAULT_MAX_PREFILL_SIZE
    )
    for size in tqdm.tqdm(
        _prefill_shape_sizes(max_size),
        desc=f"Warmup prefill shapes (up to {max_size})",
    ):
        req = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        await tokenizer_manager.generate_request(req, None).__anext__()


def _resolve_max_prefill_size(max_input_len: Optional[int]) -> int:
    max_size = _DEFAULT_MAX_PREFILL_SIZE
    env_val = os.environ.get("SGLANG_WARMUP_MAX_PREFILL_SIZE")
    if env_val is not None:
        stripped = env_val.strip()
        if stripped.isdigit():
            max_size = int(stripped)
        elif stripped:
            logger.warning(
                "Ignoring invalid SGLANG_WARMUP_MAX_PREFILL_SIZE=%r; using default %d.",
                env_val,
                _DEFAULT_MAX_PREFILL_SIZE,
            )
    if max_input_len is not None:
        # Leave a small margin below `max_input_len` when context is not tiny.
        safe_cap = max_input_len - 15 if max_input_len > 80 else max_input_len
        max_size = min(max_size, safe_cap)
    return max(0, max_size)


async def _send_warmup_request(
    disaggregation_mode: str,
    tm: TokenizerManager,
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    bootstrap_room: int = 0,
    **kwargs: Any,
) -> None:
    params = dict(sampling_params)
    base_limit = _max_warmup_input_len(tm, max_new_tokens=1)
    if base_limit is not None:
        if base_limit < 1:
            return
        max_new_tokens = max(1, int(params.get("max_new_tokens", 1) or 1))
        input_limit = _max_warmup_input_len(tm, max_new_tokens) or 0
        if input_limit < 4 and max_new_tokens > 1:
            max_new_tokens = max(1, min(max_new_tokens, (base_limit + 1) // 2))
            params["max_new_tokens"] = max_new_tokens
            input_limit = _max_warmup_input_len(tm, max_new_tokens) or 0
        if input_limit < 1:
            return
        input_ids = input_ids[:input_limit]

    if not input_ids:
        return
    if kwargs.get("logprob_start_len", -1) >= len(input_ids):
        kwargs["logprob_start_len"] = max(0, len(input_ids) - 1)

    req = GenerateReqInput(input_ids=input_ids, sampling_params=params, **kwargs)
    if disaggregation_mode != "null":
        req.bootstrap_room = bootstrap_room
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    async for _ in tm.generate_request(req, None):
        pass


async def _run_all_gpu_warmups(disaggregation_mode: str, tm: TokenizerManager) -> None:
    if not getattr(tm, "is_generation", True):
        logger.info("Skipping GPU generation warmup for non-generation model.")
        return

    vocab_size = _positive_int(
        getattr(getattr(tm, "model_config", None), "vocab_size", None)
    )
    if vocab_size is None:
        token_low, token_high = 16, 1024
    elif vocab_size > 32:
        token_low, token_high = 16, min(vocab_size - 16, 2**15)
    else:
        token_low, token_high = 0, vocab_size

    rng = np.random.default_rng(42)

    def rand_ids(length: int) -> List[int]:
        return [int(x) for x in rng.integers(token_low, token_high, size=length)]

    def send(
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        bootstrap_room: int = 0,
        **kwargs: Any,
    ):
        return _send_warmup_request(
            disaggregation_mode,
            tm,
            input_ids,
            sampling_params,
            bootstrap_room,
            **kwargs,
        )

    greedy = {"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True}
    sampled = {
        "max_new_tokens": 4,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "ignore_eos": True,
    }
    penalty = {
        **sampled,
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }
    max_input_len = _max_warmup_input_len(tm, max_new_tokens=1)

    # 1. Sampling, decode batching, and penalty paths.
    for length, params in (
        (16, greedy),
        (21, {**sampled, "max_new_tokens": 16}),
        (
            19,
            {"max_new_tokens": 4, "temperature": 0.8, "top_p": 0.9, "ignore_eos": True},
        ),
        (24, {**sampled, "temperature": 0.7, "top_k": 50, "min_p": 0.05}),
        (21, penalty),
    ):
        await send(rand_ids(length), params)

    if max_input_len is None or max_input_len >= 64:
        await asyncio.gather(
            send(rand_ids(17), greedy, bootstrap_room=0),
            send(rand_ids(23), sampled, bootstrap_room=1),
        )
        await asyncio.gather(
            send(rand_ids(19), penalty, bootstrap_room=0),
            send(rand_ids(27), penalty, bootstrap_room=1),
        )

    # 2. Output and prompt logprob paths.
    top_logprobs = min(5, vocab_size) if vocab_size is not None else 5
    logprob_kwargs = {
        "return_logprob": True,
        "top_logprobs_num": top_logprobs,
        "return_text_in_logprobs": True,
        "log_metrics": False,
    }
    await send(rand_ids(16), greedy, **logprob_kwargs)
    await send(rand_ids(21), sampled, **logprob_kwargs)
    if disaggregation_mode != "decode":
        await send(
            rand_ids(25),
            {"max_new_tokens": 1, "temperature": 0.0},
            logprob_start_len=0,
            **logprob_kwargs,
        )

    # 3. Constrained decoding (JSON schema & regex FSMs).
    grammar_backend = getattr(getattr(tm, "server_args", None), "grammar_backend", None)
    if not getattr(tm, "skip_tokenizer_init", False) and grammar_backend != "none":
        schema = '{"type":"object","properties":{"safe":{"type":"boolean"}},"required":["safe"]}'
        await send(
            rand_ids(16),
            {"max_new_tokens": 8, "temperature": 0.0, "json_schema": schema},
        )
        await send(
            rand_ids(16),
            {"max_new_tokens": 4, "temperature": 0.0, "regex": r"(safe|unsafe)"},
        )

    # 4. Radix prefix-cache extend and prefill shapes.
    max_prefill_size = _resolve_max_prefill_size(max_input_len)
    extend_cap = _max_warmup_input_len(tm, max_new_tokens=2)
    extend_max = (
        min(max_prefill_size, extend_cap)
        if extend_cap is not None
        else max_prefill_size
    )
    prefix_len = min(64, max(0, extend_max - 2))
    prefix = rand_ids(prefix_len) if prefix_len >= 4 else []
    for tail_len in (7, 2, 7, 21, 96) if prefix else ():
        if prefix_len + tail_len <= extend_max:
            await send(prefix + rand_ids(tail_len), {**sampled, "max_new_tokens": 2})

    prefill_seqs = [
        rand_ids(size) for size in (3, 7, 21, 48) if size <= max_prefill_size
    ]
    prefill_seqs += [
        list(prefix) if (size == prefix_len and prefix) else rand_ids(size)
        for size in _prefill_shape_sizes(max_prefill_size)
    ]
    if prefill_seqs:
        for input_ids in tqdm.tqdm(
            prefill_seqs, desc=f"Warmup prefill shapes (up to {max_prefill_size})"
        ):
            await send(
                input_ids,
                {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            )

    # 5. Flush synthetic warmup entries from the radix KV cache.
    flush_fn = getattr(tm, "flush_cache", None)
    if disaggregation_mode == "null" and callable(flush_fn):
        try:
            res = flush_fn()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.debug("Post-warmup flush_cache skipped", exc_info=True)


async def _warm_local_worker(tm: Any, app_state: Optional[Any]) -> None:
    """Warm per-process tokenizer, byte-level BPE cache, and chat template state."""
    if getattr(tm, "skip_tokenizer_init", False):
        return

    dyn_tokenizer = getattr(tm, "async_dynamic_batch_tokenizer", None)
    tokenizer = getattr(tm, "tokenizer", None)
    try:
        if dyn_tokenizer is not None:
            await dyn_tokenizer.encode("Warmup dynamic batch tokenizer.")
        elif tokenizer is not None and hasattr(tokenizer, "encode"):
            await asyncio.to_thread(tokenizer.encode, "Warmup tokenizer.")
    except Exception:
        logger.debug("Tokenizer warmup skipped", exc_info=True)

    if tokenizer is not None:
        try:
            from sglang.srt.entrypoints.openai.utils import token_id_to_bytes

            await asyncio.to_thread(token_id_to_bytes, tokenizer, 0)
        except Exception:
            logger.debug("Byte-level cache warmup skipped", exc_info=True)

    serving_chat = (
        getattr(app_state, "openai_serving_chat", None) if app_state else None
    )
    if serving_chat is not None and getattr(tm, "is_generation", True):
        try:
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

            req = ChatCompletionRequest(
                messages=[
                    {"role": "system", "content": "Assistant."},
                    {"role": "user", "content": "Hi."},
                ],
                max_tokens=2,
                temperature=0.6,
                top_p=0.95,
                logprobs=True,
                top_logprobs=2,
            )
            await asyncio.to_thread(
                serving_chat._convert_to_internal_request, req, None
            )
        except Exception:
            logger.debug("Chat template warmup skipped", exc_info=True)


@warmup("all")
async def warmup_all(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    info = _multi_worker_info(tokenizer_manager) if fcntl is not None else None
    if info is None:
        await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
    else:
        with open(f"{info[0]}_all.lock", "a+", encoding="utf-8") as lock_file:
            await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX)
            lock_file.seek(0)
            if not lock_file.read():
                await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
                lock_file.write("1")
                lock_file.flush()
    await _warm_local_worker(tokenizer_manager, _current_app_state.get())
