from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
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
        for raw_name in warmup_names:
            warmup_name = raw_name.strip() if raw_name else ""
            if not warmup_name:
                continue
            if warmup_name not in _warmup_registry:
                logger.warning(f"Could not find custom warmup {warmup_name}")
                continue
            logger.info(f"Running warmup {warmup_name}")
            await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)
    finally:
        _current_app_state.reset(token)


def _multi_worker_info(tokenizer_manager: Any) -> Optional[tuple[str, int, int]]:
    parent_pid = getattr(multiprocessing.current_process(), "_parent_pid", None)
    server_args = getattr(tokenizer_manager, "server_args", None)
    worker_num = getattr(server_args, "tokenizer_worker_num", 1)
    if parent_pid is None or not isinstance(worker_num, int) or worker_num <= 1:
        return None
    port = getattr(server_args, "port", 0)
    startup = getattr(tokenizer_manager, "startup_time", None)
    tag = hashlib.sha256(f"{parent_pid}:{port}:{startup!r}".encode()).hexdigest()[:12]
    prefix = os.path.join(
        tempfile.gettempdir(), f"sglang_warmup_{parent_pid}_{port}_{tag}"
    )
    return prefix, parent_pid, worker_num


def sync_multi_worker_ready_barrier(
    tokenizer_manager: Any, timeout_s: float = 0
) -> bool:
    """Wait until all multi-tokenizer HTTP workers finish warmup; returns True for the leader."""
    info = _multi_worker_info(tokenizer_manager)
    if info is None:
        return True
    prefix, main_pid, num_workers = info
    barrier_dir = f"{prefix}_ready"
    os.makedirs(barrier_dir, exist_ok=True)
    my_pid = os.getpid()
    open(os.path.join(barrier_dir, str(my_pid)), "w").close()

    deadline = time.monotonic() + (timeout_s if timeout_s > 0 else 600)
    live = [my_pid]
    while time.monotonic() < deadline:
        try:
            live = [
                int(e)
                for e in os.listdir(barrier_dir)
                if e.isdigit() and psutil.pid_exists(int(e))
            ]
        except OSError:
            live = []
        if len(live) >= num_workers:
            break
        if not psutil.pid_exists(main_pid):
            return False
        time.sleep(0.05)
    else:
        logger.warning(
            "Timed out waiting for %d tokenizer workers (%d ready); proceeding.",
            num_workers,
            len(live),
        )
    return bool(live) and my_pid == min(live)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
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
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

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
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat", rust_server=True)
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
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


def _prefill_shape_sizes(max_size: int = 32768) -> List[int]:
    page_size, sizes, base = 64, set(), 64
    while base <= max_size:
        sizes.add(base)
        mid = (base * 3 // 2 + page_size - 1) // page_size * page_size
        if mid <= max_size:
            sizes.add(mid)
        base *= 2
    return sorted(sizes)


@warmup("prefill_shapes", rust_server=True)
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    for size in tqdm.tqdm(
        _prefill_shape_sizes(), desc="Warmup prefill shapes (up to 32K)"
    ):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


async def _send(
    disaggregation_mode: str,
    tokenizer_manager: TokenizerManager,
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    *,
    room_id: int = 0,
    **kwargs: Any,
) -> None:
    req = GenerateReqInput(
        input_ids=input_ids, sampling_params=sampling_params, **kwargs
    )
    if disaggregation_mode != "null":
        req.bootstrap_room = room_id
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    async for _ in tokenizer_manager.generate_request(req, None):
        pass


async def _run_all_gpu_warmups(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
) -> None:
    vocab = getattr(getattr(tokenizer_manager, "model_config", None), "vocab_size", 0)
    high = (
        max(32, min(vocab - 16, 2**15))
        if isinstance(vocab, int) and vocab > 32
        else 1024
    )
    rng = np.random.default_rng(42)
    rand_ids = lambda n: [int(x) for x in rng.integers(16, high, size=n)]

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

    # 1. Sampling & penalties (bs=1 + concurrent mixed/dynamic bs=2)
    for size, params in (
        (16, greedy),
        (21, {**sampled, "max_new_tokens": 16}),
        (
            19,
            {"max_new_tokens": 4, "temperature": 0.8, "top_p": 0.9, "ignore_eos": True},
        ),
        (24, {**sampled, "temperature": 0.7, "top_k": 50, "min_p": 0.05}),
        (21, penalty),
    ):
        await _send(disaggregation_mode, tokenizer_manager, rand_ids(size), params)
    await asyncio.gather(
        _send(disaggregation_mode, tokenizer_manager, rand_ids(17), greedy, room_id=0),
        _send(disaggregation_mode, tokenizer_manager, rand_ids(23), sampled, room_id=1),
    )
    await asyncio.gather(
        _send(
            disaggregation_mode,
            tokenizer_manager,
            rand_ids(19),
            dict(penalty),
            room_id=0,
        ),
        _send(
            disaggregation_mode,
            tokenizer_manager,
            rand_ids(27),
            dict(penalty),
            room_id=1,
        ),
    )

    # 2. Output & prompt logprobs
    if getattr(tokenizer_manager, "is_generation", True):
        lp_kw = {
            "return_logprob": True,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
            "log_metrics": False,
        }
        await _send(
            disaggregation_mode, tokenizer_manager, rand_ids(16), greedy, **lp_kw
        )
        await _send(
            disaggregation_mode, tokenizer_manager, rand_ids(21), sampled, **lp_kw
        )
        if disaggregation_mode != "decode":
            await _send(
                disaggregation_mode,
                tokenizer_manager,
                rand_ids(25),
                {"max_new_tokens": 1, "temperature": 0.0},
                logprob_start_len=0,
                **lp_kw,
            )

    # 3. Structured output (FSM compiler + token bitmask kernel)
    if (
        not getattr(tokenizer_manager, "skip_tokenizer_init", False)
        and getattr(
            getattr(tokenizer_manager, "server_args", None), "grammar_backend", None
        )
        != "none"
    ):
        schema = '{"type":"object","properties":{"safe":{"type":"boolean"}},"required":["safe"]}'
        await _send(
            disaggregation_mode,
            tokenizer_manager,
            rand_ids(16),
            {"max_new_tokens": 8, "temperature": 0.0, "json_schema": schema},
        )
        await _send(
            disaggregation_mode,
            tokenizer_manager,
            rand_ids(16),
            {"max_new_tokens": 4, "temperature": 0.0, "regex": r"(safe|unsafe)"},
        )

    # 4. RadixCache prefix-hit extend (tile buckets + odd KV-slot alignments) & prefill shapes
    env_val = os.environ.get("SGLANG_WARMUP_MAX_PREFILL_SIZE")
    max_size = int(env_val) if env_val and env_val.isdigit() else 32768
    max_req = getattr(tokenizer_manager, "max_req_input_len", None)
    if isinstance(max_req, int) and max_req >= 64:
        max_size = min(max_size, max(64, max_req - 16))
    max_size = max(64, max_size)

    prefix = rand_ids(64)
    ext_params = {**sampled, "max_new_tokens": 2}
    for tail in (7, 2, 7, 21, 96):
        if 64 + tail <= max_size:
            await _send(
                disaggregation_mode,
                tokenizer_manager,
                prefix + rand_ids(tail),
                ext_params,
            )

    shape_seqs = [rand_ids(s) for s in (3, 7, 21, 48) if s <= max_size]
    shape_seqs += [
        list(prefix) if s == 64 else rand_ids(s) for s in _prefill_shape_sizes(max_size)
    ]
    shape_params = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}
    for ids in tqdm.tqdm(shape_seqs, desc=f"Warmup prefill shapes (up to {max_size})"):
        await _send(disaggregation_mode, tokenizer_manager, ids, shape_params)

    # Flush synthetic warmup KV / Mamba cache before serving traffic
    flush_fn = getattr(tokenizer_manager, "flush_cache", None)
    if (
        disaggregation_mode == "null"
        and getattr(tokenizer_manager, "is_generation", True)
        and callable(flush_fn)
    ):
        try:
            res = flush_fn()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.debug("Post-warmup flush_cache skipped", exc_info=True)


async def _warm_local_worker(tokenizer_manager: Any, app_state: Optional[Any]) -> None:
    if getattr(tokenizer_manager, "skip_tokenizer_init", False):
        return

    dyn_tok = getattr(tokenizer_manager, "dynamic_batch_tokenizer", None)
    tok = getattr(tokenizer_manager, "tokenizer", None)
    try:
        if dyn_tok is not None:
            await dyn_tok.encode("Warmup dynamic batch tokenizer.")
        elif tok is not None and hasattr(tok, "encode"):
            await asyncio.to_thread(tok.encode, "Warmup tokenizer.")
    except Exception:
        logger.debug("Tokenizer encode warmup skipped", exc_info=True)

    if tok is not None:
        try:
            from sglang.srt.entrypoints.openai.utils import token_id_to_bytes

            await asyncio.to_thread(token_id_to_bytes, tok, 0)
        except Exception:
            logger.debug("Tokenizer byte-level cache warmup skipped", exc_info=True)

    serving_chat = (
        getattr(app_state, "openai_serving_chat", None) if app_state else None
    )
    if serving_chat is not None and getattr(tokenizer_manager, "is_generation", True):
        try:
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

            chat_req = ChatCompletionRequest(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Warmup chat template."},
                ],
                max_tokens=2,
                temperature=0.6,
                top_p=0.95,
                logprobs=True,
                top_logprobs=2,
            )
            await asyncio.to_thread(
                serving_chat._convert_to_internal_request, chat_req, None
            )
        except Exception:
            logger.debug("OpenAIServingChat template warmup skipped", exc_info=True)


@warmup("all")
async def warmup_all(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Comprehensive warmup covering sampling, penalties, logprobs, structured
    output, RadixCache prefix hits, prefill shapes, and per-worker tokenizers."""
    info = _multi_worker_info(tokenizer_manager) if fcntl is not None else None
    if info is None:
        await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
    else:
        prefix, _, _ = info
        with open(f"{prefix}_all.lock", "a+", encoding="utf-8") as f:
            await asyncio.to_thread(fcntl.flock, f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            if not f.read():
                await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
                f.write("1")
                f.flush()
            else:
                logger.info(
                    "Skipping GPU warmup in worker %d (completed by peer)", os.getpid()
                )
    await _warm_local_worker(tokenizer_manager, _current_app_state.get())
