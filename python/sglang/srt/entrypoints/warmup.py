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


def _multi_worker_info(tm: Any) -> Optional[tuple[str, int, int]]:
    ppid = getattr(multiprocessing.current_process(), "_parent_pid", None)
    args = getattr(tm, "server_args", None)
    n = getattr(args, "tokenizer_worker_num", 1)
    if ppid is None or not isinstance(n, int) or n <= 1:
        return None
    port = getattr(args, "port", 0)
    tag = hashlib.sha256(
        f"{ppid}:{port}:{getattr(tm, 'startup_time', None)!r}".encode()
    ).hexdigest()[:12]
    return (
        os.path.join(tempfile.gettempdir(), f"sglang_warmup_{ppid}_{port}_{tag}"),
        ppid,
        n,
    )


def sync_multi_worker_ready_barrier(tm: Any, timeout_s: float = 0) -> bool:
    info = _multi_worker_info(tm)
    if info is None:
        return True
    prefix, main_pid, num_workers = info
    bdir, my_pid = f"{prefix}_ready", os.getpid()
    os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, str(my_pid)), "w").close()
    deadline, live = time.monotonic() + (timeout_s if timeout_s > 0 else 600), [my_pid]
    while time.monotonic() < deadline:
        try:
            live = [
                int(e)
                for e in os.listdir(bdir)
                if e.isdigit() and psutil.pid_exists(int(e))
            ]
        except OSError:
            live = []
        if len(live) >= num_workers or not psutil.pid_exists(main_pid):
            return psutil.pid_exists(main_pid) and bool(live) and my_pid == min(live)
        time.sleep(0.05)
    logger.warning(
        "Timed out waiting for %d tokenizer workers (%d ready).", num_workers, len(live)
    )
    return bool(live) and my_pid == min(live)


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


def _max_warmup_input_len(tm: Any, max_new: int = 1) -> Optional[int]:
    args, cfg = getattr(tm, "server_args", None), getattr(tm, "model_config", None)
    res = max(0, int(getattr(tm, "num_reserved_tokens", 0) or 0))
    out = max(1, int(max_new or 1))
    lims = [
        ctx - res - out
        for ctx in (
            getattr(tm, "context_len", None),
            getattr(cfg, "context_len", None),
            getattr(args, "context_length", None),
        )
        if isinstance(ctx, int) and ctx > 0
    ]
    if isinstance(mr := getattr(tm, "max_req_input_len", None), int) and mr > 0:
        lims.append(min(mr - 1, mr + 5 - res - out))
    cs, mp = (
        getattr(args, "chunked_prefill_size", None),
        getattr(args, "max_prefill_tokens", None),
    )
    if isinstance(cs, int) and cs <= 0 and isinstance(mp, int) and mp > 0:
        lims.append(mp)
    return max(0, min(lims)) if lims else None


def _prefill_shape_sizes(max_size: int = 32768) -> List[int]:
    sizes, base = set(), 64
    while base <= max_size:
        sizes.add(base)
        if (mid := (base * 3 // 2 + 63) // 64 * 64) <= max_size:
            sizes.add(mid)
        base *= 2
    return sorted(sizes)


@warmup("prefill_shapes", rust_server=True)
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    cap = _max_warmup_input_len(tokenizer_manager, 1)
    max_size = min(32768, cap) if cap is not None else 32768
    for size in tqdm.tqdm(
        _prefill_shape_sizes(max_size), desc=f"Warmup prefill shapes (up to {max_size})"
    ):
        req = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        if disaggregation_mode != "null":
            req.bootstrap_room, req.bootstrap_host = 0, FAKE_BOOTSTRAP_HOST
        await tokenizer_manager.generate_request(req, None).__anext__()


async def _send(
    mode: str,
    tm: TokenizerManager,
    ids: List[int],
    params: Dict[str, Any],
    room: int = 0,
    **kw: Any,
) -> None:
    params = dict(params)
    b1 = _max_warmup_input_len(tm, 1)
    if b1 is not None:
        if b1 < 1:
            return
        max_new = max(1, int(params.get("max_new_tokens", 1) or 1))
        lim = _max_warmup_input_len(tm, max_new) or 0
        if lim < 4 and max_new > 1:
            params["max_new_tokens"] = max_new = max(1, min(max_new, (b1 + 1) // 2))
            lim = _max_warmup_input_len(tm, max_new) or 0
        if lim < 1:
            return
        ids = ids[:lim]
    if kw.get("logprob_start_len", -1) >= len(ids):
        kw["logprob_start_len"] = max(0, len(ids) - 1)
    req = GenerateReqInput(input_ids=ids, sampling_params=params, **kw)
    if mode != "null":
        req.bootstrap_room, req.bootstrap_host = room, FAKE_BOOTSTRAP_HOST
    async for _ in tm.generate_request(req, None):
        pass


async def _run_all_gpu_warmups(mode: str, tm: TokenizerManager) -> None:
    vocab = getattr(getattr(tm, "model_config", None), "vocab_size", 0)
    high = (
        max(32, min(vocab - 16, 2**15))
        if isinstance(vocab, int) and vocab > 32
        else 1024
    )
    rng = np.random.default_rng(42)

    def rids(n: int) -> List[int]:
        return [int(x) for x in rng.integers(16, high, size=n)]

    def send(ids: List[int], p: Dict[str, Any], room: int = 0, **kw: Any):
        return _send(mode, tm, ids, p, room, **kw)

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
    max_in = _max_warmup_input_len(tm, 1)

    for n, p in (
        (16, greedy),
        (21, {**sampled, "max_new_tokens": 16}),
        (
            19,
            {"max_new_tokens": 4, "temperature": 0.8, "top_p": 0.9, "ignore_eos": True},
        ),
        (24, {**sampled, "temperature": 0.7, "top_k": 50, "min_p": 0.05}),
        (21, penalty),
    ):
        await send(rids(n), p)
    if max_in is None or max_in >= 64:
        await asyncio.gather(send(rids(17), greedy, 0), send(rids(23), sampled, 1))
        await asyncio.gather(send(rids(19), penalty, 0), send(rids(27), penalty, 1))

    if getattr(tm, "is_generation", True):
        lp = {
            "return_logprob": True,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
            "log_metrics": False,
        }
        await send(rids(16), greedy, **lp)
        await send(rids(21), sampled, **lp)
        if mode != "decode":
            await send(
                rids(25),
                {"max_new_tokens": 1, "temperature": 0.0},
                logprob_start_len=0,
                **lp,
            )

    if (
        not getattr(tm, "skip_tokenizer_init", False)
        and getattr(getattr(tm, "server_args", None), "grammar_backend", None) != "none"
    ):
        schema = '{"type":"object","properties":{"safe":{"type":"boolean"}},"required":["safe"]}'
        await send(
            rids(16), {"max_new_tokens": 8, "temperature": 0.0, "json_schema": schema}
        )
        await send(
            rids(16),
            {"max_new_tokens": 4, "temperature": 0.0, "regex": r"(safe|unsafe)"},
        )

    env_val = os.environ.get("SGLANG_WARMUP_MAX_PREFILL_SIZE")
    max_size = int(env_val) if env_val and env_val.isdigit() else 32768
    if max_in is not None:
        max_size = min(max_size, max_in - 15 if max_in > 80 else max_in)

    ext_cap = _max_warmup_input_len(tm, 2)
    ext_max = min(max_size, ext_cap) if ext_cap is not None else max_size
    plen = min(64, max(0, ext_max - 2))
    prefix = rids(plen) if plen >= 4 else []
    for tail in (7, 2, 7, 21, 96) if prefix else ():
        if plen + tail <= ext_max:
            await send(prefix + rids(tail), {**sampled, "max_new_tokens": 2})

    seqs = [rids(s) for s in (3, 7, 21, 48) if s <= max_size]
    seqs += [
        list(prefix) if (s == plen and prefix) else rids(s)
        for s in _prefill_shape_sizes(max_size)
    ]
    for ids in (
        tqdm.tqdm(seqs, desc=f"Warmup prefill shapes (up to {max_size})")
        if seqs
        else ()
    ):
        await send(ids, {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True})

    flush_fn = getattr(tm, "flush_cache", None)
    if mode == "null" and getattr(tm, "is_generation", True) and callable(flush_fn):
        try:
            res = flush_fn()
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.debug("Post-warmup flush_cache skipped", exc_info=True)


async def _warm_local_worker(tm: Any, app_state: Optional[Any]) -> None:
    if getattr(tm, "skip_tokenizer_init", False):
        return
    dyn_tok, tok = (
        getattr(tm, "dynamic_batch_tokenizer", None),
        getattr(tm, "tokenizer", None),
    )
    try:
        if dyn_tok is not None:
            await dyn_tok.encode("Warmup dynamic batch tokenizer.")
        elif tok is not None and hasattr(tok, "encode"):
            await asyncio.to_thread(tok.encode, "Warmup tokenizer.")
    except Exception:
        logger.debug("Tokenizer warmup skipped", exc_info=True)
    if tok is not None:
        try:
            from sglang.srt.entrypoints.openai.utils import token_id_to_bytes

            await asyncio.to_thread(token_id_to_bytes, tok, 0)
        except Exception:
            logger.debug("Byte-level cache warmup skipped", exc_info=True)
    chat = getattr(app_state, "openai_serving_chat", None) if app_state else None
    if chat is not None and getattr(tm, "is_generation", True):
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
            await asyncio.to_thread(chat._convert_to_internal_request, req, None)
        except Exception:
            logger.debug("Chat template warmup skipped", exc_info=True)


@warmup("all")
async def warmup_all(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    info = _multi_worker_info(tokenizer_manager) if fcntl is not None else None
    if info is None:
        await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
    else:
        with open(f"{info[0]}_all.lock", "a+", encoding="utf-8") as f:
            await asyncio.to_thread(fcntl.flock, f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            if not f.read():
                await _run_all_gpu_warmups(disaggregation_mode, tokenizer_manager)
                f.write("1")
                f.flush()
    await _warm_local_worker(tokenizer_manager, _current_app_state.get())
