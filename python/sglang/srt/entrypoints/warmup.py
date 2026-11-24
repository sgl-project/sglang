from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import tqdm

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}
# Warmups that only send generate requests, which Rust-server mode runs through
# its listener (`rust_server.warmups`); the others need the TokenizerManager.
_rust_server_warmups = set()


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
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


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


_DEFAULT_MAX_PREFILL_SIZE = 32768
# Short fixed prompt length used for sampling, logprob, and grammar warmup requests.
_WARMUP_PROMPT_LEN = 32
# Shared prefix length and suffix lengths used to trigger radix-cache extend paths.
_PREFIX_LEN = 64
_PREFIX_EXTEND_SUFFIX_LENS = (16, 64)
# Extra small prefill lengths below the 64-token page size used by prefill_shapes.
_SHORT_PREFILL_SIZES = (16, 32)


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _max_warmup_input_len(tm: Any, max_new_tokens: int = 1) -> Optional[int]:
    """Return the largest input token length accepted by tokenizer and scheduler."""
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

    # Scheduler rejects requests with `len(input_ids) >= max_req_input_len`.
    max_req_input_len = _positive_int(getattr(tm, "max_req_input_len", None))
    if max_req_input_len is not None:
        limits.append(max_req_input_len - 1)

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
    page_size = 64
    sizes = set()
    base = page_size
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
    input_limit = _max_warmup_input_len(tokenizer_manager, max_new_tokens=1)
    max_size = (
        min(_DEFAULT_MAX_PREFILL_SIZE, input_limit)
        if input_limit is not None
        else _DEFAULT_MAX_PREFILL_SIZE
    )
    sizes = _prefill_shape_sizes(max_size)

    for size in tqdm.tqdm(sizes, desc=f"Warmup prefill shapes (up to {max_size})"):
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
        max_size = min(max_size, max_input_len)
    return max(0, max_size)


async def _send_warmup_request(
    disaggregation_mode: str,
    tm: TokenizerManager,
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    bootstrap_room: int = 0,
    **kwargs: Any,
) -> None:
    """Send one warmup request, clamping length to stay within model context bounds."""
    params = dict(sampling_params)
    max_new_tokens = max(1, int(params.get("max_new_tokens", 1) or 1))
    max_input = _max_warmup_input_len(tm, max_new_tokens)
    if max_input is not None and max_input < 1:
        params["max_new_tokens"] = 1
        max_input = _max_warmup_input_len(tm, max_new_tokens=1)
    if max_input is not None:
        if max_input < 1:
            return
        input_ids = input_ids[:max_input]

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


async def _warm_tokenizer(tm: Any) -> None:
    """Warm tokenizer encoding, byte-level BPE table, and Jinja chat template."""
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

    if tokenizer is None:
        return

    try:
        from sglang.srt.entrypoints.openai.utils import token_id_to_bytes

        await asyncio.to_thread(token_id_to_bytes, tokenizer, 0)
    except Exception:
        logger.debug("Byte-level BPE cache warmup skipped", exc_info=True)

    if getattr(tokenizer, "chat_template", None) and hasattr(
        tokenizer, "apply_chat_template"
    ):
        try:
            await asyncio.to_thread(
                tokenizer.apply_chat_template,
                [{"role": "user", "content": "Warmup."}],
                tokenize=False,
            )
        except Exception:
            logger.debug("Chat template warmup skipped", exc_info=True)


@warmup("all")
async def warmup_all(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """End-to-end warmup covering sampling, penalties, logprobs, grammar, radix
    prefix extend, prefill shapes, and tokenizer initialization.
    """
    if getattr(tokenizer_manager, "is_generation", True):
        vocab_size = _positive_int(
            getattr(
                getattr(tokenizer_manager, "model_config", None), "vocab_size", None
            )
        )
        max_token_id = min(vocab_size, 2**15) if vocab_size is not None else 2**15
        rng = np.random.default_rng(42)

        def rand_ids(length: int) -> List[int]:
            return [int(x) for x in rng.integers(0, max_token_id, size=length)]

        def send(
            input_ids: List[int],
            sampling_params: Dict[str, Any],
            bootstrap_room: int = 0,
            **kwargs: Any,
        ):
            return _send_warmup_request(
                disaggregation_mode,
                tokenizer_manager,
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
        min_p = {
            **sampled,
            "temperature": 0.7,
            "top_k": 50,
            "min_p": 0.05,
        }
        penalty = {
            **sampled,
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
        }

        # 1. Sampling, decode batching (bs=2), and penalty paths.
        for params in (greedy, sampled, min_p, penalty):
            await send(rand_ids(_WARMUP_PROMPT_LEN), params)

        max_input_len = _max_warmup_input_len(tokenizer_manager, max_new_tokens=1)
        if max_input_len is None or max_input_len >= _WARMUP_PROMPT_LEN:
            await asyncio.gather(
                send(rand_ids(_WARMUP_PROMPT_LEN), greedy, bootstrap_room=0),
                send(rand_ids(_WARMUP_PROMPT_LEN), sampled, bootstrap_room=1),
            )
            await asyncio.gather(
                send(rand_ids(_WARMUP_PROMPT_LEN), penalty, bootstrap_room=0),
                send(rand_ids(_WARMUP_PROMPT_LEN), penalty, bootstrap_room=1),
            )

        # 2. Output and prompt logprob paths.
        top_logprobs = min(5, vocab_size) if vocab_size is not None else 5
        logprob_kwargs = {
            "return_logprob": True,
            "top_logprobs_num": top_logprobs,
            "return_text_in_logprobs": True,
            "log_metrics": False,
        }
        await send(rand_ids(_WARMUP_PROMPT_LEN), greedy, **logprob_kwargs)
        await send(rand_ids(_WARMUP_PROMPT_LEN), sampled, **logprob_kwargs)
        if disaggregation_mode != "decode":
            await send(
                rand_ids(_WARMUP_PROMPT_LEN),
                {"max_new_tokens": 1, "temperature": 0.0},
                logprob_start_len=0,
                **logprob_kwargs,
            )

        # 3. Constrained decoding (JSON schema and regex FSMs).
        grammar_backend = getattr(
            getattr(tokenizer_manager, "server_args", None), "grammar_backend", None
        )
        if (
            not getattr(tokenizer_manager, "skip_tokenizer_init", False)
            and grammar_backend != "none"
        ):
            schema = '{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}'
            await send(
                rand_ids(_WARMUP_PROMPT_LEN),
                {"max_new_tokens": 8, "temperature": 0.0, "json_schema": schema},
            )
            await send(
                rand_ids(_WARMUP_PROMPT_LEN),
                {"max_new_tokens": 4, "temperature": 0.0, "regex": r"(yes|no)"},
            )

        # 4. Radix prefix-cache extend and prefill shapes up to context limit.
        max_prefill_size = _resolve_max_prefill_size(max_input_len)
        extend_cap = _max_warmup_input_len(tokenizer_manager, max_new_tokens=2)
        extend_max = (
            min(max_prefill_size, extend_cap)
            if extend_cap is not None
            else max_prefill_size
        )
        if extend_max >= _PREFIX_LEN + min(_PREFIX_EXTEND_SUFFIX_LENS):
            prefix = rand_ids(_PREFIX_LEN)
            for suffix_len in _PREFIX_EXTEND_SUFFIX_LENS:
                if _PREFIX_LEN + suffix_len <= extend_max:
                    await send(
                        prefix + rand_ids(suffix_len),
                        {**sampled, "max_new_tokens": 2},
                    )

        prefill_sizes = [
            size for size in _SHORT_PREFILL_SIZES if size <= max_prefill_size
        ] + _prefill_shape_sizes(max_prefill_size)
        if prefill_sizes:
            for size in tqdm.tqdm(
                prefill_sizes,
                desc=f"Warmup prefill shapes (up to {max_prefill_size})",
            ):
                await send(
                    rand_ids(size),
                    {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
                )

        # 5. Flush synthetic warmup entries from the radix KV cache.
        flush_fn = getattr(tokenizer_manager, "flush_cache", None)
        if disaggregation_mode == "null" and callable(flush_fn):
            try:
                res = flush_fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.debug("Post-warmup flush_cache skipped", exc_info=True)

    await _warm_tokenizer(tokenizer_manager)
