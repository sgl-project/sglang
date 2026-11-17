from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import multiprocessing
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import tqdm

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

_warmup_registry: Dict[str, Any] = {}
# Warmups that only send generate requests, which Rust-server mode runs through
# its listener (`rust_server.warmups`); the others need the TokenizerManager.
_rust_server_warmups = set()

_current_app_state: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_current_app_state", default=None
)


def warmup(
    name: str,
    *,
    rust_server: bool = False,
    once_per_server: bool = True,
    flush_cache_after: bool = False,
):
    """Register a startup warmup routine.

    Args:
        name: Registry name passed via ``--warmups``.
        rust_server: True if the warmup only sends plain ``GenerateReqInput``
            requests supported by ``sglang.srt.rust_server.warmups``.
        once_per_server: When ``--tokenizer-worker-num > 1``, run this warmup
            only once across all worker processes (peer workers wait on a lock
            and skip redundant GPU work). Set False for worker-local CPU/Jinja
            /async-tokenizer warmups that must run in every worker process.
        flush_cache_after: Flush RadixCache once the warmup finishes so
            synthetic warmup tokens do not pollute or fragment runtime KV cache.
    """

    def decorator(fn):
        fn._once_per_server = once_per_server
        fn._flush_cache_after = flush_cache_after
        _warmup_registry[name] = fn
        if rust_server:
            _rust_server_warmups.add(name)
        else:
            _rust_server_warmups.discard(name)
        return fn

    return decorator


def _multi_worker_Coordination_key(
    tokenizer_manager: Any,
) -> Optional[tuple[str, str]]:
    """Return ``(lock_prefix, instance_token)`` when running inside a
    multi-tokenizer worker subprocess, else ``None``."""
    if fcntl is None:
        return None
    parent_pid = getattr(multiprocessing.current_process(), "_parent_pid", None)
    if parent_pid is None and getattr(tokenizer_manager, "tokenizer_ipc_name", None):
        parent_pid = os.getppid()
    if parent_pid is None:
        return None

    server_args = getattr(tokenizer_manager, "server_args", None)
    worker_num = getattr(server_args, "tokenizer_worker_num", 1)
    port = getattr(server_args, "port", 0)
    if not isinstance(worker_num, int) or worker_num <= 1:
        return None

    startup_time = getattr(tokenizer_manager, "startup_time", None)
    raw_token = f"{parent_pid}:{port}:{startup_time!r}"
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:16]
    prefix = os.path.join(tempfile.gettempdir(), f"sglang_warmup_{parent_pid}_{port}")
    return prefix, token_hash


async def _maybe_flush_warmup_cache(
    disaggregation_mode: str, tokenizer_manager: Any
) -> None:
    """Best-effort KV cache flush after synthetic token warmups."""
    if disaggregation_mode != "null":
        return
    if not getattr(tokenizer_manager, "is_generation", True):
        return
    flush_fn = getattr(tokenizer_manager, "flush_cache", None)
    if not callable(flush_fn):
        return
    try:
        res = flush_fn()
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        logger.debug("Post-warmup flush_cache skipped", exc_info=True)


async def _execute_single_warmup(
    warmup_name: str,
    disaggregation_mode: str,
    tokenizer_manager: TokenizerManager,
    *,
    defer_cache_flush: bool = False,
) -> bool:
    """Execute one registered warmup, honoring ``once_per_server`` coordination.

    Returns True if ``flush_cache_after`` is requested and was deferred.
    """
    fn = _warmup_registry[warmup_name]
    once_per_server = getattr(fn, "_once_per_server", True)
    flush_cache_after = getattr(fn, "_flush_cache_after", False)

    coord = (
        _multi_worker_Coordination_key(tokenizer_manager) if once_per_server else None
    )
    if coord is None:
        logger.info(f"Running warmup {warmup_name}")
        await fn(disaggregation_mode, tokenizer_manager)
        if flush_cache_after and not defer_cache_flush:
            await _maybe_flush_warmup_cache(disaggregation_mode, tokenizer_manager)
            return False
        return flush_cache_after

    prefix, instance_token = coord
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", warmup_name)
    lock_path = f"{prefix}_{safe_name}.lock"
    done_path = f"{prefix}_{safe_name}.done"

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        already_done = False
        if os.path.exists(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as f:
                    already_done = f.read().strip() == instance_token
            except OSError:
                already_done = False

        if already_done:
            logger.info(
                "Skipping warmup %s in worker %d (already completed by peer worker)",
                warmup_name,
                os.getpid(),
            )
            return False

        logger.info(f"Running warmup {warmup_name} (worker {os.getpid()})")
        await fn(disaggregation_mode, tokenizer_manager)
        if flush_cache_after and not defer_cache_flush:
            await _maybe_flush_warmup_cache(disaggregation_mode, tokenizer_manager)
            needs_flush = False
        else:
            needs_flush = flush_cache_after

        tmp_done = f"{done_path}.{os.getpid()}.tmp"
        with open(tmp_done, "w", encoding="utf-8") as f:
            f.write(instance_token)
        os.replace(tmp_done, done_path)
        return needs_flush
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
    app_state: Optional[Any] = None,
):
    names = [n.strip() for n in warmup_names if n and n.strip()]
    if not names:
        return

    token = _current_app_state.set(app_state)
    pending_flush = False
    try:
        for idx, warmup_name in enumerate(names):
            if warmup_name not in _warmup_registry:
                logger.warning(f"Could not find custom warmup {warmup_name}")
                continue
            is_last = idx == len(names) - 1
            needs_flush = await _execute_single_warmup(
                warmup_name,
                disaggregation_mode,
                tokenizer_manager,
                defer_cache_flush=not is_last,
            )
            pending_flush = pending_flush or needs_flush
        if pending_flush:
            await _maybe_flush_warmup_cache(disaggregation_mode, tokenizer_manager)
    finally:
        _current_app_state.reset(token)


def _safe_vocab_high(tokenizer_manager: Any) -> int:
    model_cfg = getattr(tokenizer_manager, "model_config", None)
    vocab_size = getattr(model_cfg, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 32:
        return max(32, min(vocab_size - 16, 2**15))
    return 1024


def _rand_ids(rng: np.random.Generator, size: int, high: int) -> List[int]:
    return [int(x) for x in rng.integers(16, high, size=size)]


async def _drain_generate(
    disaggregation_mode: str,
    tokenizer_manager: TokenizerManager,
    req: GenerateReqInput,
    *,
    room_id: int = 0,
) -> None:
    if disaggregation_mode != "null":
        req.bootstrap_room = room_id
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    async for _ in tokenizer_manager.generate_request(req, None):
        pass


@warmup("whisper_autodetect", once_per_server=True, flush_cache_after=False)
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
        await _drain_generate(disaggregation_mode, tokenizer_manager, req)
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat", rust_server=True, once_per_server=True, flush_cache_after=True)
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # Warms up the fused_moe triton kernels across fine-grained seq_lens.
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(17)
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=_rand_ids(rng, size, high),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        await _drain_generate(
            disaggregation_mode, tokenizer_manager, generate_req_input
        )


@warmup("sampling", rust_server=True, once_per_server=True, flush_cache_after=True)
async def sampling(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warm greedy, top-k/top-p, dense top-p, min-p, and mixed-batch decode paths."""
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(42)

    single_cases = [
        (16, {"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True}),
        (
            21,
            {
                "max_new_tokens": 16,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "ignore_eos": True,
            },
        ),
        (
            19,
            {
                "max_new_tokens": 4,
                "temperature": 0.8,
                "top_p": 0.9,
                "ignore_eos": True,
            },
        ),
        (
            24,
            {
                "max_new_tokens": 4,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "min_p": 0.05,
                "ignore_eos": True,
            },
        ),
    ]
    for size, params in single_cases:
        req = GenerateReqInput(
            input_ids=_rand_ids(rng, size, high),
            sampling_params=params,
        )
        await _drain_generate(disaggregation_mode, tokenizer_manager, req)

    # Concurrent mixed greedy + non-greedy batch (warms mixed SamplingBatchInfo,
    # _small_copy_kernel tuple specializations, and bs=2 continuous decode).
    req_greedy = GenerateReqInput(
        input_ids=_rand_ids(rng, 17, high),
        sampling_params={"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True},
    )
    req_sample = GenerateReqInput(
        input_ids=_rand_ids(rng, 23, high),
        sampling_params={
            "max_new_tokens": 4,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "ignore_eos": True,
        },
    )
    await asyncio.gather(
        _drain_generate(disaggregation_mode, tokenizer_manager, req_greedy, room_id=0),
        _drain_generate(disaggregation_mode, tokenizer_manager, req_sample, room_id=1),
    )


@warmup("penalties", rust_server=True, once_per_server=True, flush_cache_after=True)
async def penalties(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warm repetition, frequency, and presence penalizers for bs=1 and bs=2."""
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(43)
    penalty_params = {
        "max_new_tokens": 4,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "ignore_eos": True,
    }

    # bs=1
    req_single = GenerateReqInput(
        input_ids=_rand_ids(rng, 21, high),
        sampling_params=dict(penalty_params),
    )
    await _drain_generate(disaggregation_mode, tokenizer_manager, req_single)

    # bs=2 concurrent batch (warms dynamic batch inductor specialization & merge/filter)
    req_a = GenerateReqInput(
        input_ids=_rand_ids(rng, 19, high),
        sampling_params=dict(penalty_params),
    )
    req_b = GenerateReqInput(
        input_ids=_rand_ids(rng, 27, high),
        sampling_params=dict(penalty_params),
    )
    await asyncio.gather(
        _drain_generate(disaggregation_mode, tokenizer_manager, req_a, room_id=0),
        _drain_generate(disaggregation_mode, tokenizer_manager, req_b, room_id=1),
    )


@warmup("logprobs", rust_server=False, once_per_server=True, flush_cache_after=True)
async def logprobs(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warm output top-logprobs (greedy & non-greedy) and prompt input-logprobs."""
    if not getattr(tokenizer_manager, "is_generation", True):
        return
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(44)

    # 1. Greedy output logprobs
    await _drain_generate(
        disaggregation_mode,
        tokenizer_manager,
        GenerateReqInput(
            input_ids=_rand_ids(rng, 16, high),
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=5,
            return_text_in_logprobs=True,
            log_metrics=False,
        ),
    )

    # 2. Non-greedy output logprobs
    await _drain_generate(
        disaggregation_mode,
        tokenizer_manager,
        GenerateReqInput(
            input_ids=_rand_ids(rng, 21, high),
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=5,
            return_text_in_logprobs=True,
            log_metrics=False,
        ),
    )

    # 3. Prompt / scoring logprobs (_row_logsumexp_topk_kernel + IPC serialization)
    if disaggregation_mode != "decode":
        await _drain_generate(
            disaggregation_mode,
            tokenizer_manager,
            GenerateReqInput(
                input_ids=_rand_ids(rng, 25, high),
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=5,
                return_text_in_logprobs=True,
                log_metrics=False,
            ),
        )


@warmup(
    "structured_output",
    rust_server=True,
    once_per_server=True,
    flush_cache_after=True,
)
async def structured_output(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Warm constrained decoding (xgrammar / llguidance FSM + token bitmask kernel)."""
    if getattr(tokenizer_manager, "skip_tokenizer_init", False):
        logger.info("Skipping structured_output warmup (skip_tokenizer_init=True).")
        return
    server_args = getattr(tokenizer_manager, "server_args", None)
    if getattr(server_args, "grammar_backend", None) == "none":
        logger.info("Skipping structured_output warmup (grammar_backend='none').")
        return

    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(45)

    schema_req = GenerateReqInput(
        input_ids=_rand_ids(rng, 16, high),
        sampling_params={
            "max_new_tokens": 8,
            "temperature": 0.0,
            "json_schema": (
                '{"type":"object","properties":{"safe":{"type":"boolean"}},'
                '"required":["safe"]}'
            ),
        },
    )
    regex_req = GenerateReqInput(
        input_ids=_rand_ids(rng, 16, high),
        sampling_params={
            "max_new_tokens": 4,
            "temperature": 0.0,
            "regex": r"(safe|unsafe)",
        },
    )
    await _drain_generate(disaggregation_mode, tokenizer_manager, schema_req)
    await _drain_generate(disaggregation_mode, tokenizer_manager, regex_req)


@warmup("radix_prefix", rust_server=True, once_per_server=True, flush_cache_after=True)
async def radix_prefix(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warm RadixCache prefix-hit paged extend + cascade merge_state across tile buckets
    and odd KV-slot pointer alignments."""
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(46)
    prefix = _rand_ids(rng, 64, high)

    # Seed 64-token prefix + odd 7-token tail so RadixCache caches the prefix and
    # the KV pool allocator advances to an odd (8B-aligned, non-16B-aligned) slot.
    seed_req = GenerateReqInput(
        input_ids=prefix + _rand_ids(rng, 7, high),
        sampling_params={
            "max_new_tokens": 2,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "ignore_eos": True,
        },
    )
    await _drain_generate(disaggregation_mode, tokenizer_manager, seed_req)

    # Exercise cached_tokens > 0 with <4 eager fallback, cta_tile_q=16, 64, and 128.
    for tail_len in (2, 7, 21, 96):
        req = GenerateReqInput(
            input_ids=prefix + _rand_ids(rng, tail_len, high),
            sampling_params={
                "max_new_tokens": 2,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "ignore_eos": True,
            },
        )
        await _drain_generate(disaggregation_mode, tokenizer_manager, req)


def _resolve_prefill_warmup_max_size(tokenizer_manager: Any) -> int:
    env_val = os.environ.get("SGLANG_WARMUP_MAX_PREFILL_SIZE")
    max_size = int(env_val) if env_val and env_val.isdigit() else 32768

    max_req_input_len = getattr(tokenizer_manager, "max_req_input_len", None)
    if isinstance(max_req_input_len, int) and max_req_input_len >= 64:
        # Leave headroom for generated tokens.
        max_size = min(max_size, max(64, max_req_input_len - 16))
    return max(64, max_size)


@warmup(
    "prefill_shapes", rust_server=True, once_per_server=True, flush_cache_after=True
)
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup kernels across short, odd-aligned, prefix-cached, and long prefill shapes.

    Covers:
    1. Short & odd lengths ``[3, 7, 21, 48]`` (eager ``<4`` fallback, FlashInfer
       ``cta_tile_q=16`` & ``64``, odd KV pointer alignment, non-greedy sampling,
       and repetition penalty).
    2. Power-of-2 + midpoint lengths ``[64 .. max_size]`` (up to 32K or
       ``max_req_input_len``).
    3. Prefix-cached ``extend`` lengths reusing the 64-token prefix with suffixes
       ``[7, 21, 96]`` (``forward_return_lse`` + ``prefill_wrapper_paged`` +
       ``flashinfer.cascade.merge_state``).
    """
    high = _safe_vocab_high(tokenizer_manager)
    rng = np.random.default_rng(47)
    max_size = _resolve_prefill_warmup_max_size(tokenizer_manager)

    # 1. Short & odd prefill shapes + non-greedy / min-p / penalty sampling
    # 1. Short & odd prefill shapes (3, 7, 21, 48)
    for size in (3, 7, 21, 48):
        if size <= max_size:
            req = GenerateReqInput(
                input_ids=_rand_ids(rng, size, high),
                sampling_params={
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
            await _drain_generate(disaggregation_mode, tokenizer_manager, req)

    # 2. Power-of-2 + midpoint prefill shapes (64 .. max_size)
    page_size = 64
    sizes = set()
    base = 64
    while base <= max_size:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= max_size:
            sizes.add(mid)
        base *= 2
    sorted_sizes = sorted(sizes)

    shared_prefix_64 = _rand_ids(rng, 64, high)
    for size in tqdm.tqdm(
        sorted_sizes, desc=f"Warmup prefill shapes (up to {max_size})"
    ):
        input_ids = list(shared_prefix_64) if size == 64 else _rand_ids(rng, size, high)
        generate_req_input = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        await _drain_generate(
            disaggregation_mode, tokenizer_manager, generate_req_input
        )

    # 3. Prefix-cached extend shapes across cta_tile_q = 16, 64, 128
    for tail_len in (7, 21, 96):
        if 64 + tail_len <= max_size:
            extend_req = GenerateReqInput(
                input_ids=shared_prefix_64 + _rand_ids(rng, tail_len, high),
                sampling_params={
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            )
            await _drain_generate(disaggregation_mode, tokenizer_manager, extend_req)


async def warm_local_http_worker(
    tokenizer_manager: Any, app_state: Optional[Any] = None
) -> None:
    """Warm worker-local CPU/asyncio state in the current HTTP/tokenizer worker.

    Initializes ``AsyncDynamicbatchTokenizer`` background loop, HuggingFace fast
    tokenizer, and ``OpenAIServingChat`` Jinja2 template / Pydantic schemas prior
    to ``freeze_gc``.
    """
    if getattr(tokenizer_manager, "skip_tokenizer_init", False):
        return

    # 1. Warm dynamic batch tokenizer queue & background task in this worker.
    dyn_tok = getattr(tokenizer_manager, "dynamic_batch_tokenizer", None)
    if dyn_tok is not None:
        try:
            await dyn_tok.encode("Warmup dynamic batch tokenizer.")
        except Exception:
            logger.debug("Dynamic batch tokenizer warmup skipped", exc_info=True)
    else:
        tok = getattr(tokenizer_manager, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            try:
                await asyncio.to_thread(tok.encode, "Warmup tokenizer.")
            except Exception:
                logger.debug("Tokenizer encode warmup skipped", exc_info=True)

    # 2. Warm OpenAI logprobs byte-level BPE lookup cache before freeze_gc.
    tok = getattr(tokenizer_manager, "tokenizer", None)
    if tok is not None:
        try:
            from sglang.srt.entrypoints.openai.utils import token_id_to_bytes

            await asyncio.to_thread(token_id_to_bytes, tok, 0)
        except Exception:
            logger.debug(
                "Tokenizer byte-level logprob cache warmup skipped", exc_info=True
            )

    # 3. Warm OpenAIServingChat Jinja2 chat template & Pydantic v2 conversion.
    serving_chat = (
        getattr(app_state, "openai_serving_chat", None)
        if app_state is not None
        else None
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


@warmup(
    "tokenizer_worker",
    rust_server=False,
    once_per_server=False,
    flush_cache_after=False,
)
async def tokenizer_worker(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Per-worker warmup for dynamic batch tokenizer and OpenAI chat templates."""
    del disaggregation_mode
    await warm_local_http_worker(tokenizer_manager, _current_app_state.get())


_FULL_SUITE_STAGES = (
    "sampling",
    "penalties",
    "logprobs",
    "structured_output",
    "radix_prefix",
    "prefill_shapes",
    "tokenizer_worker",
)


@warmup("all", rust_server=False, once_per_server=False, flush_cache_after=False)
async def warmup_all(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Run the complete server warmup suite (sampling, penalties, logprobs,
    structured_output, radix_prefix, prefill_shapes up to 32K, and per-worker
    tokenizer/chat-template initialization)."""
    await execute_warmups(
        disaggregation_mode,
        list(_FULL_SUITE_STAGES),
        tokenizer_manager,
        app_state=_current_app_state.get(),
    )
