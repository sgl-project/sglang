from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
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


# Long enough for a step graph capture (two regular steps, then the capture) at
# any accept length.
_STEP_GRAPH_WARMUP_NEW_TOKENS = 64


def default_chat_sampling_params(model_config, max_new_tokens: int) -> Dict[str, Any]:
    """The sampling params of a chat completion request that sets no sampling
    field (the model's generation config, then the OpenAI defaults), resolved by
    the chat API's own code. The tokenizer manager merges
    ``--preferred-sampling-params`` into them as into any request."""
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": ""}],
        max_completion_tokens=max_new_tokens,
        ignore_eos=True,
    )
    return request.to_sampling_params(
        stop=[], model_generation_config=model_config.get_default_sampling_params()
    )


def _active_step_cuda_graph_stats(
    internal_states: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(scheduler state, runner stats) of every scheduler that built a step CUDA
    graph runner that did not turn itself off."""
    active = []
    for state in internal_states:
        stats = state.get("spec_step_cuda_graph")
        if stats is not None and stats["disabled_reason"] is None:
            active.append((state, stats))
    return active


def _step_graph_warmup_batch_sizes(internal_states: List[Dict[str, Any]]) -> List[int]:
    """Batch sizes 1 to N: N is the smallest, over the active runners, of the
    runner's largest batch size and its scheduler's maximum number of running
    requests. Empty without an active runner."""
    limits = [
        min(
            stats.get("max_batch_size", 1),
            state.get("effective_max_running_requests_per_dp") or 1,
        )
        for state, stats in _active_step_cuda_graph_stats(internal_states)
    ]
    return list(range(1, min(limits) + 1)) if limits else []


def _missing_batch_sizes(
    internal_states: List[Dict[str, Any]],
    batch_sizes: List[int],
    sampling_flags: Dict[str, bool],
) -> List[int]:
    """Sizes missing an unconstrained key for the warmup's sampling mode."""
    missing = set()
    for _, stats in _active_step_cuda_graph_stats(internal_states):
        captured = {
            key["batch_size"]
            for key in stats.get("captured_keys", ())
            if not key["constrained"]
            and all(key[name] == value for name, value in sampling_flags.items())
        }
        missing.update(bs for bs in batch_sizes if bs not in captured)
    return sorted(missing)


async def _run_step_graph_warmup_batch(
    tokenizer_manager: TokenizerManager,
    num_requests: int,
    sampling_params: Dict[str, Any],
):
    """Run ``num_requests`` short requests that decode as one batch: the
    tokenizer manager sends a batch of token ids to the scheduler in one
    message, so they are prefilled together. Distinct prompts keep the
    in-batch prefix caching from holding any of them back."""
    input_ids = [[10 + i, 11, 12] for i in range(num_requests)]
    req = GenerateReqInput(
        input_ids=input_ids if num_requests > 1 else input_ids[0],
        sampling_params=dict(sampling_params),
    )
    # Drain the generator so every decode step runs before the next batch.
    async for _ in tokenizer_manager.generate_request(req, None):
        pass


def _log_step_graph_warmup(
    internal_states: List[Dict[str, Any]],
    batch_sizes: List[int],
    seconds: float,
    sampling_flags: Dict[str, bool],
):
    active = _active_step_cuda_graph_stats(internal_states)
    for index, (_, stats) in enumerate(active):
        keys = stats.get("captured_keys", [])
        logger.info(
            "Speculative step CUDA graph warm-up done in %.1f s (scheduler %d of "
            "%d): %d keys, %.1f MB per GPU: %s.",
            seconds,
            index + 1,
            len(active),
            len(keys),
            sum(key["memory_mb"] for key in keys),
            ", ".join(
                f"batch size {key['batch_size']} "
                f"{'greedy' if key['greedy'] else 'sampled'} {key['memory_mb']:.1f} MB"
                for key in keys
            ),
        )
    missing = _missing_batch_sizes(internal_states, batch_sizes, sampling_flags)
    if missing:
        logger.warning(
            "Speculative step CUDA graph warm-up: no key captured for batch "
            "size(s) %s with the default sampling; they are captured at first use.",
            missing,
        )


async def speculative_step_cuda_graph_warmup(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Capture the speculative step CUDA graphs of decode steps of batch size 1
    to N with the server's default sampling, before the server takes requests.

    The step graph captures a key at its third eligible step, and only while
    enough device memory is free. SGLang's own warm-up request is greedy, so
    without this the keys of client requests that send no sampling fields are
    first met while serving, when device memory in use may have grown. Batch
    size k comes from k requests sent together (k per DP rank), after the
    smaller sizes; a size still missing afterwards gets one more batch."""
    from sglang.srt.runtime_context import get_parallel

    if disaggregation_mode != "null":
        return
    batch_sizes = _step_graph_warmup_batch_sizes(
        await tokenizer_manager.get_internal_state()
    )
    if not batch_sizes:
        return
    sampling_params = default_chat_sampling_params(
        tokenizer_manager.model_config, _STEP_GRAPH_WARMUP_NEW_TOKENS
    )
    effective_params = SamplingParams(
        **{**(tokenizer_manager.preferred_sampling_params or {}), **sampling_params}
    )
    sampling_flags = {
        "greedy": effective_params.top_k <= 1,
        "need_top_k_sampling": effective_params.top_k != TOP_K_ALL,
        "need_top_p_sampling": effective_params.top_p != 1.0,
        "need_min_p_sampling": effective_params.min_p > 0.0,
    }
    logger.info(
        "Speculative step CUDA graph warm-up with the default sampling: %s; "
        "batch sizes 1-%d.",
        {k: sampling_params[k] for k in ("temperature", "top_p", "top_k", "min_p")},
        batch_sizes[-1],
    )
    num_dp_ranks = get_parallel().num_dp_ranks
    start = time.perf_counter()
    for bs in batch_sizes:
        await _run_step_graph_warmup_batch(
            tokenizer_manager, bs * num_dp_ranks, sampling_params
        )
    states = await tokenizer_manager.get_internal_state()
    missing = _missing_batch_sizes(states, batch_sizes, sampling_flags)
    for bs in missing:
        await _run_step_graph_warmup_batch(
            tokenizer_manager, bs * num_dp_ranks, sampling_params
        )
    if missing:
        states = await tokenizer_manager.get_internal_state()
    _log_step_graph_warmup(
        states, batch_sizes, time.perf_counter() - start, sampling_flags
    )


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


@warmup("voice_chat")
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


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
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
