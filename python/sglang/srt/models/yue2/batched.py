# SPDX-License-Identifier: Apache-2.0
"""Batched semantic-phase AR decoding for YuE2.

Independent songs share one :class:`~.cuda_graph.GraphAR` decode graph whose
``branches`` dimension doubles as the batch dimension: each row owns its prefix,
KV slots, RoPE positions, and stop decision. 

The AR expert weights are loaded to GPU memory once at server start, and then the 
per-step GEMMs then **stream those same weights from HBM once per decode step**, 
and batching amortises that bandwidth across all rows.

ABC planning is still generated per request (it is short); only the expensive
semantic codec phase is batched. The NAR/VAE stages stay per request and read
each row's KV through a per-row branch index.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import torch

from .cuda_graph import GraphSessionPool
from .fastpath import MAX_GRAPH_TOPK, BatchedGraphSampler
from .protocol import ABC_END, MUSIC_END, Sampling, token_prefixes
from .sampling import generate_tokens, synchronize

_logger = logging.getLogger(__name__)

# Dedicated pool: batched sessions are keyed by (branches, capacity) and never
# shared with the single-request / CFG pool.
default_batched_pool = GraphSessionPool(max_sessions_per_key=int(
    os.environ.get("SGLANG_YUE2_BATCHED_POOL", "2")))


def batched_ar_enabled() -> bool:
    return os.environ.get("SGLANG_YUE2_BATCHED_AR", "1") == "1"


class BatchedUnsupported(Exception):
    """A request mix that cannot use the batched semantic AR path."""


class BatchedSessionLease:
    """Refcounted lease over one pooled batched session.

    The AR stage takes one lease for ``refs`` member requests; every request's
    NAR stage releases it, and the session returns to the pool after the last.
    """

    def __init__(self, session, refs: int, pool=default_batched_pool):
        self.session = session
        self.pool = pool
        self._refs = int(refs)
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            self._refs -= 1
            if self._refs > 0 or self._released:
                return
            self._released = True
        self.pool.release(self.session)


@torch.inference_mode()
def generate_semantic_batched(
    model,
    session,
    prefixes: list[list[int]],
    semantic_sampling: Sampling,
    *,
    seed: int = 831001,
    cancelled=None,
    on_token=None,
):
    """Run the semantic codec phase for ``B`` rows on one pooled session.

    ``prefixes`` are the full semantic prefixes (base + ABC + separators). The
    session is left open with each row's KV resident; the caller owns it via a
    :class:`BatchedSessionLease`.

    Returns ``(codec_ids, fed_codec, truncated, timing)`` where all outer lists
    are per row. ``codec_ids`` are raw (CODEC_OFFSET-based) codec tokens;
    ``fed_codec`` counts the tokens already decoded into the session KV.
    """
    prefixes = [list(prefix) for prefix in prefixes]
    B = len(prefixes)
    if B < 2:
        raise ValueError("generate_semantic_batched requires at least two rows")
    device = session.device

    sampler = getattr(session, "sampler", None)
    if not isinstance(sampler, BatchedGraphSampler) or sampler.B != B:
        if session.graph is not None and not session.step_mode:
            raise RuntimeError(
                "pooled session was captured without a batched sampler")
        sampler = BatchedGraphSampler(
            model.config.vocab_size, device,
            torch.Generator(device=device).manual_seed(int(seed)),
            B, max_topk=MAX_GRAPH_TOPK)
        session.sampler = sampler
        session.step_mode = True
    else:
        sampler.generator.manual_seed(int(seed))
        if sampler._k < max(1, int(semantic_sampling.top_k)):
            raise RuntimeError("pooled batched sampler top-k budget exceeded")

    budget = int(semantic_sampling.max_tokens) + 2
    session.reopen(prefixes, budget)
    synchronize(device)
    start = time.perf_counter()
    logits = session.prefill()
    prefill_seconds = time.perf_counter() - start

    min_tokens = int(semantic_sampling.min_tokens)
    sampler.configure_semantic(
        temperature=semantic_sampling.temperature,
        top_p=semantic_sampling.top_p,
        top_k=semantic_sampling.top_k,
        repetition_penalty=semantic_sampling.repetition_penalty,
        penalty_window=semantic_sampling.penalty_window,
        end=MUSIC_END,
        min_tokens=min_tokens,
    )

    # Sample t0 from the prefill logits; the in-graph transcript records every
    # later token, so the host only syncs for the periodic completion check.
    sampler.sample_in_graph(logits)
    sampler.advance_in_graph()

    blocked = min_tokens > 0
    max_steps = int(semantic_sampling.max_tokens)
    replay = 0
    finished = False
    while replay < max_steps:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during batched semantic decode")
        replay += 1
        if blocked and replay >= min_tokens:
            sampler.block_end(False)
            blocked = False
        session.graph.replay()
        if on_token is not None:
            tokens = [int(t) for t in sampler.token_buf[:, 0].tolist()]
            stops = sampler.stop_step.tolist()
            for row in range(B):
                if stops[row] < 0:
                    on_token("semantic", tokens[row])
        if replay % 8 == 0:
            finished = bool((sampler.stop_step >= 0).all().item())
            if finished:
                # One flush replay decodes each row's latched end token into
                # its KV slot before the NAR stage borrows the cache.
                if replay < max_steps:
                    session.graph.replay()
                    replay += 1
                break

    # A row that hit the budget (not finished) has no trailing MUSIC_END in its
    # KV. Force the end token for every row and decode the pending token plus
    # MUSIC_END, matching the single-request tail feed so the NAR stage sees a
    # complete ``prefix + codec + MUSIC_END`` cache instead of a truncated one.
    if not finished:
        sampler.mask.fill_(float("-inf"))
        sampler.mask[:, MUSIC_END] = 0.0
        session.graph.replay()
        session.graph.replay()

    stop = [int(value) for value in sampler.stop_step.tolist()]
    transcript = sampler.token_history[:, : replay + 2].cpu().tolist()
    codec: list[list[int]] = []
    fed: list[int] = []
    truncated: list[bool] = []
    for row in range(B):
        if stop[row] >= 0:
            codec.append([int(token) for token in transcript[row][: stop[row]]])
            fed.append(stop[row])
            truncated.append(False)
        else:
            length = replay + 1
            codec.append([int(token) for token in transcript[row][:length]])
            fed.append(replay)
            truncated.append(True)

    timing = {
        "seconds": time.perf_counter() - start,
        "prefill_seconds": prefill_seconds,
        "output_tokens": [len(tokens_) for tokens_ in codec],
        "content_tokens": [len(tokens_) for tokens_ in codec],
        "truncated": truncated,
        "execution": "cuda_graph",
        "attention": session.attention_backend,
        "prefix_tokens": [len(prefix) for prefix in prefixes],
        "cfg_branches": B,
        "sampler": "deepselect_batched_step_graph",
        "batch_size": B,
    }
    return codec, fed, truncated, timing


@torch.inference_mode()
def generate_abc_batched(model, session, prefixes, abc_sampling, *, seed=831001,
                         cancelled=None):
    """Run the ABC planning phase for ``B`` rows on one pooled session.

    Returns ``(abc_ids, truncated)`` per row. The session is only used to plan;
    its KV is discarded after (the semantic phase re-prefills the full prefix).
    """
    prefixes = [list(prefix) for prefix in prefixes]
    B = len(prefixes)
    if B < 2:
        raise ValueError("generate_abc_batched requires at least two rows")
    device = session.device

    sampler = getattr(session, "sampler", None)
    if not isinstance(sampler, BatchedGraphSampler) or sampler.B != B:
        if session.graph is not None and not session.step_mode:
            raise RuntimeError("pooled session was captured without a batched sampler")
        sampler = BatchedGraphSampler(
            model.config.vocab_size, device,
            torch.Generator(device=device).manual_seed(int(seed)),
            B, max_topk=MAX_GRAPH_TOPK)
        session.sampler = sampler
        session.step_mode = True
    else:
        sampler.generator.manual_seed(int(seed))

    session.reopen(prefixes, int(abc_sampling.max_tokens) + 2)
    synchronize(device)
    logits = session.prefill()

    min_tokens = int(abc_sampling.min_tokens)
    sampler.configure_abc(
        temperature=abc_sampling.temperature, top_p=abc_sampling.top_p,
        top_k=abc_sampling.top_k, repetition_penalty=abc_sampling.repetition_penalty,
        penalty_window=abc_sampling.penalty_window, end=ABC_END, min_tokens=min_tokens)
    sampler.sample_in_graph(logits)
    sampler.advance_in_graph()

    blocked = min_tokens > 0
    max_steps = int(abc_sampling.max_tokens)
    replay = 0
    while replay < max_steps:
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during batched ABC decode")
        replay += 1
        if blocked and replay >= min_tokens:
            sampler.block_end(False)
            blocked = False
        session.graph.replay()
        if replay % 8 == 0 and bool((sampler.stop_step >= 0).all().item()):
            break

    stop = [int(value) for value in sampler.stop_step.tolist()]
    transcript = sampler.token_history[:, : replay + 1].cpu().tolist()
    abc: list[list[int]] = []
    truncated: list[bool] = []
    for row in range(B):
        if stop[row] >= 0:
            abc.append([int(token) for token in transcript[row][: stop[row]]])
            truncated.append(False)
        else:
            abc.append([int(token) for token in transcript[row][: replay + 1]])
            truncated.append(True)
    return abc, truncated


def _abc_per_request(model, tokenizer, requests, configs, batches):
    """Per-request ABC fallback (looped path) used when batched ABC fails."""
    abc_ids_list, abc_truncated_list = [], []
    for request, config, batch in zip(requests, configs, batches):
        prefix = list(batch.extra["yue2_prefix"])
        abc_ids, _timing, abc_truncated = generate_tokens(
            model, prefix, config.abc, request.seed, phase="abc",
            cfg_scale=1.0, use_cuda_graph=True, use_fast_sampler=True)
        abc_ids_list.append(list(abc_ids))
        abc_truncated_list.append(bool(abc_truncated))
    return abc_ids_list, abc_truncated_list


@torch.inference_mode()
def run_batched_ar(model, tokenizer, batches, generation_config):
    """Generate ABC, then the semantic phase batched on one session.

    Shared by the native and plugin AR stages; both write the same
    ``batch.extra`` keys and expose API-compatible request/tokenizer objects.
    Raises :class:`BatchedUnsupported` for a mix the batched path cannot serve.
    """
    requests = [batch.extra["yue2_request"] for batch in batches]
    configs = [generation_config(batch.sampling_params) for batch in batches]

    for request in requests:
        scale = request.cfg_scale if request.cfg_scale is not None else request.guidance
        if scale != 1.0:
            raise BatchedUnsupported("cfg_scale != 1")
        if request.abc is not None:
            raise BatchedUnsupported("external ABC")
        if request.cot == "off":
            raise BatchedUnsupported("cot=off uses legacy arithmetic")

    semantic = configs[0].semantic
    signature = (semantic.temperature, semantic.top_p, semantic.top_k,
                 semantic.repetition_penalty, semantic.penalty_window,
                 semantic.min_tokens, semantic.max_tokens)
    for config in configs[1:]:
        other = config.semantic
        if signature != (other.temperature, other.top_p, other.top_k,
                         other.repetition_penalty, other.penalty_window,
                         other.min_tokens, other.max_tokens):
            raise BatchedUnsupported("semantic sampling params differ")

    base_prefixes = [list(batch.extra["yue2_prefix"]) for batch in batches]
    started = time.perf_counter()
    abc_sampling = configs[0].abc
    abc_started = time.perf_counter()
    try:
        abc_session = default_batched_pool.acquire(
            model, base_prefixes, int(abc_sampling.max_tokens) + 4)
        try:
            abc_ids_list, abc_truncated_list = generate_abc_batched(
                model, abc_session, base_prefixes, abc_sampling,
                seed=int(requests[0].seed))
        finally:
            default_batched_pool.release(abc_session)
    except Exception:
        _logger.exception("Batched ABC failed; falling back to per-request ABC")
        abc_ids_list, abc_truncated_list = _abc_per_request(
            model, tokenizer, requests, configs, batches)
    abc_seconds = time.perf_counter() - abc_started
    semantic_prefixes, abc_records = [], []
    for index, request in enumerate(requests):
        abc_ids = abc_ids_list[index]
        abc = tokenizer.decode(abc_ids) if abc_ids else None
        semantic_prefixes.append(token_prefixes(request, tokenizer, abc_ids=abc_ids))
        abc_records.append((abc, list(abc_ids), {
            "seconds": abc_seconds, "prefill_seconds": 0.0,
            "output_tokens": len(abc_ids), "content_tokens": len(abc_ids),
            "truncated": bool(abc_truncated_list[index]), "execution": "cuda_graph",
            "attention": "batch", "sampler": "deepselect_batched_abc",
        }, bool(abc_truncated_list[index])))

    budget = int(semantic.max_tokens) + 4
    session = default_batched_pool.acquire(model, semantic_prefixes, budget)
    try:
        codec, fed, truncated, semantic_timing = generate_semantic_batched(
            model, session, semantic_prefixes, semantic, seed=int(requests[0].seed))
    except Exception:
        default_batched_pool.release(session)
        raise

    lease = BatchedSessionLease(session, refs=len(batches))
    for index, batch in enumerate(batches):
        abc, abc_ids, abc_timing, abc_truncated = abc_records[index]
        batch.extra.update(
            {
                "yue2_abc": abc,
                "yue2_abc_ids": abc_ids,
                "yue2_semantic_ids": list(codec[index]),
                "yue2_semantic_prefix": semantic_prefixes[index],
                "yue2_session": session,
                "yue2_session_branch": index,
                "yue2_session_lease": lease,
                "yue2_batched": True,
                "yue2_nar_skip_feed": True,
                "yue2_batched_truncated": bool(truncated[index]),
                "yue2_fed_codec": int(fed[index]),
                "yue2_truncated": bool(abc_truncated or truncated[index]),
                "yue2_timing": {
                    "abc": abc_timing,
                    "semantic": semantic_timing,
                    "total_seconds": time.perf_counter() - started,
                },
            }
        )
    _logger.info(
        "Batched AR: %d rows in %.2fs (abc=%.2fs prefill=%.2fs semantic=%.2fs)",
        len(batches), time.perf_counter() - started, abc_seconds,
        semantic_timing.get("prefill_seconds", 0.0), semantic_timing["seconds"])
    return batches
